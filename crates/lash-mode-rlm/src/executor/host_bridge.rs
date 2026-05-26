use std::collections::HashSet;
use std::sync::Mutex;

use lash_core::{
    AttachmentRef, ExecImage, ModeExecutionContext, ModeToolReply, TextProjectionMetadata,
};
use lash_plugin_tool_output_budget::{ToolOutputBudgetConfig, project_observation_text};
use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ProcessStart, ProjectedFuture,
    Record as FlowRecord, Value as FlowValue,
};
use serde_json::Value;

use crate::projection::{flow_to_json_value, format_output_value};

pub(super) struct HostBridge<'run> {
    ctx: ModeExecutionContext<'run>,
    observe_projection: ToolOutputBudgetConfig,
    tool_result_projectors: Vec<crate::RlmToolResultProjector>,
    observations: Mutex<Vec<String>>,
    observation_truncation: Mutex<Vec<TextProjectionMetadata>>,
    printed_images: Mutex<Vec<AttachmentRef>>,
    tool_calls: Mutex<Vec<lash_core::ToolCallRecord>>,
    tool_images: Mutex<Vec<ExecImage>>,
    next_tool_index: Mutex<usize>,
}

impl<'run> HostBridge<'run> {
    pub(super) fn new(
        ctx: ModeExecutionContext<'run>,
        observe_projection: ToolOutputBudgetConfig,
        tool_result_projectors: Vec<crate::RlmToolResultProjector>,
    ) -> Self {
        Self {
            ctx,
            observe_projection,
            tool_result_projectors,
            observations: Mutex::new(Vec::new()),
            observation_truncation: Mutex::new(Vec::new()),
            printed_images: Mutex::new(Vec::new()),
            tool_calls: Mutex::new(Vec::new()),
            tool_images: Mutex::new(Vec::new()),
            next_tool_index: Mutex::new(0),
        }
    }

    fn next_index(&self) -> usize {
        let mut guard = self
            .next_tool_index
            .lock()
            .expect("tool index lock poisoned");
        let next = *guard;
        *guard += 1;
        next
    }

    fn consume_reply(
        &self,
        tool_name: &str,
        reply: ModeToolReply,
    ) -> Result<FlowValue, ExecutionHostError> {
        let projected_tool_name = reply
            .record
            .as_ref()
            .map(|record| record.tool.as_str())
            .unwrap_or(tool_name)
            .to_string();
        if let Some(record) = reply.record {
            self.tool_calls
                .lock()
                .map_err(|_| ExecutionHostError::new("tool call buffer poisoned"))?
                .push(record);
        }
        if reply.output.is_success() {
            let value = reply.output.value_for_projection();
            for projector in &self.tool_result_projectors {
                if let Some(value) = projector(&projected_tool_name, &value) {
                    return Ok(value);
                }
            }
            lash_core::lashlang_bridge::mode_tool_output_to_lashlang_value(&reply.output)
        } else {
            lash_core::lashlang_bridge::mode_tool_output_to_lashlang_value(&reply.output)
        }
    }

    pub(super) fn into_collected(self) -> CollectedExecutionOutput {
        CollectedExecutionOutput {
            observations: self.observations.into_inner().unwrap_or_default(),
            observation_truncation: self.observation_truncation.into_inner().unwrap_or_default(),
            printed_images: self.printed_images.into_inner().unwrap_or_default(),
            tool_calls: self.tool_calls.into_inner().unwrap_or_default(),
            tool_images: self.tool_images.into_inner().unwrap_or_default(),
        }
    }
}

impl HostBridge<'_> {
    async fn resource_operation(
        &self,
        operation: String,
        receiver: FlowValue,
        args: Vec<FlowValue>,
    ) -> Result<FlowValue, ExecutionHostError> {
        let resource = match &receiver {
            FlowValue::Resource(resource) => resource,
            _ => {
                return Err(ExecutionHostError::new(format!(
                    "resource operation `{operation}` requires a resource receiver"
                )));
            }
        };
        if resource.resource_type != "TOOL" || resource.alias != "default" {
            return Err(ExecutionHostError::new(format!(
                "resource `{}`.`{}` is not executable in this host",
                resource.resource_type, resource.alias
            )));
        }
        let index = self.next_index();
        let mut payload = if let [FlowValue::Record(record)] = args.as_slice() {
            flow_record_json(record).await
        } else {
            serde_json::json!({
                "args": flow_values_to_json(&args).await,
            })
        };
        payload.as_object_mut().ok_or_else(|| {
            ExecutionHostError::new("resource operation payload must be an object")
        })?;
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                operation.clone(),
                payload,
                index,
            )
            .await;
        self.consume_reply(&operation, reply)
    }

    async fn start_process(&self, start: ProcessStart) -> Result<FlowValue, ExecutionHostError> {
        let (registration, label) = self
            .ctx
            .prepare_lashlang_process_start(start)
            .map_err(ExecutionHostError::new)?;
        let reply = self.ctx.start_lashlang_process(registration, label).await;
        self.consume_reply("start_process", reply)
    }

    async fn await_handle(&self, handle: FlowValue) -> Result<FlowValue, ExecutionHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                handle_to_json(&handle).await?,
            )
            .await;
        self.consume_reply("await_handle", reply)
    }

    async fn cancel_handle(&self, handle: FlowValue) -> Result<FlowValue, ExecutionHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                handle_to_json(&handle).await?,
            )
            .await;
        self.consume_reply("cancel_handle", reply)
    }

    async fn print(&self, value: FlowValue) -> Result<(), ExecutionHostError> {
        let attachment_store = self.ctx.attachment_store();
        let images = collect_printed_images(&value, attachment_store.as_ref()).await?;
        let raw_text = format_output_value(&value).await;
        let (_projected_text, metadata) =
            project_observation_text(&raw_text, &self.observe_projection);
        self.observations
            .lock()
            .map_err(|_| ExecutionHostError::new("observation buffer poisoned"))?
            .push(raw_text);
        self.observation_truncation
            .lock()
            .map_err(|_| ExecutionHostError::new("observation metadata buffer poisoned"))?
            .push(metadata);
        if !images.is_empty() {
            self.printed_images
                .lock()
                .map_err(|_| ExecutionHostError::new("printed image buffer poisoned"))?
                .extend(images);
        }
        Ok(())
    }
}

impl ExecutionHost for HostBridge<'_> {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => self
                .resource_operation(operation.operation, operation.receiver, operation.args)
                .await
                .map(AbilityResult::Value),
            AbilityOp::StartProcess(start) => {
                self.start_process(start).await.map(AbilityResult::Value)
            }
            AbilityOp::Await(handle) => self.await_handle(handle).await.map(AbilityResult::Value),
            AbilityOp::Cancel(handle) => self.cancel_handle(handle).await.map(AbilityResult::Value),
            AbilityOp::Print(value) => {
                self.print(value).await?;
                Ok(AbilityResult::Unit)
            }
            AbilityOp::ProcessEvent(_) => Err(ExecutionHostError::new(
                "process events are only available inside lashlang process bodies",
            )),
            AbilityOp::ProcessSleep(_) | AbilityOp::WaitSignal | AbilityOp::SignalRun(_) => {
                Err(ExecutionHostError::new(
                    "process lifecycle primitives are only available inside lashlang process bodies",
                ))
            }
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
        }
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }
}

async fn handle_to_json(value: &FlowValue) -> Result<Value, ExecutionHostError> {
    match value {
        FlowValue::Projected(_) => Ok(flow_to_json_value(value).await),
        _ => lash_core::lashlang_bridge::lashlang_value_to_json(value),
    }
}

fn flow_values_to_json<'a>(values: &'a [FlowValue]) -> ProjectedFuture<'a, Vec<Value>> {
    Box::pin(async move {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(flow_to_json_value(value).await);
        }
        out
    })
}

fn flow_record_json<'a>(record: &'a FlowRecord) -> ProjectedFuture<'a, Value> {
    Box::pin(async move {
        let mut object = serde_json::Map::with_capacity(record.len());
        for (key, value) in record.iter() {
            object.insert(key.to_string(), flow_to_json_value(value).await);
        }
        Value::Object(object)
    })
}

pub(super) struct CollectedExecutionOutput {
    pub(super) observations: Vec<String>,
    pub(super) observation_truncation: Vec<TextProjectionMetadata>,
    pub(super) printed_images: Vec<AttachmentRef>,
    pub(super) tool_calls: Vec<lash_core::ToolCallRecord>,
    pub(super) tool_images: Vec<ExecImage>,
}

async fn collect_printed_images(
    value: &FlowValue,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> Result<Vec<AttachmentRef>, ExecutionHostError> {
    let mut seen = HashSet::new();
    let mut images = Vec::new();
    collect_printed_images_inner(value, attachment_store, &mut seen, &mut images).await?;
    Ok(images)
}

fn collect_printed_images_inner<'a>(
    value: &'a FlowValue,
    attachment_store: &'a dyn lash_core::AttachmentStore,
    seen: &'a mut HashSet<String>,
    images: &'a mut Vec<AttachmentRef>,
) -> ProjectedFuture<'a, Result<(), ExecutionHostError>> {
    Box::pin(async move {
        match value {
            FlowValue::Image(image) => {
                if !seen.insert(image.id.clone()) {
                    return Ok(());
                }
                let reference = attachment_store
                    .get(&lash_core::AttachmentId::new(image.id.clone()))
                    .ok()
                    .map(|stored| stored.meta.as_ref())
                    .ok_or_else(|| {
                        ExecutionHostError::new(format!(
                            "image bytes for `{}` are unavailable or were pruned",
                            image.id
                        ))
                    })?;
                images.push(reference);
            }
            FlowValue::List(values) => {
                for value in values.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Record(record) => {
                for (_, value) in record.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Projected(value) => {
                collect_printed_images_inner(
                    &value.materialize_async().await,
                    attachment_store,
                    seen,
                    images,
                )
                .await?;
            }
            FlowValue::Null
            | FlowValue::Bool(_)
            | FlowValue::Number(_)
            | FlowValue::String(_)
            | FlowValue::Resource(_) => {}
        }
        Ok(())
    })
}
