use std::collections::HashSet;
use std::sync::Mutex;

use lash_core::plugin::project_observation_text;
use lash_core::{
    AttachmentRef, ExecImage, ModeExecutionContext, ModeToolReply, TextProjectionMetadata,
    ToolOutputBudgetConfig,
};
use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ProcessBlockStart,
    ProjectedFuture, Record as FlowRecord, Value as FlowValue,
};
use serde_json::Value;

use crate::projection::{flow_record_to_tool_args, flow_to_json_value, format_output_value};

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

    async fn tool_payload(&self, tool_name: &str, args: &FlowRecord) -> Value {
        let policy = self.ctx.tool_argument_projection_policy(tool_name);
        let mut payload = flow_record_to_tool_args(args, &policy).await;
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| Value::String(self.ctx.session_id().to_string()));
        }
        payload
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
    async fn call(&self, name: String, args: FlowRecord) -> Result<FlowValue, ExecutionHostError> {
        let index = self.next_index();
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                name.clone(),
                self.tool_payload(&name, &args).await,
                index,
            )
            .await;
        self.consume_reply(&name, reply)
    }

    async fn start_call(
        &self,
        name: String,
        args: FlowRecord,
    ) -> Result<FlowValue, ExecutionHostError> {
        let reply = self
            .ctx
            .start_tool_call(
                uuid::Uuid::new_v4().to_string(),
                name.clone(),
                self.tool_payload(&name, &args).await,
            )
            .await;
        self.consume_reply(&name, reply)
    }

    async fn start_process(
        &self,
        start: ProcessBlockStart,
    ) -> Result<FlowValue, ExecutionHostError> {
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
            AbilityOp::CallTool { name, args } => {
                self.call(name, args).await.map(AbilityResult::Value)
            }
            AbilityOp::StartToolCall { name, args } => {
                self.start_call(name, args).await.map(AbilityResult::Value)
            }
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
                "process events are only available inside lashlang process blocks",
            )),
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
            FlowValue::Null | FlowValue::Bool(_) | FlowValue::Number(_) | FlowValue::String(_) => {}
        }
        Ok(())
    })
}
