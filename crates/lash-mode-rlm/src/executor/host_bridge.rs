use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use lash_core::plugin::project_observation_text;
use lash_core::{
    AttachmentRef, ExecImage, ModeExecutionContext, ModeToolBatchItem, ModeToolReply,
    TextProjectionMetadata, ToolOutputBudgetConfig,
};
use lashlang::{
    ImageValue, ProjectedFuture, Record as FlowRecord, ToolHost, ToolHostCall, ToolHostError,
    Value as FlowValue,
};
use serde_json::Value;

use crate::projection_codec::{
    flow_record_to_tool_args, flow_to_json_value, format_output_value, json_to_flow_value,
};

pub(super) struct HostBridge {
    ctx: ModeExecutionContext,
    observe_projection: ToolOutputBudgetConfig,
    tool_result_projectors: Vec<crate::RlmToolResultProjector>,
    observations: Mutex<Vec<String>>,
    observation_truncation: Mutex<Vec<TextProjectionMetadata>>,
    printed_images: Mutex<Vec<AttachmentRef>>,
    tool_calls: Mutex<Vec<lash_core::ToolCallRecord>>,
    tool_images: Mutex<Vec<ExecImage>>,
    next_tool_index: Mutex<usize>,
}

impl HostBridge {
    pub(super) fn new(
        ctx: ModeExecutionContext,
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
    ) -> Result<FlowValue, ToolHostError> {
        let projected_tool_name = reply
            .record
            .as_ref()
            .map(|record| record.tool.as_str())
            .unwrap_or(tool_name)
            .to_string();
        if let Some(record) = reply.record {
            self.tool_calls
                .lock()
                .map_err(|_| ToolHostError::new("tool call buffer poisoned"))?
                .push(record);
        }
        if reply.output.is_success() {
            let value = reply.output.value_for_projection();
            for projector in &self.tool_result_projectors {
                if let Some(value) = projector(&projected_tool_name, &value) {
                    return Ok(value);
                }
            }
            Ok(lift_tool_result_to_flow_value(
                value,
                Vec::new(),
                self.ctx.attachment_store().as_ref(),
            ))
        } else {
            Err(ToolHostError::new(tool_error_message(
                reply.output.value_for_projection(),
            )))
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

impl ToolHost for HostBridge {
    async fn call(&self, name: String, args: FlowRecord) -> Result<FlowValue, ToolHostError> {
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

    async fn call_batch(&self, calls: Vec<ToolHostCall>) -> Vec<Result<FlowValue, ToolHostError>> {
        if calls.is_empty() {
            return Vec::new();
        }
        let mut batch = Vec::with_capacity(calls.len());
        for call in &calls {
            batch.push(ModeToolBatchItem {
                id: uuid::Uuid::new_v4().to_string(),
                name: call.name.clone(),
                args: self.tool_payload(&call.name, &call.args).await,
            });
        }
        let replies = self.ctx.call_tool_batch(batch).await;
        if replies.len() != calls.len() {
            return calls
                .into_iter()
                .map(|_| {
                    Err(ToolHostError::new(
                        "tool batch returned the wrong number of results",
                    ))
                })
                .collect();
        }
        replies
            .into_iter()
            .zip(calls)
            .map(|(reply, call)| self.consume_reply(&call.name, reply))
            .collect()
    }

    async fn start_call(&self, name: String, args: FlowRecord) -> Result<FlowValue, ToolHostError> {
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

    async fn await_handle(&self, handle: FlowValue) -> Result<FlowValue, ToolHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                flow_to_json_value(&handle).await,
            )
            .await;
        self.consume_reply("await_handle", reply)
    }

    async fn cancel_handle(&self, handle: FlowValue) -> Result<FlowValue, ToolHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                flow_to_json_value(&handle).await,
            )
            .await;
        self.consume_reply("cancel_handle", reply)
    }

    async fn print(&self, value: FlowValue) -> Result<(), ToolHostError> {
        let attachment_store = self.ctx.attachment_store();
        let images = collect_printed_images(&value, attachment_store.as_ref()).await?;
        let raw_text = format_output_value(&value).await;
        let (_projected_text, metadata) =
            project_observation_text(&raw_text, &self.observe_projection);
        self.observations
            .lock()
            .map_err(|_| ToolHostError::new("observation buffer poisoned"))?
            .push(raw_text);
        self.observation_truncation
            .lock()
            .map_err(|_| ToolHostError::new("observation metadata buffer poisoned"))?
            .push(metadata);
        if !images.is_empty() {
            self.printed_images
                .lock()
                .map_err(|_| ToolHostError::new("printed image buffer poisoned"))?
                .extend(images);
        }
        Ok(())
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
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
) -> Result<Vec<AttachmentRef>, ToolHostError> {
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
) -> ProjectedFuture<'a, Result<(), ToolHostError>> {
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
                        ToolHostError::new(format!(
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

fn lift_tool_result_to_flow_value(
    result: Value,
    tool_images: Vec<ExecImage>,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> FlowValue {
    if tool_images.is_empty() {
        return json_to_flow_value(result);
    }

    let image_values = tool_images
        .into_iter()
        .filter_map(|image| register_tool_image(image, attachment_store).ok())
        .map(FlowValue::Image)
        .collect::<Vec<_>>();

    if image_values.is_empty() {
        return json_to_flow_value(result);
    }

    if is_image_only_tool_payload(&result) {
        return if image_values.len() == 1 {
            image_values.into_iter().next().unwrap_or(FlowValue::Null)
        } else {
            FlowValue::List(image_values.into())
        };
    }

    let mut value = json_to_flow_value(result);
    let images_value = FlowValue::List(image_values.into());
    match &mut value {
        FlowValue::Record(record) => {
            Arc::make_mut(record).insert("images".to_string(), images_value);
            value
        }
        _ => {
            let mut record = FlowRecord::new();
            record.insert("payload".to_string(), value);
            record.insert("images".to_string(), images_value);
            FlowValue::Record(Arc::new(record))
        }
    }
}

fn register_tool_image(
    mut image: ExecImage,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> Result<ImageValue, lash_core::AttachmentStoreError> {
    let reference = if let Some(reference) = image.reference.take() {
        reference
    } else if let Some(media_type) = lash_core::MediaType::from_mime(&image.mime) {
        let meta = lash_core::AttachmentCreateMeta::new(
            media_type,
            image.width,
            image.height,
            Some(image.label.clone()),
        );
        attachment_store.put(std::mem::take(&mut image.data), meta)?
    } else {
        let meta = lash_core::AttachmentCreateMeta::new(
            lash_core::MediaType::Image(lash_core::ImageMediaType::Png),
            image.width,
            image.height,
            Some(image.label.clone()),
        );
        attachment_store.put(std::mem::take(&mut image.data), meta)?
    };
    Ok(ImageValue::new(
        reference.id.to_string(),
        reference.label.clone().unwrap_or_default(),
        reference.byte_len,
        reference.width,
        reference.height,
    ))
}

fn is_image_only_tool_payload(result: &Value) -> bool {
    match result {
        Value::String(text) => text.trim_start().starts_with("[Image:"),
        Value::Null => true,
        Value::Array(values) => values.is_empty(),
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

fn tool_error_message(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "tool call failed".to_string()),
    }
}
