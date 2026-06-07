use std::collections::BTreeSet;
use std::sync::Arc;

use lash_core::{ChronologicalPayload, Message, MessageRole, PartKind, RuntimeExecutionContext};
use lash_rlm_types::{
    RlmAttachmentRef, RlmHistoryItem, RlmHistoryRole, RlmImageRef, RlmProtocolEvent,
    RlmTrajectoryEntry,
};
use lashlang::{
    ProjectedBindings, ProjectedFuture, ProjectedHostValue, ProjectedReadRequest,
    ProjectedReadResponse, ProjectedValue, State as FlowState, Value as FlowValue,
};

use super::bindings::{
    ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectedBindings, RlmProjectionExtension,
};
use super::transport::json_to_flow_value;

pub fn rlm_protocol_event(event: RlmProtocolEvent) -> lash_core::ProtocolEvent {
    lash_core::ProtocolEvent::typed(crate::plugin::RLM_PROTOCOL_PLUGIN_ID, event)
        .expect("RLM protocol events serialize")
}

pub fn decode_rlm_protocol_event(event: &lash_core::ProtocolEvent) -> Option<RlmProtocolEvent> {
    event
        .decode(crate::plugin::RLM_PROTOCOL_PLUGIN_ID)
        .ok()
        .flatten()
}

#[derive(Clone, Debug)]
pub struct RlmHistoryProjection {
    history: Vec<RlmHistoryItem>,
}

impl RlmHistoryProjection {
    pub fn from_chronological(projection: &lash_core::ChronologicalProjection) -> Self {
        let mut history = Vec::with_capacity(projection.entries().len());
        for entry in projection.entries() {
            match &entry.payload {
                ChronologicalPayload::Message(message) => {
                    if let Some(item) = history_item_from_message(message) {
                        history.push(item);
                    }
                }
                ChronologicalPayload::ProtocolEvent(event) => {
                    if let Some(RlmProtocolEvent::RlmTrajectoryEntry(step)) =
                        decode_rlm_protocol_event(event)
                    {
                        history.push(history_item_from_rlm_step(&step));
                    }
                }
            }
        }
        Self { history }
    }

    pub fn history(&self) -> &[RlmHistoryItem] {
        self.history.as_slice()
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    pub fn item(&self, index: usize) -> Option<RlmHistoryItem> {
        self.history.get(index).cloned()
    }

    pub fn value(&self) -> serde_json::Value {
        serde_json::to_value(&self.history).unwrap_or_else(|_| serde_json::Value::Array(vec![]))
    }
}

pub fn rlm_history_projection(
    projection: &lash_core::ChronologicalProjection,
) -> RlmHistoryProjection {
    RlmHistoryProjection::from_chronological(projection)
}

pub(crate) async fn projected_bindings(
    ctx: &RuntimeExecutionContext<'_>,
    session_bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<ProjectedBindings, String> {
    let mut bindings = ProjectedBindings::new();
    bindings
        .try_insert(
            "history",
            ProjectedValue::custom(
                "history",
                Arc::new(HistoryProjectedValue {
                    projection: Arc::new(rlm_history_projection(
                        ctx.chronological_projection().as_ref(),
                    )),
                }),
            ),
        )
        .map_err(|err| format!("`{}` is reserved as an RLM built-in binding", err.name()))?;
    insert_projected_bindings(
        &mut bindings,
        session_bindings,
        Arc::clone(&projection_resolver),
    )
    .await?;
    if let Some(extension) = ctx
        .turn_context()
        .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
    {
        insert_projected_bindings(
            &mut bindings,
            extension.bindings.clone(),
            projection_resolver,
        )
        .await?;
    }
    Ok(bindings)
}

async fn insert_projected_bindings(
    target: &mut ProjectedBindings,
    bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<(), String> {
    let host_bindings = bindings
        .into_projected_bindings(projection_resolver)
        .await
        .map_err(|err| err.to_string())?;
    for name in host_bindings.names().collect::<Vec<_>>() {
        let value = host_bindings
            .get(&name)
            .expect("name came from projected bindings");
        target.try_insert(name, value).map_err(|err| {
            format!(
                "`{}` is already bound as an RLM projected binding",
                err.name()
            )
        })?;
    }
    Ok(())
}

struct HistoryProjectedValue {
    projection: Arc<RlmHistoryProjection>,
}

impl ProjectedHostValue for HistoryProjectedValue {
    fn type_name(&self) -> &str {
        "list"
    }

    fn read_one(
        &self,
        request: ProjectedReadRequest,
    ) -> ProjectedFuture<'_, ProjectedReadResponse> {
        Box::pin(async move {
            match request {
                ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.projection.len()),
                ProjectedReadRequest::Index(index) => {
                    let Ok(Some(index)) = projected_index(&index, self.projection.len()) else {
                        return ProjectedReadResponse::Missing;
                    };
                    self.projection
                        .item(index)
                        .and_then(|item| serde_json::to_value(item).ok())
                        .map(json_to_flow_value)
                        .map(ProjectedReadResponse::Value)
                        .unwrap_or(ProjectedReadResponse::Missing)
                }
                ProjectedReadRequest::Render => ProjectedReadResponse::Text(
                    serde_json::to_string(self.projection.history())
                        .unwrap_or_else(|_| "[]".to_string()),
                ),
                ProjectedReadRequest::Materialize => {
                    ProjectedReadResponse::Value(json_to_flow_value(self.projection.value()))
                }
                _ => ProjectedReadResponse::Missing,
            }
        })
    }
}

pub(crate) fn projected_index(index: &FlowValue, len: usize) -> Result<Option<usize>, ()> {
    let FlowValue::Number(index) = index else {
        return Err(());
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return Err(());
    }
    let len = len as isize;
    let index = *index as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

pub(crate) fn prune_reserved_projected_bindings(rlm: &mut FlowState) {
    prune_protected_bindings(rlm, &BTreeSet::new());
}

pub(crate) fn prune_protected_bindings(rlm: &mut FlowState, protected_names: &BTreeSet<String>) {
    prune_projected_binding_names(
        rlm,
        std::iter::once("history").chain(protected_names.iter().map(String::as_str)),
    );
}

pub(crate) fn prune_projected_binding_names<'a>(
    rlm: &mut FlowState,
    names: impl IntoIterator<Item = &'a str>,
) {
    let mut snapshot = rlm.snapshot();
    for key in names {
        snapshot.globals.remove(key);
    }
    *rlm = FlowState::from_snapshot(snapshot);
}

fn history_item_from_message(message: &Message) -> Option<RlmHistoryItem> {
    Some(RlmHistoryItem::Message {
        id: message.id.clone(),
        role: history_role(message.role),
        content: message_history_text(message),
        attachments: message
            .parts
            .iter()
            .filter_map(|part| {
                let attachment = part.attachment.as_ref()?;
                Some(RlmAttachmentRef {
                    id: part.id.clone(),
                    media_type: attachment.reference.media_type,
                    label: attachment.reference.label.clone(),
                    reference: attachment.reference.id.to_string(),
                })
            })
            .collect(),
    })
}

fn history_item_from_rlm_step(entry: &RlmTrajectoryEntry) -> RlmHistoryItem {
    RlmHistoryItem::RlmStep {
        id: entry.id.clone(),
        protocol_iteration: entry.protocol_iteration,
        reasoning: entry.reasoning.clone(),
        code: entry.code.clone(),
        output: entry.output.clone(),
        images: entry.images.iter().map(image_ref).collect(),
        error: entry.error.clone(),
        final_output: entry.final_output.clone(),
    }
}

fn message_history_text(message: &Message) -> String {
    let chunks = message
        .parts
        .iter()
        .filter(|part| matches!(part.kind, PartKind::Text | PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    chunks.join("\n\n")
}

fn history_role(role: MessageRole) -> RlmHistoryRole {
    match role {
        MessageRole::User => RlmHistoryRole::User,
        MessageRole::System => RlmHistoryRole::System,
        MessageRole::Assistant => RlmHistoryRole::Assistant,
        MessageRole::Event => RlmHistoryRole::Event,
    }
}

fn image_ref(image: &lash_core::AttachmentRef) -> RlmImageRef {
    RlmImageRef {
        id: image.id.to_string(),
        media_type: image.media_type,
        width: image.width,
        height: image.height,
        bytes: image.byte_len as usize,
        label: image.label.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step_projection(output: &str) -> lash_core::ChronologicalProjection {
        let entry = RlmTrajectoryEntry {
            id: "rlm_step_0".to_string(),
            protocol_iteration: 0,
            reasoning: "thinking".to_string(),
            code: "print big".to_string(),
            output: vec![output.to_string()],
            images: Vec::new(),
            error: None,
            final_output: None,
        };
        let events = [lash_core::SessionEventRecord::Protocol(rlm_protocol_event(
            RlmProtocolEvent::RlmTrajectoryEntry(entry),
        ))];
        lash_core::ChronologicalProjection::from_turn_view(
            &events,
            &lash_core::MessageSequence::default(),
        )
    }

    async fn read_index(value: &HistoryProjectedValue, index: i64) -> FlowValue {
        match value
            .read_one(ProjectedReadRequest::Index(FlowValue::Number(index as f64)))
            .await
        {
            ProjectedReadResponse::Value(value) => value,
            other => panic!("expected indexed value, got {other:?}"),
        }
    }

    // The history renderer advertises a `full: history[N].output[M]` reference
    // for truncated outputs (see `truncated_ref` in driver/history.rs). This
    // proves the contract resolves: indexing the real `history` projection
    // hands back the FULL, untruncated value the prompt only previewed.
    #[tokio::test]
    async fn history_step_output_resolves_full_untruncated_value() {
        let full = "X".repeat(50_000);
        let projection = step_projection(&full);
        let value = HistoryProjectedValue {
            projection: Arc::new(rlm_history_projection(&projection)),
        };

        // history[0] -> the serialized RLM step record.
        let FlowValue::Record(step) = read_index(&value, 0).await else {
            panic!("history[0] should be a record");
        };
        // history[0].output -> the list of per-print outputs.
        let Some(FlowValue::List(outputs)) = step.get("output") else {
            panic!("step record should carry an `output` list, got {step:?}");
        };
        // history[0].output[0] -> the full untruncated string.
        let Some(FlowValue::String(text)) = outputs.first() else {
            panic!("output[0] should be a string");
        };
        assert_eq!(
            text.as_str(),
            full.as_str(),
            "re-fetched value must be the full untruncated output"
        );
    }
}
