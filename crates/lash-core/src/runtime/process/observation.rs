use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::plugin::PluginError;

use super::events::{ProcessAwaitOutput, ProcessEvent};
use super::model::{
    ProcessExecutionEnvRef, ProcessExternalRef, ProcessHandleDescriptor, ProcessId, ProcessInput,
    ProcessLifecycleStatus, ProcessListFilter, ProcessOriginator, ProcessRecord, SessionScope,
    WaitState,
};
use super::registry::ProcessRegistry;
use super::time::epoch_ms_from_system_time;

#[derive(Clone)]
pub struct ProcessWorkObserver {
    registry: Arc<dyn ProcessRegistry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessWorkSnapshot {
    pub session_id: String,
    pub visible_process_ids: Vec<ProcessId>,
    pub items: Vec<ObservedWorkItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservedWorkItem {
    pub process: ObservedProcess,
    pub descriptor: ProcessHandleDescriptor,
    pub events: Vec<ObservedProcessEvent>,
    pub kind: String,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservedProcess {
    pub process_id: ProcessId,
    pub graph_key: String,
    pub kind: String,
    pub lifecycle: ProcessLifecycleStatus,
    pub status_label: String,
    pub terminal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub input: ProcessInput,
    pub originator: ProcessOriginator,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_ref: Option<ProcessExecutionEnvRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target: Option<SessionScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caused_by: Option<crate::CausalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ProcessExternalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait: Option<WaitState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub child_session_id: Option<String>,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservedProcessEvent {
    pub sequence: u64,
    pub event_type: String,
    pub occurred_at_ms: u64,
    pub payload: serde_json::Value,
}

/// Per-item event tail in session snapshots. Snapshots are polled by
/// docks/UIs, so per-poll cost must stay bounded instead of growing with a
/// process's full event history; detail views page through `events_after`
/// with a cursor.
pub const SNAPSHOT_EVENT_TAIL: usize = 32;

impl ProcessWorkObserver {
    pub fn new(registry: Arc<dyn ProcessRegistry>) -> Self {
        Self { registry }
    }

    pub async fn snapshot_for_session(
        &self,
        session_id: impl Into<String>,
    ) -> Result<ProcessWorkSnapshot, PluginError> {
        let session_id = session_id.into();
        let session_scope = SessionScope::new(session_id.clone());
        let entries = self.registry.list_handle_grants(&session_scope).await?;
        let mut items = Vec::with_capacity(entries.len());
        for (grant, record) in entries {
            let events = self
                .registry
                .recent_events(&record.id, SNAPSHOT_EVENT_TAIL)
                .await?
                .into_iter()
                .map(ObservedProcessEvent::from)
                .collect();
            let descriptor = grant.descriptor;
            let process = ObservedProcess::from_record(record);
            let kind = descriptor
                .kind
                .clone()
                .unwrap_or_else(|| stable_kind(&process.input).to_string());
            let label = typed_label(&process.input)
                .or_else(|| descriptor.label.clone())
                .unwrap_or_else(|| kind.clone());
            items.push(ObservedWorkItem {
                process,
                descriptor,
                events,
                kind,
                label,
            });
        }
        items.sort_by(|left, right| {
            right
                .process
                .updated_at_ms
                .cmp(&left.process.updated_at_ms)
                .then_with(|| right.process.created_at_ms.cmp(&left.process.created_at_ms))
                .then_with(|| left.process.process_id.cmp(&right.process.process_id))
        });
        let visible_process_ids = items
            .iter()
            .map(|item| item.process.process_id.clone())
            .collect();
        Ok(ProcessWorkSnapshot {
            session_id,
            visible_process_ids,
            items,
        })
    }

    pub async fn process(&self, process_id: &str) -> Option<ObservedProcess> {
        self.registry
            .get_process(process_id)
            .await
            .map(ObservedProcess::from_record)
    }

    pub async fn list(
        &self,
        filter: &ProcessListFilter,
    ) -> Result<Vec<ObservedProcess>, PluginError> {
        Ok(self
            .registry
            .list_processes(filter)
            .await?
            .into_iter()
            .map(ObservedProcess::from_record)
            .collect())
    }

    pub async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ObservedProcessEvent>, PluginError> {
        Ok(self
            .registry
            .events_after(process_id, after_sequence)
            .await?
            .into_iter()
            .map(ObservedProcessEvent::from)
            .collect())
    }
}

impl ObservedProcess {
    fn from_record(record: ProcessRecord) -> Self {
        let lifecycle = ProcessLifecycleStatus::from(&record.status);
        let input = record.input.as_ref().clone();
        let kind = stable_kind(&input).to_string();
        let label = typed_label(&input).unwrap_or_else(|| kind.clone());
        let process_id = record.id;
        Self {
            graph_key: format!("process:{process_id}"),
            process_id,
            kind,
            lifecycle,
            status_label: lifecycle.label().to_string(),
            terminal: lifecycle.is_terminal(),
            error: terminal_error(&record.status),
            created_at_ms: record.created_at_ms,
            updated_at_ms: record.updated_at_ms,
            originator: record.provenance.originator,
            env_ref: record.env_ref,
            wake_target: record.wake_target,
            caused_by: record.provenance.caused_by,
            external_ref: record.external_ref,
            wait: record.wait,
            child_session_id: child_session_id(&input),
            input,
            label,
        }
    }
}

impl From<ProcessEvent> for ObservedProcessEvent {
    fn from(event: ProcessEvent) -> Self {
        Self {
            sequence: event.sequence,
            event_type: event.event_type,
            occurred_at_ms: epoch_ms_from_system_time(event.occurred_at),
            payload: event.payload,
        }
    }
}

fn terminal_error(status: &super::model::ProcessStatus) -> Option<String> {
    match status.await_output()? {
        ProcessAwaitOutput::Failure { message, .. }
        | ProcessAwaitOutput::Cancelled { message, .. } => Some(message.clone()),
        ProcessAwaitOutput::Success { .. } => None,
    }
}

fn child_session_id(input: &ProcessInput) -> Option<String> {
    match input {
        ProcessInput::SessionTurn { create_request, .. } => create_request.session_id.clone(),
        ProcessInput::ToolCall { .. }
        | ProcessInput::LashlangProcess { .. }
        | ProcessInput::External { .. } => None,
    }
}

fn typed_label(input: &ProcessInput) -> Option<String> {
    match input {
        ProcessInput::ToolCall { call } => Some(call.tool_name.clone()),
        ProcessInput::LashlangProcess { process_name, .. } => Some(process_name.clone()),
        ProcessInput::SessionTurn { create_request, .. } => create_request
            .subagent
            .as_ref()
            .map(|subagent| subagent.capability.clone())
            .or_else(|| create_request.usage_source.clone())
            .or_else(|| create_request.session_id.clone()),
        ProcessInput::External { metadata } => metadata
            .get("label")
            .or_else(|| metadata.get("name"))
            .or_else(|| metadata.get("title"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
    }
}

fn stable_kind(input: &ProcessInput) -> &'static str {
    match input {
        ProcessInput::ToolCall { .. } => "tool",
        ProcessInput::LashlangProcess { .. } => "lashlang",
        ProcessInput::SessionTurn { .. } => "session_turn",
        ProcessInput::External { .. } => "external",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::{
        InputItem, PluginOptions, PreparedToolCall, ProcessEventAppendRequest,
        ProcessExecutionEnvRef, ProcessProvenance, ProcessRegistration, SessionCreateRequest,
        SessionScope, SessionStartPoint, SubagentSessionContext, ToolFailureClass,
        ToolOutputContract, TurnInput, WaitKind,
    };

    fn observer(registry: Arc<dyn ProcessRegistry>) -> ProcessWorkObserver {
        ProcessWorkObserver::new(registry)
    }

    fn external_registration(process_id: &str, label: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            process_id,
            ProcessInput::External {
                metadata: json!({ "label": label }),
            },
            ProcessProvenance::host(),
        )
    }

    async fn register_visible(
        registry: &Arc<dyn ProcessRegistry>,
        scope: &SessionScope,
        registration: ProcessRegistration,
        descriptor: ProcessHandleDescriptor,
    ) {
        let process_id = registration.id.clone();
        registry
            .register_process(registration)
            .await
            .expect("register process");
        registry
            .grant_handle(scope, &process_id, descriptor)
            .await
            .expect("grant process handle");
    }

    #[tokio::test]
    async fn snapshot_for_session_reads_visible_grants_and_events_as_epoch_ms() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let visible_scope = SessionScope::new("visible");
        register_visible(
            &registry,
            &visible_scope,
            external_registration("visible-process", "Visible"),
            ProcessHandleDescriptor::new(Some("visible-kind"), Some("Visible descriptor")),
        )
        .await;
        register_visible(
            &registry,
            &SessionScope::new("other"),
            external_registration("hidden-process", "Hidden"),
            ProcessHandleDescriptor::new(Some("hidden-kind"), Some("Hidden")),
        )
        .await;
        registry
            .append_event(
                "visible-process",
                ProcessEventAppendRequest::new("process.cancel_requested", json!({"why": "test"}))
                    .with_replay_key("visible-process:cancel-requested"),
            )
            .await
            .expect("append event");

        let snapshot = observer(Arc::clone(&registry))
            .snapshot_for_session("visible")
            .await
            .expect("snapshot");

        assert_eq!(snapshot.session_id, "visible");
        assert_eq!(snapshot.visible_process_ids, vec!["visible-process"]);
        assert_eq!(snapshot.items.len(), 1);
        assert_eq!(snapshot.items[0].events.len(), 1);
        assert_eq!(
            snapshot.items[0].events[0].event_type,
            "process.cancel_requested"
        );
        assert!(snapshot.items[0].events[0].occurred_at_ms > 0);
    }

    #[tokio::test]
    async fn snapshot_for_session_sorts_work_by_updated_then_created_descending() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let scope = SessionScope::new("sort");
        register_visible(
            &registry,
            &scope,
            external_registration("older", "Older"),
            ProcessHandleDescriptor::new(None::<String>, None::<String>),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        register_visible(
            &registry,
            &scope,
            external_registration("newer", "Newer"),
            ProcessHandleDescriptor::new(None::<String>, None::<String>),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        registry
            .append_event(
                "older",
                ProcessEventAppendRequest::new("process.cancel_requested", json!({}))
                    .with_replay_key("older:cancel-requested"),
            )
            .await
            .expect("update older process");

        let snapshot = observer(Arc::clone(&registry))
            .snapshot_for_session("sort")
            .await
            .expect("snapshot");

        assert_eq!(snapshot.visible_process_ids, vec!["older", "newer"]);
    }

    #[tokio::test]
    async fn observed_process_reports_terminal_status_and_error_messages() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        for process_id in ["failed", "cancelled"] {
            registry
                .register_process(external_registration(process_id, process_id))
                .await
                .expect("register");
        }
        registry
            .complete_process(
                "failed",
                ProcessAwaitOutput::Failure {
                    class: ToolFailureClass::External,
                    code: "boom".to_string(),
                    message: "failed loudly".to_string(),
                    raw: None,
                    control: None,
                },
            )
            .await
            .expect("fail process");
        registry
            .complete_process(
                "cancelled",
                ProcessAwaitOutput::Cancelled {
                    message: "cancelled intentionally".to_string(),
                    raw: None,
                    control: None,
                },
            )
            .await
            .expect("cancel process");

        let observer = observer(Arc::clone(&registry));
        let failed = observer.process("failed").await.expect("failed process");
        let cancelled = observer
            .process("cancelled")
            .await
            .expect("cancelled process");

        assert_eq!(failed.status_label, "failed");
        assert!(failed.terminal);
        assert_eq!(failed.error.as_deref(), Some("failed loudly"));
        assert_eq!(cancelled.status_label, "cancelled");
        assert!(cancelled.terminal);
        assert_eq!(cancelled.error.as_deref(), Some("cancelled intentionally"));
    }

    #[tokio::test]
    async fn observed_process_exposes_current_wait_state() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let scope = SessionScope::new("wait");
        register_visible(
            &registry,
            &scope,
            external_registration("waiting-process", "Waiting"),
            ProcessHandleDescriptor::new(Some("external"), Some("Waiting")),
        )
        .await;
        let wait = WaitState {
            since_ms: 1234,
            kind: WaitKind::Signal {
                name: "ready".to_string(),
                event_type: "signal.ready".to_string(),
                key: "process:waiting-process:signal.ready:1".to_string(),
                ordinal: 1,
            },
        };
        registry
            .set_process_wait("waiting-process", wait.clone())
            .await
            .expect("set wait");

        let observer = observer(Arc::clone(&registry));
        let observed = observer
            .process("waiting-process")
            .await
            .expect("waiting process");
        let snapshot = observer
            .snapshot_for_session("wait")
            .await
            .expect("snapshot");

        assert_eq!(observed.wait, Some(wait.clone()));
        assert_eq!(snapshot.items.len(), 1);
        assert_eq!(snapshot.items[0].process.wait, Some(wait));
    }

    #[tokio::test]
    async fn snapshot_for_session_prefers_typed_labels_and_extracts_child_session_id() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;
        let scope = SessionScope::new("labels");
        let module_ref = lashlang::ModuleRef::new(&lashlang::ContentHash::new("module"));
        let host_requirements_ref =
            lashlang::HostRequirementsRef::new(&lashlang::ContentHash::new("surface"));
        let process_ref = lashlang::ProcessRef::new(lashlang::ContentHash::new("process"), 1);
        let mut child_request = SessionCreateRequest::child_session(
            "labels",
            SessionStartPoint::Empty,
            PluginOptions::default(),
        )
        .with_session_id("child-session");
        child_request.subagent = Some(SubagentSessionContext {
            parent_session_id: "labels".to_string(),
            capability: "researcher".to_string(),
            depth: 1,
            max_depth: 4,
        });
        let cases = [
            (
                "tool",
                ProcessInput::ToolCall {
                    call: PreparedToolCall::from_parts(
                        "call-1",
                        "shell.run",
                        json!({}),
                        None,
                        serde_json::Value::Null,
                    ),
                },
                "tool",
                "shell.run",
                None,
            ),
            (
                "lashlang",
                ProcessInput::LashlangProcess {
                    module_ref,
                    process_ref,
                    host_requirements_ref: host_requirements_ref,
                    process_name: "remember".to_string(),
                    args: serde_json::Map::new(),
                },
                "lashlang",
                "remember",
                None,
            ),
            (
                "session",
                ProcessInput::SessionTurn {
                    create_request: Box::new(child_request),
                    turn_input: Box::new(TurnInput::items([InputItem::text("run child")])),
                    output_contract: ToolOutputContract::Static,
                },
                "session_turn",
                "researcher",
                Some("child-session"),
            ),
            (
                "external",
                ProcessInput::External {
                    metadata: json!({ "label": "external job" }),
                },
                "external",
                "external job",
                None,
            ),
        ];
        for (process_id, input, _kind, _label, _child_session_id) in cases {
            let needs_env = matches!(
                input,
                ProcessInput::ToolCall { .. } | ProcessInput::LashlangProcess { .. }
            );
            let mut registration =
                ProcessRegistration::new(process_id, input, ProcessProvenance::host());
            if needs_env {
                registration = registration.with_execution_env_ref(Some(
                    ProcessExecutionEnvRef::new(format!("process-env:test:{process_id}")),
                ));
            }
            register_visible(
                &registry,
                &scope,
                registration,
                ProcessHandleDescriptor::new(Some("descriptor-kind"), Some("Descriptor label")),
            )
            .await;
        }

        let snapshot = observer(Arc::clone(&registry))
            .snapshot_for_session("labels")
            .await
            .expect("snapshot");
        let by_id = snapshot
            .items
            .iter()
            .map(|item| (item.process.process_id.as_str(), item))
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(by_id["tool"].label, "shell.run");
        assert_eq!(by_id["lashlang"].label, "remember");
        assert_eq!(by_id["session"].label, "researcher");
        assert_eq!(
            by_id["session"].process.child_session_id.as_deref(),
            Some("child-session")
        );
        assert_eq!(by_id["external"].label, "external job");
    }

    #[tokio::test]
    async fn observed_process_missing_lookup_returns_none() {
        let registry =
            Arc::new(super::super::TestLocalProcessRegistry::default()) as Arc<dyn ProcessRegistry>;

        assert!(observer(registry).process("missing").await.is_none());
    }
}
