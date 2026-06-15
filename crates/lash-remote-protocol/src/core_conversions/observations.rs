impl RemoteTurnActivity {
    pub fn from_core(sequence: u64, activity: lash_core::TurnActivity) -> Self {
        let lash_core::TurnActivity {
            id: lash_core::TurnActivityId(id),
            correlation_id: lash_core::TurnActivityId(correlation_id),
            event,
        } = activity;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            sequence,
            id,
            correlation_id,
            event: RemoteTurnEvent::from(event),
        }
    }
}

impl From<lash_core::SessionQueueEventKind> for RemoteSessionQueueEventKind {
    fn from(value: lash_core::SessionQueueEventKind) -> Self {
        match value {
            lash_core::SessionQueueEventKind::Enqueued => Self::Enqueued,
            lash_core::SessionQueueEventKind::Cancelled => Self::Cancelled,
        }
    }
}

impl From<lash_core::SessionProcessEventKind> for RemoteSessionProcessEventKind {
    fn from(value: lash_core::SessionProcessEventKind) -> Self {
        match value {
            lash_core::SessionProcessEventKind::Started => Self::Started,
            lash_core::SessionProcessEventKind::Cancelled => Self::Cancelled,
        }
    }
}

impl RemoteSessionObservationEvent {
    pub fn from_core(sequence: u64, event: lash_core::SessionObservationEvent) -> Self {
        let lash_core::SessionObservationEvent {
            session_id,
            revision,
            cursor,
            payload,
        } = event;
        let payload = match payload {
            lash_core::SessionObservationEventPayload::TurnActivity(activity) => {
                RemoteSessionObservationEventPayload::TurnActivity {
                    activity: RemoteTurnActivity::from_core(sequence, activity),
                }
            }
            // The committed read view is a local handle; only the commit
            // signal itself crosses the wire.
            lash_core::SessionObservationEventPayload::Committed { read_view: _ } => {
                RemoteSessionObservationEventPayload::Committed
            }
            lash_core::SessionObservationEventPayload::AgentFrameSwitched { frame_id } => {
                RemoteSessionObservationEventPayload::AgentFrameSwitched { frame_id }
            }
            lash_core::SessionObservationEventPayload::QueueChanged { kind, batch_ids } => {
                RemoteSessionObservationEventPayload::QueueChanged {
                    kind: kind.into(),
                    batch_ids,
                }
            }
            lash_core::SessionObservationEventPayload::ProcessChanged { kind, process_ids } => {
                RemoteSessionObservationEventPayload::ProcessChanged {
                    kind: kind.into(),
                    process_ids,
                }
            }
        };
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            revision: revision.as_u64(),
            cursor: cursor.to_string(),
            event: payload,
        }
    }
}

impl From<lash_core::LiveReplayGapReason> for RemoteLiveReplayGapReason {
    fn from(value: lash_core::LiveReplayGapReason) -> Self {
        match value {
            lash_core::LiveReplayGapReason::Trimmed => Self::Trimmed,
            lash_core::LiveReplayGapReason::Unavailable => Self::Unavailable,
        }
    }
}

impl From<lash_core::LiveReplayGap> for RemoteLiveReplayGap {
    fn from(value: lash_core::LiveReplayGap) -> Self {
        let lash_core::LiveReplayGap {
            session_id,
            requested_cursor,
            latest_cursor,
            latest_revision,
            reason,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            requested_cursor: requested_cursor.to_string(),
            latest_cursor: latest_cursor.to_string(),
            latest_revision: latest_revision.as_u64(),
            reason: reason.into(),
        }
    }
}

impl From<lash_core::TurnEvent> for RemoteTurnEvent {
    fn from(value: lash_core::TurnEvent) -> Self {
        match value {
            lash_core::TurnEvent::QueuedWorkStarted {
                boundary,
                batch_ids,
                causes,
            } => Self::RuntimeDiagnostic {
                kind: "queued_work_started".to_string(),
                data: serde_json::json!({
                    "boundary": boundary,
                    "batch_ids": batch_ids,
                    "causes": causes,
                }),
            },
            lash_core::TurnEvent::ModelRequestStarted { protocol_iteration } => {
                Self::ModelRequestStarted { protocol_iteration }
            }
            lash_core::TurnEvent::AssistantProseDelta { text } => {
                Self::AssistantProseDelta { text }
            }
            lash_core::TurnEvent::ReasoningDelta { text } => Self::ReasoningDelta { text },
            lash_core::TurnEvent::CodeBlockStarted {
                language,
                code,
                graph_key,
            } => Self::CodeBlockStarted {
                language,
                code,
                graph_key,
            },
            lash_core::TurnEvent::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
                graph_key,
            } => Self::CodeBlockCompleted {
                language,
                output,
                error,
                success,
                duration_ms,
                tool_call_ids,
                graph_key,
            },
            lash_core::TurnEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => Self::ToolCallStarted {
                call_id,
                name,
                args,
            },
            lash_core::TurnEvent::ToolCallCompleted {
                call_id,
                name,
                args,
                output,
                duration_ms,
            } => Self::ToolCallCompleted {
                call_id,
                name,
                args,
                output: serde_json::to_value(output).unwrap_or(serde_json::Value::Null),
                duration_ms,
            },
            lash_core::TurnEvent::SubmittedValue { value } => Self::SubmittedValue { value },
            lash_core::TurnEvent::ToolValue { tool_name, value } => {
                Self::ToolValue { tool_name, value }
            }
            lash_core::TurnEvent::Usage {
                protocol_iteration,
                usage,
                cumulative,
            } => Self::Usage {
                protocol_iteration,
                usage: usage.into(),
                cumulative: cumulative.into(),
            },
            lash_core::TurnEvent::ChildUsage {
                session_id,
                source,
                model,
                protocol_iteration,
                usage,
                cumulative,
            } => Self::ChildUsage {
                session_id,
                source,
                model,
                protocol_iteration,
                usage: usage.into(),
                cumulative: cumulative.into(),
            },
            lash_core::TurnEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            } => Self::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
            },
            lash_core::TurnEvent::PluginRuntime { plugin_id, event } => Self::RuntimeDiagnostic {
                kind: "plugin_runtime".to_string(),
                data: serde_json::json!({
                    "plugin_id": plugin_id,
                    "event": event,
                }),
            },
            lash_core::TurnEvent::QueuedInputAccepted { checkpoint, inputs } => {
                Self::RuntimeDiagnostic {
                    kind: "queued_input_accepted".to_string(),
                    data: serde_json::json!({
                        "checkpoint": checkpoint,
                        "inputs": inputs,
                    }),
                }
            }
            lash_core::TurnEvent::QueuedMessagesCommitted {
                messages,
                checkpoint,
            } => Self::RuntimeDiagnostic {
                kind: "queued_messages_committed".to_string(),
                data: serde_json::json!({
                    "messages": messages,
                    "checkpoint": checkpoint,
                }),
            },
            lash_core::TurnEvent::Error { message } => Self::Error { message },
        }
    }
}

pub fn replay_collected_activities(
    activities: impl IntoIterator<Item = lash_core::TurnActivity>,
    first_sequence: u64,
) -> Vec<RemoteTurnActivity> {
    activities
        .into_iter()
        .enumerate()
        .map(|(idx, activity)| {
            RemoteTurnActivity::from_core(first_sequence.saturating_add(idx as u64), activity)
        })
        .collect()
}

pub struct RemoteTurnActivitySink<W: Write + Send + 'static> {
    writer: Mutex<W>,
    next_sequence: AtomicU64,
    errors: Mutex<Vec<String>>,
}

impl<W: Write + Send + 'static> RemoteTurnActivitySink<W> {
    pub fn new(writer: W, first_sequence: u64) -> Self {
        Self {
            writer: Mutex::new(writer),
            next_sequence: AtomicU64::new(first_sequence),
            errors: Mutex::new(Vec::new()),
        }
    }

    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *self.errors.lock().expect("remote sink errors lock"))
    }

    pub fn into_inner(self) -> Result<W, W> {
        self.writer.into_inner().map_err(|err| err.into_inner())
    }
}

impl<W: Write + Send + 'static> lash_core::TurnActivitySink for RemoteTurnActivitySink<W> {
    fn emit<'life0, 'async_trait>(
        &'life0 self,
        activity: lash_core::TurnActivity,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
            let remote = RemoteTurnActivity::from_core(sequence, activity);
            let result = serde_json::to_writer(
                &mut *self.writer.lock().expect("remote sink writer lock"),
                &remote,
            )
            .and_then(|_| {
                self.writer
                    .lock()
                    .expect("remote sink writer lock")
                    .write_all(b"\n")
                    .map_err(serde_json::Error::io)
            });
            if let Err(err) = result {
                self.errors
                    .lock()
                    .expect("remote sink errors lock")
                    .push(err.to_string());
            }
        })
    }
}
