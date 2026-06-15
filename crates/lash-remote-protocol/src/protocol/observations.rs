#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSessionCursor {
    pub protocol_version: u32,
    pub cursor: String,
}

impl RemoteSessionCursor {
    pub fn new(cursor: impl Into<String>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            cursor: cursor.into(),
        }
    }

    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteSessionCursor", "cursor", &self.cursor)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteSessionObservationEvent {
    pub protocol_version: u32,
    pub session_id: String,
    pub revision: u64,
    pub cursor: String,
    #[serde(flatten)]
    pub event: RemoteSessionObservationEventPayload,
}

impl RemoteSessionObservationEvent {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty(
            "RemoteSessionObservationEvent",
            "session_id",
            &self.session_id,
        )?;
        require_non_empty("RemoteSessionObservationEvent", "cursor", &self.cursor)?;
        if let RemoteSessionObservationEventPayload::TurnActivity { activity } = &self.event {
            activity.validate()?;
            if activity.protocol_version != self.protocol_version {
                return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                    parent: "RemoteSessionObservationEvent",
                    child: "activity",
                    parent_version: self.protocol_version,
                    child_version: activity.protocol_version,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteSessionObservationEventPayload {
    TurnActivity {
        activity: RemoteTurnActivity,
    },
    Committed,
    AgentFrameSwitched {
        frame_id: String,
    },
    QueueChanged {
        kind: RemoteSessionQueueEventKind,
        batch_ids: Vec<String>,
    },
    ProcessChanged {
        kind: RemoteSessionProcessEventKind,
        process_ids: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionQueueEventKind {
    Enqueued,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteSessionProcessEventKind {
    Started,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RemoteLiveReplayGap {
    pub protocol_version: u32,
    pub session_id: String,
    pub requested_cursor: String,
    pub latest_cursor: String,
    pub latest_revision: u64,
    pub reason: RemoteLiveReplayGapReason,
}

impl RemoteLiveReplayGap {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteLiveReplayGap", "session_id", &self.session_id)?;
        require_non_empty(
            "RemoteLiveReplayGap",
            "requested_cursor",
            &self.requested_cursor,
        )?;
        require_non_empty("RemoteLiveReplayGap", "latest_cursor", &self.latest_cursor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RemoteLiveReplayGapReason {
    Trimmed,
    Unavailable,
}

impl RemoteTurnResult {
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        ensure_protocol_version(self.protocol_version)?;
        require_non_empty("RemoteTurnResult", "session_id", &self.session_id)?;
        require_non_empty("RemoteTurnResult", "turn_id", &self.turn_id)?;
        for activity in &self.activities {
            if activity.protocol_version != self.protocol_version {
                return Err(RemoteProtocolError::MismatchedNestedProtocolVersion {
                    parent: "RemoteTurnResult",
                    child: "activities",
                    parent_version: self.protocol_version,
                    child_version: activity.protocol_version,
                });
            }
            activity.validate()?;
        }
        Ok(())
    }
}
