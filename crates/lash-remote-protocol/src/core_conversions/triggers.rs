impl From<RemoteProtocolTurnOptions> for lash_core::ProtocolTurnOptions {
    fn from(value: RemoteProtocolTurnOptions) -> Self {
        let RemoteProtocolTurnOptions { payload } = value;
        Self { payload }
    }
}

impl From<lash_core::ProtocolTurnOptions> for RemoteProtocolTurnOptions {
    fn from(value: lash_core::ProtocolTurnOptions) -> Self {
        let lash_core::ProtocolTurnOptions { payload } = value;
        Self { payload }
    }
}

impl TryFrom<RemoteTriggerOccurrenceRequest> for lash_core::TriggerOccurrenceRequest {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerOccurrenceRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerOccurrenceRequest {
            protocol_version: _,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        } = value;
        let mut request = lash_core::TriggerOccurrenceRequest::new(
            source_type,
            source_key,
            payload,
            idempotency_key,
        );
        request.source = source;
        Ok(request)
    }
}

impl From<lash_core::TriggerOccurrenceRequest> for RemoteTriggerOccurrenceRequest {
    fn from(value: lash_core::TriggerOccurrenceRequest) -> Self {
        let lash_core::TriggerOccurrenceRequest {
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
        }
    }
}

impl From<lash_core::TriggerOccurrenceRecord> for RemoteTriggerOccurrenceRecord {
    fn from(value: lash_core::TriggerOccurrenceRecord) -> Self {
        let lash_core::TriggerOccurrenceRecord {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        } = value;
        Self {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        }
    }
}

impl From<RemoteTriggerOccurrenceRecord> for lash_core::TriggerOccurrenceRecord {
    fn from(value: RemoteTriggerOccurrenceRecord) -> Self {
        let RemoteTriggerOccurrenceRecord {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        } = value;
        Self {
            occurrence_id,
            source_type,
            source_key,
            payload,
            idempotency_key,
            source,
            occurred_at_ms,
        }
    }
}

impl From<lash_core::TriggerEmitReport> for RemoteTriggerEmitReport {
    fn from(value: lash_core::TriggerEmitReport) -> Self {
        let lash_core::TriggerEmitReport {
            occurrence_id,
            started_process_ids,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            occurrence_id,
            started_process_ids,
        }
    }
}

impl TryFrom<RemoteTriggerEmitReport> for lash_core::TriggerEmitReport {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerEmitReport) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerEmitReport {
            protocol_version: _,
            occurrence_id,
            started_process_ids,
        } = value;
        Ok(Self {
            occurrence_id,
            started_process_ids,
        })
    }
}

impl TryFrom<RemoteTriggerSubscriptionFilter> for lash_core::TriggerSubscriptionFilter {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerSubscriptionFilter) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerSubscriptionFilter {
            protocol_version: _,
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        } = value;
        let target = target
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerSubscriptionFilter",
                message: format!("invalid target identity: {err}"),
            })?;
        Ok(Self {
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        })
    }
}

impl From<lash_core::TriggerSubscriptionFilter> for RemoteTriggerSubscriptionFilter {
    fn from(value: lash_core::TriggerSubscriptionFilter) -> Self {
        let lash_core::TriggerSubscriptionFilter {
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target,
            enabled,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target: target
                .map(|target| serde_json::to_value(target).expect("target identity serializes")),
            enabled,
        }
    }
}

impl From<lash_core::TriggerRegistration> for RemoteTriggerRegistration {
    fn from(value: lash_core::TriggerRegistration) -> Self {
        let lash_core::TriggerRegistration {
            handle,
            source_key,
            name,
            source_type,
            source,
            target,
            enabled,
        } = value;
        let lash_core::TriggerTargetSummary {
            process_name,
            inputs,
        } = target;
        Self {
            handle,
            source_key,
            name,
            source_type: source_type.to_string(),
            source,
            target: RemoteTriggerTargetSummary {
                process_name,
                inputs: serde_json::to_value(inputs).expect("trigger input template serializes"),
            },
            enabled,
        }
    }
}

impl TryFrom<RemoteTriggerRegistration> for lash_core::TriggerRegistration {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerRegistration) -> Result<Self, Self::Error> {
        let RemoteTriggerRegistration {
            handle,
            source_key,
            name,
            source_type,
            source,
            target,
            enabled,
        } = value;
        let RemoteTriggerTargetSummary {
            process_name,
            inputs,
        } = target;
        let inputs =
            serde_json::from_value(inputs).map_err(|err| RemoteProtocolError::InvalidEnvelope {
                type_name: "RemoteTriggerTargetSummary",
                message: format!("invalid input template: {err}"),
            })?;
        Ok(Self {
            handle,
            source_key,
            name,
            source_type: lash_core::TriggerEventType::new(source_type),
            source,
            target: lash_core::TriggerTargetSummary {
                process_name,
                inputs,
            },
            enabled,
        })
    }
}
