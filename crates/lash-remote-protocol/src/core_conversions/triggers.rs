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
        Ok(Self {
            session_id,
            handle,
            name,
            source_type,
            source_key,
            target: target.map(Into::into),
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
            target: target.map(Into::into),
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
            label,
            identity,
            input,
            inputs,
        } = target;
        Self {
            handle,
            source_key,
            name,
            source_type: source_type.to_string(),
            source,
            target: RemoteTriggerTargetSummary {
                label,
                identity: identity.into(),
                input: input.try_into().expect("core process input serializes remotely"),
                inputs: inputs.into(),
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
            label,
            identity,
            input,
            inputs,
        } = target;
        Ok(Self {
            handle,
            source_key,
            name,
            source_type: lash_core::TriggerEventType::new(source_type),
            source,
            target: lash_core::TriggerTargetSummary {
                label,
                identity: identity.into(),
                input: input.try_into()?,
                inputs: inputs.into(),
            },
            enabled,
        })
    }
}

impl From<std::collections::BTreeMap<String, lash_core::TriggerInputBinding>>
    for RemoteTriggerInputTemplate
{
    fn from(value: std::collections::BTreeMap<String, lash_core::TriggerInputBinding>) -> Self {
        let entries = value
            .iter()
            .map(|(name, binding)| (name.to_string(), RemoteTriggerInputBinding::from(binding)))
            .collect();
        Self { entries }
    }
}

impl From<RemoteTriggerInputTemplate>
    for std::collections::BTreeMap<String, lash_core::TriggerInputBinding>
{
    fn from(value: RemoteTriggerInputTemplate) -> Self {
        let RemoteTriggerInputTemplate { entries } = value;
        entries
            .into_iter()
            .map(|(name, binding)| (name, binding.into()))
            .collect()
    }
}

impl From<&lash_core::TriggerInputBinding> for RemoteTriggerInputBinding {
    fn from(value: &lash_core::TriggerInputBinding) -> Self {
        match value {
            lash_core::TriggerInputBinding::Event => Self::Event,
            lash_core::TriggerInputBinding::Fixed { value } => Self::Fixed {
                value: value.clone(),
            },
        }
    }
}

impl From<RemoteTriggerInputBinding> for lash_core::TriggerInputBinding {
    fn from(value: RemoteTriggerInputBinding) -> Self {
        match value {
            RemoteTriggerInputBinding::Event => Self::Event,
            RemoteTriggerInputBinding::Fixed { value } => Self::Fixed { value },
        }
    }
}

impl TryFrom<RemoteTriggerSubscriptionDraft> for lash_core::TriggerSubscriptionDraft {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerSubscriptionDraft) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerSubscriptionDraft {
            protocol_version: _,
            registrant,
            env_ref,
            wake_target,
            name,
            source_type,
            source_key,
            source,
            payload_schema,
            target,
            target_identity,
            event_types,
            input_template,
            target_label,
        } = value;
        Ok(Self {
            registrant: registrant.into(),
            env_ref: lash_core::ProcessExecutionEnvRef::new(env_ref),
            wake_target: wake_target.map(Into::into),
            name,
            source_type,
            source_key,
            source,
            payload_schema: lash_core::LashSchema::new(payload_schema),
            target: target.try_into()?,
            target_identity: target_identity.into(),
            event_types: event_types.into_iter().map(Into::into).collect(),
            input_template: input_template.into(),
            target_label,
        })
    }
}

impl From<lash_core::TriggerSubscriptionDraft> for RemoteTriggerSubscriptionDraft {
    fn from(value: lash_core::TriggerSubscriptionDraft) -> Self {
        let lash_core::TriggerSubscriptionDraft {
            registrant,
            env_ref,
            wake_target,
            name,
            source_type,
            source_key,
            source,
            payload_schema,
            target,
            target_identity,
            event_types,
            input_template,
            target_label,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            registrant: registrant.into(),
            env_ref: env_ref.as_str().to_string(),
            wake_target: wake_target.map(Into::into),
            name,
            source_type,
            source_key,
            source,
            payload_schema: payload_schema.schema,
            target: target
                .try_into()
                .expect("core process input serializes remotely"),
            target_identity: target_identity.into(),
            event_types: event_types.into_iter().map(Into::into).collect(),
            input_template: input_template.into(),
            target_label,
        }
    }
}

impl From<lash_core::TriggerSubscriptionRecord> for RemoteTriggerSubscriptionRecord {
    fn from(value: lash_core::TriggerSubscriptionRecord) -> Self {
        let lash_core::TriggerSubscriptionRecord {
            subscription_id,
            registrant,
            env_ref,
            wake_target,
            handle,
            name,
            source_type,
            source_key,
            source,
            payload_schema,
            target,
            target_identity,
            event_types,
            input_template,
            target_label,
            enabled,
            created_at_ms,
            updated_at_ms,
        } = value;
        Self {
            subscription_id,
            registrant: registrant.into(),
            env_ref: env_ref.as_str().to_string(),
            wake_target: wake_target.map(Into::into),
            handle,
            name,
            source_type,
            source_key,
            source,
            payload_schema: payload_schema.schema,
            target: target
                .try_into()
                .expect("core process input serializes remotely"),
            target_identity: target_identity.into(),
            event_types: event_types.into_iter().map(Into::into).collect(),
            input_template: input_template.into(),
            target_label,
            enabled,
            created_at_ms,
            updated_at_ms,
        }
    }
}

impl TryFrom<RemoteTriggerSubscriptionRecord> for lash_core::TriggerSubscriptionRecord {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerSubscriptionRecord) -> Result<Self, Self::Error> {
        value.validate("RemoteTriggerSubscriptionRecord")?;
        let RemoteTriggerSubscriptionRecord {
            subscription_id,
            registrant,
            env_ref,
            wake_target,
            handle,
            name,
            source_type,
            source_key,
            source,
            payload_schema,
            target,
            target_identity,
            event_types,
            input_template,
            target_label,
            enabled,
            created_at_ms,
            updated_at_ms,
        } = value;
        Ok(Self {
            subscription_id,
            registrant: registrant.into(),
            env_ref: lash_core::ProcessExecutionEnvRef::new(env_ref),
            wake_target: wake_target.map(Into::into),
            handle,
            name,
            source_type,
            source_key,
            source,
            payload_schema: lash_core::LashSchema::new(payload_schema),
            target: target.try_into()?,
            target_identity: target_identity.into(),
            event_types: event_types.into_iter().map(Into::into).collect(),
            input_template: input_template.into(),
            target_label,
            enabled,
            created_at_ms,
            updated_at_ms,
        })
    }
}

impl TryFrom<RemoteTriggerRegisterSubscriptionRequest> for lash_core::TriggerSubscriptionDraft {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerRegisterSubscriptionRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerRegisterSubscriptionRequest {
            protocol_version: _,
            draft,
        } = value;
        draft.try_into()
    }
}

impl From<lash_core::TriggerSubscriptionRecord> for RemoteTriggerRegisterSubscriptionResult {
    fn from(value: lash_core::TriggerSubscriptionRecord) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            record: value.into(),
        }
    }
}

impl TryFrom<RemoteTriggerRegisterSubscriptionResult> for lash_core::TriggerSubscriptionRecord {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerRegisterSubscriptionResult) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerRegisterSubscriptionResult {
            protocol_version: _,
            record,
        } = value;
        record.try_into()
    }
}

impl From<Vec<lash_core::TriggerSubscriptionRecord>> for RemoteTriggerListSubscriptionsResponse {
    fn from(value: Vec<lash_core::TriggerSubscriptionRecord>) -> Self {
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            subscriptions: value.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<RemoteTriggerListSubscriptionsResponse>
    for Vec<lash_core::TriggerSubscriptionRecord>
{
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTriggerListSubscriptionsResponse) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTriggerListSubscriptionsResponse {
            protocol_version: _,
            subscriptions,
        } = value;
        subscriptions.into_iter().map(TryInto::try_into).collect()
    }
}

fn decode_remote_json<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    type_name: &'static str,
    field: &'static str,
) -> Result<T, RemoteProtocolError> {
    serde_json::from_value(value).map_err(|err| RemoteProtocolError::InvalidEnvelope {
        type_name,
        message: format!("invalid {field}: {err}"),
    })
}
