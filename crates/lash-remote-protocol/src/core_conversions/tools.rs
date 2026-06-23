impl From<lash_core::SchemaProjectionOverride> for RemoteSchemaProjectionOverride {
    fn from(value: lash_core::SchemaProjectionOverride) -> Self {
        let lash_core::SchemaProjectionOverride { profile, schema } = value;
        Self { profile, schema }
    }
}

impl From<RemoteSchemaProjectionOverride> for lash_core::SchemaProjectionOverride {
    fn from(value: RemoteSchemaProjectionOverride) -> Self {
        let RemoteSchemaProjectionOverride { profile, schema } = value;
        Self { profile, schema }
    }
}

impl From<RemoteToolActivation> for lash_core::ToolActivation {
    fn from(value: RemoteToolActivation) -> Self {
        match value {
            RemoteToolActivation::Always => Self::Always,
            RemoteToolActivation::Internal => Self::Internal,
        }
    }
}

impl From<RemoteToolScheduling> for lash_core::ToolScheduling {
    fn from(value: RemoteToolScheduling) -> Self {
        match value {
            RemoteToolScheduling::Parallel => Self::Parallel,
            RemoteToolScheduling::Serial => Self::Serial,
        }
    }
}

impl From<RemoteToolOutputContract> for lash_core::ToolOutputContract {
    fn from(value: RemoteToolOutputContract) -> Self {
        match value {
            RemoteToolOutputContract::Static => Self::Static,
            RemoteToolOutputContract::FromInputSchema {
                input_field,
                default_schema,
            } => Self::FromInputSchema {
                input_field,
                default_schema,
            },
        }
    }
}

impl From<lash_core::ToolOutputContract> for RemoteToolOutputContract {
    fn from(value: lash_core::ToolOutputContract) -> Self {
        match value {
            lash_core::ToolOutputContract::Static => Self::Static,
            lash_core::ToolOutputContract::FromInputSchema {
                input_field,
                default_schema,
            } => Self::FromInputSchema {
                input_field,
                default_schema,
            },
        }
    }
}

impl From<RemoteToolArgumentProjectionPolicy> for lash_core::ToolArgumentProjectionPolicy {
    fn from(value: RemoteToolArgumentProjectionPolicy) -> Self {
        match value {
            RemoteToolArgumentProjectionPolicy::MaterializeProjectedValues => {
                Self::MaterializeProjectedValues
            }
            RemoteToolArgumentProjectionPolicy::PreserveProjectedRefsInField { field } => {
                Self::PreserveProjectedRefsInField { field }
            }
        }
    }
}

impl From<RemoteToolRetryPolicy> for lash_core::ToolRetryPolicy {
    fn from(value: RemoteToolRetryPolicy) -> Self {
        match value {
            RemoteToolRetryPolicy::Never => Self::Never,
            RemoteToolRetryPolicy::Safe {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            } => Self::Safe {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            },
            RemoteToolRetryPolicy::Idempotent {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            } => Self::Idempotent {
                max_attempts,
                base_delay_ms,
                max_delay_ms,
            },
        }
    }
}

impl TryFrom<&RemoteToolGrant> for ToolDefinition {
    type Error = RemoteProtocolError;

    fn try_from(value: &RemoteToolGrant) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteToolGrant {
            protocol_version: _,
            id,
            name,
            description,
            input_schema,
            output_schema,
            input_schema_projections,
            output_schema_projections,
            output_contract,
            examples,
            activation,
            argument_projection,
            scheduling,
            retry_policy,
            bindings,
        } = value;
        let mut definition = ToolDefinition::raw(
            id.clone(),
            name.clone(),
            description.clone(),
            input_schema.clone(),
            output_schema.clone(),
        )
        .with_examples(examples.clone())
        .with_output_contract(output_contract.clone().into());
        definition.manifest.bindings = bindings.clone();
        if let Some(activation) = *activation {
            definition = definition.with_activation(activation.into());
        }
        if let Some(argument_projection) = argument_projection.clone() {
            definition = definition.with_argument_projection(argument_projection.into());
        }
        if let Some(scheduling) = *scheduling {
            definition = definition.with_scheduling(scheduling.into());
        }
        if let Some(retry_policy) = *retry_policy {
            definition = definition.with_retry_policy(retry_policy.into());
        }
        for projection in input_schema_projections {
            definition = definition.with_input_schema_projection(
                projection.profile.clone(),
                projection.schema.clone(),
            );
        }
        for projection in output_schema_projections {
            definition = definition.with_output_schema_projection(
                projection.profile.clone(),
                projection.schema.clone(),
            );
        }
        Ok(definition)
    }
}
