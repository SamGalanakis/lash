impl From<&RemoteLashlangToolBinding> for lash_core::LashlangToolBinding {
    fn from(value: &RemoteLashlangToolBinding) -> Self {
        let RemoteLashlangToolBinding {
            module_path,
            operation,
            authority_type,
            aliases,
        } = value;
        let mut binding =
            lash_core::LashlangToolBinding::new(module_path.clone(), operation.clone());
        if let Some(authority_type) = authority_type.as_ref() {
            binding = binding.with_authority_type(authority_type.clone());
        }
        binding.with_aliases(aliases.clone())
    }
}

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

impl From<RemoteToolAvailability> for lash_core::ToolAvailability {
    fn from(value: RemoteToolAvailability) -> Self {
        match value {
            RemoteToolAvailability::Off => Self::Off,
            RemoteToolAvailability::Searchable => Self::Searchable,
            RemoteToolAvailability::Callable => Self::Callable,
            RemoteToolAvailability::Showcased => Self::Showcased,
        }
    }
}

impl From<lash_core::ToolAvailability> for RemoteToolAvailability {
    fn from(value: lash_core::ToolAvailability) -> Self {
        match value {
            lash_core::ToolAvailability::Off => Self::Off,
            lash_core::ToolAvailability::Searchable => Self::Searchable,
            lash_core::ToolAvailability::Callable => Self::Callable,
            lash_core::ToolAvailability::Showcased => Self::Showcased,
        }
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
            availability,
            activation,
            argument_projection,
            scheduling,
            retry_policy,
            lashlang_binding,
        } = value;
        let mut definition = ToolDefinition::raw_with_id(
            id.clone()
                .unwrap_or_else(|| format!("remote-tool:{}", value.call_path().unwrap())),
            name.clone(),
            description.clone(),
            input_schema.clone(),
            output_schema.clone(),
        )
        .with_lashlang_binding(
            lashlang_binding
                .as_ref()
                .expect("validated lashlang binding")
                .into(),
        )
        .with_examples(examples.clone())
        .with_output_contract(output_contract.clone().into());
        if let Some(availability) = *availability {
            definition = definition
                .with_availability(lash_core::ToolAvailabilityConfig::same(availability.into()));
        }
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

impl RemoteToolCallResponse {
    pub fn into_tool_result(self) -> ToolResult {
        match self {
            Self::Success {
                protocol_version: _,
                value,
            } => ToolResult::ok(value),
            Self::Failure {
                protocol_version: _,
                code,
                message,
                raw,
                retry_after_ms,
            } => {
                let mut failure = if let Some(after_ms) = retry_after_ms {
                    lash_core::ToolFailure::safe_retry(
                        lash_core::ToolFailureClass::Execution,
                        code,
                        message,
                        Some(after_ms),
                    )
                } else {
                    lash_core::ToolFailure::tool(
                        lash_core::ToolFailureClass::Execution,
                        code,
                        message,
                    )
                };
                failure.raw = raw.map(lash_core::ToolValue::from);
                ToolResult::failure(failure)
            }
            Self::Cancelled {
                protocol_version: _,
                message,
                raw,
            } => {
                if let Some(raw) = raw {
                    ToolResult::cancelled_with_raw(message, raw)
                } else {
                    ToolResult::cancelled(message)
                }
            }
            Self::Pending {
                protocol_version: _,
                deadline_ms,
                on_timeout,
                on_cancel,
            } => {
                let mut pending = lash_core::PendingCompletion::new();
                if let Some(deadline_ms) = deadline_ms {
                    pending.deadline = Some(std::time::Duration::from_millis(deadline_ms));
                }
                pending.on_timeout = match on_timeout {
                    RemoteTimeoutBehavior::ErrorAsResult => {
                        lash_core::TimeoutBehavior::ErrorAsResult
                    }
                    RemoteTimeoutBehavior::FailTurn => lash_core::TimeoutBehavior::FailTurn,
                };
                pending.on_cancel = match on_cancel {
                    RemoteCancelHint::Ignore => lash_core::CancelHint::Ignore,
                    RemoteCancelHint::CancelExternalWork => {
                        lash_core::CancelHint::CancelExternalWork
                    }
                };
                ToolResult::pending(pending)
            }
        }
    }
}

pub trait RemoteToolTransport: Send + Sync + 'static {
    fn send<'a>(
        &'a self,
        request: RemoteToolCallRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<RemoteToolCallResponse, RemoteProtocolError>> + Send + 'a>,
    >;
}

pub struct RemoteToolProvider<T: RemoteToolTransport> {
    manifests: Vec<ToolManifest>,
    contracts: HashMap<String, Arc<ToolContract>>,
    call_paths: HashMap<String, String>,
    transport: T,
}

impl<T: RemoteToolTransport> RemoteToolProvider<T> {
    pub fn new(grants: Vec<RemoteToolGrant>, transport: T) -> Result<Self, RemoteProtocolError> {
        RemoteToolGrant::validate_all(&grants)?;
        let mut manifests = Vec::with_capacity(grants.len());
        let mut contracts = HashMap::with_capacity(grants.len());
        let mut call_paths = HashMap::with_capacity(grants.len());
        for grant in grants {
            let definition = ToolDefinition::try_from(&grant)?;
            let manifest = definition.manifest();
            let executable = lash_core::LashlangToolBinding::required_for_remote(&manifest)
                .map_err(|message| RemoteProtocolError::InvalidToolGrant {
                    tool_name: manifest.name.clone(),
                    message,
                })?;
            contracts.insert(manifest.name.clone(), Arc::new(definition.contract()));
            call_paths.insert(manifest.name.clone(), executable.call_path());
            manifests.push(manifest);
        }
        Ok(Self {
            manifests,
            contracts,
            call_paths,
            transport,
        })
    }
}

impl<T: RemoteToolTransport> ToolProvider for RemoteToolProvider<T> {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.manifests.clone()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.manifests
            .iter()
            .find(|manifest| manifest.name == name)
            .cloned()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.contracts.get(name).cloned()
    }

    fn execute<'life0, 'life1, 'async_trait>(
        &'life0 self,
        call: ToolCall<'life1>,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            if call
                .context
                .cancellation_token()
                .is_some_and(|token| token.is_cancelled())
            {
                return ToolResult::cancelled("remote tool call cancelled before dispatch");
            }
            let Some(call_path) = self.call_paths.get(call.name) else {
                return ToolResult::err_fmt(format_args!("unknown remote tool `{}`", call.name));
            };
            let completion_key = match call.context.completion_key().await {
                Ok(key) => match serde_json::to_value(key) {
                    Ok(value) => value,
                    Err(err) => return ToolResult::err_fmt(err),
                },
                Err(err) => return ToolResult::err_fmt(err),
            };
            let mut headers = HashMap::new();
            if let Some(tool_call_id) = call.context.tool_call_id() {
                headers.insert("x-lash-tool-call-id".to_string(), tool_call_id.to_string());
            }
            let replay_key = call.context.replay_key().map(str::to_string).or_else(|| {
                call.context.tool_call_id().map(|call_id| {
                    format!(
                        "lash-tool:{}:{call_id}:{}",
                        call.context.session_id(),
                        call.name
                    )
                })
            });
            if let Some(replay_key) = replay_key.as_ref() {
                headers.insert("x-lash-replay-key".to_string(), replay_key.clone());
            }
            let request = RemoteToolCallRequest {
                protocol_version: REMOTE_PROTOCOL_VERSION,
                tool_name: call.name.to_string(),
                call_path: call_path.clone(),
                args: call.args.clone(),
                session_id: call.context.session_id().to_string(),
                completion_key,
                tool_call_id: call.context.tool_call_id().map(str::to_string),
                replay_key,
                attempt_number: call.context.attempt_number(),
                max_attempts: call.context.max_attempts(),
                headers,
            };
            match self.transport.send(request).await {
                Ok(response) => match response.validate() {
                    Ok(()) => response.into_tool_result(),
                    Err(err) => ToolResult::err_fmt(err),
                },
                Err(err) => ToolResult::err_fmt(err),
            }
        })
    }
}
