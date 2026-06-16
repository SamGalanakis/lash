#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::helpers::{
        EmptyTools, MockCall, RecordingStore, mock_provider, runtime_with_plugins,
        runtime_with_plugins_and_tools, runtime_with_plugins_and_tools_and_host_and_store,
        standard_test_policy, test_host_config,
    };
    use ::lashlang::LashlangArtifactStore;
    use serde::Deserialize;
    use std::sync::Arc;

    fn test_process_op_scope(id: &str) -> crate::ProcessOpScope<'static> {
        crate::ProcessOpScope::new(
            crate::ScopedEffectController::shared(
                Arc::new(crate::InlineRuntimeEffectController),
                crate::ExecutionScope::runtime_operation(id),
            )
            .expect("test execution scope"),
        )
    }

    async fn runtime_with_processes_and_tools(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
        tools: Arc<dyn crate::ToolProvider>,
    ) -> crate::LashRuntime {
        let mut runtime =
            runtime_with_plugins_and_tools(plugins, tools, mock_provider(Vec::new())).await;
        runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
        runtime
    }

    async fn runtime_with_processes_and_tools_and_store(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
        tools: Arc<dyn crate::ToolProvider>,
    ) -> (crate::LashRuntime, Arc<RecordingStore>) {
        let store = Arc::new(RecordingStore::default());
        let runtime_store: Arc<dyn crate::RuntimePersistence> = store.clone();
        let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
            plugins,
            tools,
            mock_provider(Vec::new()),
            test_host_config(),
            runtime_store,
        )
        .await;
        runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
        (runtime, store)
    }

    async fn runtime_with_processes(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
    ) -> crate::LashRuntime {
        let mut runtime = runtime_with_plugins(plugins, mock_provider(Vec::new())).await;
        runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
        runtime
    }

    fn probe_event_type() -> crate::ProcessEventType {
        crate::ProcessEventType {
            name: "probe.event".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: crate::ProcessEventSemanticsSpec::default(),
        }
    }

    fn external_registration(process_id: &str) -> crate::ProcessRegistration {
        crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::External {
                metadata: serde_json::json!({ "process_id": process_id }),
            },
            crate::ProcessProvenance::host(),
        )
        .with_extra_event_types([probe_event_type()])
    }

    async fn lashlang_process_registration(
        process_id: &str,
        program: ::lashlang::Program,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> crate::ProcessRegistration {
        try_lashlang_process_registration(process_id, program, args)
            .await
            .expect("link lashlang test module")
    }

    async fn try_lashlang_process_registration(
        process_id: &str,
        program: ::lashlang::Program,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<crate::ProcessRegistration, ::lashlang::LinkError> {
        let mut resources = ::lashlang::LashlangHostCatalog::new();
        resources.add_module_operation(
            ["tools"],
            "Tools",
            "process_echo",
            "process_echo",
            ::lashlang::TypeExpr::Any,
            ::lashlang::TypeExpr::Any,
        );
        try_lashlang_process_registration_with_resources(process_id, program, args, resources).await
    }

    async fn try_lashlang_process_registration_with_resources(
        process_id: &str,
        program: ::lashlang::Program,
        args: serde_json::Map<String, serde_json::Value>,
        resources: ::lashlang::LashlangHostCatalog,
    ) -> Result<crate::ProcessRegistration, ::lashlang::LinkError> {
        let module = if program.process("main").is_some() {
            program
        } else {
            ::lashlang::Program {
                declarations: vec![::lashlang::Declaration::Process(::lashlang::ProcessDecl {
                    name: "main".into(),
                    params: Vec::new(),
                    signals: Vec::new(),
                    return_ty: None,
                    label: None,
                    body: program.main,
                })],
                main: ::lashlang::Expr::Block(Vec::new()),
                declaration_spans: Vec::new(),
                expression_spans: Vec::new(),
            }
        };
        let linked_module = ::lashlang::LinkedModule::link(
            module,
            ::lashlang::LashlangHostEnvironment::new(
                resources,
                ::lashlang::LashlangAbilities::default()
                    .with_processes()
                    .with_sleep()
                    .with_process_signals(),
            ),
        )?;
        ::lashlang::global_in_memory_lashlang_artifact_store()
            .put_module_artifact(&linked_module.artifact)
            .await
            .expect("store lashlang test module artifact");
        let process_ref = linked_module
            .artifact
            .process_ref("main")
            .expect("main process ref")
            .clone();
        let signal_event_types = linked_module
            .artifact
            .canonical_ir
            .process("main")
            .map(crate::lashlang_process_signal_event_types)
            .unwrap_or_default();
        Ok(crate::ProcessRegistration::session_start_draft(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_ref: linked_module.module_ref,
                process_ref,
                host_requirements_ref: linked_module.host_requirements_ref,
                process_name: "main".to_string(),
                args,
            },
        )
        .with_extra_event_types(
            crate::lashlang_process_event_types()
                .into_iter()
                .chain(signal_event_types),
        ))
    }

    struct ProcessEchoTool;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct RecordedToolContext {
        tool_call_id: Option<String>,
        replay_key: Option<String>,
        runtime_process_id: Option<String>,
    }

    #[derive(Clone, Default)]
    struct RecordingProcessEchoTool {
        calls: Arc<std::sync::Mutex<Vec<RecordedToolContext>>>,
    }

    impl RecordingProcessEchoTool {
        fn calls(&self) -> Vec<RecordedToolContext> {
            self.calls.lock().expect("recorded calls").clone()
        }
    }

    fn process_echo_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:process_echo",
            "process_echo",
            "Echo process input.",
            serde_json::json!({ "type": "object", "additionalProperties": true }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for ProcessEchoTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![process_echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "process_echo").then(|| Arc::new(process_echo_tool_definition().contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
            let value = call
                .args
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            crate::ToolResult::ok(serde_json::json!({ "payload": format!("raw:{value}") }))
        }
    }

    struct RenamedProcessEchoTool;

    fn renamed_process_echo_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:renamed_echo",
            "renamed_echo",
            "Echo process input under a changed host operation.",
            serde_json::json!({ "type": "object", "additionalProperties": true }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_lashlang_binding(crate::LashlangToolBinding::new(["tools"], "process_echo"))
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for RenamedProcessEchoTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![renamed_process_echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "renamed_echo")
                .then(|| Arc::new(renamed_process_echo_tool_definition().contract()))
        }

        async fn execute(&self, _call: crate::ToolCall<'_>) -> crate::ToolResult {
            crate::ToolResult::ok(serde_json::json!({ "payload": "renamed" }))
        }
    }

    #[derive(Deserialize)]
    struct SnapshotToolOptions {
        snapshot_ref: String,
    }

    fn snapshot_tool_options(snapshot_ref: &str) -> crate::PluginOptions {
        crate::PluginOptions::typed(
            "snapshot-tools",
            serde_json::json!({ "snapshot_ref": snapshot_ref }),
        )
        .expect("snapshot plugin options")
    }

    fn snapshot_tool_factory() -> Arc<dyn crate::PluginFactory> {
        Arc::new(crate::PluginSpecFactory::new(
            "snapshot-tools",
            Arc::new(|ctx| {
                let enabled = ctx
                    .plugin_options
                    .decode::<SnapshotToolOptions>("snapshot-tools")
                    .map_err(|err| {
                        crate::PluginError::Registration(format!(
                            "invalid snapshot tool options: {err}"
                        ))
                    })?
                    .is_some_and(|options| options.snapshot_ref == "tool-authority:sha256:ok");
                let spec = if enabled {
                    crate::PluginSpec::new().with_tool_provider(Arc::new(ProcessEchoTool))
                } else {
                    crate::PluginSpec::new()
                };
                Ok(spec)
            }),
        ))
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for RecordingProcessEchoTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![process_echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "process_echo").then(|| Arc::new(process_echo_tool_definition().contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
            self.calls
                .lock()
                .expect("recorded calls")
                .push(RecordedToolContext {
                    tool_call_id: call.context.tool_call_id().map(str::to_string),
                    replay_key: call.context.replay_key().map(str::to_string),
                    runtime_process_id: call.context.runtime_process_id().map(str::to_string),
                });
            let value = call
                .args
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            crate::ToolResult::ok(serde_json::json!({ "payload": format!("raw:{value}") }))
        }
    }

    async fn register_open_process(registry: &Arc<dyn crate::ProcessRegistry>, process_id: &str) {
        registry
            .register_process(external_registration(process_id))
            .await
            .expect("register process");
        registry
            .append_event(
                process_id,
                crate::ProcessEventAppendRequest::new(
                    "probe.event",
                    serde_json::json!({ "marker": process_id }),
                ),
            )
            .await
            .expect("append probe event");
    }

    fn process_echo_args() -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }

    fn process_echo_program() -> ::lashlang::Program {
        ::lashlang::parse(
            r#"
            process main() {
              called = await tools.process_echo({ value: "start" })?
              finish called.payload
            }
            "#,
        )
        .expect("process echo program")
    }

    fn missing_module_path_program() -> ::lashlang::Program {
        ::lashlang::parse(
            r#"
            process main() {
              called = await snapshot.tools.process_echo({ value: "start" })?
              finish called.payload
            }
            "#,
        )
        .expect("missing module path program")
    }

    fn snapshot_tools_resources() -> ::lashlang::LashlangHostCatalog {
        let mut resources = ::lashlang::LashlangHostCatalog::new();
        resources.add_module_operation(
            ["snapshot", "tools"],
            "Tools",
            "process_echo",
            "process_echo",
            ::lashlang::TypeExpr::Any,
            ::lashlang::TypeExpr::Any,
        );
        resources
    }

    async fn start_process_for_validation(
        runtime: &crate::LashRuntime,
        registration: crate::ProcessRegistration,
    ) -> Result<crate::ProcessRecord, crate::PluginError> {
        let manager =
            RuntimeSessionServices::new(runtime, true, None).expect("runtime session manager");
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("validation")),
                ),
                test_process_op_scope("validate-process-env"),
            )
            .await
    }

    async fn start_request_for_validation(
        runtime: &crate::LashRuntime,
        request: crate::ProcessStartRequest,
    ) -> Result<crate::ProcessHandleSummary, crate::PluginError> {
        let manager =
            RuntimeSessionServices::new(runtime, true, None).expect("runtime session manager");
        let service = RuntimeSessionProcessService {
            services: Arc::new(manager),
        };
        service
            .start_from_request("root", request, test_process_op_scope("validate-request-env"))
            .await
    }

    #[tokio::test]
    async fn process_start_validation_accepts_rebuilt_process_plugin_options() {
        let runtime = runtime_with_processes(vec![snapshot_tool_factory()]).await;
        let registration = lashlang_process_registration(
            "validate-snapshot-present",
            process_echo_program(),
            process_echo_args(),
        )
        .await;
        let request = crate::ProcessStartRequest::new(
            "validate-snapshot-present",
            registration.input.as_ref().clone(),
            crate::ProcessOriginator::host(),
        )
        .with_env_spec(crate::ProcessExecutionEnvSpec::new(
            snapshot_tool_options("tool-authority:sha256:ok"),
            standard_test_policy(),
        ))
        .with_event_types(registration.event_types.clone());

        let summary = start_request_for_validation(&runtime, request)
            .await
            .expect("process env should validate from snapshot options");
        assert_eq!(summary.process_id, "validate-snapshot-present");
        assert!(
            runtime
                .host
                .process_registry
                .as_ref()
                .expect("process registry")
                .get_process("validate-snapshot-present")
                .await
                .is_some()
        );
    }

    #[tokio::test]
    async fn process_start_validation_rejects_missing_module_path() {
        let runtime =
            runtime_with_processes_and_tools(Vec::new(), Arc::new(EmptyTools)).await;
        let registration = try_lashlang_process_registration_with_resources(
            "validate-missing-module",
            missing_module_path_program(),
            process_echo_args(),
            snapshot_tools_resources(),
        )
        .await
        .expect("link missing module path process");
        let err = start_process_for_validation(&runtime, registration)
            .await
            .expect_err("missing module should reject before registration");
        assert!(
            err.to_string()
                .contains("module `snapshot.tools` is not available"),
            "{err}"
        );
        assert!(
            runtime
                .host
                .process_registry
                .as_ref()
                .expect("process registry")
                .get_process("validate-missing-module")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn process_start_validation_rejects_changed_host_operation_binding() {
        let runtime =
            runtime_with_processes_and_tools(Vec::new(), Arc::new(RenamedProcessEchoTool)).await;
        let registration = lashlang_process_registration(
            "validate-changed-binding",
            process_echo_program(),
            process_echo_args(),
        )
        .await;
        let err = start_process_for_validation(&runtime, registration)
            .await
            .expect_err("changed host operation should reject before registration");
        assert!(
            err.to_string()
                .contains("resolves to `renamed_echo`, expected `process_echo`"),
            "{err}"
        );
        assert!(
            runtime
                .host
                .process_registry
                .as_ref()
                .expect("process registry")
                .get_process("validate-changed-binding")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn process_start_validation_rejects_missing_process_plugin_options() {
        let runtime = runtime_with_processes(vec![snapshot_tool_factory()]).await;
        let registration = lashlang_process_registration(
            "validate-snapshot-missing",
            process_echo_program(),
            process_echo_args(),
        )
        .await;
        let err = start_process_for_validation(&runtime, registration)
            .await
            .expect_err("missing snapshot plugin option should reject before registration");
        assert!(
            err.to_string()
                .contains("module `tools` does not expose operation `process_echo`"),
            "{err}"
        );
        assert!(
            runtime
                .host
                .process_registry
                .as_ref()
                .expect("process registry")
                .get_process("validate-snapshot-missing")
                .await
                .is_none()
        );
    }

    async fn grant_handle(
        registry: &Arc<dyn crate::ProcessRegistry>,
        session_scope: &crate::SessionScope,
        process_id: &str,
    ) {
        registry
            .grant_handle(
                session_scope,
                process_id,
                crate::ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant handle");
    }

    fn worker_registration(registration: crate::ProcessRegistration) -> crate::ProcessRegistration {
        registration.with_process_provenance(crate::ProcessProvenance::session(
            crate::SessionScope::new("root"),
        ))
    }

    async fn with_worker_execution_env_in_store(
        registration: crate::ProcessRegistration,
        store: &dyn ::lashlang::LashlangArtifactStore,
    ) -> crate::ProcessRegistration {
        match registration.input.as_ref() {
            crate::ProcessInput::ToolCall { .. } | crate::ProcessInput::LashlangProcess { .. } => {
                let spec = crate::ProcessExecutionEnvSpec::new(
                    crate::PluginOptions::default(),
                    standard_test_policy(),
                );
                let env_ref = crate::persist_process_execution_env(store, &spec)
                    .await
                    .expect("persist worker process execution env");
                registration.with_execution_env_ref(Some(env_ref))
            }
            crate::ProcessInput::SessionTurn { .. } | crate::ProcessInput::External { .. } => {
                registration
            }
        }
    }

    async fn worker_registration_with_env(
        registration: crate::ProcessRegistration,
    ) -> crate::ProcessRegistration {
        with_worker_execution_env_in_store(
            worker_registration(registration),
            ::lashlang::global_in_memory_lashlang_artifact_store().as_ref(),
        )
        .await
    }

    fn process_worker(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
    ) -> crate::DurableProcessWorker {
        process_worker_with_core(registry, factory, {
            let mut config = crate::RuntimeHostConfig::in_memory();
            config.providers.provider_resolver = Arc::new(crate::SingleProviderResolver::new(
                mock_provider(vec![MockCall {
                    stream_events: Vec::new(),
                    response: Ok(successful_text_response("child done")),
                }])
                .into_handle(),
            ));
            config
        })
    }

    fn process_worker_with_core(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
        runtime_host: crate::RuntimeHostConfig,
    ) -> crate::DurableProcessWorker {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(ProcessEchoTool);
        process_worker_with_tools(registry, factory, runtime_host, tools)
    }

    fn process_worker_with_tools(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
        runtime_host: crate::RuntimeHostConfig,
        tools: Arc<dyn crate::ToolProvider>,
    ) -> crate::DurableProcessWorker {
        let plugin_host =
            crate::PluginHost::new(vec![Arc::new(crate::plugin::StaticPluginFactory::new(
                "worker-test-tools",
                crate::PluginSpec::new().with_tool_provider(tools),
            ))]);
        crate::DurableProcessWorker::new(
            crate::DurableProcessWorkerConfig::new(
                Arc::new(plugin_host),
                runtime_host,
                factory,
                registry,
            )
            .with_session_policy(standard_test_policy()),
        )
    }

    fn successful_text_response(text: &str) -> crate::LlmResponse {
        crate::LlmResponse {
            full_text: text.to_string(),
            parts: vec![crate::LlmOutputPart::Text {
                text: text.to_string(),
                response_meta: None,
            }],
            ..crate::LlmResponse::default()
        }
    }

    #[derive(Default)]
    struct RecordingTraceSink {
        records: std::sync::Mutex<Vec<lash_trace::TraceRecord>>,
    }

    impl RecordingTraceSink {
        fn records(&self) -> Vec<lash_trace::TraceRecord> {
            self.records.lock().expect("trace records").clone()
        }
    }

    impl lash_trace::TraceSink for RecordingTraceSink {
        fn append(
            &self,
            record: &lash_trace::TraceRecord,
        ) -> Result<(), lash_trace::TraceSinkError> {
            self.records
                .lock()
                .map_err(|_| lash_trace::TraceSinkError::LockPoisoned)?
                .push(record.clone());
            Ok(())
        }
    }

    /// A `SessionStoreFactory` that hands every rebuilt worker runtime the same
    /// shared store, so a worker-run process enqueues its wakes into the store
    /// the test inspects (the inline spawn used the live runtime's own store).
    struct SharedSessionStoreFactory {
        store: Arc<RecordingStore>,
    }

    #[async_trait::async_trait]
    impl crate::SessionStoreFactory for SharedSessionStoreFactory {
        async fn create_store(
            &self,
            _request: &crate::SessionStoreCreateRequest,
        ) -> Result<Arc<dyn crate::RuntimePersistence>, String> {
            Ok(Arc::clone(&self.store) as Arc<dyn crate::RuntimePersistence>)
        }

        async fn open_existing_store(
            &self,
            _request: &crate::SessionStoreCreateRequest,
        ) -> Result<Option<Arc<dyn crate::RuntimePersistence>>, String> {
            Ok(Some(
                Arc::clone(&self.store) as Arc<dyn crate::RuntimePersistence>
            ))
        }

        async fn delete_session(&self, _session_id: &str) -> Result<(), String> {
            Ok(())
        }
    }

    /// Spawn `count` independent inline [`ProcessWorkRunner`]s over the same
    /// registry, returning their pokes.
    ///
    /// One runner suffices for any nesting depth: the worker runs each claimed
    /// process on its own task, so a parent blocked awaiting a child does not
    /// park the runner — a later drive claims and runs the child. (The `count`
    /// parameter is retained for tests that deliberately exercise concurrent
    /// competing owners, where the lease fences double-execution.)
    fn spawn_inline_process_runners(
        worker: &crate::DurableProcessWorker,
        count: usize,
    ) -> Vec<crate::ProcessWorkPoke> {
        (0..count)
            .map(|_| crate::ProcessWorkRunner::inline(worker.clone()).spawn())
            .collect()
    }

    #[tokio::test]
    async fn lashlang_process_registration_serializes_refs_only() {
        let registration = lashlang_process_registration(
            "refs-only",
            ::lashlang::parse("process main() { finish 1 }").expect("parse module"),
            serde_json::Map::new(),
        )
        .await;

        let json = serde_json::to_string(&registration).expect("serialize registration");

        assert!(json.contains("module_ref"));
        assert!(json.contains("process_ref"));
        assert!(json.contains("host_requirements_ref"));
        assert!(!json.contains("linked_module"));
        assert!(!json.contains("canonical_ir"));
    }

    #[tokio::test]
    async fn lashlang_process_fails_clearly_when_module_artifact_is_missing() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let empty_artifact_store: Arc<dyn ::lashlang::LashlangArtifactStore> =
            Arc::new(::lashlang::InMemoryLashlangArtifactStore::new());
        let worker = process_worker_with_core(
            Arc::clone(&registry_dyn),
            factory as Arc<dyn crate::SessionStoreFactory>,
            {
                let mut config = crate::RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store = Arc::clone(&empty_artifact_store);
                config
            },
        );
        let registration = with_worker_execution_env_in_store(
            worker_registration(
                lashlang_process_registration(
                    "missing-artifact",
                    ::lashlang::parse("process main() { finish 1 }").expect("parse module"),
                    serde_json::Map::new(),
                )
                .await,
            ),
            empty_artifact_store.as_ref(),
        )
        .await;

        let output = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("worker returns process failure output");

        assert!(matches!(
            output,
            crate::ProcessAwaitOutput::Failure { code, .. }
                if code == "process_module_artifact_missing"
        ));
    }

    #[tokio::test]
    async fn durable_process_worker_runs_host_originated_tool_process() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let worker = process_worker(
            Arc::clone(&registry_dyn),
            factory as Arc<dyn crate::SessionStoreFactory>,
        );
        let registration = with_worker_execution_env_in_store(
            crate::ProcessRegistration::new(
                "worker-unstructured-scope",
                crate::ProcessInput::ToolCall {
                    call: crate::PreparedToolCall::from_parts(
                        "tool-call-unstructured",
                        "process_echo",
                        serde_json::json!({ "value": "tool" }),
                        None,
                        serde_json::Value::Null,
                    ),
                },
                crate::ProcessProvenance::host(),
            ),
            ::lashlang::global_in_memory_lashlang_artifact_store().as_ref(),
        )
        .await;

        let output = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("worker should run host-originated tool process");

        assert!(matches!(
            output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value == serde_json::json!({ "payload": "raw:tool" })
        ));
    }

    #[tokio::test]
    async fn durable_process_worker_rejects_empty_process_id_on_execution() {
        // Process execution identity is the persisted process_id; a retry/
        // recovery that presents an empty (fresh, non-persisted) id has lost its
        // idempotency anchor and must fail loudly, mirroring the empty-turn-id
        // rejection in `ExecutionScope`.
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let worker = process_worker(
            Arc::clone(&registry_dyn),
            factory as Arc<dyn crate::SessionStoreFactory>,
        );
        let registration =
            worker_registration_with_env(crate::ProcessRegistration::session_start_draft(
                "",
                crate::ProcessInput::ToolCall {
                    call: crate::PreparedToolCall::from_parts(
                        "tool-call-empty-id",
                        "process_echo",
                        serde_json::json!({ "value": "tool" }),
                        None,
                        serde_json::Value::Null,
                    ),
                },
            ))
            .await;

        let err = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect_err("worker should reject an empty/fresh process id on execution");

        assert!(
            err.to_string()
                .contains(crate::RuntimeErrorCode::MissingExecutionScopeId.as_str()),
            "{err}"
        );
    }

    /// Echo tool that also counts its invocations, so a test can witness how
    /// many times a process body actually ran (exactly-once vs. double-run).
    struct CountingEchoTool {
        runs: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for CountingEchoTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![process_echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "process_echo").then(|| Arc::new(process_echo_tool_definition().contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
            self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let value = call
                .args
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            crate::ToolResult::ok(serde_json::json!({ "payload": format!("raw:{value}") }))
        }
    }

    /// A worker whose only tool counts its runs into the shared counter. Two
    /// such workers over one registry are distinct owners sharing one witness.
    fn counting_worker(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
        runs: Arc<std::sync::atomic::AtomicUsize>,
    ) -> crate::DurableProcessWorker {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(CountingEchoTool { runs });
        let plugin_host =
            crate::PluginHost::new(vec![Arc::new(crate::plugin::StaticPluginFactory::new(
                "worker-counting-tools",
                crate::PluginSpec::new().with_tool_provider(tools),
            ))]);
        crate::DurableProcessWorker::new(
            crate::DurableProcessWorkerConfig::new(
                Arc::new(plugin_host),
                {
                    let mut config = crate::RuntimeHostConfig::in_memory();
                    config.providers.provider_resolver = Arc::new(
                        crate::SingleProviderResolver::new(mock_provider(Vec::new()).into_handle()),
                    );
                    config
                },
                factory,
                registry,
            )
            .with_session_policy(standard_test_policy()),
        )
    }

    /// A trigger/trigger-shaped registry row: a tool process written straight to
    /// the registry with no turn lease and no manager-driven start.
    async fn counting_registration(process_id: &str) -> crate::ProcessRegistration {
        worker_registration_with_env(crate::ProcessRegistration::session_start_draft(
            process_id,
            crate::ProcessInput::ToolCall {
                call: crate::PreparedToolCall::from_parts(
                    process_id,
                    "process_echo",
                    serde_json::json!({ "value": "out-of-turn" }),
                    None,
                    serde_json::Value::Null,
                ),
            },
        ))
        .await
    }

    #[tokio::test]
    async fn process_work_runner_drives_directly_registered_process_to_terminal_on_poke() {
        // Out-of-turn execution: a process is registered straight into the
        // registry (the trigger/trigger shape — no turn, no manager) and reaches
        // terminal promptly because the control seam's poke wakes the runner,
        // not because a separate boot-time recovery sweep eventually finds it.
        let registry: Arc<dyn crate::ProcessRegistry> =
            Arc::new(crate::TestLocalProcessRegistry::default());
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker = counting_worker(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            Arc::clone(&runs),
        );
        let poke = crate::ProcessWorkRunner::inline(worker).spawn();

        registry
            .register_process(counting_registration("proc-poke").await)
            .await
            .expect("register out-of-turn process");
        poke.poke();

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            registry.await_process("proc-poke"),
        )
        .await
        .expect("process reaches terminal promptly via the poke")
        .expect("await terminal output");
        assert!(matches!(output, crate::ProcessAwaitOutput::Success { .. }));
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the lease-protected runner runs the process exactly once"
        );
    }

    #[tokio::test]
    async fn concurrent_workers_run_a_directly_registered_process_exactly_once() {
        // Two independent workers (two lease owners) over the SAME registry, each
        // with its own runtime but a shared run counter. The `ProcessLease`
        // single-owner contract must fence them: exactly one claims and runs; the
        // other skips on claim conflict (or finds the row already terminal). This
        // is the no-double-execution guarantee at the worker layer — the registry
        // lease primitive itself is covered by the conformance suite
        // (`active_process_lease_fences_competing_owner`).
        let registry: Arc<dyn crate::ProcessRegistry> =
            Arc::new(crate::TestLocalProcessRegistry::default());
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_a = counting_worker(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            Arc::clone(&runs),
        );
        let worker_b = counting_worker(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            Arc::clone(&runs),
        );

        registry
            .register_process(counting_registration("proc-race").await)
            .await
            .expect("register out-of-turn process");

        let (a, b) = tokio::join!(
            worker_a.drive_pending_processes(),
            worker_b.drive_pending_processes(),
        );
        a.expect("worker a drive");
        b.expect("worker b drive");

        let output = registry
            .await_process("proc-race")
            .await
            .expect("await terminal output");
        assert!(matches!(output, crate::ProcessAwaitOutput::Success { .. }));
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the process must run exactly once across competing lease owners"
        );
    }

    #[derive(Clone, Default)]
    struct RecordingProcessEffectController {
        records: Arc<std::sync::Mutex<Vec<crate::RuntimeInvocation>>>,
    }

    impl RecordingProcessEffectController {
        fn records(&self) -> Vec<crate::RuntimeInvocation> {
            self.records.lock().expect("process effect records").clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for RecordingProcessEffectController {
        async fn execute_effect(
            &self,
            envelope: crate::RuntimeEffectEnvelope,
            local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            self.records
                .lock()
                .expect("process effect records")
                .push(envelope.invocation.clone());
            crate::InlineRuntimeEffectController
                .execute_effect(envelope, local_executor)
                .await
        }

        async fn await_event_key(
            &self,
            scope: &crate::ExecutionScope,
            wait: crate::AwaitEventWaitIdentity,
        ) -> Result<crate::AwaitEventKey, crate::RuntimeError> {
            crate::InlineRuntimeEffectController
                .await_event_key(scope, wait)
                .await
        }

        async fn resolve_await_event(
            &self,
            key: &crate::AwaitEventKey,
            resolution: crate::Resolution,
        ) -> Result<crate::ResolveOutcome, crate::RuntimeError> {
            crate::InlineRuntimeEffectController
                .resolve_await_event(key, resolution)
                .await
        }

        async fn await_await_event(
            &self,
            key: &crate::AwaitEventKey,
            cancel: tokio_util::sync::CancellationToken,
            deadline: Option<std::time::Instant>,
        ) -> Result<crate::Resolution, crate::RuntimeError> {
            crate::InlineRuntimeEffectController
                .await_await_event(key, cancel, deadline)
                .await
        }

        async fn revoke_await_events_for_session(
            &self,
            session_id: &str,
        ) -> Result<(), crate::RuntimeError> {
            crate::InlineRuntimeEffectController
                .revoke_await_events_for_session(session_id)
                .await
        }
    }

    #[derive(Clone)]
    struct RejectingDeploymentEffectHost {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::EffectHost for RejectingDeploymentEffectHost {
        fn scoped<'run>(
            &'run self,
            scope: crate::ExecutionScope,
        ) -> Result<crate::ScopedEffectController<'run>, crate::RuntimeError> {
            crate::ScopedEffectController::shared(
                Arc::new(RejectingEffectController {
                    attempts: Arc::clone(&self.attempts),
                }),
                scope,
            )
        }

        fn scoped_static(
            &self,
            scope: crate::ExecutionScope,
        ) -> Result<Option<crate::ScopedEffectController<'static>>, crate::RuntimeError> {
            Ok(Some(crate::ScopedEffectController::shared(
                Arc::new(RejectingEffectController {
                    attempts: Arc::clone(&self.attempts),
                }),
                scope,
            )?))
        }
    }

    struct RejectingEffectController {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::RuntimeEffectController for RejectingEffectController {
        async fn execute_effect(
            &self,
            _envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            self.attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(crate::RuntimeEffectControllerError::new(
                "rejecting_deployment_effect_host",
                "deployment effect host must not execute this child turn",
            ))
        }
    }

    fn rejecting_host_worker(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
    ) -> (
        crate::DurableProcessWorker,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker = process_worker_with_core(registry, factory, {
            let mut config = crate::RuntimeHostConfig::in_memory();
            config.control.effect_host = Arc::new(RejectingDeploymentEffectHost {
                attempts: Arc::clone(&attempts),
            });
            config.providers.provider_resolver = Arc::new(crate::SingleProviderResolver::new(
                mock_provider(vec![MockCall {
                    stream_events: Vec::new(),
                    response: Ok(successful_text_response("child done")),
                }])
                .into_handle(),
            ));
            config
        });
        (worker, attempts)
    }

    fn session_turn_registration(
        process_id: &str,
        child_session_id: &str,
    ) -> crate::ProcessRegistration {
        let child_policy = standard_test_policy();
        worker_registration(crate::ProcessRegistration::session_start_draft(
            process_id,
            crate::ProcessInput::SessionTurn {
                create_request: Box::new(
                    crate::SessionCreateRequest::child(
                        "root",
                        crate::SessionStartPoint::Empty,
                        child_policy,
                        crate::PluginOptions::default(),
                        "worker-test",
                    )
                    .with_session_id(child_session_id),
                ),
                turn_input: Box::new(crate::TurnInput::text("run child")),
                output_contract: crate::ToolOutputContract::Static,
            },
        ))
    }

    #[tokio::test]
    async fn session_turn_process_uses_supplied_scoped_controller() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let factory_dyn = Arc::clone(&factory) as Arc<dyn crate::SessionStoreFactory>;
        let (worker, rejecting_attempts) =
            rejecting_host_worker(Arc::clone(&registry_dyn), factory_dyn);
        let registration = session_turn_registration("scoped-session-turn", "scoped-child");
        let controller = RecordingProcessEffectController::default();
        let scoped_effect_controller = crate::ScopedEffectController::borrowed(
            &controller,
            crate::ExecutionScope::process("scoped-session-turn"),
        )
        .expect("process scoped controller");

        let output = worker
            .run_process_with_scoped_effect_controller(
                registration,
                crate::ProcessExecutionContext::default(),
                scoped_effect_controller,
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("session turn process should run");

        assert!(matches!(output, crate::ProcessAwaitOutput::Success { .. }));
        assert_eq!(
            rejecting_attempts.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "explicit scoped execution must not touch the deployment effect host"
        );
        let records = controller.records();
        assert!(
            records
                .iter()
                .any(|record| record.scope.turn_id.as_deref() == Some("scoped-session-turn")),
            "child turn effects should use the process id as the turn id: {records:#?}"
        );
        assert!(
            records
                .iter()
                .all(|record| { record.scope.turn_id.as_deref() == Some("scoped-session-turn") }),
            "all child turn effects should stay in the child turn scope: {records:#?}"
        );
    }

    #[tokio::test]
    async fn session_turn_process_fails_on_deployment_effect_host_fallback() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let factory_dyn = Arc::clone(&factory) as Arc<dyn crate::SessionStoreFactory>;
        let (worker, rejecting_attempts) =
            rejecting_host_worker(Arc::clone(&registry_dyn), factory_dyn);
        let registration = session_turn_registration("fallback-session-turn", "fallback-child");

        let output = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("process runner should return process output");

        let crate::ProcessAwaitOutput::Failure { code, message, .. } = output else {
            panic!("expected fallback to fail, got {output:#?}");
        };
        assert_eq!(code, "process_session_turn_failed");
        assert!(
            !message.trim().is_empty(),
            "fallback should surface a process failure message"
        );
        assert!(
            rejecting_attempts.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "fallback path should execute through the rejecting deployment effect host"
        );
    }

    #[tokio::test]
    async fn durable_process_worker_rebuilds_context_for_tool_lashlang_and_session_turn() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let factory_dyn = Arc::clone(&factory) as Arc<dyn crate::SessionStoreFactory>;
        let worker = process_worker(Arc::clone(&registry_dyn), factory_dyn);

        let tool_registration =
            worker_registration_with_env(crate::ProcessRegistration::session_start_draft(
                "worker-tool",
                crate::ProcessInput::ToolCall {
                    call: crate::PreparedToolCall::from_parts(
                        "tool-call-1",
                        "process_echo",
                        serde_json::json!({ "value": "tool" }),
                        None,
                        serde_json::Value::Null,
                    ),
                },
            ))
            .await;
        registry_dyn
            .register_process(tool_registration.clone())
            .await
            .expect("register tool process");
        let tool_output = worker
            .run_process(
                tool_registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("run tool process");
        assert!(matches!(
            tool_output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value == serde_json::json!({ "payload": "raw:tool" })
        ));

        let mut args = serde_json::Map::new();
        args.insert("value".to_string(), serde_json::json!("linked"));
        let lashlang_registration = worker_registration_with_env(
            lashlang_process_registration(
                "worker-lashlang",
                ::lashlang::parse(
                    r#"
                process main(value: str) {
                  finish { value: value }
                }
                "#,
                )
                .expect("process module"),
                args,
            )
            .await,
        )
        .await;
        let crate::ProcessInput::LashlangProcess { module_ref, .. } =
            lashlang_registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        let artifact = ::lashlang::global_in_memory_lashlang_artifact_store()
            .get_module_artifact(module_ref)
            .await
            .expect("load lashlang module artifact")
            .expect("lashlang module artifact exists");
        let requirements = ::lashlang::host_requirements_for_program(&artifact.canonical_ir);
        let relink_without_process_abilities = ::lashlang::LinkedModule::link(
            artifact.canonical_ir.clone(),
            ::lashlang::LashlangHostEnvironment::new(
                requirements.resources,
                ::lashlang::LashlangAbilities::default(),
            ),
        )
        .expect_err("current worker host abilities should not be required to relink");
        assert!(
            relink_without_process_abilities
                .to_string()
                .contains("lashlang feature `processes` is disabled by this host"),
            "{relink_without_process_abilities}"
        );
        registry_dyn
            .register_process(lashlang_registration.clone())
            .await
            .expect("register lashlang process");
        let lashlang_output = worker
            .run_process(
                lashlang_registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("run lashlang process");
        assert!(matches!(
            lashlang_output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value == serde_json::json!({ "value": "linked" })
        ));

        let child_policy = standard_test_policy();
        let session_registration =
            worker_registration(crate::ProcessRegistration::session_start_draft(
                "worker-session",
                crate::ProcessInput::SessionTurn {
                    create_request: Box::new(crate::SessionCreateRequest::child(
                        "root",
                        crate::SessionStartPoint::Empty,
                        child_policy,
                        crate::PluginOptions::default(),
                        "worker-test",
                    )),
                    turn_input: Box::new(crate::TurnInput::text("run child")),
                    output_contract: crate::ToolOutputContract::Static,
                },
            ));
        registry_dyn
            .register_process(session_registration.clone())
            .await
            .expect("register session process");
        let session_output = worker
            .run_process(
                session_registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("run session process");
        assert!(matches!(
            session_output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value.get("turn").is_some()
                    && value.get("child_session_id").and_then(serde_json::Value::as_str).is_some()
        ));
    }

    #[tokio::test]
    async fn lashlang_process_runs_with_input_events_wake_and_receiver_operation() {
        let (runtime, store) =
            runtime_with_processes_and_tools_and_store(Vec::new(), Arc::new(ProcessEchoTool)).await;
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        // The runner is the sole executor now that the inline per-start spawn is
        // gone. Run its worker against the live runtime's own store so a
        // worker-run process enqueues its wake into the store this test inspects.
        let worker = process_worker_with_core(
            Arc::clone(&registry),
            Arc::new(SharedSessionStoreFactory {
                store: Arc::clone(&store),
            }),
            crate::RuntimeHostConfig::in_memory(),
        );
        let poke = spawn_inline_process_runners(&worker, 1);
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        let mut input = serde_json::Map::new();
        input.insert("root".to_string(), serde_json::json!("seed"));
        input.insert(
            "tool".to_string(),
            serde_json::to_value(::lashlang::Value::Resource(
                ::lashlang::ResourceHandle::new("Tools", "tools"),
            ))
            .expect("resource handle json"),
        );
        let program = ::lashlang::parse(
            r#"
            process main(root: str, tool: Tools) {
              yield root
              called = await tool.process_echo({ value: root })?
              wake called.payload
              nested = await tool.process_echo({ value: "nested" })?
              finish { first: called.payload, nested: nested.payload }
            }
            "#,
        )
        .expect("lashlang process body");
        let registration = lashlang_process_registration("process-1", program, input).await;

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("block")),
                ),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("start process");
        poke[0].poke();
        let output = manager
            .processes
            .await_process(
                &manager.current,
                "process-1",
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("await process");

        let crate::ProcessAwaitOutput::Success { value, .. } = output else {
            panic!("process should succeed");
        };
        assert_eq!(
            value,
            serde_json::json!({ "first": "raw:seed", "nested": "raw:nested" })
        );
        let registry = manager
            .current
            .host
            .process_registry
            .as_ref()
            .expect("process registry");
        let events = registry.events_after("process-1", 0).await.expect("events");
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "process.yield"
                    && event.payload["value"] == serde_json::json!("seed"))
        );
        assert!(events.iter().any(|event| event.event_type == "process.wake"
            && event.payload["text"] == serde_json::json!("raw:seed")));
        let wake_event = events
            .iter()
            .find(|event| event.event_type == "process.wake")
            .expect("wake event");
        // M1: wake/yield emissions carry a deterministic per-process ordinal
        // replay key, so a crash-recovery replay re-issues the same key and the
        // append dedupes instead of redelivering a duplicate wake.
        assert!(
            wake_event
                .invocation
                .replay_key()
                .is_some_and(|key| key.starts_with("process:process-1:event:")),
            "wake emission must carry a deterministic replay key, got {:?}",
            wake_event.invocation.replay_key(),
        );
        let wake_sequence = wake_event.sequence;
        let completed_sequence = events
            .iter()
            .find(|event| event.event_type == "process.completed")
            .expect("completed event")
            .sequence;
        assert!(
            wake_sequence < completed_sequence,
            "process.wake should be committed before process completion"
        );
        // The runner is the sole executor and reconstructs a process's wake
        // target from its persisted provenance (the creator scope), so a
        // runner-driven process wakes its creator session `root`. (The registry
        // record does not persist a start-time custom wake target, so wake
        // delivery follows provenance, matching the durable recovery path.)
        let queued = crate::store::RuntimePersistence::list_queued_work(store.as_ref(), "root")
            .await
            .expect("queued wake");
        assert_eq!(queued.len(), 1);
        assert_eq!(
            queued[0].delivery_policy,
            crate::DeliveryPolicy::EarliestSafeBoundary
        );
        assert_eq!(queued[0].items.len(), 1);
        let crate::QueuedWorkPayload::ProcessWake { wake } = &queued[0].items[0].payload else {
            panic!("expected process wake queue payload");
        };
        assert_eq!(wake.input, "raw:seed");
    }

    #[tokio::test]
    async fn lashlang_resource_operation_tool_call_ids_are_deterministic_for_call_site() {
        let registry: Arc<dyn crate::ProcessRegistry> =
            Arc::new(crate::TestLocalProcessRegistry::default());
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let tool = RecordingProcessEchoTool::default();
        let worker = process_worker_with_tools(
            Arc::clone(&registry),
            factory as Arc<dyn crate::SessionStoreFactory>,
            crate::RuntimeHostConfig::in_memory(),
            Arc::new(tool.clone()),
        );
        let mut input = serde_json::Map::new();
        input.insert(
            "tool".to_string(),
            serde_json::to_value(::lashlang::Value::Resource(
                ::lashlang::ResourceHandle::new("Tools", "tools"),
            ))
            .expect("resource handle json"),
        );
        let program = ::lashlang::parse(
            r#"
            process main(tool: Tools) {
              first = await tool.process_echo({ value: "first" })?
              second = await tool.process_echo({ value: "second" })?
              finish { first: first.payload, second: second.payload }
            }
            "#,
        )
        .expect("lashlang process body");
        let registration = worker_registration_with_env(
            lashlang_process_registration("resource-determinism", program, input).await,
        )
        .await;

        for _ in 0..2 {
            let output = worker
                .run_process(
                    registration.clone(),
                    crate::ProcessExecutionContext::default(),
                    tokio_util::sync::CancellationToken::new(),
                )
                .await
                .expect("run lashlang process");
            assert!(matches!(output, crate::ProcessAwaitOutput::Success { .. }));
        }

        let calls = tool.calls();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0].tool_call_id, calls[2].tool_call_id);
        assert_eq!(calls[1].tool_call_id, calls[3].tool_call_id);
        assert_eq!(calls[0].replay_key, calls[2].replay_key);
        assert_eq!(calls[1].replay_key, calls[3].replay_key);
        assert_ne!(calls[0].tool_call_id, calls[1].tool_call_id);
        for call in &calls {
            let call_id = call.tool_call_id.as_deref().expect("tool call id");
            assert!(
                call_id.starts_with("lashlang:resource-determinism:resource:process_echo:"),
                "unexpected lashlang resource call id: {call_id}"
            );
            assert_eq!(
                call.runtime_process_id.as_deref(),
                Some("resource-determinism")
            );
            assert!(
                call.replay_key
                    .as_deref()
                    .is_some_and(|key| key.contains(call_id)),
                "replay key should include deterministic call id: {:?}",
                call.replay_key
            );
        }
    }

    #[test]
    fn lashlang_process_event_payload_is_stable_without_timestamp() {
        let value = ::lashlang::Value::String("ready".into());

        let first = crate::lashlang_bridge::process_event_payload(&value).expect("payload");
        let second = crate::lashlang_bridge::process_event_payload(&value).expect("payload");

        assert_eq!(first, second);
        assert_eq!(first["value"], serde_json::json!("ready"));
        assert_eq!(first["text"], serde_json::json!("ready"));
        assert!(first.get("timestamp").is_none());
    }

    #[tokio::test]
    async fn lashlang_process_uses_artifact_refs_for_nested_starts() {
        let runtime = runtime_with_processes(Vec::new()).await;
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        // The parent process blocks awaiting its nested child, so a single
        // sequential runner drive would starve the child. Two lease-fenced
        // runners restore the per-process concurrency the deleted detached spawn
        // provided without double-running either process.
        let worker = process_worker_with_core(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            crate::RuntimeHostConfig::in_memory(),
        );
        let pokes = spawn_inline_process_runners(&worker, 1);
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        assert!(
            !manager.current.plugins.lashlang_abilities().processes,
            "current plugin surface should not provide process linking"
        );

        let program = ::lashlang::parse(
            r#"
            process child(value: str) {
              finish { child: value }
            }
            process main() {
              handle = start child(value: "stored")
              result = (await handle)?
              finish { nested: result.child }
            }
            "#,
        )
        .expect("process module");
        let registration =
            lashlang_process_registration("snapshot-parent", program, serde_json::Map::new()).await;
        let crate::ProcessInput::LashlangProcess { module_ref, .. } = registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        let artifact = ::lashlang::global_in_memory_lashlang_artifact_store()
            .get_module_artifact(module_ref)
            .await
            .expect("load lashlang module artifact")
            .expect("lashlang module artifact exists");
        let requirements = ::lashlang::host_requirements_for_program(&artifact.canonical_ir);
        let relink = ::lashlang::LinkedModule::link(
            artifact.canonical_ir.clone(),
            ::lashlang::LashlangHostEnvironment::new(
                requirements.resources,
                manager.current.plugins.lashlang_abilities(),
            ),
        )
        .expect_err("current host surface should not be able to link this module");
        assert!(
            relink
                .to_string()
                .contains("lashlang feature `processes` is disabled by this host"),
            "{relink}"
        );

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("snapshot-parent")),
                ),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("start process");
        pokes[0].poke();
        let output = manager
            .processes
            .await_process(
                &manager.current,
                "snapshot-parent",
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("await process");

        assert!(matches!(
            output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value == serde_json::json!({ "nested": "stored" })
        ));
    }

    #[tokio::test]
    async fn lashlang_execution_uses_dedicated_sink_only() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let normal_trace = Arc::new(RecordingTraceSink::default());
        let process_trace = Arc::new(RecordingTraceSink::default());
        let worker = process_worker_with_core(
            Arc::clone(&registry_dyn),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            {
                let mut config = crate::RuntimeHostConfig::in_memory();
                config.tracing.trace_sink =
                    Some(Arc::clone(&normal_trace) as Arc<dyn lash_trace::TraceSink>);
                config.tracing.lashlang_execution_sink =
                    Some(Arc::clone(&process_trace) as Arc<dyn lash_trace::TraceSink>);
                config
            },
        );

        let mut args = serde_json::Map::new();
        args.insert("flag".to_string(), serde_json::json!(true));
        let registration = worker_registration_with_env(
            lashlang_process_registration(
                "tracking-parent",
                ::lashlang::parse(
                    r#"
                process child() {
                  finish "child"
                }
                process main(flag: bool) {
                  if flag {
                    handle = start child()
                    yield "started"
                    finish "parent"
                  } else {
                    fail "else"
                  }
                }
                "#,
                )
                .expect("tracking process"),
                args,
            )
            .await,
        )
        .await;
        registry_dyn
            .register_process(registration.clone())
            .await
            .expect("register tracking parent");

        let output = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("run tracking parent");

        assert!(matches!(
            output,
            crate::ProcessAwaitOutput::Success { value, .. }
                if value == serde_json::json!("parent")
        ));
        assert!(
            normal_trace.records().iter().all(|record| {
                !matches!(
                    record.event,
                    lash_trace::TraceEvent::LashlangExecution { .. }
                )
            }),
            "normal trace sink must not receive Lashlang execution events"
        );

        let records = process_trace.records();
        let events = records
            .iter()
            .filter_map(|record| match &record.event {
                lash_trace::TraceEvent::LashlangExecution { event } => Some(event),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            events.iter().any(|event| matches!(
                event,
                lash_trace::TraceLashlangExecutionEvent::ExecutionStarted { execution_map, .. }
                    if execution_map.nodes.iter().any(|node| node.kind == "branch")
                        && execution_map.edges.iter().any(|edge| edge.label == "then")
            )),
            "started event should carry the static Lashlang map: {events:?}"
        );
        assert!(events.iter().any(|event| {
            matches!(
                event,
                lash_trace::TraceLashlangExecutionEvent::BranchSelected {
                    selected: lash_trace::TraceBranchSelection::Then,
                    ..
                }
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                lash_trace::TraceLashlangExecutionEvent::ChildStarted {
                    child,
                    ..
                } if matches!(
                    &child.subject,
                    lash_trace::TraceRuntimeSubject::Process { process_id }
                        if process_id != "tracking-parent"
                ) && child.entry_name.as_deref() == Some("child")
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                lash_trace::TraceLashlangExecutionEvent::NodeCompleted { label, .. }
                    if label == "result"
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                lash_trace::TraceLashlangExecutionEvent::ExecutionFinished {
                    status: lash_trace::TraceLashlangStatus::Completed,
                    ..
                }
            )
        }));
    }

    #[tokio::test]
    async fn lashlang_process_lifecycle_wait_signal_signal_run_and_sleep() {
        let mut runtime = runtime_with_processes(Vec::new()).await;
        let controller = RecordingProcessEffectController::default();
        runtime.host.core.control.effect_host =
            Arc::new(crate::InlineEffectHost::new(Arc::new(controller.clone())));
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        // The runner is the sole executor; run its worker against the recording
        // controller so the signaler's sleep effect is observed exactly as the
        // inline run observed it. Two lease-fenced runners run the target (which
        // blocks on `wait_signal`) and the signaler concurrently.
        let worker = process_worker_with_core(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            {
                let mut config = crate::RuntimeHostConfig::in_memory();
                config.control.effect_host =
                    Arc::new(crate::InlineEffectHost::new(Arc::new(controller.clone())));
                config
            },
        );
        let pokes = spawn_inline_process_runners(&worker, 1);

        let target = lashlang_process_registration(
            "signal-target",
            ::lashlang::parse(
                r#"
                process main() signals { ready: any } {
                  value = wait_signal("ready")
                  finish value
                }
                "#,
            )
            .expect("target process"),
            serde_json::Map::new(),
        )
        .await;
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                target,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("target")),
                ),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("start target");
        pokes[0].poke();

        let mut signaler_args = serde_json::Map::new();
        signaler_args.insert(
            "target".to_string(),
            crate::lashlang_bridge::process_handle_json("signal-target"),
        );
        let signaler = lashlang_process_registration(
            "signal-sender",
            ::lashlang::parse(
                r#"
                process main(target: any) {
                  sleep for "0ms"
                  signal_run(target, "ready", { ok: true })
                  finish "sent"
                }
                "#,
            )
            .expect("signaler process"),
            signaler_args,
        )
        .await;
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                signaler,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("signaler")),
                ),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("start signaler");
        pokes[0].poke();

        let signaler_output = manager
            .processes
            .await_process(
                &manager.current,
                "signal-sender",
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("await signaler");
        assert!(
            matches!(
                &signaler_output,
                crate::ProcessAwaitOutput::Success { value, .. } if *value == serde_json::json!("sent")
            ),
            "unexpected signaler output: {signaler_output:?}"
        );

        let target_output = manager
            .processes
            .await_process(
                &manager.current,
                "signal-target",
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("await target");
        assert!(matches!(
            target_output,
            crate::ProcessAwaitOutput::Success { value, .. } if value == serde_json::json!({ "ok": true })
        ));
        let events = registry
            .events_after("signal-target", 0)
            .await
            .expect("target events");
        assert!(events.iter().any(|event| event.event_type == "signal.ready"
            && event.payload == serde_json::json!({ "ok": true })));
        let waiting_events = events
            .iter()
            .filter(|event| event.event_type == "process.waiting")
            .collect::<Vec<_>>();
        assert_eq!(waiting_events.len(), 1);
        let wait = serde_json::from_value::<crate::WaitState>(
            waiting_events[0]
                .payload
                .get("wait")
                .expect("wait payload")
                .clone(),
        )
        .expect("decode wait state");
        assert!(wait.since_ms > 0);
        assert!(matches!(
            wait.kind,
            crate::WaitKind::Signal {
                ref name,
                ref event_type,
                ordinal: 1,
                ..
            } if name == "ready" && event_type == "signal.ready"
        ));
        assert!(events.iter().any(|event| {
            event.event_type == "process.resumed"
                && event
                    .payload
                    .get("signal")
                    .and_then(serde_json::Value::as_str)
                    == Some("ready")
        }));
        assert_eq!(
            registry
                .get_process("signal-target")
                .await
                .expect("target record")
                .wait,
            None
        );
        let sleep_records = controller
            .records()
            .into_iter()
            .filter(|record| record.effect_kind() == Some(crate::RuntimeEffectKind::Sleep))
            .collect::<Vec<_>>();
        assert_eq!(sleep_records.len(), 1);
        assert!(
            sleep_records[0]
                .replay_key()
                .expect("replay key")
                .contains("signal-sender")
        );
    }

    #[tokio::test]
    async fn lashlang_process_failure_retains_raw_value() {
        let runtime = runtime_with_processes_and_tools(Vec::new(), Arc::new(ProcessEchoTool)).await;
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        let worker = process_worker_with_core(
            Arc::clone(&registry),
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            crate::RuntimeHostConfig::in_memory(),
        );
        let pokes = spawn_inline_process_runners(&worker, 1);
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        let program = ::lashlang::Program::block(vec![::lashlang::Expr::Fail(Box::new(
            ::lashlang::Expr::Record(vec![(
                "reason".into(),
                ::lashlang::Expr::String("bad".into()),
            )]),
        ))]);
        let registration =
            lashlang_process_registration("process-fail", program, serde_json::Map::new()).await;

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("fail")),
                ),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("start process");
        pokes[0].poke();
        let output = manager
            .processes
            .await_process(
                &manager.current,
                "process-fail",
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("await process");

        let crate::ProcessAwaitOutput::Failure {
            message, raw, code, ..
        } = output
        else {
            panic!("process should fail");
        };
        assert_eq!(code, "process_failed");
        assert_eq!(raw, Some(serde_json::json!({ "reason": "bad" })));
        assert!(message.contains("reason"));
        assert!(message.contains("bad"));
    }

    #[tokio::test]
    async fn lashlang_process_has_no_parent_capture_or_tool_name_fallback() {
        let no_parent_program = ::lashlang::Program::block(vec![::lashlang::Expr::Finish(Some(
            Box::new(::lashlang::Expr::Variable("parent".into())),
        ))]);
        let err = try_lashlang_process_registration(
            "process-no-parent",
            no_parent_program,
            serde_json::Map::new(),
        )
        .await
        .expect_err("linked process should reject parent capture");
        assert!(err.to_string().contains("unknown name `parent`"), "{err}");

        let fallback_program =
            ::lashlang::parse(r#"process main() { finish await process_echo({ value: "x" })? }"#)
                .expect("fallback process body");
        let err = try_lashlang_process_registration(
            "process-no-fallback",
            fallback_program,
            serde_json::Map::new(),
        )
        .await
        .expect_err("bare operation-name fallback should be rejected by linker");
        assert!(
            err.to_string()
                .contains("tools must be called through module paths, e.g. `tools.process_echo`"),
            "{err}"
        );
    }

    async fn process_event_projection(
        registry: &Arc<dyn crate::ProcessRegistry>,
        process_ids: &[&str],
    ) -> Vec<(String, Vec<(u64, String, serde_json::Value)>)> {
        let mut projection = Vec::new();
        for process_id in process_ids {
            let events = registry
                .events_after(process_id, 0)
                .await
                .expect("process events")
                .into_iter()
                .map(|event| (event.sequence, event.event_type, event.payload))
                .collect();
            projection.push(((*process_id).to_string(), events));
        }
        projection
    }

    #[tokio::test]
    async fn cancel_unreferenced_process_handles_revokes_current_grants_and_cancels_only_unowned() {
        let runtime = runtime_with_processes(Vec::new()).await;
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        let current_scope = crate::SessionScope::new("root");
        let other_scope = crate::SessionScope::new("other");
        let process_ids = ["keep", "sole", "shared"];

        for process_id in process_ids {
            register_open_process(&registry, process_id).await;
            grant_handle(&registry, &current_scope, process_id).await;
        }
        grant_handle(&registry, &other_scope, "shared").await;

        let events_before = process_event_projection(&registry, &process_ids).await;
        let cancelled = manager
            .processes
            .cancel_unreferenced_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                vec!["keep".to_string()],
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("cancel unreferenced handles");

        assert_eq!(
            cancelled
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            vec!["sole"]
        );
        assert_eq!(
            registry
                .list_handle_grants(&current_scope)
                .await
                .expect("current grants")
                .into_iter()
                .map(|(grant, _)| grant.process_id)
                .collect::<Vec<_>>(),
            vec!["keep".to_string()]
        );
        assert!(
            registry
                .handle_grants_for_process("sole")
                .await
                .expect("sole grants")
                .is_empty()
        );
        assert_eq!(
            registry
                .handle_grants_for_process("shared")
                .await
                .expect("shared grants")
                .into_iter()
                .map(|grant| grant.session_id)
                .collect::<Vec<_>>(),
            vec!["other".to_string()]
        );
        let events_after = process_event_projection(&registry, &process_ids).await;
        assert_eq!(events_after[0], events_before[0]);
        assert_eq!(events_after[2], events_before[2]);
        assert!(events_after[1].1.iter().any(|(_, event_type, payload)| {
            event_type == "process.cancel_requested"
                && payload["reason"] == serde_json::json!("requested by host")
        }));
    }

    #[tokio::test]
    async fn scoped_transfer_and_cleanup_use_process_effect_controller_metadata() {
        let runtime = runtime_with_processes(Vec::new()).await;
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        let controller = RecordingProcessEffectController::default();
        let metadata = crate::RuntimeInvocation::effect(
            crate::RuntimeScope::for_turn("root", "turn-process-scope", 1, 0),
            "parent-process-control",
            crate::RuntimeEffectKind::Process,
            "root:turn-process-scope:process-control",
        );
        let scoped_request = || {
            crate::ProcessOpScope::new(
                crate::ScopedEffectController::borrowed(
                    &controller,
                    crate::ExecutionScope::runtime_operation("scoped-process-transfer-test"),
                )
                .expect("scoped process transfer scope"),
            )
            .with_parent_invocation(Some(metadata.clone()))
        };
        let root_scope = crate::SessionScope::new("root");
        let target_scope = crate::SessionScope::new("target");

        register_open_process(&registry, "transfer-me").await;
        grant_handle(&registry, &root_scope, "transfer-me").await;
        manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                "target",
                vec!["transfer-me".to_string()],
                scoped_request(),
            )
            .await
            .expect("transfer handles");
        assert!(
            registry
                .list_handle_grants(&target_scope)
                .await
                .expect("target grants")
                .into_iter()
                .any(|(grant, _)| { grant.process_id == "transfer-me" })
        );

        register_open_process(&registry, "cleanup-me").await;
        grant_handle(&registry, &root_scope, "cleanup-me").await;
        manager
            .processes
            .cancel_unreferenced_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                Vec::<String>::new(),
                scoped_request(),
            )
            .await
            .expect("cleanup handles");

        let records = controller.records();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| {
            record.effect_kind() == Some(crate::RuntimeEffectKind::Process)
                && record.scope.turn_id.as_deref() == Some("turn-process-scope")
                && record
                    .replay_key()
                    .expect("replay key")
                    .starts_with("root:turn-process-scope:process-control:process:")
        }));
    }

    #[tokio::test]
    async fn processes_fails_loudly_when_process_registry_is_unavailable() {
        let mut runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
        runtime.host.process_registry = None;
        let manager =
            RuntimeSessionServices::new(&runtime, true, None).expect("runtime session manager");

        manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                "target",
                Vec::<String>::new(),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect("empty transfer remains a no-op");

        let validation = manager
            .processes
            .validate_process_handles_visible(
                &manager.current,
                &manager.managed,
                "root",
                &["missing".to_string()],
                test_process_op_scope("host-configured"),
            )
            .await
            .expect_err("validation should fail");
        assert!(
            validation
                .to_string()
                .contains("process registry is unavailable")
        );

        let transfer = manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                "target",
                vec!["missing".to_string()],
                test_process_op_scope("host-configured"),
            )
            .await
            .expect_err("transfer should fail");
        assert!(
            transfer
                .to_string()
                .contains("process registry is unavailable")
        );

        let cleanup = manager
            .processes
            .cancel_unreferenced_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                Vec::<String>::new(),
                test_process_op_scope("host-configured"),
            )
            .await
            .expect_err("cleanup should fail");
        assert!(
            cleanup
                .to_string()
                .contains("process registry is unavailable")
        );
    }
}
