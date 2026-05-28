use super::*;

mod control;
mod lashlang;
mod runner;
mod session;
mod tool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::helpers::{
        MockCall, RecordingStore, mock_provider, runtime_with_plugins,
        runtime_with_plugins_and_tools, runtime_with_plugins_and_tools_and_host_and_store,
        standard_test_policy, test_host_config,
    };
    use ::lashlang::LashlangArtifactStore;
    use std::sync::Arc;

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
        )
        .with_extra_event_types([probe_event_type()])
    }

    fn lashlang_process_registration(
        process_id: &str,
        program: ::lashlang::Program,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> crate::ProcessRegistration {
        try_lashlang_process_registration(process_id, program, args)
            .expect("link lashlang test module")
    }

    fn try_lashlang_process_registration(
        process_id: &str,
        program: ::lashlang::Program,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Result<crate::ProcessRegistration, ::lashlang::LinkError> {
        let module = if program.process("main").is_some() {
            program
        } else {
            ::lashlang::Program {
                declarations: vec![::lashlang::Declaration::Process(::lashlang::ProcessDecl {
                    name: "main".into(),
                    params: Vec::new(),
                    return_ty: None,
                    body: program.main,
                })],
                main: ::lashlang::Expr::Block(Vec::new()),
                declaration_spans: Vec::new(),
                expression_spans: Vec::new(),
            }
        };
        let mut resources = ::lashlang::ResourceCatalog::new();
        resources.add_module_instance(["tools"], "Tools");
        resources.add_operation("Tools", "process_echo", "process_echo");
        let linked_module = ::lashlang::LinkedModule::link(
            module,
            ::lashlang::LashlangSurface::new(
                resources,
                ::lashlang::LashlangAbilities::default()
                    .with_processes()
                    .with_process_lifecycle(),
            ),
        )?;
        ::lashlang::global_in_memory_lashlang_artifact_store()
            .put_module_artifact(&linked_module.artifact)
            .expect("store lashlang test module artifact");
        let process_ref = linked_module
            .artifact
            .process_ref("main")
            .expect("main process ref")
            .clone();
        Ok(crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_ref: linked_module.module_ref,
                process_ref,
                required_surface_ref: linked_module.required_surface_ref,
                process_name: "main".to_string(),
                args,
            },
        )
        .with_extra_event_types(crate::lashlang_process_event_types()))
    }

    struct ProcessEchoTool;

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

    async fn grant_handle(
        registry: &Arc<dyn crate::ProcessRegistry>,
        owner_scope: &crate::ProcessScope,
        process_id: &str,
    ) {
        registry
            .grant_handle(
                owner_scope,
                process_id,
                crate::ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant handle");
    }

    fn worker_registration(registration: crate::ProcessRegistration) -> crate::ProcessRegistration {
        registration.with_process_provenance(crate::ProcessProvenance::new(
            crate::ProcessScope::new("root"),
            "worker-profile",
        ))
    }

    fn process_worker(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
    ) -> crate::DurableProcessWorker {
        process_worker_with_core(
            registry,
            factory,
            crate::RuntimeCoreConfig::default().with_host_profile_id("worker-profile"),
        )
    }

    fn process_worker_with_core(
        registry: Arc<dyn crate::ProcessRegistry>,
        factory: Arc<dyn crate::SessionStoreFactory>,
        runtime_core: crate::RuntimeCoreConfig,
    ) -> crate::DurableProcessWorker {
        let tools: Arc<dyn crate::ToolProvider> = Arc::new(ProcessEchoTool);
        let plugin_host =
            crate::PluginHost::new(vec![Arc::new(crate::plugin::StaticPluginFactory::new(
                "worker-test-tools",
                crate::PluginSpec::new().with_tool_provider(tools),
            ))]);
        crate::DurableProcessWorker::new(
            crate::DurableProcessWorkerConfig::new(
                Arc::new(plugin_host),
                runtime_core,
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

    #[test]
    fn lashlang_process_registration_serializes_refs_only() {
        let registration = lashlang_process_registration(
            "refs-only",
            ::lashlang::parse("process main() { finish 1 }").expect("parse module"),
            serde_json::Map::new(),
        );

        let json = serde_json::to_string(&registration).expect("serialize registration");

        assert!(json.contains("module_ref"));
        assert!(json.contains("process_ref"));
        assert!(json.contains("required_surface_ref"));
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
            crate::RuntimeCoreConfig::default()
                .with_host_profile_id("worker-profile")
                .with_lashlang_artifact_store(empty_artifact_store),
        );
        let registration = worker_registration(lashlang_process_registration(
            "missing-artifact",
            ::lashlang::parse("process main() { finish 1 }").expect("parse module"),
            serde_json::Map::new(),
        ));

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
    async fn durable_process_worker_requires_structured_creator_scope() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let worker = process_worker(
            Arc::clone(&registry_dyn),
            factory as Arc<dyn crate::SessionStoreFactory>,
        );
        let mut registration = crate::ProcessRegistration::new(
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
        );
        registration.provenance =
            crate::ProcessProvenance::new(crate::ProcessScope::new(""), "worker-profile");

        let err = worker
            .run_process(
                registration,
                crate::ProcessExecutionContext::default(),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect_err("worker should require creator session provenance");

        assert!(
            err.to_string().contains("missing a structured owner scope"),
            "{err}"
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
            crate::InlineRuntimeEffectController::default()
                .execute_effect(envelope, local_executor)
                .await
        }
    }

    #[tokio::test]
    async fn durable_process_worker_rebuilds_context_for_tool_lashlang_and_session_turn() {
        let registry = Arc::new(crate::TestLocalProcessRegistry::default());
        let registry_dyn = Arc::clone(&registry) as Arc<dyn crate::ProcessRegistry>;
        let factory =
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
        let factory_dyn = Arc::clone(&factory) as Arc<dyn crate::SessionStoreFactory>;
        let worker = process_worker(Arc::clone(&registry_dyn), factory_dyn);

        let tool_registration = worker_registration(crate::ProcessRegistration::new(
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
        ));
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
        let lashlang_registration = worker_registration(lashlang_process_registration(
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
        ));
        let crate::ProcessInput::LashlangProcess { module_ref, .. } =
            lashlang_registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        let artifact = ::lashlang::global_in_memory_lashlang_artifact_store()
            .get_module_artifact(module_ref)
            .expect("load lashlang module artifact")
            .expect("lashlang module artifact exists");
        let requirements = ::lashlang::surface_requirements_for_program(&artifact.canonical_ir);
        let relink_without_process_abilities = ::lashlang::LinkedModule::link(
            artifact.canonical_ir.clone(),
            ::lashlang::LashlangSurface::new(
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

        let child_policy = crate::SessionPolicy {
            provider: mock_provider(vec![MockCall {
                stream_events: Vec::new(),
                response: Ok(successful_text_response("child done")),
            }])
            .into_handle(),
            ..standard_test_policy()
        };
        let session_registration = worker_registration(crate::ProcessRegistration::new(
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
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let wake_target = manager
            .managed
            .create_session(
                &manager.current,
                &manager.usage,
                crate::SessionCreateRequest::root(
                    crate::SessionStartPoint::Empty,
                    crate::PluginOptions::default(),
                )
                .with_session_id("wake-target")
                .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
            )
            .await
            .expect("wake target session");
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
        let registration = lashlang_process_registration("process-1", program, input);

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                "root",
                registration,
                crate::ProcessStartOptions::new()
                    .with_wake_session_id(wake_target.session_id.clone())
                    .with_descriptor(crate::ProcessHandleDescriptor::new(
                        Some("lashlang"),
                        Some("block"),
                    )),
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(&manager.current, "process-1", crate::ProcessOpScope::new())
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
        let queued = crate::store::RuntimePersistence::list_queued_work(
            store.as_ref(),
            &wake_target.session_id,
        )
        .await
        .expect("queued wake");
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].items.len(), 1);
        let crate::QueuedWorkPayload::ProcessWake { wake } = &queued[0].items[0].payload else {
            panic!("expected process wake queue payload");
        };
        assert_eq!(wake.input, "raw:seed");
    }

    #[tokio::test]
    async fn lashlang_process_uses_artifact_refs_for_nested_starts() {
        let runtime = runtime_with_processes(Vec::new()).await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
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
            lashlang_process_registration("snapshot-parent", program, serde_json::Map::new());
        let crate::ProcessInput::LashlangProcess { module_ref, .. } = registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        let artifact = ::lashlang::global_in_memory_lashlang_artifact_store()
            .get_module_artifact(module_ref)
            .expect("load lashlang module artifact")
            .expect("lashlang module artifact exists");
        let requirements = ::lashlang::surface_requirements_for_program(&artifact.canonical_ir);
        let relink = ::lashlang::LinkedModule::link(
            artifact.canonical_ir.clone(),
            ::lashlang::LashlangSurface::new(
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
                Arc::new(manager.clone()),
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("snapshot-parent")),
                ),
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(
                &manager.current,
                "snapshot-parent",
                crate::ProcessOpScope::new(),
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
    async fn lashlang_process_lifecycle_wait_signal_signal_run_and_sleep() {
        let mut runtime = runtime_with_processes(Vec::new()).await;
        let controller = RecordingProcessEffectController::default();
        runtime.host.core.effect_controller = Arc::new(controller.clone());
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();

        let target = lashlang_process_registration(
            "signal-target",
            ::lashlang::parse(
                r#"
                process main() {
                  value = wait signal
                  finish value
                }
                "#,
            )
            .expect("target process"),
            serde_json::Map::new(),
        );
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                "root",
                target,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("target")),
                ),
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("start target");

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
                  signal run target with { ok: true }
                  finish "sent"
                }
                "#,
            )
            .expect("signaler process"),
            signaler_args,
        );
        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                "root",
                signaler,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("signaler")),
                ),
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("start signaler");

        let signaler_output = manager
            .processes
            .await_process(
                &manager.current,
                "signal-sender",
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("await signaler");
        assert!(matches!(
            signaler_output,
            crate::ProcessAwaitOutput::Success { value, .. } if value == serde_json::json!("sent")
        ));

        let target_output = manager
            .processes
            .await_process(
                &manager.current,
                "signal-target",
                crate::ProcessOpScope::new(),
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
        assert!(
            events
                .iter()
                .any(|event| event.event_type == "process.signal"
                    && event.payload["payload"] == serde_json::json!({ "ok": true }))
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
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let program = ::lashlang::Program::block(vec![::lashlang::Expr::Fail(Box::new(
            ::lashlang::Expr::Record(vec![(
                "reason".into(),
                ::lashlang::Expr::String("bad".into()),
            )]),
        ))]);
        let registration =
            lashlang_process_registration("process-fail", program, serde_json::Map::new());

        manager
            .processes
            .start_process(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                "root",
                registration,
                crate::ProcessStartOptions::new().with_descriptor(
                    crate::ProcessHandleDescriptor::new(Some("lashlang"), Some("fail")),
                ),
                crate::ProcessOpScope::new(),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(
                &manager.current,
                "process-fail",
                crate::ProcessOpScope::new(),
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
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let registry = runtime
            .host
            .process_registry
            .as_ref()
            .expect("process registry")
            .clone();
        let current_scope = manager.processes.process_scope("root");
        let other_scope = manager.processes.process_scope("other");
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
                Arc::new(manager.clone()),
                "root",
                vec!["keep".to_string()],
                crate::ProcessOpScope::new(),
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
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
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
            Some("0".repeat(64)),
        );
        let scoped_request = || {
            crate::ProcessOpScope::new()
                .with_parent_invocation(Some(metadata.clone()))
                .with_effect_controller(&controller)
        };
        let root_scope = manager.processes.process_scope("root");
        let successor_scope = manager.processes.process_scope("successor");

        register_open_process(&registry, "transfer-me").await;
        grant_handle(&registry, &root_scope, "transfer-me").await;
        manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                "successor",
                vec!["transfer-me".to_string()],
                scoped_request(),
            )
            .await
            .expect("transfer handles");
        assert!(
            registry
                .list_handle_grants(&successor_scope)
                .await
                .expect("successor grants")
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
                Arc::new(manager.clone()),
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
    async fn process_control_fails_loudly_when_process_registry_is_unavailable() {
        let mut runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
        runtime.host.process_registry = None;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");

        manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                "root",
                "successor",
                Vec::<String>::new(),
                crate::ProcessOpScope::new(),
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
                "successor",
                vec!["missing".to_string()],
                crate::ProcessOpScope::new(),
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
                Arc::new(manager.clone()),
                "root",
                Vec::<String>::new(),
                crate::ProcessOpScope::new(),
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
