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
        mock_provider, runtime_with_plugins, runtime_with_plugins_and_tools,
    };
    use std::sync::Arc;

    async fn runtime_with_processes_and_tools(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
        tools: Arc<dyn crate::ToolProvider>,
    ) -> crate::LashRuntime {
        let mut runtime =
            runtime_with_plugins_and_tools(plugins, tools, mock_provider(Vec::new())).await;
        runtime.host.process_registry = Some(Arc::new(crate::LocalProcessRegistry::default()));
        runtime
    }

    async fn runtime_with_processes(
        plugins: Vec<Arc<dyn crate::PluginFactory>>,
    ) -> crate::LashRuntime {
        let mut runtime = runtime_with_plugins(plugins, mock_provider(Vec::new())).await;
        runtime.host.process_registry = Some(Arc::new(crate::LocalProcessRegistry::default()));
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
        resources.add_alias("TOOL", "default");
        resources.add_operation("TOOL", "process_echo", "process_echo");
        let linked_module = ::lashlang::LinkedModule::link(
            module,
            ::lashlang::LashlangSurface::new(
                resources,
                ::lashlang::LashlangAbilities::default()
                    .with_processes()
                    .with_process_lifecycle(),
            ),
        )?;
        let module_version = linked_module.module_version.clone();
        Ok(crate::ProcessRegistration::new(
            process_id,
            crate::ProcessInput::LashlangProcess {
                module_version,
                linked_module,
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
                "probe.event".to_string(),
                serde_json::json!({ "marker": process_id }),
            )
            .await
            .expect("append probe event");
    }

    async fn grant_handle(
        registry: &Arc<dyn crate::ProcessRegistry>,
        scope_key: &str,
        process_id: &str,
    ) {
        registry
            .grant_handle(
                scope_key,
                process_id,
                crate::ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
            )
            .await
            .expect("grant handle");
    }

    #[derive(Clone, Default)]
    struct RecordingProcessEffectController {
        records: Arc<std::sync::Mutex<Vec<crate::EffectInvocationMetadata>>>,
    }

    impl RecordingProcessEffectController {
        fn records(&self) -> Vec<crate::EffectInvocationMetadata> {
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
                .push(envelope.metadata.clone());
            crate::InlineRuntimeEffectController::default()
                .execute_effect(envelope, local_executor)
                .await
        }
    }

    #[tokio::test]
    async fn lashlang_process_runs_with_input_events_wake_and_receiver_operation() {
        let runtime = runtime_with_processes_and_tools(Vec::new(), Arc::new(ProcessEchoTool)).await;
        let manager = RuntimeSessionManager::new(&runtime, true, None, None)
            .expect("runtime session manager");
        let wake_target = manager
            .managed
            .create_session(
                &manager.current,
                &manager.usage,
                crate::SessionCreateRequest {
                    session_id: Some("wake-target".to_string()),
                    relation: crate::SessionRelation::Root,
                    start: crate::SessionStartPoint::Empty,
                    policy: None,
                    plugin_mode: crate::SessionPluginMode::InheritCurrent,
                    initial_nodes: Vec::new(),
                    first_turn_input: None,
                    tool_access: crate::SessionToolAccess::default(),
                    subagent: None,
                    context_surface: crate::SessionContextSurface::default(),
                    mode_extras: crate::ModeExtras::default(),
                    usage_source: None,
                },
            )
            .await
            .expect("wake target session");
        let mut input = serde_json::Map::new();
        input.insert("root".to_string(), serde_json::json!("seed"));
        input.insert(
            "tool".to_string(),
            serde_json::to_value(::lashlang::Value::Resource(
                ::lashlang::ResourceHandle::new("TOOL", "default"),
            ))
            .expect("resource handle json"),
        );
        let program = ::lashlang::parse(
            r#"
            process main(root: str, tool: TOOL) {
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
                crate::ProcessStartRequest::new(
                    "root",
                    registration,
                    crate::ProcessExecutionContext::default()
                        .with_wake_session_id(wake_target.session_id.clone()),
                )
                .with_descriptor(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("block"),
                )),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(
                &manager.current,
                crate::ProcessAwaitRequest::new("process-1"),
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
        assert!(
            registry
                .wake_events_after("process-1", 0)
                .await
                .expect("wake events")
                .is_empty()
        );
        let child_runtime = manager
            .managed
            .registry
            .lock()
            .await
            .get(&wake_target.session_id)
            .cloned()
            .expect("wake target runtime");
        let injected = child_runtime
            .runtime
            .lock()
            .await
            .session
            .as_ref()
            .expect("wake target session")
            .turn_input_injection_bridge()
            .drain()
            .expect("injected input");
        assert_eq!(injected.len(), 1);
        assert_eq!(injected[0].message.content, "raw:seed");
    }

    #[tokio::test]
    async fn lashlang_process_uses_stored_linked_module_for_nested_starts() {
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
        let crate::ProcessInput::LashlangProcess { linked_module, .. } =
            registration.input.as_ref()
        else {
            panic!("expected lashlang process input");
        };
        let relink = ::lashlang::LinkedModule::link(
            linked_module.program.clone(),
            ::lashlang::LashlangSurface::new(
                linked_module.surface.resources.clone(),
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
                crate::ProcessStartRequest::new(
                    "root",
                    registration,
                    crate::ProcessExecutionContext::default(),
                )
                .with_descriptor(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("snapshot-parent"),
                )),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(
                &manager.current,
                crate::ProcessAwaitRequest::new("snapshot-parent"),
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
                crate::ProcessStartRequest::new(
                    "root",
                    target,
                    crate::ProcessExecutionContext::default(),
                )
                .with_descriptor(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("target"),
                )),
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
                crate::ProcessStartRequest::new(
                    "root",
                    signaler,
                    crate::ProcessExecutionContext::default(),
                )
                .with_descriptor(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("signaler"),
                )),
            )
            .await
            .expect("start signaler");

        let signaler_output = manager
            .processes
            .await_process(
                &manager.current,
                crate::ProcessAwaitRequest::new("signal-sender"),
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
                crate::ProcessAwaitRequest::new("signal-target"),
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
            .filter(|record| record.effect_kind == crate::RuntimeEffectKind::Sleep)
            .collect::<Vec<_>>();
        assert_eq!(sleep_records.len(), 1);
        assert!(sleep_records[0].idempotency_key.contains("signal-sender"));
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
                crate::ProcessStartRequest::new(
                    "root",
                    registration,
                    crate::ProcessExecutionContext::default(),
                )
                .with_descriptor(crate::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("fail"),
                )),
            )
            .await
            .expect("start process");
        let output = manager
            .processes
            .await_process(
                &manager.current,
                crate::ProcessAwaitRequest::new("process-fail"),
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
            err.to_string().contains("unknown builtin `process_echo`"),
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
        let current_scope = manager.processes.process_scope_key("root");
        let other_scope = manager.processes.process_scope_key("other");
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
                crate::ProcessCleanupRequest::new("root", vec!["keep".to_string()]),
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
            vec![other_scope]
        );
        assert_eq!(
            process_event_projection(&registry, &process_ids).await,
            events_before
        );
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
        let metadata = crate::EffectInvocationMetadata {
            session_id: "root".to_string(),
            origin: crate::EffectOrigin::Turn,
            turn_id: Some("turn-process-scope".to_string()),
            turn_index: Some(1),
            mode_iteration: Some(0),
            effect_id: "parent-process-control".to_string(),
            effect_kind: crate::RuntimeEffectKind::Process,
            idempotency_key: "root:turn-process-scope:process-control".to_string(),
            turn_checkpoint_hash: Some("0".repeat(64)),
        };
        let scoped_request = || {
            crate::ProcessRequestScope::new()
                .with_effect_metadata(Some(metadata.clone()))
                .with_effect_controller(&controller)
        };
        let root_scope = manager.processes.process_scope_key("root");
        let successor_scope = manager.processes.process_scope_key("successor");

        register_open_process(&registry, "transfer-me").await;
        grant_handle(&registry, &root_scope, "transfer-me").await;
        manager
            .processes
            .transfer_process_handles(
                &manager.current,
                &manager.managed,
                crate::ProcessTransferRequest::new(
                    "root",
                    "successor",
                    vec!["transfer-me".to_string()],
                )
                .with_scope(scoped_request()),
            )
            .await
            .expect("transfer handles");
        assert!(
            registry
                .list_handle_grants(&successor_scope)
                .await
                .expect("successor grants")
                .into_iter()
                .any(|(grant, _)| grant.process_id == "transfer-me")
        );

        register_open_process(&registry, "cleanup-me").await;
        grant_handle(&registry, &root_scope, "cleanup-me").await;
        manager
            .processes
            .cancel_unreferenced_process_handles(
                &manager.current,
                &manager.managed,
                Arc::new(manager.clone()),
                crate::ProcessCleanupRequest::new("root", Vec::<String>::new())
                    .with_scope(scoped_request()),
            )
            .await
            .expect("cleanup handles");

        let records = controller.records();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| {
            record.effect_kind == crate::RuntimeEffectKind::Process
                && record.turn_id.as_deref() == Some("turn-process-scope")
                && record
                    .idempotency_key
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
                crate::ProcessTransferRequest::new("root", "successor", Vec::<String>::new()),
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
                crate::ProcessTransferRequest::new(
                    "root",
                    "successor",
                    vec!["missing".to_string()],
                ),
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
                crate::ProcessCleanupRequest::new("root", Vec::<String>::new()),
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
