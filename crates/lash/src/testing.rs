//! Test helpers for embedders. Enable with `lash = { ..., features = ["testing"] }`
//! to script model responses in integration tests without a live provider.

pub use lash_core::testing::{TestProvider, TestProviderBuilder};

/// Backend-agnostic conformance suites: validate a custom backend implementation
/// against a contract by running the same suite the in-tree backends run.
///
/// Re-exports the lash-core trait suites ([`process_registry`], [`runtime_persistence`])
/// and adds [`runtime_rebuild_and_worker_recovery`] — a runtime-level suite
/// that proves cold session rebuild and durable worker recovery use the same
/// reconstructed runtime surface.
pub mod conformance {
    pub use lash_core::testing::conformance::*;

    use std::sync::Arc;
    use std::time::Duration;

    use crate::core::LashCoreBuilder;
    use crate::plugins::{
        PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    };
    use crate::testing::TestProvider;
    use crate::{LashCore, ModeId, ModePreset};

    /// Stores + registry for one run of the
    /// [`runtime_rebuild_and_worker_recovery`] suite.
    ///
    /// `build_core` receives a builder pre-loaded with the mode, provider, model,
    /// plugins, and `process_registry`, and must wire the stores (and, for a
    /// durable store factory, an effect controller) and `build()`. `process_registry`
    /// is the same registry the builder is given, retained so the suite can drive
    /// and await processes. `make` in
    /// [`runtime_rebuild_and_worker_recovery`] must return a fresh backend
    /// (fresh stores) on each call.
    pub struct RuntimeRebuildBackend {
        pub process_registry: Arc<dyn lash_core::ProcessRegistry>,
        pub build_core: Box<dyn Fn(LashCoreBuilder) -> LashCore + Send + Sync>,
    }

    /// Run the full cold-rebuild + worker-recovery conformance suite against
    /// the backend produced by `make`. `make` must return a fresh backend on
    /// each call.
    ///
    /// Each scenario registers a Lashlang trigger route through the runtime
    /// `triggers` module, drops and reopens the session (a restart from cold
    /// durable storage), then drives an out-of-turn process through the
    /// lease-protected worker. The worker reconstructs the session purely from
    /// persisted state; the suite asserts that trigger registry state and the
    /// process runtime surface survive rebuild across the
    /// [`ProcessInput`](lash_core::ProcessInput) variants the worker runs.
    pub async fn runtime_rebuild_and_worker_recovery<F>(make: F)
    where
        F: Fn() -> RuntimeRebuildBackend,
    {
        reopen_restores_trigger_registry_state(make()).await;
        worker_runs_trigger_started_lashlang_process_after_restart(make()).await;
        trigger_triggered_process_wake_provenance_survives_restart(make()).await;
        worker_recovers_tool_call_process_in_restarted_session(make()).await;
        worker_recovers_session_turn_process_in_restarted_session(make()).await;
    }

    const TRIGGER_SOURCE: &str = r#"
process remember(tick: clock.Tick) {
  wake { id: tick.id, scheduled_at: tick.scheduled_at }
  finish { id: tick.id, ok: true }
}

source = clock.Alarm({ at: "08:00" })
handle = await triggers.register({
  source: source,
  target: remember,
  inputs: { tick: trigger.event },
  name: "remembered"
})?
submit "registered"
"#;

    const BUTTON_TRIGGER_SOURCE: &str = r#"
process remember_button(event: ui.button.Pressed) {
  wake { button: event.button, message: event.message }
  finish { button: event.button, ok: true }
}

source = ui.button.pressed({})
handle = await triggers.register({
  source: source,
  target: remember_button,
  inputs: { event: trigger.event },
  name: "button remembered"
})?
submit "registered"
"#;

    const SESSION_ID: &str = "rebuild-conformance";

    fn clock_tick_event_type() -> crate::modes::NamedDataType {
        crate::modes::NamedDataType::object(
            "clock.Tick",
            vec![
                crate::modes::TypeField {
                    name: "id".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                },
                crate::modes::TypeField {
                    name: "scheduled_at".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                },
            ],
        )
        .expect("valid clock tick type")
    }

    fn button_pressed_event_type() -> crate::modes::NamedDataType {
        crate::modes::NamedDataType::object(
            "ui.button.Pressed",
            vec![
                crate::modes::TypeField {
                    name: "button".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                },
                crate::modes::TypeField {
                    name: "message".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                },
                crate::modes::TypeField {
                    name: "pressed_at".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                },
            ],
        )
        .expect("valid button event type")
    }

    fn rebuild_abilities() -> crate::modes::LashlangAbilities {
        crate::modes::LashlangAbilities::default()
            .with_processes()
            .with_sleep()
            .with_process_signals()
            .with_triggers()
    }

    /// Installs the trigger abilities used by the rebuild conformance program.
    struct TriggerResourcePluginFactory;

    impl PluginFactory for TriggerResourcePluginFactory {
        fn id(&self) -> &'static str {
            "rebuild-conformance-trigger"
        }

        fn lashlang_abilities(&self) -> crate::modes::LashlangAbilities {
            rebuild_abilities()
        }

        fn lashlang_resources(&self) -> crate::modes::LashlangHostCatalog {
            let mut resources = crate::modes::LashlangHostCatalog::new();
            resources
                .add_trigger_source_constructor(
                    ["clock", "Alarm"],
                    crate::modes::TypeExpr::Object(vec![crate::modes::TypeField {
                        name: "at".into(),
                        ty: crate::modes::TypeExpr::Str,
                        optional: false,
                    }]),
                    clock_tick_event_type(),
                )
                .expect("valid clock trigger source");
            resources
                .add_trigger_source_constructor(
                    ["ui", "button", "pressed"],
                    crate::modes::TypeExpr::Object(Vec::new()),
                    button_pressed_event_type(),
                )
                .expect("valid button trigger source");
            resources
        }

        fn build(
            &self,
            _ctx: &PluginSessionContext,
        ) -> std::result::Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(TriggerResourceSessionPlugin))
        }
    }

    struct TriggerResourceSessionPlugin;

    impl SessionPlugin for TriggerResourceSessionPlugin {
        fn id(&self) -> &'static str {
            "rebuild-conformance-trigger"
        }

        fn register(&self, reg: &mut PluginRegistrar) -> std::result::Result<(), PluginError> {
            reg.triggers().declare(crate::triggers::TriggerEvent::new(
                "Button",
                "ui.button",
                "pressed",
                button_pressed_event_type(),
            ))?;
            Ok(())
        }
    }

    /// Echo tool for the [`ProcessInput::ToolCall`](lash_core::ProcessInput) scenario.
    struct EchoToolProvider;

    fn echo_tool_definition() -> lash_core::ToolDefinition {
        lash_core::ToolDefinition::raw(
            "tool:rebuild_echo",
            "rebuild_echo",
            "Echo the input value.",
            serde_json::json!({ "type": "object", "additionalProperties": true }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
    }

    #[async_trait::async_trait]
    impl lash_core::ToolProvider for EchoToolProvider {
        fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
            vec![echo_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
            (name == "rebuild_echo").then(|| Arc::new(echo_tool_definition().contract()))
        }

        async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
            let value = call
                .args
                .get("value")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            lash_core::ToolResult::ok(serde_json::json!({ "echoed": value }))
        }
    }

    fn rebuild_model() -> crate::ModelSpec {
        crate::ModelSpec::from_token_limits("rebuild-conformance-model", None, 4096, None)
            .expect("model spec")
    }

    /// Provider used both to register the trigger route through a normal RLM
    /// turn and to finish the SessionTurn child. The child inherits this
    /// provider, exercising provider re-supply after rebuild.
    fn rebuild_provider() -> crate::provider::ProviderHandle {
        TestProvider::builder()
            .kind("rebuild-conformance")
            .complete(|req| async move {
                let rendered_messages = format!("{:?}", req.messages);
                let text = if rendered_messages.contains("run child") {
                    "```lashlang\nsubmit \"child done\"\n```".to_string()
                } else if rendered_messages.contains("register rebuild button trigger") {
                    format!("```lashlang\n{}\n```", BUTTON_TRIGGER_SOURCE.trim())
                } else {
                    format!("```lashlang\n{}\n```", TRIGGER_SOURCE.trim())
                };
                Ok(crate::direct::LlmResponse {
                    full_text: text.clone(),
                    parts: vec![crate::direct::LlmOutputPart::Text {
                        text,
                        response_meta: None,
                    }],
                    ..crate::direct::LlmResponse::default()
                })
            })
            .build()
            .into_handle()
    }

    fn base_builder(registry: Arc<dyn lash_core::ProcessRegistry>) -> LashCoreBuilder {
        LashCore::builder()
            .install_mode(ModePreset::rlm_with_config(
                crate::modes::RlmProtocolPluginConfig::default()
                    .with_lashlang_abilities(rebuild_abilities()),
            ))
            .default_mode(ModeId::rlm())
            .provider(rebuild_provider())
            .model(rebuild_model())
            .plugin(Arc::new(TriggerResourcePluginFactory))
            .tools(Arc::new(EchoToolProvider))
            .process_registry(registry)
    }

    fn worker_registration(
        input: lash_core::ProcessInput,
        id: &str,
    ) -> lash_core::ProcessRegistration {
        lash_core::ProcessRegistration::new(
            id,
            input,
            lash_core::ProcessProvenance::session(
                lash_core::SessionScope::new(SESSION_ID),
                "default",
            ),
        )
    }

    async fn attach_rebuild_process_env(
        core: &LashCore,
        registration: lash_core::ProcessRegistration,
    ) -> lash_core::ProcessRegistration {
        let env_ref = lash_core::runtime::persist_process_execution_env(
            core.env.core.durability.lashlang_artifact_store.as_ref(),
            &lash_core::ProcessExecutionEnvSpec::new(
                lash_core::PluginOptions::default(),
                lash_core::SessionPolicy {
                    model: rebuild_model(),
                    ..lash_core::SessionPolicy::default()
                },
            ),
        )
        .await
        .expect("persist rebuild process env");
        registration.with_execution_env_ref(Some(env_ref))
    }

    /// Open the session, register the trigger route through Lashlang, optionally
    /// register an out-of-turn process, then drop and reopen from cold durable
    /// storage.
    async fn open_mutate_and_restart(
        core: &LashCore,
        register: Option<lash_core::ProcessRegistration>,
        registry: &Arc<dyn lash_core::ProcessRegistry>,
    ) {
        open_mutate_and_restart_with_prompt(core, "register rebuild trigger", register, registry)
            .await;
    }

    async fn open_mutate_and_restart_with_prompt(
        core: &LashCore,
        prompt: &str,
        register: Option<lash_core::ProcessRegistration>,
        registry: &Arc<dyn lash_core::ProcessRegistry>,
    ) {
        let session = core
            .session(SESSION_ID)
            .rlm()
            .open()
            .await
            .expect("open session");
        let output = session
            .turn(lash_core::TurnInput::text(prompt))
            .run()
            .await
            .expect("register trigger route");
        assert_eq!(
            output.submitted_value(),
            Some(&serde_json::json!("registered"))
        );
        if let Some(registration) = register {
            registry
                .register_process(registration)
                .await
                .expect("register out-of-turn process");
        }
        drop(session);
        // Reopen from cold storage — spawns the default work runner and forces the
        // worker to reconstruct the trigger-mutated surface purely from persistence.
        core.session(SESSION_ID)
            .rlm()
            .open()
            .await
            .expect("reopen session");
    }

    async fn await_success(registry: &Arc<dyn lash_core::ProcessRegistry>, process_id: &str) {
        let outcome =
            tokio::time::timeout(Duration::from_secs(10), registry.await_process(process_id))
                .await
                .expect("worker runs the process to terminal promptly")
                .expect("await terminal output");
        assert!(
            matches!(outcome, lash_core::ProcessAwaitOutput::Success { .. }),
            "process `{process_id}` must reach terminal SUCCESS via the worker's rebuilt \
             runtime, got: {outcome:?}"
        );
    }

    fn inline_trigger_scope(
        scope_id: impl Into<String>,
    ) -> lash_core::ScopedEffectController<'static> {
        lash_core::ScopedEffectController::shared(
            Arc::new(lash_core::InlineRuntimeEffectController),
            lash_core::EffectScope::runtime_operation(scope_id.into()),
        )
        .expect("inline trigger occurrence effect scope")
    }

    async fn emit_first_clock_alarm(
        core: &LashCore,
        session: &crate::LashSession,
        payload: serde_json::Value,
    ) -> lash_core::TriggerEmitReport {
        let registrations = session
            .triggers()
            .by_source_type("clock.Alarm")
            .await
            .expect("list clock trigger registrations");
        let handle = registrations
            .iter()
            .find(|registration| registration.enabled)
            .expect("registered clock trigger");
        let idempotency_key = format!(
            "test-clock:{}",
            payload
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("occurrence")
        );
        core.triggers()
            .emit(
                crate::triggers::TriggerOccurrenceRequest::new(
                    "clock.Alarm",
                    handle.source_key.clone(),
                    payload,
                    idempotency_key.clone(),
                ),
                inline_trigger_scope(format!("trigger:{idempotency_key}")),
            )
            .await
            .expect("emit clock trigger occurrence")
    }

    /// Differential baseline: a live reopen restores the trigger registry route
    /// installed through a normal turn — the same reconstruction the worker
    /// must use for out-of-turn process starts.
    async fn reopen_restores_trigger_registry_state(backend: RuntimeRebuildBackend) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        open_mutate_and_restart(&core, None, &registry).await;

        let reopened = core
            .session(SESSION_ID)
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let report = emit_first_clock_alarm(
            &core,
            &reopened,
            serde_json::json!({
                "id": "daily-2026-06-01",
                "scheduled_at": "2026-06-01T08:00:00Z"
            }),
        )
        .await;
        assert_eq!(report.started_process_ids.len(), 1);
    }

    async fn worker_runs_trigger_started_lashlang_process_after_restart(
        backend: RuntimeRebuildBackend,
    ) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        open_mutate_and_restart(&core, None, &registry).await;

        let session = core
            .session(SESSION_ID)
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let report = emit_first_clock_alarm(
            &core,
            &session,
            serde_json::json!({
                "id": "daily-2026-06-01",
                "scheduled_at": "2026-06-01T08:00:00Z"
            }),
        )
        .await;
        assert_eq!(report.started_process_ids.len(), 1);
        await_success(&registry, &report.started_process_ids[0]).await;
    }

    async fn trigger_triggered_process_wake_provenance_survives_restart(
        backend: RuntimeRebuildBackend,
    ) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        open_mutate_and_restart_with_prompt(
            &core,
            "register rebuild button trigger",
            None,
            &registry,
        )
        .await;

        let source_key = crate::triggers::empty_trigger_source_key("ui.button.pressed")
            .expect("button source key");
        let idempotency_key = "runtime-rebuild-trigger";
        let report = core
            .triggers()
            .emit(
                crate::triggers::TriggerOccurrenceRequest::new(
                    "ui.button.pressed",
                    source_key,
                    serde_json::json!({
                    "button": "Red",
                    "message": "user pressed the red button",
                    "pressed_at": "2026-06-02T12:00:00Z"
                    }),
                    idempotency_key,
                )
                .with_source(serde_json::json!({})),
                inline_trigger_scope(format!("trigger:{idempotency_key}")),
            )
            .await
            .expect("emit trigger occurrence");
        let process_records = registry
            .list_non_terminal()
            .await
            .expect("trigger-triggered process records");
        assert_eq!(process_records.len(), 1);
        let process_id = process_records[0].id.clone();
        let record = registry
            .get_process(&process_id)
            .await
            .expect("trigger-triggered process record");
        let process_caused_by = record
            .provenance
            .caused_by
            .clone()
            .expect("triggered process cause");
        assert!(matches!(
            &process_caused_by,
            lash_core::CausalRef::TriggerOccurrence { occurrence_id }
                if occurrence_id == &report.occurrence_id
        ));

        await_success(&registry, &process_id).await;
        let session = core
            .session(SESSION_ID)
            .rlm()
            .open()
            .await
            .expect("reopen session");
        let queued = session.queued_work().await.expect("queued wake");
        let wake = queued
            .iter()
            .flat_map(|batch| &batch.items)
            .find_map(|item| match &item.payload {
                lash_core::runtime::QueuedWorkPayload::ProcessWake { wake } => Some(wake),
                _ => None,
            })
            .expect("process wake queued for trigger-triggered process");
        assert_eq!(wake.process_id, process_id);
        assert_eq!(wake.process_caused_by, Some(process_caused_by));
        assert!(matches!(
            &wake.event_invocation.subject,
            lash_core::runtime::RuntimeSubject::ProcessEvent {
                process_id: wake_process_id,
                event_type,
                ..
            } if wake_process_id == &process_id && event_type == "process.wake"
        ));
    }

    async fn worker_recovers_tool_call_process_in_restarted_session(
        backend: RuntimeRebuildBackend,
    ) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        let registration = worker_registration(
            lash_core::ProcessInput::ToolCall {
                call: lash_core::PreparedToolCall::from_parts(
                    "rebuild-tool-call",
                    "rebuild_echo",
                    serde_json::json!({ "value": "recovered" }),
                    None,
                    serde_json::Value::Null,
                ),
            },
            "proc-tool-call",
        );
        let registration = attach_rebuild_process_env(&core, registration).await;
        open_mutate_and_restart(&core, Some(registration), &registry).await;
        await_success(&registry, "proc-tool-call").await;
    }

    async fn worker_recovers_session_turn_process_in_restarted_session(
        backend: RuntimeRebuildBackend,
    ) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        let child_policy = lash_core::SessionPolicy {
            model: rebuild_model(),
            ..lash_core::SessionPolicy::default()
        };
        let registration = worker_registration(
            lash_core::ProcessInput::SessionTurn {
                create_request: Box::new(lash_core::SessionCreateRequest::child(
                    SESSION_ID,
                    lash_core::SessionStartPoint::Empty,
                    child_policy,
                    lash_core::PluginOptions::default(),
                    "rebuild-conformance",
                )),
                turn_input: Box::new(lash_core::TurnInput::text("run child")),
                output_contract: lash_core::ToolOutputContract::Static,
            },
            "proc-session-turn",
        );
        open_mutate_and_restart(&core, Some(registration), &registry).await;
        await_success(&registry, "proc-session-turn").await;
    }
}
