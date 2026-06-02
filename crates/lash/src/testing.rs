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
  name: "remembered"
})?
submit "registered"
"#;

    const SESSION_ID: &str = "rebuild-conformance";

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

        fn lashlang_resources(&self) -> crate::modes::ResourceCatalog {
            let mut resources = crate::modes::ResourceCatalog::new();
            resources.add_trigger_source_constructor(
                ["clock", "Alarm"],
                crate::modes::TypeExpr::Object(vec![crate::modes::TypeField {
                    name: "at".into(),
                    ty: crate::modes::TypeExpr::Str,
                    optional: false,
                }]),
                crate::modes::TypeExpr::Ref("clock.Tick".into()),
            );
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

        fn register(&self, _reg: &mut PluginRegistrar) -> std::result::Result<(), PluginError> {
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
        crate::ModelSpec::from_token_limits("rebuild-conformance-model", None, 4096, None, None)
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
        lash_core::ProcessRegistration::new(id, input).with_process_provenance(
            lash_core::ProcessProvenance::new(lash_core::ProcessScope::new(SESSION_ID), "default"),
        )
    }

    /// Open the session, register the trigger route through Lashlang, optionally
    /// register an out-of-turn process, then drop and reopen from cold durable
    /// storage.
    async fn open_mutate_and_restart(
        core: &LashCore,
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
            .turn(lash_core::TurnInput::text("register rebuild trigger"))
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

    async fn activate_first_clock_trigger(
        session: &crate::LashSession,
        payload: serde_json::Value,
    ) -> lash_core::HostEventEmitReport {
        let registrations = session
            .triggers()
            .by_source_type("clock.Alarm")
            .await
            .expect("list clock trigger registrations");
        let handle = registrations
            .iter()
            .find(|registration| registration.enabled)
            .map(|registration| registration.handle.as_str())
            .expect("registered clock trigger handle")
            .to_string();
        session
            .triggers()
            .activate(handle, payload)
            .await
            .expect("activate trigger")
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
        let report = activate_first_clock_trigger(
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
        let report = activate_first_clock_trigger(
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
