//! Test helpers for embedders. Enable with `lash = { ..., features = ["testing"] }`
//! to script model responses in integration tests without a live provider.

pub use lash_core::testing::{TestProvider, TestProviderBuilder};

/// Backend-agnostic conformance suites: validate a custom backend implementation
/// against a contract by running the same suite the in-tree backends run.
///
/// Re-exports the lash-core trait suites ([`process_registry`], [`runtime_persistence`])
/// and adds [`runtime_rebuild`] — a runtime-level suite that proves the durable
/// worker reconstructs a session identically to a live `session().open()`.
pub mod conformance {
    pub use lash_core::testing::conformance::*;

    use std::sync::Arc;
    use std::time::Duration;

    use crate::core::LashCoreBuilder;
    use crate::plugins::{
        HostEvent, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    };
    use crate::testing::TestProvider;
    use crate::{LashCore, ModeId, ModePreset};

    /// Stores + registry for one run of the [`runtime_rebuild`] suite.
    ///
    /// `build_core` receives a builder pre-loaded with the mode, provider, model,
    /// plugins, and `process_registry`, and must wire the stores (and, for a
    /// durable store factory, an effect controller) and `build()`. `process_registry`
    /// is the same registry the builder is given, retained so the suite can drive
    /// and await processes. `make` in [`runtime_rebuild`] must return a fresh
    /// backend (fresh stores) on each call.
    pub struct RebuildBackend {
        pub process_registry: Arc<dyn lash_core::ProcessRegistry>,
        pub build_core: Box<dyn Fn(LashCoreBuilder) -> LashCore + Send + Sync>,
    }

    /// Run the full runtime-rebuild conformance suite against the backend
    /// produced by `make`. `make` must return a fresh backend on each call.
    ///
    /// Each scenario mutates a session's tool surface (installs a lashlang
    /// trigger → generation ≥ 2), drops and reopens it (a restart from cold
    /// durable storage), then drives an out-of-turn process through the
    /// lease-protected worker. The worker reconstructs the session purely from
    /// persisted state; the suite asserts that reconstruction matches a live
    /// `session().open()` — the surface is restored (not reset to the base
    /// generation) and the process reaches terminal success — across the
    /// [`ProcessInput`](lash_core::ProcessInput) variants the worker runs.
    pub async fn runtime_rebuild<F>(make: F)
    where
        F: Fn() -> RebuildBackend,
    {
        reopen_restores_mutated_tool_surface(make()).await;
        worker_runs_trigger_started_lashlang_process_after_restart(make()).await;
        worker_recovers_tool_call_process_in_restarted_session(make()).await;
        worker_recovers_session_turn_process_in_restarted_session(make()).await;
    }

    const TRIGGER_SOURCE: &str = r#"
type ButtonChoice = enum["Red", "Blue"]
type ButtonPressed = { button: ButtonChoice, message: str, pressed_at: str }

process remember(event: ButtonPressed) {
  checked = validate(event, Type {
    button: enum["Red", "Blue"],
    message: str,
    pressed_at: str
  })
  wake { button: checked.button, message: checked.message }
  finish { button: checked.button, ok: true }
}

trigger remembered on ui.button.pressed as event
  -> remember(event: event)
"#;

    const SESSION_ID: &str = "rebuild-conformance";

    fn rebuild_abilities() -> crate::modes::LashlangAbilities {
        crate::modes::LashlangAbilities::default()
            .with_processes()
            .with_process_lifecycle()
            .with_triggers()
    }

    fn trigger_payload_type() -> crate::modes::TypeExpr {
        crate::modes::TypeExpr::Object(vec![
            crate::modes::TypeField {
                name: "button".into(),
                ty: crate::modes::TypeExpr::Enum(vec!["Red".into(), "Blue".into()]),
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
        ])
    }

    /// Makes `ui.button.pressed` a linkable trigger event source so the trigger
    /// source installs and mutates the tool surface.
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
            resources.add_module_instance(["ui", "button"], "Button");
            resources.add_trigger_event("Button", "pressed", trigger_payload_type());
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
            reg.host_events().declare(
                HostEvent::new("Button", "ui.button", "pressed").payload(trigger_payload_type()),
            )
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

    /// Provider that finishes an RLM child turn with a lashlang submit. Used only
    /// by the SessionTurn child (the lashlang/tool processes never call the LLM);
    /// the recovered child inherits this provider, exercising provider re-supply.
    fn rebuild_provider() -> crate::provider::ProviderHandle {
        TestProvider::builder()
            .kind("rebuild-conformance")
            .complete(|_| async {
                Ok(crate::direct::LlmResponse {
                    full_text: "```lashlang\nsubmit \"child done\"\n```".to_string(),
                    parts: vec![crate::direct::LlmOutputPart::Text {
                        text: "```lashlang\nsubmit \"child done\"\n```".to_string(),
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

    fn worker_registration(input: lash_core::ProcessInput, id: &str) -> lash_core::ProcessRegistration {
        lash_core::ProcessRegistration::new(id, input).with_process_provenance(
            lash_core::ProcessProvenance::new(lash_core::ProcessScope::new(SESSION_ID), "default"),
        )
    }

    /// Open the session, install the trigger (mutating the tool surface to
    /// generation ≥ 2), optionally register an out-of-turn process, then drop and
    /// reopen from cold durable storage. Returns the reopened core.
    async fn open_mutate_and_restart(
        core: &LashCore,
        register: Option<lash_core::ProcessRegistration>,
        registry: &Arc<dyn lash_core::ProcessRegistry>,
    ) {
        let session = core.session(SESSION_ID).rlm().open().await.expect("open session");
        let install = session
            .triggers()
            .install_lashlang_source(TRIGGER_SOURCE)
            .await
            .expect("install trigger source");
        assert_eq!(install.installed, vec!["remembered"]);
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
        let outcome = tokio::time::timeout(
            Duration::from_secs(10),
            registry.await_process(process_id),
        )
        .await
        .expect("worker runs the process to terminal promptly")
        .expect("await terminal output");
        assert!(
            matches!(outcome, lash_core::ProcessAwaitOutput::Success { .. }),
            "process `{process_id}` must reach terminal SUCCESS via the worker's rebuilt \
             runtime, got: {outcome:?}"
        );
    }

    /// Differential baseline: a live reopen restores the mutated tool surface
    /// (the trigger removed the `attach_button_trigger` tool), not the base
    /// surface — the same reconstruction the worker must produce.
    async fn reopen_restores_mutated_tool_surface(backend: RebuildBackend) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        open_mutate_and_restart(&core, None, &registry).await;

        let reopened = core.session(SESSION_ID).rlm().open().await.expect("reopen session");
        let tool_names = reopened
            .tools()
            .active_definitions()
            .await
            .expect("active tools")
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        // The installed trigger removes the surface-mutating installer tool; a
        // base-generation surface (rebuild failure) would still advertise it.
        assert!(
            !tool_names.iter().any(|name| name == "attach_button_trigger"),
            "reopened surface should reflect the installed trigger, got tools: {tool_names:?}"
        );
    }

    async fn worker_runs_trigger_started_lashlang_process_after_restart(backend: RebuildBackend) {
        let registry = Arc::clone(&backend.process_registry);
        let core = (backend.build_core)(base_builder(Arc::clone(&registry)));
        open_mutate_and_restart(&core, None, &registry).await;

        let session = core.session(SESSION_ID).rlm().open().await.expect("reopen session");
        let report = session
            .host_events()
            .emit(
                "Button",
                "ui.button",
                "pressed",
                serde_json::json!({
                    "button": "Blue",
                    "message": "user pressed the blue button",
                    "pressed_at": "2026-05-29T00:00:00Z"
                }),
            )
            .await
            .expect("emit host event");
        assert_eq!(report.started_process_ids.len(), 1);
        await_success(&registry, &report.started_process_ids[0]).await;
    }

    async fn worker_recovers_tool_call_process_in_restarted_session(backend: RebuildBackend) {
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

    async fn worker_recovers_session_turn_process_in_restarted_session(backend: RebuildBackend) {
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
