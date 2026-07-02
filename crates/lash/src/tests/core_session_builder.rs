use super::*;
use crate::rlm::{RlmFinalAnswerFormat, RlmSessionBuilderExt as _, RlmTurnBuilderExt as _};
use lash_lashlang_runtime::LashlangArtifactStore as _;

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct CompileSurfaceToolConfig {
    tool_name: String,
}

struct CompileSurfaceToolFactory {
    id: &'static str,
    default_tool_name: &'static str,
}

impl CompileSurfaceToolFactory {
    fn new(id: &'static str, default_tool_name: &'static str) -> Self {
        Self {
            id,
            default_tool_name,
        }
    }
}

impl lash_core::PluginFactory for CompileSurfaceToolFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn build(
        &self,
        ctx: &lash_core::PluginSessionContext,
    ) -> std::result::Result<Arc<dyn lash_core::SessionPlugin>, lash_core::PluginError> {
        let config = ctx
            .plugin_options
            .decode::<CompileSurfaceToolConfig>(self.id)
            .map_err(|err| lash_core::PluginError::Registration(err.to_string()))?;
        let tool_name = config
            .map(|config| config.tool_name)
            .unwrap_or_else(|| self.default_tool_name.to_string());
        Ok(Arc::new(CompileSurfaceToolPlugin {
            plugin_id: self.id,
            tool_name,
        }))
    }
}

struct CompileSurfaceToolPlugin {
    plugin_id: &'static str,
    tool_name: String,
}

impl lash_core::SessionPlugin for CompileSurfaceToolPlugin {
    fn id(&self) -> &'static str {
        self.plugin_id
    }

    fn register(
        &self,
        reg: &mut lash_core::PluginRegistrar,
    ) -> std::result::Result<(), lash_core::PluginError> {
        reg.tools().provider(Arc::new(CompileSurfaceToolProvider {
            tool_name: self.tool_name.clone(),
        }))?;
        Ok(())
    }
}

struct CompileSurfaceToolProvider {
    tool_name: String,
}

#[async_trait]
impl lash_core::ToolProvider for CompileSurfaceToolProvider {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![compile_surface_tool_definition(&self.tool_name).manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == self.tool_name)
            .then(|| Arc::new(compile_surface_tool_definition(&self.tool_name).contract()))
    }

    async fn execute(&self, _call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        lash_core::ToolResult::ok(serde_json::json!({ "ok": true }))
    }
}

fn compile_surface_tool_definition(name: &str) -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        format!("tool:{name}"),
        name.to_string(),
        "Compile-surface test tool.",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object" }),
    )
    .with_lashlang_binding(lash_lashlang_runtime::LashlangToolBinding::new(
        ["tools"],
        name.to_string(),
    ))
}

#[tokio::test]
async fn standard_core_runs_mock_turn() -> Result<()> {
    let core = standard_core();
    let session = core.session("main").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("hello"))
        .stream_to(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(
        events
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::AssistantProseDelta { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { .. }))
    );
    Ok(())
}

#[test]
fn typed_core_builders_require_explicit_store_choice() {
    let err = match LashCore::standard_builder()
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()
    {
        Ok(_) => panic!("standard preset must not install implicit in-memory stores"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::MissingEffectHost));

    let err = match LashCore::standard_builder()
        .provider(mock_provider())
        .model(mock_model_spec())
        .effect_host(Arc::new(crate::durability::InlineEffectHost::default()))
        .build()
    {
        Ok(_) => panic!("attachment store must be explicit after effect host is wired"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::MissingAttachmentStore));

    // The RLM factory now requires the Lashlang artifact store at construction,
    // so a missing-artifact-store error is unrepresentable. The RLM preset still
    // must not install implicit generic stores.
    let err = match rlm_core_builder()
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()
    {
        Ok(_) => panic!("rlm preset must not install implicit generic stores"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::MissingEffectHost));
}

#[test]
fn generic_lash_core_builder_requires_protocol_plugin() {
    let err = match explicit_ephemeral_facets(LashCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()
    {
        Ok(_) => panic!("generic LashCore must require an explicit protocol plugin"),
        Err(err) => err,
    };

    assert!(matches!(err, EmbedError::MissingProtocolPlugin));
}

#[tokio::test]
async fn prompt_layers_apply_across_core_session_turn_and_mutation_scopes() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(recording_prompt_provider(Arc::clone(&seen)))
        .model(mock_model_spec())
        .prompt_contribution(PromptContribution::guidance("Core", "core guidance"))
        .build()?;
    let session = core
        .session("prompt-api")
        .prompt_contribution(PromptContribution::guidance("Session", "session guidance"))
        .open()
        .await?;

    session
        .turn(TurnInput::text("first"))
        .prompt_contribution(PromptContribution::guidance("Turn", "turn guidance"))
        .run()
        .await?;
    session
        .admin()
        .config()
        .replace_prompt_slot(
            PromptSlot::Guidance,
            [PromptContribution::guidance(
                "Replacement",
                "replacement guidance",
            )],
        )
        .await?;
    session.turn(TurnInput::text("second")).run().await?;
    session
        .admin()
        .config()
        .clear_prompt_slot(PromptSlot::Guidance)
        .await?;
    session.turn(TurnInput::text("third")).run().await?;

    let prompts = seen.lock().expect("seen prompts");
    assert!(prompts[0].contains("core guidance"));
    assert!(prompts[0].contains("session guidance"));
    assert!(prompts[0].contains("turn guidance"));
    assert!(prompts[1].contains("replacement guidance"));
    assert!(!prompts[1].contains("core guidance"));
    assert!(!prompts[1].contains("session guidance"));
    assert!(!prompts[2].contains("core guidance"));
    assert!(!prompts[2].contains("replacement guidance"));
    Ok(())
}

#[tokio::test]
async fn provider_overrides_apply_at_core_session_turn_and_config_scopes() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(text_provider("core-provider", "core-model", "core"))
        .model(model_spec("core-model", None, 200_000))
        .build()
        .expect("standard core");
    let session = core
        .session("main")
        .provider(text_provider(
            "session-provider",
            "session-model",
            "session",
        ))
        .open()
        .await?;

    let session_result = session.turn(TurnInput::text("hello")).run().await?;
    assert_eq!(assistant_prose(&session_result.activities), "session");

    let turn_result = session
        .turn(TurnInput::text("hello"))
        .provider(text_provider("turn-provider", "turn-model", "turn"))
        .run()
        .await?;
    assert_eq!(assistant_prose(&turn_result.activities), "turn");

    let after_turn = session.turn(TurnInput::text("hello")).run().await?;
    assert_eq!(assistant_prose(&after_turn.activities), "session");

    session
        .admin()
        .config()
        .update(SessionConfigPatch {
            provider: Some(text_provider(
                "updated-provider",
                "updated-model",
                "updated",
            )),
            model: Some(model_spec("updated-model", None, 200_000)),
            ..SessionConfigPatch::default()
        })
        .await?;

    let updated = session.turn(TurnInput::text("hello")).run().await?;
    assert_eq!(assistant_prose(&updated.activities), "updated");
    Ok(())
}

#[tokio::test]
async fn provider_only_overrides_use_provider_default_model_and_variant() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(recording_text_provider(
            "core-provider",
            "core-model",
            Some("core-variant"),
            "core",
            Arc::clone(&seen),
        ))
        .model(model_spec(
            "core-model",
            Some("core-variant".to_string()),
            200_000,
        ))
        .build()
        .expect("standard core");
    let session = core
        .session("main")
        .provider(recording_text_provider(
            "session-provider",
            "session-model",
            Some("session-variant"),
            "session",
            Arc::clone(&seen),
        ))
        .open()
        .await?;

    session.turn(TurnInput::text("hello")).run().await?;
    session
        .turn(TurnInput::text("hello"))
        .provider(recording_text_provider(
            "turn-provider",
            "turn-model",
            Some("turn-variant"),
            "turn",
            Arc::clone(&seen),
        ))
        .run()
        .await?;
    session
        .turn(TurnInput::text("hello"))
        .provider(recording_text_provider(
            "manual-provider",
            "manual-default-model",
            Some("turn-variant"),
            "manual",
            Arc::clone(&seen),
        ))
        .model(model_spec(
            "manual-model",
            Some("manual-variant".to_string()),
            200_000,
        ))
        .run()
        .await?;
    session
        .admin()
        .config()
        .update(SessionConfigPatch {
            provider: Some(recording_text_provider(
                "updated-provider",
                "updated-model",
                Some("updated-variant"),
                "updated",
                Arc::clone(&seen),
            )),
            ..SessionConfigPatch::default()
        })
        .await?;
    session.turn(TurnInput::text("hello")).run().await?;

    assert_eq!(
        *seen.lock().expect("seen requests"),
        vec![
            ("core-model".to_string(), Some("core-variant".to_string())),
            ("core-model".to_string(), Some("core-variant".to_string())),
            (
                "manual-model".to_string(),
                Some("manual-variant".to_string())
            ),
            ("core-model".to_string(), Some("core-variant".to_string())),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn rlm_core_opens_rlm_session() -> Result<()> {
    let core = explicit_ephemeral_facets(rlm_core_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;

    core.session("rlm").open().await?;
    Ok(())
}

#[tokio::test]
async fn rlm_protocol_config_lashlang_abilities_drive_prompt_surface() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let provider = lash_core::testing::TestProvider::builder()
        .kind("rlm-abilities-prompt-test")
        .complete({
            let seen = Arc::clone(&seen);
            move |request| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.lock()
                        .expect("seen prompts")
                        .push(system_text(&request));
                    Ok(text_response(&lashlang_block("finish \"ok\"")))
                }
            }
        })
        .build()
        .into_handle();
    let config: crate::rlm::RlmProtocolPluginConfig = serde_json::from_value(serde_json::json!({
        "lashlang_abilities": { "processes": true, "triggers": true }
    }))
    .expect("rlm config");
    let factory = lash_protocol_rlm::RlmProtocolPluginFactory::new(config, inmem_artifact_store());
    let core = LashCore::rlm_builder(factory)
        .provider(provider)
        .model(mock_model_spec())
        .effect_host(Arc::new(crate::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(crate::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            crate::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("rlm-abilities-prompt").open().await?;

    session
        .turn(TurnInput::text("hello"))
        .require_finish()?
        .run()
        .await?;

    let prompts = seen.lock().expect("seen prompts");
    assert!(prompts[0].contains("Trigger registry"));
    assert!(prompts[0].contains("trigger registration connects"));
    assert!(prompts[0].contains("process definition"));
    assert!(prompts[0].contains("triggers.list({})"));
    assert!(!prompts[0].contains("TRIGGER."));
    Ok(())
}

#[tokio::test]
async fn rlm_compile_surface_uses_core_plugins_extra_plugins_and_request_options() -> Result<()> {
    // The compile APIs are now operations over the RLM factory and a plugin host
    // the caller builds. The plugin host carries the core tool plugin plus any
    // extra tool plugins; the request's execution env plugin options configure
    // them (here `compile-extra-tool` resolves to `lookup`).
    let artifact_store = Arc::new(crate::persistence::InMemoryLashlangArtifactStore::new());
    let factory = Arc::new(lash_protocol_rlm::RlmProtocolPluginFactory::new(
        lash_protocol_rlm::RlmProtocolPluginConfig::default(),
        artifact_store.clone(),
    ));
    let plugin_host = lash_core::PluginHost::new(vec![
        Arc::clone(&factory) as Arc<dyn PluginFactory>,
        Arc::new(CompileSurfaceToolFactory::new(
            "compile-core-tool",
            "compile_core_tool",
        )),
        Arc::new(CompileSurfaceToolFactory::new(
            "compile-extra-tool",
            "fallback",
        )),
    ]);
    // Process lifecycle available for the compile surface (parity with the old
    // core that wired a process registry).
    let process_lifecycle_available = true;
    let plugin_options = || {
        lash_core::PluginOptions::typed(
            "compile-extra-tool",
            CompileSurfaceToolConfig {
                tool_name: "lookup".to_string(),
            },
        )
        .expect("compile plugin options serialize")
    };
    let request = crate::rlm::LashlangCompileSurfaceRequest::new(
        "compile-surface",
        lash_core::ProcessExecutionEnvSpec::new(
            plugin_options(),
            lash_core::SessionPolicy::default(),
        ),
    );

    let surface =
        factory.lashlang_compile_surface(&plugin_host, process_lifecycle_available, request)?;

    assert!(surface.host_environment.abilities.processes);
    assert!(surface.host_environment.abilities.sleep);
    assert!(surface.host_environment.abilities.process_signals);
    assert!(surface.tool_catalog.has_callable_tool("compile_core_tool"));
    assert!(surface.tool_catalog.has_callable_tool("lookup"));
    assert!(!surface.tool_catalog.has_callable_tool("fallback"));
    assert!(
        surface
            .host_environment
            .resources
            .resolve_module_operation("Tools", "tools", "compile_core_tool")
            .is_some()
    );
    assert!(
        surface
            .host_environment
            .resources
            .resolve_module_operation("Tools", "tools", "lookup")
            .is_some()
    );

    let compiled = factory
        .compile_lashlang_module(
            &plugin_host,
            process_lifecycle_available,
            crate::rlm::LashlangModuleCompileRequest::new(
                "compile-module",
                r#"
value = tools.lookup({})
finish value
"#,
                lash_core::ProcessExecutionEnvSpec::new(
                    plugin_options(),
                    lash_core::SessionPolicy::default(),
                ),
            ),
        )
        .await
        .expect("compile module through the RLM factory");
    assert!(
        artifact_store
            .get_module_artifact(&compiled.module_ref)
            .await
            .expect("load persisted module artifact")
            .is_some(),
        "compile_lashlang_module should persist through the configured artifact store"
    );
    Ok(())
}

#[tokio::test]
async fn rlm_root_session_final_answer_format_defaults_to_markdown_and_can_be_raw() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = explicit_ephemeral_facets(rlm_core_builder())
        .provider(recording_request_provider(Arc::clone(&seen)))
        .model(mock_model_spec())
        .build()?;

    let markdown = core.session("rlm-root-markdown").open().await?;
    markdown.turn(TurnInput::text("hello")).run().await?;

    let raw = core
        .session("rlm-root-raw")
        .final_answer_format(RlmFinalAnswerFormat::RawFinalValue)
        .open()
        .await?;
    raw.turn(TurnInput::text("hello"))
        .require_finish()?
        .run()
        .await?;

    let prompts = seen.lock().expect("seen prompts");
    assert!(prompts[0].contains("=== FINAL ANSWER FORMAT ==="));
    assert!(prompts[0].contains("Markdown string"));
    assert!(!prompts[1].contains("=== FINAL ANSWER FORMAT ==="));
    assert!(!prompts[1].contains("Markdown string"));
    Ok(())
}

#[tokio::test]
async fn malformed_rlm_create_extras_fail_child_session_creation() -> Result<()> {
    let core = explicit_ephemeral_facets(rlm_core_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("rlm-root").open().await?;
    let mut plugin_options = lash_core::PluginOptions {
        plugins: BTreeMap::new(),
    };
    plugin_options.plugins.insert(
        lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID.to_string(),
        serde_json::json!({
            "termination": {
                "kind": "unknown"
            }
        }),
    );

    let err = session
        .admin()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("rlm-child-bad-extras".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "rlm-root".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_overlay: lash_core::SessionContextOverlay::default(),
            plugin_options,
            usage_source: None,
        })
        .await
        .expect_err("malformed RLM create extras should fail session creation");

    assert!(err.to_string().contains("invalid RLM create options"));
    Ok(())
}

#[tokio::test]
async fn rlm_projection_errors_surface_from_protocol_extensions() -> Result<()> {
    use lash_protocol_rlm::{RlmProjectedBindings, RlmTurnInputExt};

    let core = explicit_ephemeral_facets(rlm_core_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("rlm").open().await?;
    session
        .admin()
        .protocol()
        .apply_session_extension(lash_protocol_rlm::rlm_session_projection_extension(
            RlmProjectedBindings::new()
                .bind_json("current_query", serde_json::json!("session"))
                .expect("session bind"),
        ))
        .await?;

    let input = TurnInput::text("hello")
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("current_query", serde_json::json!("turn"))
                .expect("turn bind"),
        )
        .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
    let err = match session.turn(input).run().await {
        Ok(_) => panic!("duplicate session and turn projection should fail"),
        Err(err) => err,
    };
    assert!(
        matches!(err, EmbedError::Session(message) if message.to_string().contains("current_query"))
    );
    Ok(())
}

#[tokio::test]
async fn store_factory_reopens_persisted_session_state() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "persisted".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "already stored",
    )]);
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let reopened = core.session("persisted").open().await?;
    let messages = reopened.read_view().messages().to_vec();
    assert_eq!(messages.len(), 1);
    assert_eq!(message_text(&messages[0]), "already stored");
    Ok(())
}

#[tokio::test]
async fn open_fresh_ignores_persisted_state_and_replaces_it_on_commit() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "fresh-start".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "already stored",
    )]);
    let store = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: store.clone(),
        }))
        .build()?;

    let resumed = core.session("fresh-start").open().await?;
    assert_eq!(
        message_text(&resumed.read_view().messages()[0]),
        "already stored"
    );
    drop(resumed);

    let scopes_before_fresh = store.scopes().len();
    let fresh = core.session("fresh-start").open_fresh().await?;
    assert!(
        fresh.read_view().messages().is_empty(),
        "fresh opens must not expose persisted messages"
    );
    assert_eq!(fresh.policy_snapshot().recorded_provider_id(), "embed-test");
    assert_eq!(
        store.scopes().len(),
        scopes_before_fresh,
        "open_fresh must not load persisted session state"
    );

    fresh.turn(TurnInput::text("new root")).run().await?;
    drop(fresh);

    let reopened = core.session("fresh-start").open().await?;
    let texts = reopened
        .read_view()
        .messages()
        .iter()
        .map(message_text)
        .collect::<Vec<_>>();
    assert!(texts.contains(&"new root".to_string()));
    assert!(
        !texts.contains(&"already stored".to_string()),
        "fresh commit must replace the prior persisted graph"
    );
    Ok(())
}

#[test]
fn session_policy_serializes_provider_id_without_provider_config() -> Result<()> {
    let provider = crate::testing::TestProvider::builder()
        .kind("secret-provider")
        .serialize_config(|| serde_json::json!({ "api_key": "should-not-persist" }))
        .build()
        .into_handle();
    let policy = lash_core::SessionPolicy {
        provider_id: provider.kind().to_string(),
        model: mock_model_spec(),
        ..Default::default()
    };

    let value = serde_json::to_value(&policy)?;
    assert_eq!(value["provider_id"], "secret-provider");
    assert!(value.get("provider").is_none());
    assert!(!value.to_string().contains("should-not-persist"));

    let decoded: lash_core::SessionPolicy = serde_json::from_value(value)?;
    assert_eq!(decoded.recorded_provider_id(), "secret-provider");
    Ok(())
}

#[tokio::test]
async fn persisted_provider_id_rebinds_to_live_provider_on_open() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "provider-rebind".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: "embed-test".to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        current_agent_frame_id: String::new(),
        agent_frames: Vec::new(),
        ..Default::default()
    };
    state.ensure_agent_frame_initialized();
    state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "stored",
    )]);
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let reopened = core.session("provider-rebind").open().await?;
    let persisted = reopened.admin().state().persist_current().await?;

    assert_eq!(persisted.policy.recorded_provider_id(), "embed-test");
    assert!(
        persisted
            .agent_frames
            .iter()
            .all(|frame| frame.assignment.policy.recorded_provider_id() == "embed-test")
    );
    Ok(())
}

#[tokio::test]
async fn persisted_provider_id_mismatch_fails_at_turn_execution() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "provider-mismatch".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: "other-provider".to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        current_agent_frame_id: String::new(),
        agent_frames: Vec::new(),
        ..Default::default()
    };
    state.ensure_agent_frame_initialized();
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let session = core.session("provider-mismatch").open().await?;
    let err = match session.turn(TurnInput::text("must not run")).run().await {
        Ok(_) => panic!("provider mismatch should fail at turn execution"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EmbedError::Runtime(lash_core::RuntimeError {
            code: lash_core::RuntimeErrorCode::Other(code),
            message,
        }) if code == "llm_provider"
            && message.contains("other-provider")
            && message.contains("provider-mismatch")
    ));
    Ok(())
}

#[tokio::test]
async fn agent_frame_provider_id_mismatch_fails_at_turn_execution() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "frame-provider-mismatch".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: "embed-test".to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        current_agent_frame_id: String::new(),
        agent_frames: Vec::new(),
        ..Default::default()
    };
    state.ensure_agent_frame_initialized();
    state
        .current_agent_frame_mut()
        .expect("initial frame")
        .assignment
        .policy
        .provider_id = "other-provider".to_string();
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let session = core.session("frame-provider-mismatch").open().await?;
    let err = match session.turn(TurnInput::text("must not run")).run().await {
        Ok(_) => panic!("agent-frame provider mismatch should fail at turn execution"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EmbedError::Runtime(lash_core::RuntimeError {
            code: lash_core::RuntimeErrorCode::Other(code),
            message,
        }) if code == "llm_provider"
            && message.contains("other-provider")
            && message.contains("frame-provider-mismatch")
    ));
    Ok(())
}

#[tokio::test]
async fn refreshed_head_provider_id_mismatch_fails_before_turn() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "refresh-provider-mismatch".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: "embed-test".to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        current_agent_frame_id: String::new(),
        agent_frames: Vec::new(),
        ..Default::default()
    };
    state.ensure_agent_frame_initialized();
    let store = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let runtime_store: Arc<dyn lash_core::RuntimePersistence> = store.clone();
    let session = core
        .session("refresh-provider-mismatch")
        .store(runtime_store)
        .open()
        .await?;

    store.set_head_provider_id("other-provider");
    let err = match session.turn(TurnInput::text("must not run")).run().await {
        Ok(_) => panic!("head-refresh provider mismatch should fail before turn"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EmbedError::Runtime(lash_core::RuntimeError {
            code: lash_core::RuntimeErrorCode::Other(code),
            message,
        }) if code == "llm_provider"
            && message.contains("other-provider")
    ));
    Ok(())
}

#[tokio::test]
async fn explicit_provider_persists_reopens_and_runs_second_turn() -> Result<()> {
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::default());
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;

    let first = core
        .session("provider-reload")
        .store(Arc::clone(&store))
        .open()
        .await?;
    first.turn(TurnInput::text("first")).run().await?;
    drop(first);

    let reopened = core
        .session("provider-reload")
        .store(Arc::clone(&store))
        .open()
        .await?;
    let second = reopened.turn(TurnInput::text("second")).run().await?;

    assert_eq!(assistant_prose(&second.activities), "echo: second");
    assert_eq!(
        reopened.policy_snapshot().recorded_provider_id(),
        "embed-test"
    );
    Ok(())
}

#[tokio::test]
async fn core_delete_session_removes_factory_backed_session_state() -> Result<()> {
    let factory = Arc::new(DeletingStoreFactory::default());
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(factory)
        .build()?;
    let session = core.session("delete-session").open().await?;
    session
        .turn(TurnInput::text("stored before delete"))
        .run()
        .await?;
    assert!(!session.read_view().messages().is_empty());
    drop(session);

    let report = core
        .delete_session("delete-session", session_delete_scope("delete-session"))
        .await?;
    let reopened = core.session("delete-session").open().await?;

    assert_eq!(report.session_id, "delete-session");
    assert!(reopened.read_view().messages().is_empty());
    Ok(())
}

#[tokio::test]
async fn active_path_residency_opens_with_active_path_scope() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "active-path".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut root_message = text_message(lash_core::MessageRole::User, "root");
    root_message.id = "root-message".to_string();
    state.append_active_conversation_messages(&[root_message]);
    let root = state.session_graph.leaf_node_id.clone();
    let mut inactive_message = text_message(lash_core::MessageRole::User, "inactive branch");
    inactive_message.id = "inactive-message".to_string();
    state.append_active_conversation_messages(&[inactive_message]);
    state.session_graph.branch_to(root);
    let mut active_message = text_message(lash_core::MessageRole::User, "active branch");
    active_message.id = "active-message".to_string();
    state.append_active_conversation_messages(&[active_message]);

    let store = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: store.clone(),
        }))
        .residency(lash_core::Residency::ActivePathOnly)
        .build()?;

    let reopened = core.session("active-path").open().await?;
    let messages = reopened.read_view().messages().to_vec();
    let texts = messages.iter().map(message_text).collect::<Vec<_>>();
    assert_eq!(texts, vec!["root", "active branch"]);
    assert_eq!(
        store.scopes(),
        vec![lash_core::SessionReadScope::ActivePath { leaf_node_id: None }]
    );
    Ok(())
}

#[tokio::test]
async fn keep_all_residency_opens_with_full_graph_scope() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "keep-all".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "already stored",
    )]);
    let store = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: store.clone(),
        }))
        .build()?;

    let reopened = core.session("keep-all").open().await?;
    assert_eq!(reopened.read_view().messages().len(), 1);
    assert_eq!(store.scopes(), vec![lash_core::SessionReadScope::FullGraph]);
    Ok(())
}

#[tokio::test]
async fn store_session_id_mismatch_is_rejected() -> Result<()> {
    let state = RuntimeSessionState {
        session_id: "actual-session".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let err = match core.session("requested-session").open().await {
        Ok(_) => panic!("mismatched store should fail"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EmbedError::StoreSessionMismatch {
            loaded,
            requested
        } if loaded == "actual-session" && requested == "requested-session"
    ));
    Ok(())
}

#[tokio::test]
async fn open_with_state_uses_manual_state_and_persists_tool_state() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "manual-state".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "manual input",
    )]);
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::default());
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;

    let opened = core
        .session("manual-state")
        .store(Arc::clone(&store))
        .open_with_state(state)
        .await?;
    assert_eq!(
        message_text(&opened.read_view().messages().to_vec()[0]),
        "manual input"
    );
    opened
        .admin()
        .tools()
        .set_membership("tool:app_lookup", false)
        .await?;
    let mut persisted = opened.admin().state().persist_current().await?;
    let expected_generation = opened
        .admin()
        .tools()
        .state()
        .await?
        .generation()
        .saturating_add(5);
    persisted.tool_state_generation = Some(expected_generation);
    persisted.tool_state_snapshot = Some(
        opened
            .admin()
            .tools()
            .state()
            .await?
            .with_generation(expected_generation),
    );
    drop(opened);

    let reopened = core
        .session("manual-state")
        .store(Arc::clone(&store))
        .open_with_state(persisted)
        .await?;
    let state = reopened.admin().tools().state().await?;
    assert_eq!(state.generation(), expected_generation);
    assert!(
        !state
            .get(&lash_core::ToolId::from("tool:app_lookup"))
            .expect("app tool")
            .is_member(),
        "the host-removed tool is restored as a non-member"
    );
    Ok(())
}

#[tokio::test]
async fn core_store_factory_is_used_for_managed_child_sessions() -> Result<()> {
    let factory = Arc::new(RecordingStoreFactory::default());
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(factory.clone())
        .build()?;
    let session = core.session("root-with-child-store").open().await?;

    session
        .admin()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("managed-child-store".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "root-with-child-store".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_overlay: lash_core::SessionContextOverlay::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await?;

    assert_eq!(
        factory.session_ids(),
        vec![
            "root-with-child-store".to_string(),
            "managed-child-store".to_string()
        ]
    );
    Ok(())
}

#[tokio::test]
async fn reused_root_store_factory_reports_child_store_guidance() -> Result<()> {
    let reused_store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(BoundSessionStore {
        session_id: "root-store".to_string(),
    });
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: reused_store,
        }))
        .build()?;
    let session = core.session("root-store").open().await?;

    let err = session
        .admin()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("child-needs-own-store".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "root-store".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_overlay: lash_core::SessionContextOverlay::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await
        .expect_err("reused root store should not open a child session");
    let message = err.to_string();

    assert!(message.contains("configured child session store is already bound"));
    assert!(message.contains("SessionBuilder::store"));
    assert!(message.contains("LashCoreBuilder::child_store_factory"));
    Ok(())
}

#[tokio::test]
async fn explicit_root_store_keeps_configured_child_store_factory() -> Result<()> {
    let factory = Arc::new(RecordingStoreFactory::default());
    let explicit_store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::default());
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(factory.clone())
        .build()?;
    let session = core
        .session("explicit-root-store")
        .store(explicit_store)
        .open()
        .await?;

    session
        .admin()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("explicit-root-child".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "explicit-root-store".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_overlay: lash_core::SessionContextOverlay::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await?;

    assert_eq!(
        factory.session_ids(),
        vec!["explicit-root-child".to_string()]
    );
    Ok(())
}

#[tokio::test]
async fn explicit_session_store_takes_precedence_over_core_store_factory() -> Result<()> {
    let mut explicit_state = RuntimeSessionState {
        session_id: "store-precedence".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    explicit_state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::User,
        "explicit store",
    )]);
    let mut factory_state = explicit_state.clone();
    factory_state.append_active_conversation_messages(&[text_message(
        lash_core::MessageRole::Assistant,
        "factory store",
    )]);
    let explicit_store: Arc<dyn lash_core::RuntimePersistence> =
        Arc::new(SnapshotStore::with_state(explicit_state));
    let factory_store: Arc<dyn lash_core::RuntimePersistence> =
        Arc::new(SnapshotStore::with_state(factory_state));
    let core = explicit_ephemeral_facets(LashCore::standard_builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: factory_store,
        }))
        .build()?;

    let reopened = core
        .session("store-precedence")
        .store(explicit_store)
        .open()
        .await?;
    let messages = reopened.read_view().messages().to_vec();

    assert_eq!(messages.len(), 1);
    assert_eq!(message_text(&messages[0]), "explicit store");
    Ok(())
}

#[test]
fn turn_result_total_usage_sums_parent_and_children() {
    use lash_core::{
        ExecutionSummary, OutputState, SessionPolicy, SessionSnapshot, TurnFinish, TurnOutcome,
    };

    let result = TurnResult {
        state: SessionSnapshot {
            session_id: "s".to_string(),
            policy: SessionPolicy::default(),
            ..Default::default()
        },
        outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "ok".to_string(),
        }),
        assistant_output: AssistantOutput {
            safe_text: "ok".to_string(),
            raw_text: "ok".to_string(),
            state: OutputState::Usable,
        },
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 2,
            cache_write_input_tokens: 0,
            reasoning_output_tokens: 1,
        },
        children_usage: vec![
            TokenLedgerEntry {
                source: "subagent".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 7,
                    output_tokens: 3,
                    cache_read_input_tokens: 4,
                    cache_write_input_tokens: 0,
                    reasoning_output_tokens: 0,
                },
            },
            TokenLedgerEntry {
                source: "compaction".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 0,
                    cache_read_input_tokens: 0,
                    cache_write_input_tokens: 0,
                    reasoning_output_tokens: 0,
                },
            },
        ],
        tool_calls: Vec::new(),
        execution: ExecutionSummary {
            had_tool_calls: false,
            had_code_execution: false,
        },
        errors: Vec::new(),
    };

    let total = result.total_usage();
    assert_eq!(total.input_tokens, 10 + 7 + 1);
    assert_eq!(total.output_tokens, 5 + 3);
    assert_eq!(total.cache_read_input_tokens, 2 + 4);
    assert_eq!(total.reasoning_output_tokens, 1);
    // Parent's own usage is unchanged.
    assert_eq!(result.usage.input_tokens, 10);
}

// =============================================================================
// Phase-A: facade build store peer-coherence (durable store consistency)
// =============================================================================
//
// `LashCore::builder().build()` validates store peer-coherence only — it never
// inspects the effect controller (the build-time controller is inline by
// construction; the durable controller is per-invocation). These tests use the
// real durable backends so the durability tier each store reports is the
// production tier, not a faked one:
//
// - durable session store factory => `lash_sqlite_store::SqliteSessionStoreFactory`
// - durable attachment store       => `lash::FileAttachmentStore`
// - durable artifact store         => `lash_sqlite_store::Store`
// - durable process registry       => `lash_sqlite_store::SqliteProcessRegistry`
// - durable trigger store       => `lash_sqlite_store::SqliteTriggerStore`
//
// Ephemeral peers are the named in-memory implementations.

/// An RLM builder with a model + provider already named, ready for the
/// peer-coherence dependency under test. The artifact store rides the RLM
/// protocol factory (ADR-0013): its durability tier surfaces through the
/// contributed Lashlang process engine during the store-peer coherence sweep.
fn peer_coherence_builder(
    artifact_store: Arc<dyn lash_lashlang_runtime::LashlangArtifactStore>,
) -> crate::core::LashCoreBuilder {
    LashCore::rlm_builder(lash_protocol_rlm::RlmProtocolPluginFactory::new(
        lash_protocol_rlm::RlmProtocolPluginConfig::default(),
        artifact_store,
    ))
    .provider(mock_provider())
    .model(mock_model_spec())
}

/// The named in-memory (Inline-tier) Lashlang artifact store used by
/// peer-coherence tests that do not exercise the artifact facet itself.
fn inline_artifact_store() -> Arc<dyn lash_lashlang_runtime::LashlangArtifactStore> {
    Arc::new(lash_lashlang_runtime::InMemoryLashlangArtifactStore::new())
}

fn durable_session_store_factory(dir: &std::path::Path) -> Arc<dyn lash_core::SessionStoreFactory> {
    Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        dir.join("sessions"),
    ))
}

fn durable_attachment_store(dir: &std::path::Path) -> Arc<dyn lash_core::AttachmentStore> {
    Arc::new(crate::persistence::FileAttachmentStore::new(
        dir.join("attachments"),
    ))
}

/// `LashCore` is not `Debug`, so `Result::expect_err` is unavailable; this
/// extracts the build error or panics with the given message.
fn expect_build_error<T>(result: std::result::Result<T, EmbedError>, message: &str) -> EmbedError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}

async fn durable_artifact_store(
    dir: &std::path::Path,
) -> Arc<dyn lash_lashlang_runtime::LashlangArtifactStore> {
    Arc::new(
        lash_sqlite_store::Store::open(&dir.join("artifacts.db"))
            .await
            .expect("open durable artifact store"),
    )
}

async fn durable_process_env_store(
    dir: &std::path::Path,
) -> Arc<dyn lash_core::ProcessExecutionEnvStore> {
    Arc::new(
        lash_sqlite_store::Store::open(&dir.join("process-env.db"))
            .await
            .expect("open durable process env store"),
    )
}

async fn durable_trigger_store(dir: &std::path::Path) -> Arc<dyn lash_core::TriggerStore> {
    Arc::new(
        lash_sqlite_store::SqliteTriggerStore::open(&dir.join("triggers.db"))
            .await
            .expect("open durable trigger store"),
    )
}

#[tokio::test]
async fn durable_session_store_rejects_ephemeral_attachment_store_at_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = explicit_ephemeral_facets(peer_coherence_builder(
        durable_artifact_store(dir.path()).await,
    ))
    .store_factory(durable_session_store_factory(dir.path()))
    // Explicit ephemeral attachment store overrides the in-memory default
    // so the coherence check reads its Inline tier.
    .attachment_store(Arc::new(lash_core::InMemoryAttachmentStore::new()))
    .build();
    let err = expect_build_error(
        result,
        "durable session store + ephemeral attachment store must be rejected",
    );

    assert!(matches!(
        err,
        EmbedError::DurableStorePeerRequired {
            facet: "attachment store"
        }
    ));
}

#[tokio::test]
async fn builder_requires_explicit_process_env_store_at_build() {
    let result = peer_coherence_builder(inline_artifact_store())
        .effect_host(Arc::new(lash_core::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash_core::InMemoryAttachmentStore::new()))
        .build();
    let err = expect_build_error(
        result,
        "builder must reject missing process execution environment store",
    );

    assert!(matches!(err, EmbedError::MissingProcessEnvStore));
}

#[tokio::test]
async fn durable_session_store_rejects_ephemeral_process_env_store_at_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = explicit_ephemeral_facets(peer_coherence_builder(
        durable_artifact_store(dir.path()).await,
    ))
    .store_factory(durable_session_store_factory(dir.path()))
    .attachment_store(durable_attachment_store(dir.path()))
    .build();
    let err = expect_build_error(
        result,
        "durable session store + ephemeral process env store must be rejected",
    );

    assert!(matches!(
        err,
        EmbedError::DurableStorePeerRequired {
            facet: "process execution environment store"
        }
    ));
}

#[tokio::test]
async fn durable_session_store_rejects_ephemeral_artifact_store_at_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Explicit ephemeral (Inline) artifact store rides the RLM factory; the
    // durable attachment + process-env peers clear their facets first, so the
    // contributed Lashlang process engine's Inline tier is the one that fails.
    let result = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .store_factory(durable_session_store_factory(dir.path()))
        .attachment_store(durable_attachment_store(dir.path()))
        .process_env_store(durable_process_env_store(dir.path()).await)
        .build();
    let err = expect_build_error(
        result,
        "durable session store + ephemeral artifact store must be rejected",
    );

    // The artifact tier now surfaces through the contributed process engine, so
    // the violated facet is the engine kind (ADR-0013), not "artifact store".
    assert!(matches!(
        err,
        EmbedError::DurableStorePeerRequired {
            facet: lash_lashlang_runtime::LASHLANG_ENGINE_KIND
        }
    ));
}

#[tokio::test]
async fn durable_process_registry_rejects_missing_durable_store_factory_at_build() {
    // A durable process registry is meaningless without a durable session store
    // behind it. With no store factory (the in-memory default), the session
    // store tier is unknown/non-durable, so the registry must be rejected.
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.path().join("processes.db"))
            .await
            .expect("open durable registry"),
    );
    let result = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .process_registry(registry)
        .build();
    let err = expect_build_error(
        result,
        "durable process registry without durable store factory must be rejected",
    );

    assert!(matches!(
        err,
        EmbedError::DurableProcessRegistryRequiresStoreFactory
    ));
}

#[tokio::test]
async fn all_durable_stores_build_successfully() -> Result<()> {
    // Positive control: a coherent durable wiring (durable session store +
    // durable attachment + durable artifact + durable process registry +
    // durable trigger store) builds without error.
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.path().join("processes.db"))
            .await
            .expect("open durable registry"),
    );
    peer_coherence_builder(durable_artifact_store(dir.path()).await)
        .effect_host(Arc::new(lash_core::InlineEffectHost::default()))
        .store_factory(durable_session_store_factory(dir.path()))
        .attachment_store(durable_attachment_store(dir.path()))
        .process_env_store(durable_process_env_store(dir.path()).await)
        .trigger_store(durable_trigger_store(dir.path()).await)
        .process_registry(registry)
        .build()?;
    Ok(())
}

#[tokio::test]
async fn durable_process_registry_rejects_ephemeral_trigger_store_at_build() {
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.path().join("processes.db"))
            .await
            .expect("open durable registry"),
    );
    let result = peer_coherence_builder(durable_artifact_store(dir.path()).await)
        .effect_host(Arc::new(lash_core::InlineEffectHost::default()))
        .store_factory(durable_session_store_factory(dir.path()))
        .attachment_store(durable_attachment_store(dir.path()))
        .process_env_store(durable_process_env_store(dir.path()).await)
        .process_registry(registry)
        .build();
    let err = expect_build_error(
        result,
        "durable process registry without durable trigger store must be rejected",
    );

    assert!(matches!(
        err,
        EmbedError::DurableStorePeerRequired {
            facet: "trigger store"
        }
    ));
}

#[tokio::test]
async fn durable_registry_with_only_child_store_factory_builds() -> Result<()> {
    // C2 regression: the CLI wires a durable process registry + a durable *child*
    // store factory (managed child sessions) and NO root `store_factory`. Since
    // `build()` installs `child_store_factory.or(store_factory)` as the session
    // store, this wiring is durable end-to-end and must build. The coherence
    // guard and the work-runner resolver therefore have to read that same
    // effective factory; reading `store_factory` alone wrongly rejected it with
    // `DurableProcessRegistryRequiresStoreFactory` even though `build()` would
    // wire the child factory durably.
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.path().join("processes.db"))
            .await
            .expect("open durable registry"),
    );
    peer_coherence_builder(durable_artifact_store(dir.path()).await)
        .effect_host(Arc::new(lash_core::InlineEffectHost::default()))
        .child_store_factory(durable_session_store_factory(dir.path()))
        .attachment_store(durable_attachment_store(dir.path()))
        .process_env_store(durable_process_env_store(dir.path()).await)
        .trigger_store(durable_trigger_store(dir.path()).await)
        .process_registry(registry)
        .build()?;
    Ok(())
}

#[tokio::test]
async fn explicit_ephemeral_facets_build_successfully() -> Result<()> {
    // The durable-first guard must not regress inline/in-memory hosts: an
    // all-ephemeral build (the named in-memory implementations) succeeds,
    // including the explicit in-memory session store factory that backs
    // ephemeral process execution.
    explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    Ok(())
}

struct NoopProcessRunHandle;

#[async_trait]
impl lash_core::ProcessRunHandle for NoopProcessRunHandle {
    async fn claim_and_run_pending(&self) -> std::result::Result<(), lash_core::PluginError> {
        Ok(())
    }
}

#[tokio::test]
async fn process_work_driver_configures_external_runner_without_inline_store_factory() -> Result<()>
{
    let registry =
        Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn lash_core::ProcessRegistry>;
    let driver =
        lash_core::ProcessWorkDriver::new(Arc::clone(&registry), Arc::new(NoopProcessRunHandle));
    let core = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .process_work_driver(driver)
        .build()?;

    let configured = core
        .process_registry()
        .expect("external driver configures the core registry");
    assert!(Arc::ptr_eq(&configured, &registry));
    assert!(core.processes().observer().is_ok());
    assert!(core.work_driver.drivers().await.process.is_some());
    Ok(())
}

#[tokio::test]
async fn default_process_work_driver_resolves_when_registry_and_store_factory_present() -> Result<()>
{
    // Zero-ceremony path: a registry + a store factory (so the inline worker can
    // rebuild session runtimes) and no explicit driver constructs the default
    // inline process work driver on first `session().open()`. The driver's actual
    // lease-protected execution of out-of-turn processes is covered in lash-core
    // (`concurrent_workers_run_a_directly_registered_process_exactly_once`).
    let state = RuntimeSessionState {
        session_id: "main".to_string(),
        policy: lash_core::SessionPolicy {
            provider_id: mock_provider().kind().to_string(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    core.session("main").open().await?;
    assert!(
        core.work_driver.drivers().await.process.is_some(),
        "the default inline process driver must resolve when a registry + store factory are wired"
    );
    Ok(())
}

#[tokio::test]
async fn durable_process_worker_config_uses_core_process_registry() -> Result<()> {
    let registry =
        Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn lash_core::ProcessRegistry>;
    let trigger_store =
        Arc::new(lash_core::InMemoryTriggerStore::default()) as Arc<dyn lash_core::TriggerStore>;
    let core = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .trigger_store(Arc::clone(&trigger_store))
        .process_registry(Arc::clone(&registry))
        .build()?;

    assert!(core.processes().observer().is_ok());
    let config = core.durable_process_worker_config()?;
    assert!(Arc::ptr_eq(&config.process_registry, &registry));
    assert!(Arc::ptr_eq(&config.trigger_store, &trigger_store));
    Ok(())
}

#[tokio::test]
async fn durable_process_worker_config_requires_core_process_registry() {
    let core = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()
        .expect("build core without process support");

    let Err(err) = core.durable_process_worker_config() else {
        panic!("worker config must require process support");
    };
    assert!(matches!(err, EmbedError::MissingProcessRegistry));
}

#[tokio::test]
async fn registry_without_store_factory_fails_loudly() {
    // A registry but no store factory: the default work runner rebuilds a
    // session runtime per process and cannot do so without a store factory, so
    // build must fail loudly rather than silently leave processes unexecuted
    // (a process started in such a host would otherwise hang forever).
    let result = explicit_ephemeral_facets(peer_coherence_builder(inline_artifact_store()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build();
    let err = expect_build_error(
        result,
        "a process registry with no store factory must be rejected",
    );
    assert!(matches!(
        err,
        EmbedError::ProcessRegistryRequiresStoreFactory
    ));
}
