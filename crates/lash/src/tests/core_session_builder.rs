use super::*;

#[tokio::test]
async fn standard_core_runs_mock_turn() -> Result<()> {
    let core = standard_core();
    let session = core.session("main").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("hello"))
        .stream(&events)
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

#[tokio::test]
async fn prompt_layers_apply_across_core_session_turn_and_mutation_scopes() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = LashCore::standard()
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
        .control()
        .config()
        .replace_prompt_slot(
            PromptSlot::Guidance,
            [PromptContribution::guidance(
                "Replacement",
                "replacement guidance",
            )],
        )
        .await?;
    session.run(TurnInput::text("second")).await?;
    session
        .control()
        .config()
        .clear_prompt_slot(PromptSlot::Guidance)
        .await?;
    session.run(TurnInput::text("third")).await?;

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
    let core = LashCore::standard()
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

    let session_result = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&session_result.activities), "session");

    let turn_result = session
        .turn(TurnInput::text("hello"))
        .provider(text_provider("turn-provider", "turn-model", "turn"))
        .run()
        .await?;
    assert_eq!(assistant_prose(&turn_result.activities), "turn");

    let after_turn = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&after_turn.activities), "session");

    session
        .control()
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

    let updated = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&updated.activities), "updated");
    Ok(())
}

#[tokio::test]
async fn provider_only_overrides_use_provider_default_model_and_variant() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = LashCore::standard()
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

    session.run(TurnInput::text("hello")).await?;
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
        .control()
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
    session.run(TurnInput::text("hello")).await?;

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
async fn rlm_core_opens_rlm_session_and_rejects_standard_session() -> Result<()> {
    let core = LashCore::rlm()
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;

    let rlm = core.session("rlm").open().await?;
    assert_eq!(rlm.mode(), &ModeId::rlm());

    let err = match core.session("standard").standard().open().await {
        Ok(_) => panic!("standard mode should not be installed"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::standard()));
    Ok(())
}

#[tokio::test]
async fn rlm_mode_config_lashlang_abilities_drive_prompt_surface() -> Result<()> {
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
                    Ok(text_response("```lashlang\nsubmit \"ok\"\n```"))
                }
            }
        })
        .build()
        .into_handle();
    let config: crate::modes::RlmProtocolPluginConfig = serde_json::from_value(serde_json::json!({
        "lashlang_abilities": { "processes": true, "triggers": true }
    }))
    .expect("rlm config");
    let core = LashCore::builder()
        .install_mode(ModePreset::rlm_with_config(config))
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(mock_model_spec())
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("rlm-abilities-prompt").open().await?;

    session
        .turn(TurnInput::text("hello"))
        .require_submit()?
        .run()
        .await?;

    let prompts = seen.lock().expect("seen prompts");
    assert!(prompts[0].contains("Triggers: resource-event declarations"));
    assert!(!prompts[0].contains("TRIGGER.button.pressed"));
    Ok(())
}

#[tokio::test]
async fn malformed_rlm_create_extras_fail_child_session_creation() -> Result<()> {
    let core = LashCore::rlm()
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
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("rlm-child-bad-extras".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "rlm-root".to_string(),
                originating_tool_call_id: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash_core::SessionContextSurface::default(),
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

    let core = LashCore::rlm()
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("rlm").open().await?;
    session
        .control()
        .mode()
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
async fn explicit_dual_mode_install_allows_standard_parent_and_rlm_child() -> Result<()> {
    let core = LashCore::builder()
        .install_mode(ModePreset::standard())
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;

    let parent = core.session("main").standard().open().await?;
    let child = core.session("child").rlm().parent("main").open().await?;

    assert_eq!(parent.mode(), &ModeId::standard());
    assert_eq!(child.mode(), &ModeId::rlm());
    assert_eq!(child.parent_session_id(), Some("main"));
    Ok(())
}

#[tokio::test]
async fn uninstalled_mode_fails_at_open_time() -> Result<()> {
    let core = standard_core();
    let err = match core.session("rlm").rlm().open().await {
        Ok(_) => panic!("rlm mode should not be installed"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::rlm()));
    Ok(())
}

#[tokio::test]
async fn store_factory_reopens_persisted_session_state() -> Result<()> {
    let mut state = RuntimeSessionState {
        session_id: "persisted".to_string(),
        policy: lash_core::SessionPolicy {
            provider: mock_provider(),
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
    let core = LashCore::standard()
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
async fn core_delete_session_removes_factory_backed_session_state() -> Result<()> {
    let factory = Arc::new(DeletingStoreFactory::default());
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(factory)
        .build()?;
    let session = core.session("delete-session").open().await?;
    session.run(TurnInput::text("stored before delete")).await?;
    assert!(!session.read_view().messages().is_empty());
    drop(session);

    let report = core.delete_session("delete-session").await?;
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
            provider: mock_provider(),
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
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: store.clone(),
        }))
        .advanced()
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
            provider: mock_provider(),
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
    let core = LashCore::standard()
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
            provider: mock_provider(),
            model: mock_model_spec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = LashCore::standard()
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
            provider: mock_provider(),
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
    let core = LashCore::standard()
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
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let mut persisted = opened.control().state().persist_current().await?;
    persisted.tool_state_snapshot = Some(opened.control().tools().state().await?);
    drop(opened);

    let reopened = core
        .session("manual-state")
        .store(Arc::clone(&store))
        .open_with_state(persisted)
        .await?;
    let state = reopened.control().tools().state().await?;
    assert_eq!(
        state
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn core_store_factory_is_used_for_managed_child_sessions() -> Result<()> {
    let factory = Arc::new(RecordingStoreFactory::default());
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(factory.clone())
        .build()?;
    let session = core.session("root-with-child-store").open().await?;

    session
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("managed-child-store".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "root-with-child-store".to_string(),
                originating_tool_call_id: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash_core::SessionContextSurface::default(),
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
    let core = LashCore::standard()
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(ReusableStoreFactory {
            store: reused_store,
        }))
        .build()?;
    let session = core.session("root-store").open().await?;

    let err = session
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("child-needs-own-store".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "root-store".to_string(),
                originating_tool_call_id: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash_core::SessionContextSurface::default(),
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
    let core = LashCore::standard()
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
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("explicit-root-child".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "explicit-root-store".to_string(),
                originating_tool_call_id: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash_core::SessionContextSurface::default(),
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
            provider: mock_provider(),
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
    let core = LashCore::standard()
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
        ExecutionSummary, OutputState, SessionPolicy, SessionStateEnvelope, TurnFinish, TurnOutcome,
    };

    let result = TurnResult {
        state: SessionStateEnvelope {
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
            cached_input_tokens: 2,
            reasoning_tokens: 1,
        },
        children_usage: vec![
            TokenLedgerEntry {
                source: "subagent".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 7,
                    output_tokens: 3,
                    cached_input_tokens: 4,
                    reasoning_tokens: 0,
                },
            },
            TokenLedgerEntry {
                source: "compaction".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
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
    assert_eq!(total.cached_input_tokens, 2 + 4);
    assert_eq!(total.reasoning_tokens, 1);
    // Parent's own usage is unchanged.
    assert_eq!(result.usage.input_tokens, 10);
}
