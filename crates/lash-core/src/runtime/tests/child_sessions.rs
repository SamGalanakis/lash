use super::*;
use crate::AttachmentStore as _;

struct AttachmentWritingTool;

#[async_trait::async_trait]
impl crate::ToolProvider for AttachmentWritingTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        vec![attachment_writing_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "write_attachment")
            .then(|| Arc::new(attachment_writing_tool_definition().contract()))
    }

    async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
        let reference = match call
            .context
            .attachments()
            .put(
                vec![4, 2, 4, 2],
                crate::AttachmentCreateMeta::new(
                    crate::MediaType::Image(crate::ImageMediaType::Png),
                    Some(2),
                    Some(2),
                    Some("child.png".to_string()),
                ),
            )
            .await
        {
            Ok(reference) => reference,
            Err(err) => return crate::ToolResult::err_fmt(err),
        };
        crate::ToolResult::ok(json!({ "attachment_id": reference.id }))
    }
}

fn attachment_writing_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:write_attachment",
        "write_attachment",
        "write a test attachment",
        crate::ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
}

#[tokio::test]
async fn session_manager_create_session_accepts_custom_context_overlay() {
    let runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_state_service().expect("session manager");
    let lifecycle = runtime
        .session_lifecycle_service()
        .expect("session lifecycle");
    let handle = lifecycle
        .create_session(
            crate::SessionCreateRequest::root(
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("memory-child")
            .with_plugin_source(crate::SessionPluginSource::CurrentHostFresh)
            .with_context_overlay(crate::SessionContextOverlay {
                include_base_tools: false,
                tool_providers: vec![Arc::new(MemoryProbeTool)],
                prompt_contributions: vec![crate::PromptContribution::guidance(
                    "Memory Context",
                    "memory child",
                )],
            }),
        )
        .await
        .expect("child session");

    let catalog = manager
        .tool_catalog(&handle.session_id)
        .await
        .expect("tool catalog");
    let tool_names = catalog
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(tool_names, vec!["memory_probe"]);
}

#[tokio::test]
async fn inherited_child_session_carries_parent_tool_state() {
    let plugin_host = crate::PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "memory_probe",
        crate::PluginSpec::new().with_tool_provider(Arc::new(MemoryProbeTool)),
    ))]);
    let plugin_session = plugin_host.build_session("root", None).expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    set_runtime_provider(&mut runtime, mock_provider(Vec::new()).into_handle());
    let manager = runtime.session_state_service().expect("session manager");
    let lifecycle = runtime
        .session_lifecycle_service()
        .expect("session lifecycle");
    let mut snapshot = manager.tool_state("root").await.expect("tool state");
    assert!(
        snapshot
            .remove(&crate::ToolId::from("tool:memory_probe"))
            .is_some()
    );
    manager
        .apply_tool_state("root", snapshot)
        .await
        .expect("apply dynamic state");

    let handle = lifecycle
        .create_session(
            crate::SessionCreateRequest::child_session(
                "root",
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("dynamic-child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("child session");

    let catalog = manager
        .tool_catalog(&handle.session_id)
        .await
        .expect("tool catalog");
    let tool_names = catalog
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(
        !tool_names.contains(&"memory_probe"),
        "inherited child should receive the parent's dynamic snapshot, got {tool_names:?}"
    );
}

#[tokio::test]
async fn durable_managed_child_writes_to_its_own_attachment_namespace() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                call_id: "child-attachment-call".to_string(),
                tool_name: "write_attachment".to_string(),
                input_json: "{}".to_string(),
                replay: None,
            })],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let child_factory = RecordingSessionStoreFactory::default();
    let root_store = Arc::new(RecordingStore::default());
    let bytes = Arc::new(crate::InMemoryAttachmentStore::new());
    let mut host_config = crate::RuntimeHostConfig::in_memory();
    host_config.durability.attachment_store =
        Arc::new(crate::SessionAttachmentStore::ephemeral(bytes.clone()));
    let host = crate::EmbeddedRuntimeHost::new(host_config)
        .with_session_store_factory(Arc::new(child_factory.clone()));
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        host,
        crate::PersistentRuntimeServices::new(
            plugin_session_with_tools("root", Arc::new(AttachmentWritingTool)),
            Arc::clone(&root_store) as Arc<dyn crate::store::RuntimePersistence>,
        ),
        state,
    )
    .await
    .expect("durable root runtime");
    set_runtime_provider(&mut runtime, transport.into_handle());

    let lifecycle = runtime
        .session_lifecycle_service()
        .expect("session lifecycle");
    let child = lifecycle
        .create_session(
            crate::SessionCreateRequest::child_session(
                "root",
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("attachment-child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("durable child session");
    let turn_id = "attachment-child-turn";
    let controller = crate::ScopedEffectController::shared(
        Arc::new(crate::InlineRuntimeEffectController),
        crate::ExecutionScope::turn(&child.session_id, turn_id),
    )
    .expect("child effect controller");
    let request = crate::SessionTurnRequest::new(
        &child.session_id,
        turn_id,
        TurnInput::text("write the attachment"),
        controller,
    )
    .expect("child turn request");
    lifecycle.start_turn(request).await.expect("child turn");

    let id = crate::attachments::content_id(&[4, 2, 4, 2]);
    // The blob lives exactly once in the shared, flat backend...
    assert_eq!(
        bytes.get(&id).await.expect("child attachment bytes").bytes,
        vec![4, 2, 4, 2]
    );
    // ...but reference isolation is now manifest-based: the child session holds
    // the ref, the root session never does. A managed child starts with its own
    // empty manifest, so it cannot resolve a blob the root put and vice versa.
    let child_store = child_factory
        .stores()
        .into_iter()
        .find(|store| {
            store
                .session_meta
                .lock()
                .expect("lock session meta")
                .as_ref()
                .is_some_and(|meta| meta.session_id == "attachment-child")
        })
        .expect("child store");
    assert!(
        crate::AttachmentManifest::holds_ref(&*child_store, "attachment-child", &id)
            .expect("child manifest lookup"),
        "child session must hold the ref it wrote"
    );
    assert!(
        !crate::AttachmentManifest::holds_ref(&*root_store, "root", &id)
            .expect("root manifest lookup"),
        "root session must not hold a ref for the child's attachment"
    );
}

#[tokio::test]
async fn same_session_rebuild_preserves_only_that_sessions_pending_attachment_ids() {
    let root_store = Arc::new(RecordingStore::default());
    let bytes = Arc::new(crate::InMemoryAttachmentStore::new());
    let mut host_config = crate::RuntimeHostConfig::in_memory();
    host_config.durability.attachment_store =
        Arc::new(crate::SessionAttachmentStore::ephemeral(bytes.clone()));
    let state = RuntimeSessionState {
        session_id: "root".to_string(),
        ..RuntimeSessionState::default()
    };
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        crate::EmbeddedRuntimeHost::new(host_config),
        crate::PersistentRuntimeServices::new(
            plugin_session_with_tools("root", Arc::new(EmptyTools)),
            root_store,
        ),
        state,
    )
    .await
    .expect("durable root runtime");

    let first = runtime
        .host
        .core
        .durability
        .attachment_store
        .put(
            vec![1, 2, 3],
            crate::AttachmentCreateMeta::new(
                crate::MediaType::Image(crate::ImageMediaType::Png),
                Some(1),
                Some(1),
                Some("before-rebuild.png".to_string()),
            ),
        )
        .await
        .expect("attachment before rebuild");
    assert_eq!(
        runtime
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids(),
        vec![first.id.clone()]
    );

    runtime
        .branch_to_node(None)
        .await
        .expect("same-session rebuild");
    assert_eq!(
        runtime
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids(),
        vec![first.id.clone()],
        "same-session rebinding must carry uncommitted attachment intents"
    );

    let second = runtime
        .host
        .core
        .durability
        .attachment_store
        .put(
            vec![4, 5, 6],
            crate::AttachmentCreateMeta::new(
                crate::MediaType::Image(crate::ImageMediaType::Png),
                Some(1),
                Some(1),
                Some("after-rebuild.png".to_string()),
            ),
        )
        .await
        .expect("attachment after rebuild");
    let mut expected = vec![first.id.clone(), second.id.clone()];
    expected.sort();
    assert_eq!(
        runtime
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids(),
        expected
    );
    runtime
        .host
        .core
        .durability
        .attachment_store
        .mark_manifest_committed(&[first.id.clone(), second.id.clone()]);
    assert!(
        runtime
            .host
            .core
            .durability
            .attachment_store
            .pending_manifest_commit_ids()
            .is_empty()
    );
    assert_eq!(
        bytes
            .get(&second.id)
            .await
            .expect("rebuilt runtime shares the flat backend blob")
            .bytes,
        vec![4, 5, 6]
    );
}

struct RootOnlyMemoryProbeFactory;

impl crate::plugin::PluginFactory for RootOnlyMemoryProbeFactory {
    fn id(&self) -> &'static str {
        "root_only_memory_probe"
    }

    fn build(
        &self,
        ctx: &crate::plugin::PluginSessionContext,
    ) -> Result<Arc<dyn crate::plugin::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(RootOnlyMemoryProbePlugin {
            active: ctx.is_root_session(),
        }))
    }
}

struct RootOnlyMemoryProbePlugin {
    active: bool,
}

impl crate::plugin::SessionPlugin for RootOnlyMemoryProbePlugin {
    fn id(&self) -> &'static str {
        "root_only_memory_probe"
    }

    fn register(&self, reg: &mut crate::plugin::PluginRegistrar) -> Result<(), crate::PluginError> {
        if self.active {
            reg.tools().provider(Arc::new(MemoryProbeTool))?;
        }
        Ok(())
    }
}

#[tokio::test]
async fn forked_child_session_filters_hidden_tool_state_before_rebind() {
    let plugin_host = crate::PluginHost::new(vec![Arc::new(RootOnlyMemoryProbeFactory)]);
    let plugin_session = plugin_host.build_session("root", None).expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    set_runtime_provider(&mut runtime, mock_provider(Vec::new()).into_handle());
    let manager = runtime.session_state_service().expect("session manager");
    let lifecycle = runtime
        .session_lifecycle_service()
        .expect("session lifecycle");
    assert!(
        manager
            .tool_state("root")
            .await
            .expect("tool state")
            .contains(&crate::ToolId::from("tool:memory_probe"))
    );

    let handle = lifecycle
        .create_session(
            crate::SessionCreateRequest::child_session(
                "root",
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("filtered-child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork)
            .with_tool_access(crate::SessionToolAccess {
                tools: Vec::new(),
                hidden_tools: ["memory_probe".to_string()].into_iter().collect(),
            }),
        )
        .await
        .expect("hidden root-only state should not poison fork");

    let catalog = manager
        .tool_catalog(&handle.session_id)
        .await
        .expect("tool catalog");
    let tool_names = catalog
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(!tool_names.contains(&"memory_probe"));
}

#[tokio::test]
async fn parent_turn_receives_live_child_token_usage_events() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
                    replay: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 11,
                    output_tokens: 3,
                    cache_read_input_tokens: 0,
                    cache_write_input_tokens: 0,
                    reasoning_output_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: vec![LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 7,
                output_tokens: 2,
                cache_read_input_tokens: 4,
                cache_write_input_tokens: 0,
                reasoning_output_tokens: 1,
            })],
            response: Ok(LlmResponse {
                full_text: "child session".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "child session".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(ChildSessionTool);
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let sink = RecordingSink::default();
    let turn_events = RecordingTurnEvents::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run child".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(
                CancellationToken::new(),
                named_turn_scope("root", "child-session-usage-parent"),
            )
            .with_events(&sink)
            .with_turn_events(&turn_events),
        )
        .await
        .expect("parent turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
    ));
    let events = sink.snapshot();
    let child_usage_event = events
        .clone()
        .into_iter()
        .find_map(|event| match event {
            SessionEvent::ChildTokenUsage {
                session_id,
                source,
                model,
                usage,
                cumulative,
                ..
            } => Some((session_id, source, model, usage, cumulative)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("child token usage event missing from {events:?}"));
    assert_eq!(child_usage_event.0, "subagent-child");
    assert_eq!(child_usage_event.1, "subagent");
    assert_eq!(child_usage_event.2, "mock-model");
    assert_eq!(child_usage_event.3.input_tokens, 7);
    assert_eq!(child_usage_event.3.output_tokens, 2);
    assert_eq!(child_usage_event.3.cache_read_input_tokens, 4);
    assert_eq!(child_usage_event.3.reasoning_output_tokens, 1);
    assert_eq!(child_usage_event.4.cache_read_input_tokens, 4);

    // The session-event projection should also surface a TurnEvent::ChildUsage
    // on the embed-facing TurnActivity stream.
    let activities = turn_events.snapshot();
    let projected = activities
        .iter()
        .find_map(|activity| match &activity.event {
            crate::TurnEvent::ChildUsage {
                session_id,
                source,
                model,
                usage,
                cumulative,
                ..
            } => Some((
                session_id.clone(),
                source.clone(),
                model.clone(),
                usage.clone(),
                cumulative.clone(),
            )),
            _ => None,
        })
        .unwrap_or_else(|| panic!("TurnEvent::ChildUsage missing from {activities:?}"));
    assert_eq!(projected.0, "subagent-child");
    assert_eq!(projected.1, "subagent");
    assert_eq!(projected.2, "mock-model");
    assert_eq!(projected.3.input_tokens, 7);
    assert_eq!(projected.4.cache_read_input_tokens, 4);

    // AssembledTurn carries per-(source, model) child entries so embed
    // consumers can compute per-turn breakdowns without diffing reports.
    let child_entry = turn
        .children_usage
        .iter()
        .find(|entry| entry.source == "subagent" && entry.model == "mock-model")
        .unwrap_or_else(|| panic!("missing subagent ledger entry: {:?}", turn.children_usage));
    assert_eq!(child_entry.usage.input_tokens, 7);
    assert_eq!(child_entry.usage.output_tokens, 2);
    assert_eq!(child_entry.usage.cache_read_input_tokens, 4);
    assert_eq!(child_entry.usage.reasoning_output_tokens, 1);

    let usage = runtime.usage_report();
    assert_eq!(usage.by_source["subagent"].usage.input_tokens, 7);
    assert_eq!(usage.by_source["subagent"].usage.output_tokens, 2);
    assert_eq!(usage.by_source["subagent"].usage.cache_read_input_tokens, 4);
    assert_eq!(usage.by_source["subagent"].usage.reasoning_output_tokens, 1);
}

#[tokio::test]
async fn parent_turn_keeps_cached_only_child_usage_live() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
                    replay: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 5,
                    output_tokens: 1,
                    cache_read_input_tokens: 0,
                    cache_write_input_tokens: 0,
                    reasoning_output_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: vec![LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_read_input_tokens: 9,
                cache_write_input_tokens: 0,
                reasoning_output_tokens: 0,
            })],
            response: Ok(LlmResponse {
                full_text: "cached child".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "cached child".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(ChildSessionTool);
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let sink = RecordingSink::default();

    runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run child".to_string(),
                }],
                image_blobs: HashMap::new(),
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            TurnOptions::new(
                CancellationToken::new(),
                named_turn_scope("root", "child-session-event-parent"),
            )
            .with_events(&sink),
        )
        .await
        .expect("parent turn");

    let events = sink.snapshot();
    let child_usage_event = events
        .clone()
        .into_iter()
        .find_map(|event| match event {
            SessionEvent::ChildTokenUsage {
                usage, cumulative, ..
            } => Some((usage, cumulative)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("child token usage event missing from {events:?}"));
    assert_eq!(child_usage_event.0.input_tokens, 0);
    assert_eq!(child_usage_event.0.output_tokens, 0);
    assert_eq!(child_usage_event.0.cache_read_input_tokens, 9);
    assert_eq!(child_usage_event.0.reasoning_output_tokens, 0);
    assert_eq!(child_usage_event.1.cache_read_input_tokens, 9);

    let usage = runtime.usage_report();
    assert_eq!(usage.by_source["subagent"].usage.input_tokens, 0);
    assert_eq!(usage.by_source["subagent"].usage.output_tokens, 0);
    assert_eq!(usage.by_source["subagent"].usage.cache_read_input_tokens, 9);
    assert_eq!(usage.by_source["subagent"].usage.reasoning_output_tokens, 0);
}
