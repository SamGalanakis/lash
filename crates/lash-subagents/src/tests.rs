use super::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::rlm_support::{
    SpawnCreateRequestInput, build_session_policy, build_spawn_create_request,
};
use lash_core::llm::types::{LlmContentBlock, LlmOutputPart, LlmRequest, LlmResponse, LlmRole};
use lash_core::{
    LashRuntime, PluginFactory, PluginHost, ProcessRuntimeHost, RuntimeCoreConfig, RuntimeServices,
    RuntimeSessionState, SessionPolicy, TestLocalProcessRegistry,
};
use lash_core::{ToolArgumentProjectionPolicy, ToolDefinition, ToolOutputContract, TurnInput};
use lash_protocol_rlm::RlmTurnInputExt;
use serde_json::json;

fn model_spec(
    model: impl Into<String>,
    variant: Option<String>,
    context_window_tokens: usize,
) -> lash_core::ModelSpec {
    lash_core::ModelSpec::from_token_limits(model, variant, context_window_tokens, None, None)
        .expect("valid model spec")
}

struct SeedProbeState {
    parent_response: String,
    captured_child_prompt: Arc<Mutex<Option<String>>>,
}

#[test]
fn static_capability_policy_fields_distinguish_inherit_set_and_clear() {
    let current = SessionPolicy {
        model: model_spec("parent-model", Some("parent-variant".to_string()), 200_000),
        ..SessionPolicy::default()
    };
    let spec = SessionSpec::inherit().model(model_spec("child-model", None, 100_000));
    let registry = CapabilityRegistry::new().with(Arc::new(StaticCapability::new("child", spec)));

    let policy = build_session_policy(&registry, &current, "child").expect("policy");

    assert_eq!(policy.model.id, "child-model");
    assert_eq!(policy.model.variant, None);
}

#[test]
fn rlm_definitions_expose_spawn_without_mini_api() {
    let registry = default_registry(&BTreeMap::new());
    let rlm_defs = rlm::rlm_subagent_tool_definitions(&registry.names());

    assert!(rlm_defs.iter().any(|tool| tool.name() == "spawn_agent"));
    assert_eq!(
        rlm_defs.iter().map(|tool| tool.name()).collect::<Vec<_>>(),
        vec!["spawn_agent"]
    );

    let rlm_spawn = rlm_defs
        .iter()
        .find(|tool| tool.name() == "spawn_agent")
        .expect("rlm spawn_agent");
    assert_eq!(
        rlm_spawn.effective_availability(),
        lash_core::ToolAvailability::Showcased
    );
    assert_eq!(
        rlm_spawn.contract.output_contract,
        ToolOutputContract::from_input_schema("output", None)
    );
    assert_eq!(
        rlm_spawn.manifest.argument_projection,
        ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed")
    );
    assert!(
        rlm_spawn
            .contract
            .examples
            .iter()
            .any(|example| example.contains("await agents.spawn"))
    );
    assert!(rlm_spawn.description().contains("module operation"));
    assert!(rlm_spawn.description().contains("agents: Agents"));
    assert!(!rlm_spawn.description().contains("use `start spawn_agent"));
}

#[test]
fn spawn_schema_is_strict_and_nameless() {
    let registry = default_registry(&BTreeMap::new());
    let tool = rlm::spawn_agent_tool_definition(&registry.names());
    let schema = tool.contract.input_schema;
    let retired_key = ["agent", "_", "name"].concat();

    let properties = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .expect("spawn schema properties");
    assert!(
        !properties.contains_key(&retired_key),
        "retired model-authored identity field leaked into spawn schema"
    );
    assert_eq!(
        schema.get("additionalProperties"),
        Some(&serde_json::Value::Bool(false))
    );

    let compiled = jsonschema::JSONSchema::compile(&schema).expect("spawn schema compiles");
    assert!(
        compiled
            .validate(&json!({ "task": "inspect routing", "capability": "explore" }))
            .is_ok()
    );
    let mut rejected = serde_json::Map::new();
    rejected.insert(
        "task".to_string(),
        serde_json::Value::String("inspect routing".to_string()),
    );
    rejected.insert(
        "capability".to_string(),
        serde_json::Value::String("explore".to_string()),
    );
    rejected.insert(
        retired_key,
        serde_json::Value::String("retired".to_string()),
    );
    assert!(
        compiled
            .validate(&serde_json::Value::Object(rejected))
            .is_err(),
        "strict spawn schema must reject retired identity arguments"
    );
}

#[test]
fn subagents_source_does_not_reintroduce_retired_lifecycle_surface() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source_files = [
        "src/lib.rs",
        "src/rlm.rs",
        "src/rlm_support.rs",
        "src/capability.rs",
    ];
    let banned = [
        ["agent", "_", "name"].concat(),
        ["Subagent", "Host"].concat(),
        ["Local", "Subagent", "Host"].concat(),
        ["Wait", "Agent"].concat(),
        ["Close", "Agent"].concat(),
        ["Spawn", "Agent", "Request"].concat(),
        ["Spawn", "Agent", "Response"].concat(),
        ["Agent", "Metadata"].concat(),
        ["normalize", "_async", "_subagent", "_name"].concat(),
        ["subagent:<", "agent", "_", "name", ">"].concat(),
    ];

    for relative in source_files {
        let text = std::fs::read_to_string(manifest_dir.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        for needle in &banned {
            assert!(
                !text.contains(needle),
                "{relative} reintroduced retired subagent surface `{needle}`"
            );
        }
    }
}

#[test]
fn single_capability_spawn_can_omit_capability_field() {
    let registry = CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
        "explore",
        lash_core::SessionSpec::inherit(),
    )));
    let rlm_spawn = rlm::spawn_agent_tool_definition(&registry.names());

    assert!(
        !rlm_spawn
            .contract
            .input_schema
            .get("required")
            .and_then(serde_json::Value::as_array)
            .expect("required fields")
            .iter()
            .any(|field| field.as_str() == Some("capability")),
        "single-capability spawn should not require explicit capability"
    );
    assert!(
        rlm_spawn
            .contract
            .examples
            .iter()
            .all(|example| !example.contains("capability:")),
        "single-capability examples should not teach redundant capability args"
    );
}

#[tokio::test]
async fn spawn_uses_live_parent_provider_when_selecting_subagent_model() {
    // Two distinct stub providers so we can verify that spawn
    // resolves against the *live* policy, not the factory's stale
    // one. The final child policy inherits the live policy's explicit
    // model spec.
    fn tiered_provider(tag: &'static str) -> lash_core::testing::TestProvider {
        let kind = match tag {
            "stale" => "stale-stub",
            "live" => "live-stub",
            _ => "stub",
        };
        lash_core::testing::TestProvider::builder()
            .kind(kind)
            .complete_error("stub")
            .build()
    }
    let stale_policy = SessionPolicy {
        provider: tiered_provider("stale").into_handle(),
        model: model_spec("stale-parent", None, 200_000),
        ..SessionPolicy::default()
    };
    let live_policy = SessionPolicy {
        provider: tiered_provider("live").into_handle(),
        model: model_spec("live-parent", None, 1234),
        ..SessionPolicy::default()
    };
    let registry = Arc::new(default_registry(&BTreeMap::new()));
    let current_snapshot = RuntimeSessionState {
        policy: live_policy.clone(),
        ..RuntimeSessionState::default()
    };

    let request = build_spawn_create_request(SpawnCreateRequestInput {
        registry: &registry,
        parent_session_id: "root",
        current_snapshot: current_snapshot.clone(),
        session_spec: &SessionSpec::inherit(),
        capability_name: "explore",
        output_schema: None,
        seed: Default::default(),
        parent_subagent: None,
        caused_by: None,
    })
    .await
    .expect("spawn request");
    let child_policy = request.policy.expect("child policy");

    // The capability looked up the live policy's provider, not
    // the stale one. This pins the behaviour where the spawn
    // pipeline always resolves models against the *current* session
    // policy snapshot, even when the factory was built earlier.
    let stale_choice = build_session_policy(&registry, &stale_policy, "explore")
        .expect("stale policy")
        .model;
    assert_eq!(child_policy.provider, live_policy.provider);
    assert_eq!(
        child_policy.model.context_window_tokens(),
        live_policy.model.context_window_tokens()
    );
    assert_ne!(child_policy.model.id, stale_choice.id);
    assert_eq!(child_policy.model.id, "live-parent");
    assert!(request.tool_access.tools.is_empty());

    let structured_request = build_spawn_create_request(SpawnCreateRequestInput {
        registry: &registry,
        parent_session_id: "root",
        current_snapshot,
        session_spec: &SessionSpec::inherit(),
        capability_name: "explore",
        output_schema: Some(json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"]
        })),
        seed: Default::default(),
        parent_subagent: None,
        caused_by: None,
    })
    .await
    .expect("structured spawn request");
    let structured_policy = structured_request
        .policy
        .as_ref()
        .expect("structured child policy");
    assert_eq!(structured_policy.model.id, "live-parent");
    let extras = structured_request
        .plugin_options
        .decode::<lash_rlm_types::RlmCreateExtras>(lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID)
        .expect("decode rlm extras")
        .expect("rlm extras");
    assert!(matches!(
        extras.termination,
        lash_rlm_types::RlmTermination::SubmitRequired { .. }
    ));
    assert!(structured_request.tool_access.tools.is_empty());
}

#[tokio::test]
async fn rlm_spawn_seed_is_visible_to_child_executor_and_prompt() {
    let (outcome, prompt) = run_seed_probe(
        r#"```lashlang
result = await agents.spawn({
  capability: "default",
  task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
  seed: { chunk: ["a", "b"] },
  output: Type { len: int }
})?
submit result
```"#,
        TurnInput::text("spawn a child with a seeded chunk"),
    )
    .await;

    assert_eq!(
        outcome,
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue {
            value: json!({ "len": 2 })
        })
    );
    assert!(
        prompt.contains("- `chunk`:"),
        "child prompt did not advertise seeded `chunk` variable:\n{prompt}"
    );
}

#[tokio::test]
async fn rlm_spawn_process_handle_returns_child_submitted_value() {
    let (outcome, prompt) = run_seed_probe(
        r#"```lashlang
process spawn_child(agents: Agents) {
  result = await agents.spawn({
    capability: "default",
    task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
    seed: { chunk: ["a", "b"] },
    output: Type { len: int }
  })?
  finish result
}
handle = start spawn_child(agents: agents)
result = (await handle)?
submit result
```"#,
        TurnInput::text("spawn a child with a seeded chunk through start/await"),
    )
    .await;

    assert_eq!(
        outcome,
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue {
            value: json!({ "len": 2 })
        })
    );
    assert!(
        prompt.contains("- `chunk`:"),
        "child prompt did not advertise seeded `chunk` variable:\n{prompt}"
    );
}

#[tokio::test]
async fn rlm_spawn_defaults_single_capability_when_omitted() {
    let (outcome, prompt) = run_seed_probe(
        r#"```lashlang
result = await agents.spawn({
  task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
  seed: { chunk: ["a", "b"] },
  output: Type { len: int }
})?
submit result
```"#,
        TurnInput::text("spawn a child with the default capability"),
    )
    .await;

    assert_eq!(
        outcome,
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue {
            value: json!({ "len": 2 })
        })
    );
    assert!(
        prompt.contains("Subagent capability: default. Depth: 1/5."),
        "child prompt did not render subagent authority:\n{prompt}"
    );
}

#[tokio::test]
async fn rlm_spawn_seed_derived_from_projected_binding_is_visible_to_child_prompt() {
    let input = TurnInput::text("spawn a child with a chunk from projected input")
            .rlm_project(
                lash_protocol_rlm::RlmProjectedBindings::new()
                    .bind_json(
                        "input",
                        json!({
                            "context": "Header\nDate: Jan 01, 2026 || Instance: A\nDate: Jan 02, 2026 || Instance: B\n",
                        }),
                    )
                    .expect("bind input"),
            )
            .expect("project input");
    let (outcome, prompt) = run_seed_probe(
        r#"```lashlang
ctx = to_string(input.context)
lines = split(ctx, "\n")
data = []
for line in lines {
  if starts_with(line, "Date: ") {
    data = push(data, line)
  }
}
chunk = slice(data, 0, 2)
result = await agents.spawn({
  capability: "default",
  task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
  seed: { chunk: chunk },
  output: Type { len: int }
})?
submit result
```"#,
        input,
    )
    .await;

    assert_eq!(
        outcome,
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue {
            value: json!({ "len": 2 })
        })
    );
    assert!(
        prompt.contains("- `chunk`:"),
        "child prompt did not advertise projected-derived seeded `chunk` variable:\n{prompt}"
    );
}

fn seed_probe_provider(state: Arc<SeedProbeState>) -> lash_core::testing::TestProvider {
    // The child subagent inherits the parent's live provider handle through the
    // runtime (deployment-level binding); there is no factory rematerialization,
    // so this provider needs no serializable config.
    lash_core::testing::TestProvider::builder()
        .kind("seed-probe")
        .complete(move |request| {
            let state = Arc::clone(&state);
            async move { complete_seed_probe_request(state, request).await }
        })
        .build()
}

async fn complete_seed_probe_request(
    state: Arc<SeedProbeState>,
    request: LlmRequest,
) -> Result<LlmResponse, lash_core::llm::transport::LlmTransportError> {
    let prompt = request_text(&request);
    let is_child = prompt.contains("Subagent capability: default. Depth: 1/5.")
        || prompt.contains("- `chunk`:");
    if is_child {
        *state.captured_child_prompt.lock().expect("captured prompt") = Some(prompt);
        Ok(LlmResponse {
            full_text: "```lashlang\nsubmit { len: len(chunk) }\n```".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "```lashlang\nsubmit { len: len(chunk) }\n```".to_string(),
                response_meta: None,
            }],
            ..Default::default()
        })
    } else {
        Ok(LlmResponse {
            full_text: state.parent_response.clone(),
            parts: vec![LlmOutputPart::Text {
                text: state.parent_response.clone(),
                response_meta: None,
            }],
            ..Default::default()
        })
    }
}

async fn run_seed_probe(
    parent_response: &'static str,
    input: TurnInput,
) -> (lash_core::TurnOutcome, String) {
    let captured_child_prompt: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let state = Arc::new(SeedProbeState {
        parent_response: parent_response.to_string(),
        captured_child_prompt: Arc::clone(&captured_child_prompt),
    });
    let provider = seed_probe_provider(Arc::clone(&state));

    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(lash_protocol_rlm::RlmProtocolPluginFactory::default()),
        Arc::new(SubagentsPluginFactory::new(Arc::new(
            CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                "default",
                lash_core::SessionSpec::inherit(),
            ))),
        ))),
    ];
    let registry = Arc::new(TestLocalProcessRegistry::default());
    let host_plugins = PluginHost::new(factories.clone());
    let process_abilities = host_plugins
        .lashlang_abilities()
        .with_processes()
        .with_process_lifecycle();
    let plugins = host_plugins
        .with_lashlang_abilities(process_abilities)
        .build_session("root", None)
        .expect("plugin session");
    let host = ProcessRuntimeHost::new(
        lash_core::EmbeddedRuntimeHost::new(RuntimeCoreConfig::in_memory()),
        Arc::clone(&registry) as Arc<dyn lash_core::ProcessRegistry>,
    );
    let policy = SessionPolicy {
        provider: provider.into_handle(),
        model: model_spec("seed-probe-model", None, 64_000),
        max_turns: Some(4),
        ..SessionPolicy::default()
    };
    // `agents.spawn(...)` starts a SessionTurn (subagent) process that the
    // lease-protected worker executes — not inline. A SINGLE inline runner over
    // the same registry + an explicit in-memory store factory runs it (and
    // provider re-supply reaches the child). One runner suffices even for the
    // nested case here (`handle = start spawn_child` then `await handle`) because
    // the worker runs each process on its own task, so the parent's await never
    // parks the runner away from the child.
    let worker = lash_core::DurableProcessWorker::new(
        lash_core::DurableProcessWorkerConfig::from_plugin_factories(
            factories,
            RuntimeCoreConfig::in_memory(),
            Arc::new(lash_core::InMemorySessionStoreFactory::new()),
            Arc::clone(&registry) as Arc<dyn lash_core::ProcessRegistry>,
        )
        .with_session_policy(policy.clone()),
    );
    let _poke = lash_core::ProcessWorkRunner::inline(worker).spawn();
    let mut runtime = LashRuntime::from_background_state(
        policy.clone(),
        host,
        RuntimeServices::new(plugins),
        RuntimeSessionState {
            session_id: "root".to_string(),
            policy,
            ..RuntimeSessionState::default()
        },
    )
    .await
    .expect("runtime");

    let turn =
        Box::pin(runtime.run_turn_assembled(input, tokio_util::sync::CancellationToken::new()))
            .await
            .expect("turn");

    let prompt = captured_child_prompt
        .lock()
        .expect("captured prompt")
        .clone()
        .expect("child prompt was captured");
    (turn.outcome, prompt)
}

fn request_text(request: &LlmRequest) -> String {
    let mut out = String::new();
    for message in &request.messages {
        let role = match message.role {
            LlmRole::System => "system",
            LlmRole::User => "user",
            LlmRole::Assistant => "assistant",
        };
        out.push_str(role);
        out.push('\n');
        for block in message.blocks.iter() {
            match block {
                LlmContentBlock::Text { text, .. } => out.push_str(text),
                LlmContentBlock::ToolCall { input_json, .. } => out.push_str(input_json),
                LlmContentBlock::ToolResult { content, .. } => out.push_str(content),
                LlmContentBlock::Reasoning { text, .. } => out.push_str(text),
                LlmContentBlock::Image { .. } => {}
            }
            out.push('\n');
        }
    }
    out
}

#[tokio::test]
async fn subagents_plugin_builds_without_mode_context() {
    let factory = SubagentsPluginFactory::new(Arc::new(default_registry(&BTreeMap::new())));
    let ctx = PluginSessionContext {
        session_id: "parent".to_string(),
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: None,
        lashlang_abilities: Default::default(),
        parent_session_id: None,
    };
    let plugin = factory.build(&ctx).expect("plugin");
    assert_eq!(plugin.id(), "subagents");
}

#[tokio::test]
async fn rlm_provider_does_not_require_process_support() {
    let factory = SubagentsPluginFactory::new(Arc::new(default_registry(&BTreeMap::new())));
    let ctx = PluginSessionContext {
        session_id: "parent".to_string(),
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: None,
        lashlang_abilities: Default::default(),
        parent_session_id: None,
    };

    let plugin = factory.build(&ctx).expect("rlm plugin");
    assert_eq!(plugin.id(), "subagents");
}

fn dummy_tool(name: &str) -> ToolDefinition {
    ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        format!("{name} description"),
        ToolDefinition::default_input_schema(),
        json!({ "type": "null" }),
    )
}

#[test]
fn subagent_surface_reports_authority_notes() {
    use lash_core::plugin::ToolSurfaceContext;

    let tools = vec![
        dummy_tool("read_file"),
        dummy_tool("ask"),
        dummy_tool("show_snippet_to_user"),
        dummy_tool("showcase"),
        dummy_tool("plan_exit"),
        dummy_tool("apply_patch"),
        dummy_tool("spawn_agent"),
    ];
    let contracts = tools
        .iter()
        .map(|tool| {
            (
                tool.name().to_string(),
                std::sync::Arc::new(tool.contract()),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let ctx = ToolSurfaceContext {
        session_id: "child".to_string(),
        tools: tools.into_iter().map(|tool| tool.manifest()).collect(),
        resolve_contract: Some(std::sync::Arc::new(move |name| {
            contracts.get(name).cloned()
        })),
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: Some(lash_core::SubagentSessionContext {
            parent_session_id: "root".to_string(),
            capability: "explore".to_string(),
            depth: 1,
            max_depth: 5,
        }),
        lashlang_abilities: Default::default(),
    };

    let contribution =
        rlm_support::subagent_surface_contribution(ctx).expect("surface contribution");
    assert!(contribution.overrides.is_empty());
    assert_eq!(
        contribution.tool_list_notes,
        vec!["Subagent capability: explore. Depth: 1/5."]
    );
}
