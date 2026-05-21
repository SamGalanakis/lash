use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, Once};

use crate::rlm_support::{
    SpawnCreateRequestInput, build_session_policy, build_spawn_create_request,
};
use lash_core::llm::types::{LlmContentBlock, LlmOutputPart, LlmRequest, LlmResponse, LlmRole};
use lash_core::{
    ExecutionMode, LashRuntime, LocalProcessRegistry, RuntimeSessionState, PluginFactory,
    PluginHost, ProcessRuntimeHost, RuntimeCoreConfig, RuntimeServices, SessionPolicy,
};
use lash_core::{ToolArgumentProjectionPolicy, ToolDefinition, ToolOutputContract, TurnInput};
use lash_mode_rlm::RlmTurnInputExt;
use serde_json::json;

static SEED_PROBE_FACTORY: Once = Once::new();
static SEED_PROBE_NEXT_ID: AtomicUsize = AtomicUsize::new(0);
static SEED_PROBE_STATES: LazyLock<Mutex<BTreeMap<String, Arc<SeedProbeState>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

struct SeedProbeState {
    parent_response: String,
    captured_child_prompt: Arc<Mutex<Option<String>>>,
    request_count: AtomicUsize,
}

struct SeedProbeProviderFactory;

impl lash_core::ProviderFactory for SeedProbeProviderFactory {
    fn kind(&self) -> &'static str {
        "seed-probe"
    }

    fn deserialize(
        &self,
        config: serde_json::Value,
    ) -> Result<lash_core::ProviderComponents, String> {
        let id = config
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "seed-probe provider config missing `id`".to_string())?
            .to_string();
        let state = SEED_PROBE_STATES
            .lock()
            .expect("seed probe states")
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("unknown seed-probe provider state `{id}`"))?;
        Ok(seed_probe_provider(id, state)
            .into_handle()
            .components()
            .clone())
    }
}

fn ensure_seed_probe_provider_factory() {
    SEED_PROBE_FACTORY.call_once(|| {
        lash_core::register_provider_factory(Arc::new(SeedProbeProviderFactory));
    });
}

#[test]
fn static_capability_policy_fields_distinguish_inherit_set_and_clear() {
    let current = SessionPolicy {
        model: "parent-model".to_string(),
        model_variant: Some("parent-variant".to_string()),
        execution_mode: lash_core::ExecutionMode::standard(),
        ..SessionPolicy::default()
    };
    let spec = SessionSpec::inherit()
        .model("child-model", None)
        .mode(lash_core::ExecutionMode::new("rlm"));
    let registry = CapabilityRegistry::new().with(Arc::new(StaticCapability::new("child", spec)));

    let policy = build_session_policy(&registry, &current, "child").expect("policy");

    assert_eq!(policy.model, "child-model");
    assert_eq!(policy.model_variant, None);
    assert_eq!(policy.execution_mode, lash_core::ExecutionMode::new("rlm"));
}

#[test]
fn rlm_definitions_expose_spawn_without_mini_api() {
    let registry = default_registry(&BTreeMap::new());
    let rlm_defs = rlm::rlm_subagent_tool_definitions(&registry.names());

    assert!(rlm_defs.iter().any(|tool| tool.name == "spawn_agent"));
    assert_eq!(
        rlm_defs
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["spawn_agent"]
    );

    let rlm_spawn = rlm_defs
        .iter()
        .find(|tool| tool.name == "spawn_agent")
        .expect("rlm spawn_agent");
    assert_eq!(
        rlm_spawn.output_contract,
        ToolOutputContract::from_input_schema("output", None)
    );
    assert_eq!(
        rlm_spawn.argument_projection,
        ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed")
    );
    assert!(
        rlm_spawn
            .examples
            .iter()
            .any(|example| example.contains("start call spawn_agent"))
    );
}

#[test]
fn spawn_schema_is_strict_and_nameless() {
    let registry = default_registry(&BTreeMap::new());
    let tool = rlm::spawn_agent_tool_definition(&registry.names());
    let schema = tool.input_schema;
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
    rejected.insert(retired_key, serde_json::Value::String("legacy".to_string()));
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
        lash_core::SessionSpec::inherit().mode(ExecutionMode::new("rlm")),
    )));
    let rlm_spawn = rlm::spawn_agent_tool_definition(&registry.names());

    assert!(
        !rlm_spawn
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
    // one. Each stub returns a different per-tier model from
    // `default_agent_model` so the final child policy's model shows
    // which provider the capability lookup resolved against.
    fn tiered_provider(tag: &'static str) -> lash_core::testing::TestProvider {
        let (kind, default_model, explore_model) = match tag {
            "stale" => ("stale-stub", "stale-model", "stale-explore"),
            "live" => ("live-stub", "live-model", "live-explore"),
            _ => ("stub", "mock-model", "mock-explore"),
        };
        lash_core::testing::TestProvider::builder()
            .kind(kind)
            .default_model(default_model)
            .default_agent_model(move |tier| {
                if tier == "explore" {
                    Some(lash_core::AgentModelSelection {
                        model: explore_model.to_string(),
                        variant: None,
                    })
                } else {
                    None
                }
            })
            .complete_error("stub")
            .build()
    }
    let stale_policy = SessionPolicy {
        provider: tiered_provider("stale").into_handle(),
        execution_mode: lash_core::ExecutionMode::standard(),
        ..SessionPolicy::default()
    };
    let live_policy = SessionPolicy {
        provider: tiered_provider("live").into_handle(),
        execution_mode: lash_core::ExecutionMode::standard(),
        max_context_tokens: Some(1234),
        ..SessionPolicy::default()
    };
    let registry = Arc::new(default_registry(&BTreeMap::new()));
    let current_snapshot = RuntimeSessionState {
        policy: live_policy.clone(),
        ..RuntimeSessionState::default()
    };

    let noop = NoopSubagentSessionConfigurator;
    let request = build_spawn_create_request(SpawnCreateRequestInput {
        registry: &registry,
        parent_session_id: "root",
        current_snapshot: current_snapshot.clone(),
        session_spec: &SessionSpec::inherit(),
        capability_name: "explore",
        output_schema: None,
        seed: Default::default(),
        parent_subagent: None,
        originating_tool_call_id: None,
        configurator: &noop,
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
        child_policy.max_context_tokens,
        live_policy.max_context_tokens
    );
    assert_ne!(child_policy.model, stale_choice);
    assert_eq!(child_policy.model, "live-explore");
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
        originating_tool_call_id: None,
        configurator: &noop,
    })
    .await
    .expect("structured spawn request");
    let structured_policy = structured_request.policy.expect("structured child policy");
    assert_eq!(
        structured_policy.execution_mode,
        lash_core::ExecutionMode::new("rlm"),
        "explore runs in RLM so typed output uses native submit"
    );
    assert!(structured_request.tool_access.tools.is_empty());
}

#[tokio::test]
async fn rlm_spawn_seed_is_visible_to_child_executor_and_prompt() {
    let (outcome, prompt) = run_seed_probe(
        r#"```lashlang
result = (call spawn_agent {
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
handle = start call spawn_agent {
  capability: "default",
  task: "Submit `{ len: len(chunk) }` using the seeded `chunk` variable.",
  seed: { chunk: ["a", "b"] },
  output: Type { len: int }
}
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
result = (call spawn_agent {
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
                lash_mode_rlm::RlmProjectedBindings::new()
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
result = (call spawn_agent {
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

fn seed_probe_provider(id: String, state: Arc<SeedProbeState>) -> lash_core::testing::TestProvider {
    let config_id = id.clone();
    lash_core::testing::TestProvider::builder()
        .kind("seed-probe")
        .default_model("seed-probe-model")
        .serialize_config(move || json!({ "id": config_id.clone() }))
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
    let is_child = state.request_count.fetch_add(1, Ordering::SeqCst) > 0;
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
    ensure_seed_probe_provider_factory();
    let captured_child_prompt: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let state = Arc::new(SeedProbeState {
        parent_response: parent_response.to_string(),
        captured_child_prompt: Arc::clone(&captured_child_prompt),
        request_count: AtomicUsize::new(0),
    });
    let provider_id = format!(
        "seed-probe-{}",
        SEED_PROBE_NEXT_ID.fetch_add(1, Ordering::SeqCst)
    );
    SEED_PROBE_STATES
        .lock()
        .expect("seed probe states")
        .insert(provider_id.clone(), Arc::clone(&state));
    let provider = seed_probe_provider(provider_id.clone(), state);

    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(lash_mode_rlm::BuiltinRlmModePluginFactory::default()),
        Arc::new(SubagentsPluginFactory::new(Arc::new(
            CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                "default",
                lash_core::SessionSpec::inherit().mode(ExecutionMode::new("rlm")),
            ))),
        ))),
    ];
    let plugins = PluginHost::new(factories)
        .with_processes()
        .build_session("root", ExecutionMode::new("rlm"), None, None)
        .expect("plugin session");
    let host = ProcessRuntimeHost::new(
        lash_core::EmbeddedRuntimeHost::new(RuntimeCoreConfig::default()),
        Arc::new(LocalProcessRegistry::default()),
    );
    let policy = SessionPolicy {
        provider: provider.into_handle(),
        model: "seed-probe-model".to_string(),
        execution_mode: ExecutionMode::new("rlm"),
        max_context_tokens: Some(64_000),
        max_turns: Some(4),
        ..SessionPolicy::default()
    };
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

    let turn = runtime
        .run_turn_assembled(input, tokio_util::sync::CancellationToken::new())
        .await
        .expect("turn");

    let prompt = captured_child_prompt
        .lock()
        .expect("captured prompt")
        .clone()
        .expect("child prompt was captured");
    SEED_PROBE_STATES
        .lock()
        .expect("seed probe states")
        .remove(&provider_id);
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
async fn standard_provider_does_not_expose_subagent_tools() {
    let factory = SubagentsPluginFactory::new(Arc::new(default_registry(&BTreeMap::new())));
    let ctx = PluginSessionContext {
        session_id: "parent".to_string(),
        execution_mode: lash_core::ExecutionMode::standard(),
        standard_context_approach: None,
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: None,
        processes_available: false,
        parent_session_id: None,
    };
    let plugin = factory.build(&ctx).expect("plugin");
    assert_eq!(plugin.id(), "subagents");
}

#[tokio::test]
async fn rlm_provider_requires_process_support() {
    let factory = SubagentsPluginFactory::new(Arc::new(default_registry(&BTreeMap::new())));
    let ctx = PluginSessionContext {
        session_id: "parent".to_string(),
        execution_mode: lash_core::ExecutionMode::new("rlm"),
        standard_context_approach: None,
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: None,
        processes_available: false,
        parent_session_id: None,
    };

    let err = match factory.build(&ctx) {
        Ok(_) => panic!("rlm build should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("subagents require session process support"),
        "{err}"
    );
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
        .map(|tool| (tool.name.clone(), std::sync::Arc::new(tool.contract())))
        .collect::<std::collections::BTreeMap<_, _>>();
    let ctx = ToolSurfaceContext {
        session_id: "child".to_string(),
        mode: lash_core::ExecutionMode::standard(),
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
    };

    let contribution =
        rlm_support::subagent_surface_contribution(ctx).expect("surface contribution");
    assert!(contribution.overrides.is_empty());
    assert_eq!(
        contribution.tool_list_notes,
        vec!["Subagent capability: explore. Depth: 1/5."]
    );
}
