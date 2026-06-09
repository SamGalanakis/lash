//! Startup stage: assemble the CLI plugin stack, the CLI prompt layer, and
//! the autonomous-mode tool policy.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lash::ModeId;
use lash::plugins::PluginFactory;
use lash::prompt::{
    PromptBuiltin, PromptContribution, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection,
};
use lash::{ModelSpec, PluginStack, SessionSpec};
use lash_core::{PromptLayer, SessionToolAccess, ToolState};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_plugin_plan_mode::{PlanModePluginFactory, UpdatePlanPluginFactory};
use lash_standard_plugins::{
    StandardContextApproach, StandardToolStackOptions, standard_tool_stack,
};
use lash_subagents::{SubagentsPluginFactory, default_registry};

use crate::prompt_context_plugin::{
    InstructionSource, PromptContextPluginConfig, PromptContextPluginFactory,
};
use crate::prompt_tool::{CliPromptBridge, cli_ask_plugin_factory};

const CLI_AUTONOMOUS_INTRO: &str = "You are an autonomous AI coding assistant running without a human in the loop.\nComplete the task end-to-end without asking for user input.\nIf the task is incomplete and concrete next actions are available, continue executing them instead of stopping to summarize incompletion.";
const CLI_AUTONOMOUS_EXECUTION: &str = "- No user is available during this run. Default to acting without asking. Ask only when progress is blocked and user intervention is strictly required; otherwise make the best reasonable decision from local context and continue.\n- Do not stop merely to report that work remains. If concrete actions are still available, keep executing them.\n- Only summarize remaining work when you are blocked, need a decision, or have exhausted feasible actions for this turn.\n- Do not claim completion unless you have actually verified the required end state.";
const CLI_RLM_SUBMISSION_GUIDANCE: &str = "- When calling `submit`, keep the submitted value concise. Do not include large variables such as diffs, full logs, raw command output, or other bulky dumps unless the user explicitly asks for them. Prefer short prose. If you use `format`, use it with small values rather than large captured variables.";

pub(super) struct PluginFactorySurfaceInput<'a> {
    pub(super) autonomous: bool,
    pub(super) execution_mode: ModeId,
    pub(super) standard_context_approach: Option<StandardContextApproach>,
    pub(super) tavily_key: String,
    pub(super) instruction_source: Arc<dyn InstructionSource>,
    pub(super) agent_model_specs: &'a BTreeMap<String, ModelSpec>,
    pub(super) host_docs_dir: Option<std::path::PathBuf>,
    pub(super) prompt_bridge: CliPromptBridge,
}

pub(super) fn plugin_factories_for_surface(input: PluginFactorySurfaceInput<'_>) -> PluginStack {
    let PluginFactorySurfaceInput {
        autonomous,
        execution_mode,
        standard_context_approach,
        tavily_key,
        instruction_source,
        agent_model_specs,
        host_docs_dir,
        prompt_bridge,
    } = input;
    let capability_registry = Arc::new(default_registry(agent_model_specs));

    let runtime_options = StandardToolStackOptions {
        standard_context_approach: standard_context_approach.clone(),
        include_cancel_process: execution_mode == ModeId::standard(),
        tavily_api_key: if tavily_key.is_empty() {
            None
        } else {
            Some(tavily_key)
        },
    };
    let mut plugin_stack = standard_tool_stack(runtime_options);
    plugin_stack.push(Arc::new(PromptContextPluginFactory::new(
        Arc::clone(&instruction_source),
        PromptContextPluginConfig::default(),
        execution_mode.clone(),
    )) as Arc<dyn PluginFactory>);
    if !autonomous {
        if let Some(host_docs_dir) = host_docs_dir {
            plugin_stack.push(
                Arc::new(crate::host_docs::HostDocsPluginFactory::new(host_docs_dir))
                    as Arc<dyn PluginFactory>,
            );
        }
        plugin_stack.push(Arc::new(
            PlanModePluginFactory::new(Default::default())
                .with_prompt(Arc::new(prompt_bridge.clone())),
        ));
        plugin_stack.push(cli_ask_plugin_factory(prompt_bridge));
        // `update_plan` drives the sticky plan dock at the bottom of
        // the TUI. Interactive-only here; root-only inside the plugin
        // itself (the factory returns an inert plugin for subagent
        // / compaction / other non-root sessions).
        plugin_stack.push(Arc::new(UpdatePlanPluginFactory));
    }
    plugin_stack.push(Arc::new(lash_autoresearch::AutoresearchPluginFactory));
    if execution_mode == ModeId::rlm() {
        plugin_stack.push(Arc::new(LlmToolsPluginFactory::default()));
        plugin_stack.push(Arc::new(
            SubagentsPluginFactory::new(capability_registry)
                .with_session_spec(SessionSpec::inherit())
                .with_tool_access(cli_child_tool_access()),
        ));
    }
    plugin_stack
}

fn cli_child_tool_access() -> SessionToolAccess {
    SessionToolAccess {
        tools: Vec::new(),
        hidden_tools: cli_child_hidden_tools(),
    }
}

fn cli_child_hidden_tools() -> BTreeSet<String> {
    [
        "ask",
        "showcase",
        "request_user_input",
        "plan_exit",
        "update_plan",
    ]
    .into_iter()
    .map(ToOwned::to_owned)
    .collect()
}

fn autonomous_tool_allowed(name: &str) -> bool {
    !matches!(name, "ask" | "showcase")
        && !name.starts_with("plan_")
        && name != "request_user_input"
}

pub(super) async fn apply_autonomous_tool_policy(
    session: &lash::LashSession,
) -> anyhow::Result<()> {
    let mut snapshot = session.control().tools().state().await?;
    retain_autonomous_tools(&mut snapshot);
    session
        .control()
        .tools()
        .advanced()
        .apply_state(snapshot)
        .await?;
    Ok(())
}

fn retain_autonomous_tools(snapshot: &mut ToolState) {
    snapshot.retain(|name, _| autonomous_tool_allowed(name));
}

pub(super) fn cli_prompt_config(autonomous: bool, execution_mode: &ModeId) -> PromptLayer {
    let mut intro_entries = vec![PromptTemplateEntry::builtin(PromptBuiltin::MainAgentIntro)];
    intro_entries.push(PromptTemplateEntry::slot(PromptSlot::Intro));

    let execution_entries = vec![
        PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
        PromptTemplateEntry::slot(PromptSlot::Execution),
    ];

    let template = PromptTemplate::new(vec![
        PromptTemplateSection::untitled(intro_entries),
        PromptTemplateSection::titled("Execution", execution_entries),
        PromptTemplateSection::titled(
            "Guidance",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::CoreGuidance),
                PromptTemplateEntry::slot(PromptSlot::ProjectInstructions),
                PromptTemplateEntry::slot(PromptSlot::Guidance),
            ],
        ),
        PromptTemplateSection::titled(
            "Environment",
            vec![
                PromptTemplateEntry::slot(PromptSlot::RuntimeContext),
                PromptTemplateEntry::slot(PromptSlot::Environment),
            ],
        ),
    ]);

    let mut layer = PromptLayer::with_template(template);
    if autonomous {
        layer.add_contribution(
            PromptContribution::new(PromptSlot::Intro, "", CLI_AUTONOMOUS_INTRO)
                .with_priority(-100),
        );
        layer.add_contribution(
            PromptContribution::new(PromptSlot::Execution, "", CLI_AUTONOMOUS_EXECUTION)
                .with_priority(100),
        );
    }
    if *execution_mode == ModeId::rlm() {
        layer.add_contribution(
            PromptContribution::new(
                PromptSlot::Execution,
                "RLM Submit Output",
                CLI_RLM_SUBMISSION_GUIDANCE,
            )
            .with_priority(200),
        );
    }
    layer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::FsInstructionSource;
    use lash::tools::{
        ToolAvailabilityConfig, ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider,
        ToolResult, ToolScheduling,
    };
    use lash_core::ToolRegistry;

    struct DummyToolProvider;

    fn dummy_tool(name: &str) -> ToolDefinition {
        ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            format!("{name} description"),
            ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "null" }),
        )
        .with_availability(ToolAvailabilityConfig::callable())
        .with_scheduling(ToolScheduling::Parallel)
    }

    #[async_trait::async_trait]
    impl ToolProvider for DummyToolProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            dummy_tools()
                .into_iter()
                .map(|tool| tool.manifest())
                .collect()
        }

        fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
            dummy_tools()
                .into_iter()
                .find(|tool| tool.name() == name)
                .map(|tool| std::sync::Arc::new(tool.contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            ToolResult::err_fmt(format_args!("unexpected tool call: {}", call.name))
        }
    }

    fn dummy_tools() -> Vec<ToolDefinition> {
        vec![
            dummy_tool("read_file"),
            dummy_tool("ask"),
            dummy_tool("plan_exit"),
            dummy_tool("showcase"),
        ]
    }

    fn plugin_factory_ids_for_autonomous(autonomous: bool) -> Vec<&'static str> {
        let agent_model_specs = BTreeMap::new();
        plugin_factories_for_surface(PluginFactorySurfaceInput {
            autonomous,
            execution_mode: ModeId::standard(),
            standard_context_approach: None,
            tavily_key: String::new(),
            instruction_source: Arc::new(FsInstructionSource::default()),
            agent_model_specs: &agent_model_specs,
            host_docs_dir: (!autonomous)
                .then(|| std::path::PathBuf::from("/tmp/lash-home/docs/lash-cli")),
            prompt_bridge: CliPromptBridge::default(),
        })
        .into_factories()
        .into_iter()
        .map(|factory| factory.id())
        .collect()
    }

    #[test]
    fn cli_surface_stack_does_not_install_mode_factories() {
        let ids = plugin_factory_ids_for_autonomous(false);

        assert!(!ids.contains(&"standard_protocol"));
        assert!(!ids.contains(&"rlm_protocol"));
    }

    #[test]
    fn host_docs_prompt_is_interactive_only() {
        let interactive_ids = plugin_factory_ids_for_autonomous(false);
        assert!(interactive_ids.contains(&"lash_cli_host_docs"));

        let autonomous_ids = plugin_factory_ids_for_autonomous(true);
        assert!(!autonomous_ids.contains(&"lash_cli_host_docs"));
    }

    #[test]
    fn rlm_prompt_config_uses_execution_slot_and_contribution() {
        let layer = cli_prompt_config(false, &ModeId::rlm());
        let template = layer.template.as_ref().expect("cli prompt template");
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        let execution = template
            .sections
            .iter()
            .find(|section| section.title.as_deref() == Some("Execution"))
            .expect("execution section");
        assert!(execution.entries.iter().any(|entry| {
            matches!(
                entry,
                PromptTemplateEntry::Slot {
                    slot: PromptSlot::Execution
                }
            )
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::Execution
                && contribution.content.as_ref() == CLI_RLM_SUBMISSION_GUIDANCE
        }));
    }

    #[test]
    fn standard_prompt_config_omits_rlm_contribution() {
        let layer = cli_prompt_config(false, &ModeId::standard());
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        assert!(
            !contributions
                .iter()
                .any(|contribution| contribution.content.as_ref() == CLI_RLM_SUBMISSION_GUIDANCE)
        );
    }

    #[test]
    fn autonomous_prompt_config_uses_neutral_slots() {
        let layer = cli_prompt_config(true, &ModeId::standard());
        let template = layer.template.as_ref().expect("cli prompt template");
        let contributions = layer
            .slots
            .values()
            .flat_map(|slot| slot.contributions.iter())
            .collect::<Vec<_>>();

        assert!(template.sections.iter().any(|section| {
            section.entries.iter().any(|entry| {
                matches!(
                    entry,
                    PromptTemplateEntry::Slot {
                        slot: PromptSlot::Intro
                    }
                )
            })
        }));
        assert!(template.sections.iter().any(|section| {
            section.entries.iter().any(|entry| {
                matches!(
                    entry,
                    PromptTemplateEntry::Slot {
                        slot: PromptSlot::Execution
                    }
                )
            })
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::Intro
                && contribution.content.as_ref() == CLI_AUTONOMOUS_INTRO
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.slot == PromptSlot::Execution
                && contribution.content.as_ref() == CLI_AUTONOMOUS_EXECUTION
        }));
    }

    #[test]
    fn autonomous_policy_disables_interactive_tools() {
        let tool_registry = ToolRegistry::from_tool_provider(Arc::new(DummyToolProvider)).unwrap();

        let mut snapshot = tool_registry.export_state();
        retain_autonomous_tools(&mut snapshot);
        assert!(snapshot.contains("read_file"));
        assert!(!snapshot.contains("ask"));
        assert!(!snapshot.contains("plan_exit"));
        assert!(!snapshot.contains("showcase"));
    }

    #[test]
    fn cli_child_tool_access_hides_interactive_root_tools() {
        let hidden = cli_child_hidden_tools();

        assert!(hidden.contains("ask"));
        assert!(hidden.contains("plan_exit"));
        assert!(hidden.contains("update_plan"));
        assert!(hidden.contains("showcase"));
        assert!(hidden.contains("request_user_input"));
    }
}
