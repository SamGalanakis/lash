use std::sync::{Arc, RwLock};

use lash_core::plugin::{
    PluginDirective, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    ProtocolDriverPlugin, SessionPlugin, ToolCallHookContext,
};
use lash_core::{ProtocolBuildInput, TurnDriverPreamble};
use lash_plugin_tool_output_budget::ToolOutputBudgetConfig;

use crate::driver::{RlmProjectorConfig, SharedPromptUsage, build_rlm_preamble};
use crate::projection::{
    ProjectionRegistry, ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectionExtension,
};
use crate::rlm_support::BoundVariablesCache;
#[cfg(test)]
use crate::rlm_support::format_budget_suffix;
use crate::stream_mask;

pub const RLM_PROTOCOL_PLUGIN_ID: &str = "rlm_protocol";

mod budget_warning;
mod forced_continuation;
mod prose_projector;
mod protocol_session;
mod runtime_state;

use budget_warning::BudgetUsageObserver;
#[cfg(test)]
use forced_continuation::parse_forced_continue_as_args;
use prose_projector::RlmAssistantProseProjector;
use protocol_session::RlmProtocolSession;
use runtime_state::{RlmCodeExecutor, RlmRuntimeState};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmProtocolPluginConfig {
    pub observe_projection: ToolOutputBudgetConfig,
    #[serde(default)]
    pub prompt_features: crate::protocol::RlmPromptFeatures,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_continue_as_soft_warn_tokens")]
    pub continue_as_soft_warn_tokens: Option<usize>,
    #[serde(default = "default_continue_as_forced_fallback_tokens")]
    pub continue_as_forced_fallback_tokens: Option<usize>,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_continue_as_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

fn default_continue_as_forced_fallback_tokens() -> Option<usize> {
    Some(140_000)
}

impl Default for RlmProtocolPluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolOutputBudgetConfig::default(),
            prompt_features: crate::protocol::RlmPromptFeatures::default(),
            max_output_chars: default_max_output_chars(),
            continue_as_soft_warn_tokens: default_continue_as_soft_warn_tokens(),
            continue_as_forced_fallback_tokens: default_continue_as_forced_fallback_tokens(),
        }
    }
}

impl RlmProtocolPluginConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let (Some(soft), Some(forced)) = (
            self.continue_as_soft_warn_tokens,
            self.continue_as_forced_fallback_tokens,
        ) && forced < soft
        {
            return Err(format!(
                "continue_as_forced_fallback_tokens ({forced}) must be greater than or equal to continue_as_soft_warn_tokens ({soft})"
            ));
        }
        Ok(())
    }
}

pub struct RlmProtocolPluginFactory {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
}

impl RlmProtocolPluginFactory {
    pub fn new(config: RlmProtocolPluginConfig) -> Self {
        Self {
            config,
            projection_resolver: Arc::new(ProjectionRegistry::default()),
        }
    }

    pub fn with_projection_resolver(
        mut self,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
        self
    }
}

impl Default for RlmProtocolPluginFactory {
    fn default() -> Self {
        Self::new(RlmProtocolPluginConfig::default())
    }
}

impl PluginFactory for RlmProtocolPluginFactory {
    fn id(&self) -> &'static str {
        RLM_PROTOCOL_PLUGIN_ID
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmProtocolPlugin {
            config: self.config.clone(),
            projection_resolver: Arc::clone(&self.projection_resolver),
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }))
    }
}

struct RlmProtocolPlugin {
    config: RlmProtocolPluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    last_prompt_usage: SharedPromptUsage,
}

impl SessionPlugin for RlmProtocolPlugin {
    fn id(&self) -> &'static str {
        RLM_PROTOCOL_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let runtime_state = Arc::new(
            RlmRuntimeState::new(self.config.clone(), Arc::clone(&self.projection_resolver))
                .map_err(|err| PluginError::Session(err.to_string()))?,
        );
        let code_executor = Arc::new(RlmCodeExecutor::new(Arc::clone(&runtime_state)));
        let protocol_session = Arc::new(
            RlmProtocolSession::new(self.config.clone(), Arc::clone(&runtime_state))
                .map_err(|err| PluginError::Session(err.to_string()))?,
        );
        reg.protocol().session(protocol_session.clone())?;
        reg.execution().code_executor(code_executor)?;
        reg.output()
            .assistant_prose_projector(Arc::new(RlmAssistantProseProjector))?;
        reg.protocol().protocol_driver(Arc::new(RlmProtocolDriver {
            config: self.config.clone(),
            last_prompt_usage: Arc::clone(&self.last_prompt_usage),
        }))?;
        reg.tools()
            .provider(Arc::new(crate::control_tools::RlmControlToolsProvider))?;
        reg.tool_calls().before(Arc::new(|ctx| {
            Box::pin(async move { normalize_projected_tool_args(ctx) })
        }));

        let bound_vars_cache = Arc::new(BoundVariablesCache::new());
        let bound_vars_hook: lash_core::plugin::PromptContributor = Arc::new(move |ctx| {
            let cache = Arc::clone(&bound_vars_cache);
            Box::pin(async move { Ok(cache.contributions(&ctx)) })
        });
        reg.prompt().contribute(bound_vars_hook);

        let projected_session = protocol_session.clone();
        reg.prompt().contribute(Arc::new(move |ctx| {
            let session = projected_session.clone();
            Box::pin(async move {
                let mut contributions = session.projected_binding_prompt_contributions().await;
                if let Some(extension) = ctx
                    .turn_context
                    .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
                {
                    contributions.extend(RlmProjectionExtension::prompt_contributions_for(
                        &extension.bindings,
                    ));
                }
                Ok(contributions)
            })
        }));

        // Per-turn `prompt_usage` is captured here and passed to the
        // projector via a shared cell so the budget line can ride in the
        // volatile turn-tail message instead of poisoning the cached
        // system prefix.
        reg.history().prepare_turn(
            10,
            Arc::new(BudgetUsageObserver {
                cell: Arc::clone(&self.last_prompt_usage),
            }),
        );

        let warn_session = protocol_session.clone();
        reg.turn().checkpoint(Arc::new(move |ctx| {
            let session = warn_session.clone();
            Box::pin(async move { session.soft_warn_directives(ctx) })
        }));

        stream_mask::register_stream_mask(reg)?;
        Ok(())
    }
}

struct RlmProtocolDriver {
    config: RlmProtocolPluginConfig,
    last_prompt_usage: SharedPromptUsage,
}

impl ProtocolDriverPlugin for RlmProtocolDriver {
    fn build_preamble(&self, input: ProtocolBuildInput) -> TurnDriverPreamble {
        build_rlm_preamble(
            input,
            RlmProjectorConfig {
                max_output_chars: self.config.max_output_chars,
                max_budget_tokens: self.config.continue_as_soft_warn_tokens,
                last_prompt_usage: Arc::clone(&self.last_prompt_usage),
                prompt_features: self.config.prompt_features,
            },
        )
    }
}

fn normalize_projected_tool_args(
    ctx: ToolCallHookContext,
) -> Result<Vec<PluginDirective>, PluginError> {
    let original = ctx.args;
    let normalized = crate::projection::normalize_tool_args_for_projection(
        original.clone(),
        &ctx.argument_projection,
    );
    if normalized == original {
        Ok(Vec::new())
    } else {
        Ok(vec![PluginDirective::ReplaceToolArgs { args: normalized }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::budget_warning::BUDGET_WARNING_STATUS;
    use crate::projection::{RlmProjectedBindings, RlmSeed};
    use lash_core::plugin::ProtocolSessionPlugin;

    struct NoopPromptManager;

    #[async_trait::async_trait]
    impl lash_core::plugin::runtime_host::RuntimeSessionHost for NoopPromptManager {
        async fn snapshot_current(
            &self,
        ) -> Result<lash_core::RuntimeSessionState, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash_core::RuntimeSessionState, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash_core::plugin::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            _request: lash_core::SessionCreateRequest,
        ) -> Result<lash_core::SessionHandle, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn close_session(
            &self,
            _session_id: &str,
        ) -> Result<(), lash_core::plugin::PluginError> {
            Ok(())
        }
    }
    fn prompt_usage(context_budget_tokens: usize) -> lash_core::PromptUsage {
        lash_core::PromptUsage {
            prompt_context_tokens: context_budget_tokens,
            input_tokens: context_budget_tokens,
            cached_input_tokens: 0,
            context_budget_tokens,
        }
    }

    fn test_session(config: RlmProtocolPluginConfig) -> RlmProtocolSession {
        let runtime_state = Arc::new(
            RlmRuntimeState::new(config.clone(), Arc::new(ProjectionRegistry::default()))
                .expect("runtime state"),
        );
        RlmProtocolSession::new(config, runtime_state).expect("session should build")
    }

    #[test]
    fn rlm_config_defaults_budget_thresholds() {
        let config = RlmProtocolPluginConfig::default();

        assert_eq!(config.continue_as_soft_warn_tokens, Some(100_000));
        assert_eq!(config.continue_as_forced_fallback_tokens, Some(140_000));
        config.validate().expect("default config should validate");
    }

    #[test]
    fn rlm_config_validation_accepts_equal_and_disabled_thresholds() {
        RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: Some(100_000),
            ..Default::default()
        }
        .validate()
        .expect("equal thresholds are allowed");

        RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: None,
            continue_as_forced_fallback_tokens: Some(80_000),
            ..Default::default()
        }
        .validate()
        .expect("soft warning can be disabled independently");

        RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: None,
            ..Default::default()
        }
        .validate()
        .expect("forced fallback can be disabled independently");
    }

    #[test]
    fn rlm_config_validation_rejects_forced_below_soft_warn() {
        let err = RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: Some(99_999),
            ..Default::default()
        }
        .validate()
        .expect_err("forced threshold below soft warn should fail");

        assert!(err.contains("continue_as_forced_fallback_tokens"));
    }

    #[test]
    fn forced_fallback_accepts_omitted_seed() {
        let parsed = parse_forced_continue_as_args(r#"{"task":"continue from compact state"}"#)
            .expect("valid fallback args");

        assert_eq!(
            parsed.get("task").and_then(serde_json::Value::as_str),
            Some("continue from compact state")
        );
        assert!(parsed.get("seed").is_none());
    }

    #[test]
    fn forced_fallback_rejects_schema_extra_fields() {
        let err = parse_forced_continue_as_args(
            r#"{"task":"continue from compact state","unexpected":true}"#,
        )
        .expect_err("extra properties should fail");

        assert!(err.to_string().contains("schema-invalid"));
    }

    fn projected(value: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "__projected__": value })
    }

    fn received_tool_args(
        policy: lash_core::ToolArgumentProjectionPolicy,
        args: serde_json::Value,
    ) -> serde_json::Value {
        crate::projection::normalize_tool_args_for_projection(args, &policy)
    }

    fn materializing_args(args: serde_json::Value) -> serde_json::Value {
        received_tool_args(
            lash_core::ToolArgumentProjectionPolicy::MaterializeProjectedValues,
            args,
        )
    }

    fn seed_preserving_args(args: serde_json::Value) -> serde_json::Value {
        received_tool_args(
            lash_core::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
            args,
        )
    }

    fn classify_received_seed(received: &serde_json::Value) -> RlmSeed {
        RlmSeed::from_tool_args(received).expect("seed should classify")
    }

    #[test]
    fn projected_tool_arg_normalization_materializes_ordinary_tools_recursively() {
        let args = serde_json::json!({
            "path": projected(serde_json::json!("/tmp/projected.txt")),
            "nested": {
                "items": [
                    projected(serde_json::json!("a")),
                    { "plain": projected(serde_json::json!(true)) }
                ]
            }
        });

        let normalized = materializing_args(args);

        assert_eq!(
            normalized,
            serde_json::json!({
                "path": "/tmp/projected.txt",
                "nested": {
                    "items": [
                        "a",
                        { "plain": true }
                    ]
                }
            })
        );
    }

    #[test]
    fn projected_tool_arg_normalization_preserves_seed_roots_for_projection_aware_tools() {
        let args = serde_json::json!({
            "task": projected(serde_json::json!("inspect the file")),
            "capability": "explore",
            "seed": {
                "projected_root": projected(serde_json::json!("carry-over")),
                "computed_record": {
                    "field": projected(serde_json::json!("materialize me"))
                }
            }
        });

        let normalized = seed_preserving_args(args);

        assert_eq!(
            normalized,
            serde_json::json!({
                "task": "inspect the file",
                "capability": "explore",
                "seed": {
                    "projected_root": { "__projected__": "carry-over" },
                    "computed_record": {
                        "field": "materialize me"
                    }
                }
            })
        );
    }

    #[test]
    fn projected_tool_arg_normalization_preserves_continue_as_seed_roots() {
        let args = serde_json::json!({
            "task": projected(serde_json::json!("continue")),
            "seed": {
                "problem": projected(serde_json::json!({ "prompt": "large prompt" }))
            }
        });

        let normalized = seed_preserving_args(args);
        let seed = classify_received_seed(&normalized);

        assert_eq!(
            normalized.get("task").and_then(serde_json::Value::as_str),
            Some("continue")
        );
        assert_eq!(
            seed.projected.entries.as_slice(),
            &[(
                "problem".to_string(),
                serde_json::json!({ "prompt": "large prompt" })
            )]
        );
        assert!(seed.globals.is_empty());
    }

    #[test]
    fn ordinary_tool_receives_non_projected_input_without_materialization() {
        let args = serde_json::json!({
            "query": "plain",
            "options": { "limit": 3, "exact": true }
        });

        let received = materializing_args(args.clone());

        assert_eq!(received, args);
    }

    #[test]
    fn ordinary_tool_receives_projected_input_materialized_as_plain_json() {
        let received = materializing_args(serde_json::json!({
            "query": projected(serde_json::json!("lazy query")),
            "options": {
                "limit": projected(serde_json::json!(3)),
                "filters": [
                    projected(serde_json::json!("rust")),
                    "tests"
                ]
            }
        }));

        assert_eq!(
            received,
            serde_json::json!({
                "query": "lazy query",
                "options": {
                    "limit": 3,
                    "filters": ["rust", "tests"]
                }
            })
        );
    }

    #[test]
    fn projection_aware_tool_receives_non_projected_seed_as_plain_input() {
        let received = seed_preserving_args(serde_json::json!({
            "task": "continue from facts",
            "capability": "explore",
            "seed": {
                "facts": { "count": 2 },
                "label": "plain"
            }
        }));

        let seed = classify_received_seed(&received);

        assert!(seed.projected.is_empty());
        assert_eq!(
            seed.globals,
            serde_json::Map::from_iter([
                ("facts".to_string(), serde_json::json!({ "count": 2 })),
                ("label".to_string(), serde_json::json!("plain")),
            ])
        );
    }

    #[test]
    fn projection_aware_tool_receives_projected_seed_roots_without_materializing_them() {
        let received = seed_preserving_args(serde_json::json!({
            "task": projected(serde_json::json!("continue from projected context")),
            "capability": "explore",
            "seed": {
                "problem": projected(serde_json::json!("large parent context")),
                "computed": {
                    "summary": projected(serde_json::json!("short summary"))
                }
            }
        }));

        let seed = classify_received_seed(&received);

        assert_eq!(
            received.get("task").and_then(serde_json::Value::as_str),
            Some("continue from projected context")
        );
        assert_eq!(
            seed.projected.entries.as_slice(),
            &[(
                "problem".to_string(),
                serde_json::json!("large parent context")
            )]
        );
        assert_eq!(
            seed.globals,
            serde_json::Map::from_iter([(
                "computed".to_string(),
                serde_json::json!({ "summary": "short summary" })
            )])
        );
    }

    #[test]
    fn projection_policy_cutover_has_no_name_based_projection_checks() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let plugin_src = std::fs::read_to_string(manifest_dir.join("src/plugin.rs"))
            .expect("read plugin source");
        let executor_src = std::fs::read_to_string(manifest_dir.join("src/executor.rs"))
            .expect("read executor source");
        let host_bridge_src =
            std::fs::read_to_string(manifest_dir.join("src/executor/host_bridge.rs"))
                .expect("read host bridge source");
        let projected_bindings_src =
            std::fs::read_to_string(manifest_dir.join("src/projection/bindings.rs"))
                .expect("read projected bindings source");

        let old_hook_call = ["normalize_tool_args_for_projection", "(&ctx.tool_name"].concat();
        let old_name_match = [
            "matches!",
            "(tool_name, ",
            "\"continue_as\" | \"spawn_agent\")",
        ]
        .concat();
        let old_invalid_ref = [
            "ProjectedBindingError::duplicate(",
            "\"invalid_projection_ref\")",
        ]
        .concat();

        assert!(!plugin_src.contains(&old_hook_call));
        assert!(!executor_src.contains(&old_name_match));
        assert!(!host_bridge_src.contains(&old_name_match));
        assert!(!projected_bindings_src.contains(&old_invalid_ref));
    }

    #[test]
    fn budget_prompt_contribution_below_advisory_floor_emits_status_only() {
        // 23% of the configured handoff threshold — emit the status line so
        // the model has continuous context-size awareness, but no
        // `continue_as` nag.
        let usage = prompt_usage(47_213);
        let content = format_budget_suffix(0, Some(&usage), Some(200_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 47213 · handoff threshold: 200000 (23%)"));
        assert!(content.contains("Turn:"));
        assert!(!content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[tokio::test]
    async fn session_projection_extension_rejects_duplicate_names() {
        let session = test_session(RlmProtocolPluginConfig::default());
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("first bind"),
            ))
            .await
            .expect("first projection");

        let duplicate = session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("second"))
                    .expect("second bind"),
            ))
            .await;
        let Err(err) = duplicate else {
            panic!("duplicate session projection should fail");
        };
        assert!(err.to_string().contains("current_query"));
    }

    #[tokio::test]
    async fn session_projection_prompt_contribution_lists_names() {
        let session = test_session(RlmProtocolPluginConfig::default());
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("bind"),
            ))
            .await
            .expect("projection");

        let contributions = session.projected_binding_prompt_contributions().await;
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].content.contains("`current_query`"));
        assert!(contributions[0].content.contains("Readonly: true"));
    }

    #[test]
    fn budget_prompt_contribution_advisory_tier_60_to_89_pct() {
        // 75% of threshold — advisory: scout for a clean handoff point.
        let usage = prompt_usage(75_000);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 75000 · handoff threshold: 100000 (75%)"));
        assert!(content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[test]
    fn budget_prompt_contribution_tight_tier_90_to_99_pct() {
        // 95% of threshold — tight: wrap the current step, then continue_as.
        let usage = prompt_usage(95_000);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 95000 · handoff threshold: 100000 (95%)"));
        assert!(content.contains("Budget tight"));
        assert!(content.contains("`continue_as`"));
        assert!(!content.contains("Past the handoff threshold"));
        assert!(!content.contains("Look for a clean handoff point"));
    }

    #[test]
    fn budget_prompt_contribution_over_threshold_forces_handoff() {
        let usage = prompt_usage(120_292);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 120292 · handoff threshold: 100000 (120%)"));
        assert!(content.contains("Past the handoff threshold"));
        assert!(content.contains("End this block with `continue_as` now"));
        assert!(content.contains("do not call `submit`"));
        assert!(content.contains("`task` + `seed`"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = test_session(RlmProtocolPluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            ..Default::default()
        });
        let state = lash_core::SessionStateEnvelope {
            token_usage: lash_core::TokenUsage {
                input_tokens: 120_292,
                ..Default::default()
            },
            ..Default::default()
        };
        let directives = session
            .soft_warn_directives(lash_core::plugin::CheckpointHookContext {
                session_id: "root".to_string(),
                checkpoint: lash_core::CheckpointKind::AfterWork,
                state: lash_core::SessionReadView::from_exported_state(&state),
                host: std::sync::Arc::new(NoopPromptManager),
            })
            .expect("warning directives");

        assert_eq!(directives.len(), 1);
        let lash_core::plugin::PluginDirective::EmitRuntimeEvents { events } = &directives[0]
        else {
            panic!("budget warning must be a runtime event, not an injected message");
        };
        assert_eq!(events.len(), 1);
        let lash_core::PluginRuntimeEvent::Status { key, label, detail } = &events[0] else {
            panic!("budget warning should use a typed status runtime event");
        };
        assert_eq!(key, BUDGET_WARNING_STATUS);
        assert_eq!(label, "context budget");
        assert!(detail.as_deref().is_some_and(|text| {
            text.contains("120292 tokens used") && text.contains("choose handoff path")
        }));
    }

    #[test]
    fn budget_prompt_contribution_omits_without_configured_budget() {
        let usage = prompt_usage(47_213);

        assert!(format_budget_suffix(0, Some(&usage), None).is_none());
    }

    #[test]
    fn budget_prompt_contribution_omits_without_used_tokens() {
        let usage = prompt_usage(0);

        assert!(format_budget_suffix(0, Some(&usage), Some(200_000)).is_none());
    }
}
