use std::sync::Arc;

use lash::agent::{Message, MessageRole};
use lash::provider::Provider;
use lash::*;
use sha2::{Digest, Sha256};

use crate::ROOT_SESSION_ID;
use crate::app::{App, DisplayBlock, PersistedUiState, PreparedTurn};

#[derive(Debug, Clone)]
pub(crate) struct ModelSelection {
    pub(crate) model: String,
}

pub(crate) fn controls_text() -> String {
    [
        "Controls:",
        "  Esc                Cancel agent (while running)",
        "  Enter              Submit; inject at next checkpoint while running",
        "  Tab                Queue next turn; submit plain draft when idle",
        "  Up (empty draft)   Edit last queued turn",
        "  Shift+Tab          Toggle persistent plan mode",
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down",
        "  PgUp / PgDn        Scroll page up / down",
        "  Shift+Enter        Insert newline",
        "  Ctrl+V             Paste image as inline [Image #n]",
        "  Ctrl+Shift+V       Paste text only",
        "  Ctrl+Y             Copy last response to clipboard",
        "  Ctrl+O             Cycle tool expansion (ghost ↔ compact)",
        "  Alt+O              Full expansion (code + stdout)",
        "  Up / Down          Input history",
        "  Shift+Drag         Select text (terminal native)",
        "  Ctrl+C             Quit",
    ]
    .join("\n")
}

pub(crate) fn models_dev_catalog() -> Result<Arc<CachedModelCatalog>, String> {
    CachedModelCatalog::models_dev(
        Arc::new(FileModelCatalogStore::default_models_dev()),
        Some(Arc::new(ModelsDevHttpSource::default_models_dev())),
    )
    .map(Arc::new)
}

pub(crate) fn parse_model_selection(input: &str) -> Result<ModelSelection, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Model cannot be empty.".to_string());
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts.as_slice() {
        [model] => Ok(ModelSelection {
            model: (*model).to_string(),
        }),
        _ => Err("Model input must be a single `<model>` token.".to_string()),
    }
}

fn parse_variant_input(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Variant cannot be empty.".to_string());
    }
    if trimmed.contains(char::is_whitespace) {
        return Err("Variant must be a single token.".to_string());
    }
    Ok(trimmed.to_ascii_lowercase())
}

pub(crate) fn parse_execution_mode(input: &str) -> Result<ExecutionMode, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Execution mode cannot be empty.".to_string()),
        "repl" => Ok(ExecutionMode::Repl),
        "standard" | "tools" => Ok(ExecutionMode::Standard),
        other => Err(format!(
            "Unknown execution mode `{other}`. Expected `repl` or `standard`."
        )),
    }
}

pub(crate) fn execution_mode_usage() -> &'static str {
    if lash::execution_mode_supported(ExecutionMode::Repl) {
        "<repl|standard>"
    } else {
        "<standard>"
    }
}

pub(crate) fn ensure_supported_execution_mode(
    mode: ExecutionMode,
) -> Result<ExecutionMode, String> {
    if lash::execution_mode_supported(mode) {
        Ok(mode)
    } else {
        Err(match mode {
            ExecutionMode::Repl => "REPL mode is not available in this build.".to_string(),
            ExecutionMode::Standard => "Execution mode is not available.".to_string(),
        })
    }
}

pub(crate) fn execution_mode_label(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Repl => "repl",
        ExecutionMode::Standard => "standard",
    }
}

pub(crate) fn parse_context_strategy(input: &str) -> Result<ContextStrategy, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Context strategy cannot be empty.".to_string()),
        "rolling_context" | "rolling-context" | "rolling" => Ok(ContextStrategy::RollingContext),
        "recall_agent" | "recall-agent" | "recall_agent_context" | "recall-agent-context" => {
            Ok(ContextStrategy::recall_agent_default())
        }
        other => Err(format!(
            "Unknown context strategy `{other}`. Expected `rolling_context` or `recall_agent`."
        )),
    }
}

pub(crate) fn resolve_context_strategy(
    configured: ContextStrategy,
    requested: Option<ContextStrategy>,
) -> Result<ContextStrategy, String> {
    match requested.unwrap_or(configured) {
        ContextStrategy::RollingContext => Ok(ContextStrategy::RollingContext),
        ContextStrategy::RecallAgent { keep_recent_pct } => {
            ContextStrategy::recall_agent(keep_recent_pct)
        }
    }
}

pub(crate) fn validate_model_selection(
    provider: &Provider,
    selection: &ModelSelection,
) -> Result<(), String> {
    provider
        .validate_model_name(&selection.model)
        .map_err(|err| {
            let normalized = provider.resolve_model(&selection.model);
            if normalized != selection.model {
                format!(
                    "{err}
Resolved provider model ID: `{normalized}`"
                )
            } else {
                err
            }
        })
}

pub(crate) fn resolve_model_selection(
    provider: &Provider,
    selection: &ModelSelection,
    catalog: &CachedModelCatalog,
) -> Result<ResolvedModelSpec, String> {
    let snapshot = catalog.snapshot();
    provider
        .resolve_model_spec(&selection.model, &snapshot)
        .map_err(|err| {
            let normalized = provider.resolve_model(&selection.model);
            if normalized != selection.model {
                format!(
                    "{err}
Resolved provider model ID: `{normalized}`"
                )
            } else {
                err
            }
        })
}

pub(crate) fn resolve_model_variant(
    provider: &Provider,
    model: &str,
    requested: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(raw) = requested else {
        return Ok(provider.default_model_variant(model).map(str::to_string));
    };
    let variant = parse_variant_input(raw)?;
    if variant == "default" {
        return Ok(provider.default_model_variant(model).map(str::to_string));
    }
    provider.validate_variant(model, &variant)?;
    Ok(Some(variant))
}

pub(crate) fn variant_lines(
    provider: &Provider,
    model: &str,
    current_variant: Option<&str>,
) -> Vec<String> {
    let supported = provider.supported_variants(model);
    let mut lines = Vec::new();
    if supported.is_empty() {
        lines.push(format!(
            "`{}` on {} does not expose configurable variants.",
            model,
            provider.label()
        ));
        return lines;
    }
    lines.push(format!(
        "Current variant: `{}`",
        current_variant.unwrap_or("(none)")
    ));
    if let Some(default_variant) = provider.default_model_variant(model) {
        lines.push(format!("Recommended default: `{}`", default_variant));
    }
    lines.push(format!("Available variants: {}", supported.join(", ")));
    lines.push("Usage: `/variant <name>` or `/variant default`".to_string());
    lines
}

pub(crate) fn hash12(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{:x}", digest)[..12].to_string()
}

pub(crate) fn latest_user_prompt_hash(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User)
        .map(|m| {
            let text = m
                .parts
                .iter()
                .map(|p| p.content.as_str())
                .collect::<Vec<_>>()
                .join(
                    "
",
                );
            hash12(text.as_bytes())
        })
}

struct ReplayManifest {
    version: u8,
    saved_at: String,
    provider: String,
    configured_model: String,
    resolved_model: String,
    context_window: u64,
    model_variant: Option<String>,
    toolset_hash: String,
    prompt_hash: Option<String>,
    snapshot_hash: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn persist_root_agent_state(
    store: &Store,
    state: &mut AgentStateEnvelope,
    ui_state: &PersistedUiState,
    dynamic_state: &DynamicStateSnapshot,
    provider: &Provider,
    configured_model: &str,
    context_window: u64,
    execution_mode: ExecutionMode,
    context_strategy: ContextStrategy,
    model_variant: Option<&str>,
    toolset_hash: &str,
    prompt_hash: Option<String>,
    snapshot_hash: Option<String>,
) {
    let manifest = ReplayManifest {
        version: 3,
        saved_at: chrono::Utc::now().to_rfc3339(),
        provider: provider.id().to_string(),
        configured_model: configured_model.to_string(),
        resolved_model: provider.resolve_model(configured_model),
        context_window,
        model_variant: model_variant.map(str::to_string),
        toolset_hash: toolset_hash.to_string(),
        prompt_hash,
        snapshot_hash,
    };
    let manifest_json = serde_json::json!({
        "version": manifest.version,
        "saved_at": manifest.saved_at,
        "provider": manifest.provider,
        "configured_model": manifest.configured_model,
        "resolved_model": manifest.resolved_model,
        "context_window": manifest.context_window,
        "execution_mode": execution_mode_label(execution_mode),
        "context_strategy": context_strategy,
        "model_variant": manifest.model_variant,
        "toolset_hash": manifest.toolset_hash,
        "prompt_hash": manifest.prompt_hash,
        "snapshot_hash": manifest.snapshot_hash,
    });
    state.replay_manifest = Some(manifest_json.clone());
    let config_json = serde_json::json!({
        "manifest": manifest_json,
        "context_strategy": context_strategy,
        "last_prompt_usage": state.last_prompt_usage.clone(),
        "plugin_snapshot": state.plugin_snapshot,
        "task_state": state.task_state,
        "dynamic_state": dynamic_state,
    })
    .to_string();
    let messages_json = serde_json::to_string(&state.messages).unwrap_or_else(|_| "[]".to_string());
    let tool_calls_json =
        serde_json::to_string(&state.tool_calls).unwrap_or_else(|_| "[]".to_string());
    let ui_json = serde_json::to_string(ui_state).unwrap_or_else(|_| "{}".to_string());
    store.save_agent_state(lash::store::AgentStateSave {
        agent_id: ROOT_SESSION_ID,
        messages_json: &messages_json,
        tool_calls_json: &tool_calls_json,
        ui_json: &ui_json,
        iteration: state.iteration as i64,
        config_json: &config_json,
        repl_snapshot: state.repl_snapshot.as_deref(),
        input_tokens: state.token_usage.input_tokens,
        output_tokens: state.token_usage.output_tokens,
        cached_input_tokens: state.token_usage.cached_input_tokens,
        reasoning_tokens: state.token_usage.reasoning_tokens,
    });
}

fn persisted_plugin_snapshot(state: &lash::AgentState) -> Option<PluginSessionSnapshot> {
    serde_json::from_str::<serde_json::Value>(&state.config_json)
        .ok()
        .and_then(|config| config.get("plugin_snapshot").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn persist_live_runtime_snapshot(
    store: &Store,
    snapshot: DurableTurnSnapshot,
    ui_state: &PersistedUiState,
    dynamic_state: &DynamicStateSnapshot,
    provider: &Provider,
    configured_model: &str,
    context_window: u64,
    execution_mode: ExecutionMode,
    context_strategy: ContextStrategy,
    model_variant: Option<&str>,
    toolset_hash: &str,
    token_usage: TokenUsage,
    last_prompt_usage: Option<PromptUsage>,
) {
    let existing = store.load_agent_state(ROOT_SESSION_ID);
    let mut state = AgentStateEnvelope {
        agent_id: ROOT_SESSION_ID.to_string(),
        messages: snapshot.messages,
        tool_calls: snapshot.tool_calls,
        iteration: snapshot.iteration,
        token_usage,
        last_prompt_usage,
        task_state: Some(serde_json::json!({
            "kind": "live_resume",
            "status": "running",
            "saved_at": chrono::Utc::now().to_rfc3339(),
        })),
        plugin_snapshot: existing.as_ref().and_then(persisted_plugin_snapshot),
        repl_snapshot: existing
            .as_ref()
            .and_then(|state| state.repl_snapshot.clone()),
        ..Default::default()
    };
    let prompt_hash = latest_user_prompt_hash(&state.messages);
    persist_root_agent_state(
        store,
        &mut state,
        ui_state,
        dynamic_state,
        provider,
        configured_model,
        context_window,
        execution_mode,
        context_strategy,
        model_variant,
        toolset_hash,
        prompt_hash,
        None,
    );
}

pub(crate) fn push_system_message(app: &mut App, msg: impl Into<String>) {
    let msg = msg.into();
    let duplicate = matches!(
        app.blocks.last(),
        Some(DisplayBlock::SystemMessage(existing)) if existing == &msg
    );
    if duplicate {
        return;
    }
    app.blocks.push(DisplayBlock::SystemMessage(msg));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}

pub(crate) fn version_text() -> String {
    format!(
        "lash {}
lash-sansio {}",
        crate::APP_VERSION,
        lash::SANSIO_VERSION
    )
}

pub(crate) fn info_text_unconfigured(execution_mode: ExecutionMode, cwd: &str) -> String {
    [
        format!("lash: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash::SANSIO_VERSION),
        "provider: (not configured)".to_string(),
        "configured model: (not configured)".to_string(),
        "resolved model: (not configured)".to_string(),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        "context window: unknown".to_string(),
        format!("cwd: {}", cwd),
        "session: (not started)".to_string(),
    ]
    .join(
        "
",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn info_text(
    provider: &Provider,
    configured_model: &str,
    model_variant: Option<&str>,
    execution_mode: ExecutionMode,
    context_window: Option<u64>,
    tool_count: usize,
    toolset_hash: &str,
    context_strategy: ContextStrategy,
    cwd: &str,
    session_name: Option<&str>,
) -> String {
    let resolved_model = provider.resolve_model(configured_model);
    let mut lines = vec![
        format!("lash: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash::SANSIO_VERSION),
        format!("provider: {} ({})", provider.label(), provider.id()),
        format!("configured model: {}", configured_model),
        format!("resolved model: {}", resolved_model),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
    ];

    if let Some(variant) = model_variant {
        lines.push(format!("variant: {}", variant));
    }
    if let Some(window) = context_window {
        lines.push(format!("context window: {}", window));
    } else {
        lines.push("context window: unknown".to_string());
    }

    let context_line = match context_strategy {
        ContextStrategy::RollingContext => "context strategy: rolling_context".to_string(),
        ContextStrategy::RecallAgent { keep_recent_pct } => {
            format!(
                "context strategy: recall_agent (keep_recent={}%)",
                keep_recent_pct
            )
        }
    };
    lines.extend([
        context_line,
        format!("tools: {} (hash {})", tool_count, toolset_hash),
        format!("cwd: {}", cwd),
        format!("session: {}", session_name.unwrap_or("(not started)")),
    ]);

    lines.join(
        "
",
    )
}

pub(crate) fn help_text(skills: &SkillCatalog) -> String {
    let mut lines = vec![
        "Commands:".to_string(),
        "  /clear, /new       Reset conversation".to_string(),
        "  /fork [prompt]     Open a forked session in a new terminal".to_string(),
        "  /version           Show Lash and lash-sansio versions".to_string(),
        "  /info              Show current runtime/session info".to_string(),
        "  /model [name]      Show or switch LLM model".to_string(),
        "  /variant [name]    Show or switch provider-native model variant".to_string(),
        format!(
            "  /mode [name]       Show current execution mode; new session required to change {}",
            execution_mode_usage()
        ),
        "  /provider          Switch, add, or re-authenticate providers".to_string(),
        "  /login             Sign in or reconfigure provider".to_string(),
        "  /logout            Remove stored credentials for active provider".to_string(),
        "  /retry             Replay the previous turn payload".to_string(),
        "  /resume [name]     Browse or load a previous session".to_string(),
        "  /skills            Browse loaded skills".to_string(),
        "  /<skill> [text]    Invoke a loaded skill directly".to_string(),
        "  /tools ...         Inspect/edit dynamic tools".to_string(),
        "  /reconfigure ...   Apply/status/clear pending changes".to_string(),
        "  /help, /?          Show this help".to_string(),
        "  /exit, /quit       Quit".to_string(),
    ];

    if !skills.is_empty() {
        lines.push(String::new());
        lines.push("Installed skills:".to_string());
        for skill in skills.iter() {
            let desc = if skill.description.is_empty() {
                String::new()
            } else {
                format!("  {}", skill.description)
            };
            lines.push(format!("  ${}{}", skill.name, desc));
        }
        lines
            .push("  Use /skills to browse or `/<skill-name>` to invoke one directly.".to_string());
    }

    lines.extend([
        String::new(),
        "Dynamic Runtime:".to_string(),
        "  /tools".to_string(),
        "  /tools add <name> <handler> [description]".to_string(),
        "  /tools rm <name>".to_string(),
        "  /tools update <name> key=value ...".to_string(),
        "  /tools enable <name> | /tools disable <name>".to_string(),
        "  /reconfigure status|apply|clear".to_string(),
    ]);

    lines.extend([
        String::new(),
        "Shortcuts:".to_string(),
        "  Esc                Cancel agent (while running)".to_string(),
        "  Enter              Submit; inject at next checkpoint while running".to_string(),
        "  Tab                Queue next turn; submit plain draft when idle".to_string(),
        "  Up (empty draft)   Edit last queued turn".to_string(),
        "  Shift+Tab          Toggle persistent plan mode".to_string(),
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down".to_string(),
        "  PgUp / PgDn        Scroll page up / down".to_string(),
        "  Shift+Enter        Insert newline".to_string(),
        "  Ctrl+V             Paste image as inline [Image #n]".to_string(),
        "  Ctrl+Shift+V       Paste text only".to_string(),
        "  Ctrl+Y             Copy last response to clipboard".to_string(),
        "  Ctrl+O             Cycle tool expansion (ghost ↔ compact)".to_string(),
        "  Alt+O              Full expansion (code + stdout)".to_string(),
        "  Shift+Drag         Select text (terminal native)".to_string(),
        "  Up/Down            Input history".to_string(),
        "  Ctrl+C             Quit".to_string(),
    ]);

    lines.join(
        "
",
    )
}

fn plan_mode_enabled_from_result(result: ToolResult) -> Result<bool, String> {
    if !result.success {
        return Err(result.result.to_string());
    }
    result
        .result
        .get("enabled")
        .and_then(|value| value.as_bool())
        .ok_or_else(|| "plan mode API response missing `enabled`".to_string())
}

async fn plan_mode_status(
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) -> Result<bool, String> {
    let result = plugin_host
        .invoke_external_for_session(
            ROOT_SESSION_ID,
            "plan_mode.status",
            serde_json::json!({}),
            session_manager,
        )
        .await
        .map_err(|err| err.to_string())?;
    plan_mode_enabled_from_result(result)
}

pub(crate) async fn plan_mode_toggle(
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) -> Result<bool, String> {
    let result = plugin_host
        .invoke_external_for_session(
            ROOT_SESSION_ID,
            "plan_mode.toggle",
            serde_json::json!({}),
            session_manager,
        )
        .await
        .map_err(|err| err.to_string())?;
    plan_mode_enabled_from_result(result)
}

pub(crate) async fn sync_plan_mode(
    app: &mut App,
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) {
    match plan_mode_status(plugin_host, session_manager).await {
        Ok(enabled) => app.set_plan_mode_enabled(enabled),
        Err(err) => push_system_message(app, format!("Failed to read plan mode: {}", err)),
    }
}

pub(crate) fn normalize_prepared_turn_for_dispatch(
    turn: PreparedTurn,
    _skills: &SkillCatalog,
) -> PreparedTurn {
    turn
}

pub(crate) fn shell_escape_command(input: &str) -> Option<&str> {
    input
        .strip_prefix('!')
        .map(str::trim)
        .filter(|cmd| !cmd.is_empty())
}
