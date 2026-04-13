use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash::provider::Provider;
use lash::*;
use lash_ui::{UiContext, UiExtensions, UiHostEffect};
use sha2::{Digest, Sha256};

use crate::ROOT_SESSION_ID;
use crate::app::{App, DisplayBlock, PreparedTurn, UiResumeState};
use crate::command;
use crate::ui_resume;

#[derive(Debug, Clone)]
pub(crate) struct ModelSelection {
    pub(crate) model: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QueuedTurnEditBinding {
    AltUp,
    ShiftLeft,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CopyBinding {
    CtrlC,
    CtrlShiftC,
    CtrlY,
}

impl CopyBinding {
    pub(crate) fn display(self) -> &'static str {
        match self {
            Self::CtrlC => "Ctrl+C",
            Self::CtrlShiftC => "Ctrl+Shift+C",
            Self::CtrlY => "Ctrl+Y",
        }
    }

    pub(crate) fn matches(self, key: KeyEvent) -> bool {
        match self {
            Self::CtrlC => {
                key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
            }
            Self::CtrlShiftC => {
                key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.modifiers.contains(KeyModifiers::SHIFT)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
            }
            Self::CtrlY => {
                key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'y'))
            }
        }
    }
}

impl QueuedTurnEditBinding {
    pub(crate) fn display(self) -> &'static str {
        match self {
            Self::AltUp => "Alt+Up",
            Self::ShiftLeft => "Shift+Left",
        }
    }

    pub(crate) fn matches(self, key: KeyEvent) -> bool {
        match self {
            Self::AltUp => key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Up,
            Self::ShiftLeft => {
                key.modifiers.contains(KeyModifiers::SHIFT) && key.code == KeyCode::Left
            }
        }
    }
}

fn queued_turn_edit_binding_from_hints(
    term_program: Option<&str>,
    inside_tmux: bool,
    inside_vscode: bool,
) -> QueuedTurnEditBinding {
    if inside_tmux {
        return QueuedTurnEditBinding::ShiftLeft;
    }

    match term_program.map(|value| value.to_ascii_lowercase()) {
        Some(value)
            if matches!(
                value.as_str(),
                "apple_terminal" | "warpterminal" | "warp" | "vscode"
            ) =>
        {
            QueuedTurnEditBinding::ShiftLeft
        }
        _ if inside_vscode => QueuedTurnEditBinding::ShiftLeft,
        _ => QueuedTurnEditBinding::AltUp,
    }
}

pub(crate) fn queued_turn_edit_binding() -> QueuedTurnEditBinding {
    queued_turn_edit_binding_from_hints(
        std::env::var("TERM_PROGRAM").ok().as_deref(),
        std::env::var_os("TMUX").is_some(),
        std::env::var_os("VSCODE_INJECTION").is_some(),
    )
}

pub(crate) fn copy_binding() -> CopyBinding {
    copy_binding_from_env(std::env::var("LASH_COPY_BINDING").ok().as_deref())
}

pub(crate) fn copy_binding_from_env(value: Option<&str>) -> CopyBinding {
    match value
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("ctrl-shift-c") | Some("ctrl_shift_c") => CopyBinding::CtrlShiftC,
        Some("ctrl-y") | Some("ctrl_y") => CopyBinding::CtrlY,
        Some("ctrl-c") | Some("ctrl_c") | None | Some("") => CopyBinding::CtrlC,
        Some(_) => CopyBinding::CtrlC,
    }
}

pub(crate) fn controls_text(ui_extensions: &UiExtensions) -> String {
    let mut lines = vec!["Controls:".to_string()];
    lines.extend(render_shortcut_lines(ui_extensions, true));
    lines.join("\n")
}

fn render_shortcut_lines(ui_extensions: &UiExtensions, spaced_history_arrows: bool) -> Vec<String> {
    let history_arrows = if spaced_history_arrows {
        "  Up / Down          Input history"
    } else {
        "  Up/Down            Input history"
    };

    let mut lines = vec![
        "  Esc                Cancel session (while running)".to_string(),
        "  Enter              Submit; inject at next checkpoint while running".to_string(),
        "  Tab                Queue next turn; submit plain draft when idle".to_string(),
        "  Up (empty draft)   Edit last queued turn".to_string(),
        format!(
            "  {:<18} Edit last queued turn",
            queued_turn_edit_binding().display()
        ),
    ];

    for shortcut in ui_extensions.shortcut_specs() {
        lines.push(format!(
            "  {:<18} {}",
            shortcut.chord.display(),
            shortcut.description
        ));
    }

    lines.extend([
        "  Ctrl+U / Ctrl+D    Scroll half-page up / down".to_string(),
        "  PgUp / PgDn        Scroll page up / down".to_string(),
        "  Shift+Enter        Insert newline".to_string(),
        "  Ctrl+V             Paste image as inline [Image #n]".to_string(),
        "  Ctrl+Shift+V       Paste text only".to_string(),
        format!(
            "  {:<18} Copy selection or last response",
            copy_binding().display()
        ),
        "  Ctrl+O             Cycle tool expansion (ghost ↔ compact)".to_string(),
        "  Alt+O              Full expansion (code + stdout)".to_string(),
        history_arrows.to_string(),
        "  Ctrl+C             Quit".to_string(),
    ]);

    lines
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

fn log_filter_enables_debug(filter: &str) -> bool {
    let lowered = filter.to_ascii_lowercase();
    lowered.contains("debug") || lowered.contains("trace")
}

pub(crate) fn effective_lash_log_filter(debug_flag: bool) -> String {
    let env_filter = std::env::var("LASH_LOG")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    match (debug_flag, env_filter) {
        (true, Some(filter)) if log_filter_enables_debug(&filter) => filter,
        (true, _) => "debug".to_string(),
        (false, Some(filter)) => filter,
        (false, None) => "warn".to_string(),
    }
}

pub(crate) fn detailed_debug_logging_enabled(debug_flag: bool) -> bool {
    log_filter_enables_debug(&effective_lash_log_filter(debug_flag))
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
        "rlm" => Ok(ExecutionMode::Rlm),
        "standard" | "tools" => Ok(ExecutionMode::Standard),
        other => Err(format!(
            "Unknown execution mode `{other}`. Expected `rlm` or `standard`."
        )),
    }
}

pub(crate) fn parse_context_approach(input: &str) -> Result<lash::ContextApproach, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Context approach cannot be empty.".to_string()),
        "rolling" | "rolling-history" | "rolling_history" => {
            Ok(lash::ContextApproach::RollingHistory(Default::default()))
        }
        "om" | "observational" | "observational-memory" | "observational_memory" => Ok(
            lash::ContextApproach::ObservationalMemory(Default::default()),
        ),
        other => Err(format!(
            "Unknown context approach `{other}`. Expected `rolling_history` or `observational_memory`."
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_context_approach_overrides(
    mut approach: lash::ContextApproach,
    om_observation_message_tokens: Option<usize>,
    om_observation_buffer_tokens: Option<usize>,
    om_observation_block_after_tokens: Option<usize>,
    om_observation_max_tokens_per_batch: Option<usize>,
    om_previous_observer_tokens: Option<usize>,
    om_reflection_observation_tokens: Option<usize>,
    om_reflection_buffer_activation_percent: Option<u16>,
    om_reflection_block_after_tokens: Option<usize>,
) -> Result<lash::ContextApproach, String> {
    let has_om_overrides = om_observation_message_tokens.is_some()
        || om_observation_buffer_tokens.is_some()
        || om_observation_block_after_tokens.is_some()
        || om_observation_max_tokens_per_batch.is_some()
        || om_previous_observer_tokens.is_some()
        || om_reflection_observation_tokens.is_some()
        || om_reflection_buffer_activation_percent.is_some()
        || om_reflection_block_after_tokens.is_some();
    if !has_om_overrides {
        return Ok(approach);
    }

    let lash::ContextApproach::ObservationalMemory(config) = &mut approach else {
        return Err(
            "OM tuning flags require `--context-approach observational_memory`.".to_string(),
        );
    };

    if let Some(value) = om_observation_message_tokens {
        if value == 0 {
            return Err("`--om-observation-message-tokens` must be greater than 0.".to_string());
        }
        config.observation_message_tokens = value;
    }
    if let Some(value) = om_observation_buffer_tokens {
        config.observation_buffer_tokens = value;
    }
    if let Some(value) = om_observation_block_after_tokens {
        if value == 0 {
            return Err(
                "`--om-observation-block-after-tokens` must be greater than 0.".to_string(),
            );
        }
        config.observation_block_after_tokens = value;
    }
    if let Some(value) = om_observation_max_tokens_per_batch {
        if value == 0 {
            return Err(
                "`--om-observation-max-tokens-per-batch` must be greater than 0.".to_string(),
            );
        }
        config.observation_max_tokens_per_batch = value;
    }
    if let Some(value) = om_previous_observer_tokens {
        config.previous_observer_tokens = value;
    }
    if let Some(value) = om_reflection_observation_tokens {
        if value == 0 {
            return Err("`--om-reflection-observation-tokens` must be greater than 0.".to_string());
        }
        config.reflection_observation_tokens = value;
    }
    if let Some(value) = om_reflection_buffer_activation_percent {
        if value > 100 {
            return Err(
                "`--om-reflection-buffer-activation-percent` must be between 0 and 100."
                    .to_string(),
            );
        }
        config.reflection_buffer_activation_bps = value.saturating_mul(100);
    }
    if let Some(value) = om_reflection_block_after_tokens {
        if value == 0 {
            return Err("`--om-reflection-block-after-tokens` must be greater than 0.".to_string());
        }
        config.reflection_block_after_tokens = value;
    }

    if config.observation_buffer_tokens >= config.observation_block_after_tokens {
        return Err(
            "`--om-observation-buffer-tokens` must be smaller than `--om-observation-block-after-tokens`."
                .to_string(),
        );
    }
    if config.observation_activation_tokens() == 0 {
        return Err(
            "`--om-observation-buffer-tokens` must be smaller than `--om-observation-message-tokens`."
                .to_string(),
        );
    }
    if config.reflection_buffer_activation_bps > 10_000 {
        return Err(
            "`--om-reflection-buffer-activation-percent` must be between 0 and 100.".to_string(),
        );
    }

    Ok(approach)
}

pub(crate) fn execution_mode_usage() -> &'static str {
    if lash::execution_mode_supported(ExecutionMode::Rlm) {
        "<rlm|standard>"
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
            ExecutionMode::Rlm => "RLM mode is not available in this build.".to_string(),
            ExecutionMode::Standard => "Execution mode is not available.".to_string(),
        })
    }
}

pub(crate) fn execution_mode_label(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Rlm => "rlm",
        ExecutionMode::Standard => "standard",
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

pub(crate) fn persist_root_session_state(
    store: &Store,
    state: &mut SessionStateEnvelope,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
) {
    state.stamp_runtime_state(Some(dynamic_state), None);
    let message_count = state.projected_messages().len();
    let tool_call_count = state.projected_tool_calls().len();
    tracing::debug!(
        iteration = state.iteration,
        messages = message_count,
        tool_calls = tool_call_count,
        plugin_panels = ui_state.plugin_panels.len(),
        input_tokens = state.token_usage.input_tokens,
        output_tokens = state.token_usage.output_tokens,
        cached_input_tokens = state.token_usage.cached_input_tokens,
        reasoning_tokens = state.token_usage.reasoning_tokens,
        "persisting root session state"
    );
    ui_resume::save_ui_resume_state(store, ui_state);
    let checkpoint = lash::HydratedSessionCheckpoint {
        turn_state: lash::PersistedTurnState {
            iteration: state.iteration,
            token_usage: state.token_usage.clone(),
            last_prompt_usage: state.last_prompt_usage.clone(),
        },
        dynamic_state_ref: state.dynamic_state_ref.clone(),
        dynamic_state: state.dynamic_state_snapshot.clone(),
        plugin_snapshot_ref: state.plugin_snapshot_ref.clone(),
        plugin_snapshot_revision: state.plugin_snapshot_revision,
        plugin_snapshot: state.plugin_snapshot.clone(),
    };
    let stored_checkpoint = store.put_checkpoint(&checkpoint);
    state.checkpoint_ref = Some(stored_checkpoint.checkpoint_ref.clone());
    state.dynamic_state_ref = stored_checkpoint.manifest.dynamic_state_ref;
    state.dynamic_state_generation = state
        .dynamic_state_snapshot
        .as_ref()
        .map(|snapshot| snapshot.base_generation);
    state.plugin_snapshot_ref = stored_checkpoint.manifest.plugin_snapshot_ref;
    state.execution_state_snapshot = None;
    state.dynamic_state_snapshot = None;
    state.plugin_snapshot = None;
    let node_count = state.session_graph.nodes.len();
    if state.graph_replace_required || state.persisted_graph_node_count > node_count {
        store.replace_session_graph(&state.session_graph);
    } else if state.persisted_graph_node_count < node_count {
        store.append_session_graph_nodes(
            &state.session_graph.nodes[state.persisted_graph_node_count..],
        );
    }
    store.save_session_head_meta(lash::SessionHeadMeta {
        config: lash::PersistedSessionConfig {
            provider_id: state.policy.provider.id().to_string(),
            configured_model: state.policy.model.clone(),
            context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: state.policy.execution_mode,
            context_approach: state.policy.context_approach.clone(),
            model_variant: state.policy.model_variant.clone(),
        },
        checkpoint_ref: Some(stored_checkpoint.checkpoint_ref),
        leaf_node_id: state.session_graph.leaf_node_id.clone(),
        graph_node_count: state.session_graph.nodes.len(),
        token_ledger: state.token_ledger.clone(),
    });
    state.persisted_graph_node_count = node_count;
    state.graph_replace_required = false;
    store.clear_live_resume();
}

pub(crate) fn persist_live_runtime_snapshot(
    store: &Store,
    seed_graph: Option<lash::SessionGraph>,
    live_graph: lash::SessionGraph,
    ui_state: &UiResumeState,
    dynamic_state: &DynamicStateSnapshot,
    policy: &lash::SessionPolicy,
    turn_state: lash::PersistedTurnState,
    token_ledger: &[lash::TokenLedgerEntry],
    plugin_snapshot: Option<&lash::PluginSessionSnapshot>,
    execution_state_snapshot: Option<&[u8]>,
) {
    let base = store
        .load_live_resume()
        .map(|snapshot| (snapshot.graph, snapshot.checkpoint_ref))
        .or_else(|| {
            store
                .load_session_head()
                .map(|head| (head.graph, head.checkpoint_ref))
        });
    let (graph, checkpoint_ref) = base.unwrap_or_else(|| (seed_graph.unwrap_or_default(), None));
    if !lash::messages_are_live_resume_safe(&live_graph.project_messages()) {
        tracing::debug!(
            iteration = turn_state.iteration,
            graph_nodes = live_graph.nodes.len(),
            "skipping unsafe live runtime snapshot"
        );
        return;
    }
    tracing::debug!(
        iteration = turn_state.iteration,
        graph_nodes = live_graph.nodes.len(),
        plugin_mode_indicators = ui_state.plugin_mode_indicators.len(),
        plugin_panels = ui_state.plugin_panels.len(),
        "persisting live runtime snapshot"
    );
    ui_resume::save_ui_resume_state(store, ui_state);
    let delta_ref = store.put_typed_blob(&lash::LiveResumeDelta {
        appended_graph_nodes: live_graph.nodes[graph.nodes.len()..].to_vec(),
        leaf_node_id: live_graph.leaf_node_id.clone(),
        turn_state,
        dynamic_state: Some(dynamic_state.clone()),
        plugin_snapshot: plugin_snapshot.cloned(),
        execution_state_snapshot: execution_state_snapshot.map(ToOwned::to_owned),
        token_ledger: token_ledger.to_vec(),
    });
    store.save_live_resume(lash::LiveResumeSnapshot {
        graph,
        config: lash::PersistedSessionConfig {
            provider_id: policy.provider.id().to_string(),
            configured_model: policy.model.clone(),
            context_window: policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: policy.execution_mode,
            context_approach: policy.context_approach.clone(),
            model_variant: policy.model_variant.clone(),
        },
        checkpoint_ref,
        delta_ref: Some(delta_ref),
    });
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
    crate::ui_trace::record_system_message_aux(&msg);
    app.blocks.push(DisplayBlock::SystemMessage(msg));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}

pub(crate) fn version_text() -> String {
    format!(
        "lash-cli {}
lash-sansio {}",
        crate::APP_VERSION,
        lash::SANSIO_VERSION
    )
}

pub(crate) fn info_text_unconfigured(execution_mode: ExecutionMode, cwd: &str) -> String {
    [
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash::SANSIO_VERSION),
        "provider: (not configured)".to_string(),
        "configured model: (not configured)".to_string(),
        "resolved model: (not configured)".to_string(),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        format!(
            "context approach: {}",
            match lash::ContextApproach::default() {
                lash::ContextApproach::RollingHistory(_) => "rolling_history",
                lash::ContextApproach::ObservationalMemory(_) => "observational_memory",
            }
        ),
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
    context_approach: &lash::ContextApproach,
    context_window: Option<u64>,
    tool_count: usize,
    toolset_hash: &str,
    cwd: &str,
    session_name: Option<&str>,
) -> String {
    let resolved_model = provider.resolve_model(configured_model);
    let mut lines = vec![
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash::SANSIO_VERSION),
        format!("provider: {} ({})", provider.label(), provider.id()),
        format!("configured model: {}", configured_model),
        format!("resolved model: {}", resolved_model),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        format!(
            "context approach: {}",
            match context_approach {
                lash::ContextApproach::RollingHistory(_) => "rolling_history",
                lash::ContextApproach::ObservationalMemory(_) => "observational_memory",
            }
        ),
    ];

    if let Some(variant) = model_variant {
        lines.push(format!("variant: {}", variant));
    }
    if let Some(window) = context_window {
        lines.push(format!("context window: {}", window));
    } else {
        lines.push("context window: unknown".to_string());
    }

    lines.extend([
        format!("tools: {} (hash {})", tool_count, toolset_hash),
        format!("cwd: {}", cwd),
        format!("session: {}", session_name.unwrap_or("(not started)")),
    ]);

    lines.join(
        "
",
    )
}

pub(crate) fn help_text(skills: &SkillCatalog, ui_extensions: &UiExtensions) -> String {
    let mut lines = vec!["Commands:".to_string()];
    for spec in command::catalog() {
        let aliases = if spec.aliases.is_empty() {
            String::new()
        } else {
            format!(", {}", spec.aliases.join(", "))
        };
        let description = if spec.name == "/mode" {
            format!(
                "{}; new session required to change {}",
                spec.description,
                execution_mode_usage()
            )
        } else {
            spec.description.to_string()
        };
        lines.push(format!(
            "  {:<18} {}",
            format!("{}{}", spec.usage, aliases),
            description
        ));
    }
    for spec in ui_extensions.command_specs() {
        let aliases = if spec.aliases.is_empty() {
            String::new()
        } else {
            format!(", {}", spec.aliases.join(", "))
        };
        lines.push(format!(
            "  {:<18} {}",
            format!("{}{}", spec.usage, aliases),
            spec.description
        ));
    }
    lines.push("  /<skill> [text]    Invoke a loaded skill directly".to_string());

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

    lines.push(String::new());
    lines.push("Shortcuts:".to_string());
    lines.extend(render_shortcut_lines(ui_extensions, false));

    lines.join(
        "
",
    )
}

pub(crate) fn apply_ui_host_effects(app: &mut App, effects: Vec<UiHostEffect>) {
    for effect in effects {
        match effect {
            UiHostEffect::PushSystemMessage(message) => push_system_message(app, message),
            UiHostEffect::DesktopNotification {
                title,
                body,
                only_when_unfocused,
            } => {
                if !only_when_unfocused || !app.focused {
                    crate::interactive::notify_desktop(&title, &body);
                }
            }
            UiHostEffect::UpsertModeIndicator { key, label } => {
                app.upsert_mode_indicator(key, label);
            }
            UiHostEffect::ClearModeIndicator { key } => {
                app.clear_mode_indicator(&key);
            }
            UiHostEffect::UpsertPanel {
                plugin_id,
                key,
                title,
                content,
            } => {
                app.handle_session_event(SessionEvent::PluginEvent {
                    plugin_id,
                    event: PluginSurfaceEvent::PanelUpsert {
                        key,
                        title,
                        content,
                    },
                });
            }
            UiHostEffect::ClearPanel { plugin_id, key } => {
                app.handle_session_event(SessionEvent::PluginEvent {
                    plugin_id,
                    event: PluginSurfaceEvent::PanelClear { key },
                });
            }
            UiHostEffect::QueueTurn { input } => {
                app.queue_turn(PreparedTurn::prepare(input, Vec::new(), &app.skills));
                app.dirty = true;
            }
            UiHostEffect::SwitchToNewSession { .. } => {
                app.dirty = true;
            }
        }
    }
}

pub(crate) async fn sync_ui_extensions(
    app: &mut App,
    ui_extensions: &UiExtensions,
    plugin_host: &PluginHost,
    session_manager: Arc<dyn SessionManager>,
) {
    match ui_extensions
        .sync_all(UiContext {
            plugin_host,
            session_id: ROOT_SESSION_ID,
            session_manager,
        })
        .await
    {
        Ok(effects) => apply_ui_host_effects(app, effects),
        Err(err) => push_system_message(app, format!("Failed to sync UI extensions: {err}")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};
    use lash::SessionMeta;

    #[test]
    fn desktop_notification_effect_respects_focus() {
        let mut app = App::new("test-model".into(), "test".into());
        app.focused = true;

        apply_ui_host_effects(
            &mut app,
            vec![UiHostEffect::DesktopNotification {
                title: "lash".into(),
                body: "Response complete".into(),
                only_when_unfocused: true,
            }],
        );
    }

    #[test]
    fn file_backed_store_creates_wal_file() {
        let _env_guard = env_lock().blocking_lock();
        let temp = TempDirGuard::new("lash-cli-store-wal");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let db_path = temp.path().join("session.db");
        let store = Store::open(&db_path).expect("store");

        store.save_session_meta(SessionMeta {
            session_id: "s1".to_string(),
            session_name: "demo".to_string(),
            created_at: "2026-03-26T10:00:00Z".to_string(),
            model: "gpt-5".to_string(),
            cwd: Some("/tmp/demo".to_string()),
            parent_session_id: None,
        });

        assert!(db_path.with_extension("db-wal").exists());
    }

    #[test]
    fn effective_lash_log_filter_defaults_and_promotes_debug_flag() {
        let _env_guard = env_lock().blocking_lock();
        unsafe { std::env::remove_var("LASH_LOG") };
        assert_eq!(effective_lash_log_filter(false), "warn");
        assert_eq!(effective_lash_log_filter(true), "debug");

        unsafe { std::env::set_var("LASH_LOG", "trace") };
        assert_eq!(effective_lash_log_filter(false), "trace");
        assert_eq!(effective_lash_log_filter(true), "trace");

        unsafe { std::env::set_var("LASH_LOG", "warn") };
        assert_eq!(effective_lash_log_filter(true), "debug");
    }

    #[test]
    fn queued_turn_edit_binding_defaults_to_alt_up() {
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, false, false),
            QueuedTurnEditBinding::AltUp
        );
        assert_eq!(
            queued_turn_edit_binding_from_hints(Some("ghostty"), false, false),
            QueuedTurnEditBinding::AltUp
        );
    }

    #[test]
    fn queued_turn_edit_binding_falls_back_when_alt_up_is_unreliable() {
        for term_program in ["Apple_Terminal", "WarpTerminal", "warp", "vscode"] {
            assert_eq!(
                queued_turn_edit_binding_from_hints(Some(term_program), false, false),
                QueuedTurnEditBinding::ShiftLeft
            );
        }
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, true, false),
            QueuedTurnEditBinding::ShiftLeft
        );
        assert_eq!(
            queued_turn_edit_binding_from_hints(None, false, true),
            QueuedTurnEditBinding::ShiftLeft
        );
    }

    #[test]
    fn observational_memory_overrides_apply_to_context_approach() {
        let approach = apply_context_approach_overrides(
            lash::ContextApproach::ObservationalMemory(Default::default()),
            Some(45_000),
            Some(8_000),
            None,
            Some(12_000),
            Some(3_000),
            None,
            Some(60),
            None,
        )
        .expect("overrides");
        let lash::ContextApproach::ObservationalMemory(config) = approach else {
            panic!("expected observational_memory");
        };
        assert_eq!(config.observation_message_tokens, 45_000);
        assert_eq!(config.observation_buffer_tokens, 8_000);
        assert_eq!(config.observation_max_tokens_per_batch, 12_000);
        assert_eq!(config.previous_observer_tokens, 3_000);
        assert_eq!(config.reflection_buffer_activation_bps, 6_000);
    }

    #[test]
    fn observational_memory_overrides_require_om_context_approach() {
        let err = apply_context_approach_overrides(
            lash::ContextApproach::RollingHistory(Default::default()),
            Some(45_000),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect_err("expected validation error");
        assert!(err.contains("observational_memory"));
    }
}
