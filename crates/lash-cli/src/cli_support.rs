use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lash::advanced::ExecutionMode;
use lash::provider::ProviderHandle;
#[cfg(test)]
use lash_sqlite_store::Store;
use lash_standard_plugins::{StandardContextApproach, StandardContextApproachKind};
use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiHostEffect};
use sha2::{Digest, Sha256};

use crate::SkillCatalog;
use crate::app::{App, PluginPanelBlock, PreparedTurn, UiTimelineItem};
use crate::command;
use crate::model_catalog::{
    CachedModelCatalog, FileModelCatalogStore, ModelsDevHttpSource, ResolvedModelSpec,
};

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

pub(crate) fn controls_text(ui_extensions: &TuiExtensions) -> String {
    let mut lines = vec!["Controls:".to_string()];
    lines.extend(render_shortcut_lines(ui_extensions, true));
    lines.join("\n")
}

fn render_shortcut_lines(
    ui_extensions: &TuiExtensions,
    spaced_history_arrows: bool,
) -> Vec<String> {
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
        Arc::new(FileModelCatalogStore::new(
            crate::paths::model_catalog_cache_file(),
        )),
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
        "rlm" => Ok(ExecutionMode::rlm()),
        "standard" | "tools" => Ok(ExecutionMode::standard()),
        other => Err(format!(
            "Unknown execution mode `{other}`. Expected `rlm` or `standard`."
        )),
    }
}

pub(crate) fn parse_standard_context_approach(
    input: &str,
) -> Result<StandardContextApproach, String> {
    match input.trim().to_ascii_lowercase().as_str() {
        "" => Err("Context approach cannot be empty.".to_string()),
        "rolling" | "rolling-history" | "rolling_history" => {
            Ok(StandardContextApproach::RollingHistory(Default::default()))
        }
        "om" | "observational" | "observational-memory" | "observational_memory" => Ok(
            StandardContextApproach::ObservationalMemory(Default::default()),
        ),
        other => Err(format!(
            "Unknown context approach `{other}`. Expected `rolling_history` or `observational_memory`."
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_standard_context_approach_overrides(
    mut approach: StandardContextApproach,
    om_observation_message_tokens: Option<usize>,
    om_observation_buffer_tokens: Option<usize>,
    om_observation_block_after_tokens: Option<usize>,
    om_observation_max_tokens_per_batch: Option<usize>,
    om_previous_observer_tokens: Option<usize>,
    om_reflection_observation_tokens: Option<usize>,
    om_reflection_buffer_activation_percent: Option<u16>,
    om_reflection_block_after_tokens: Option<usize>,
) -> Result<StandardContextApproach, String> {
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

    let StandardContextApproach::ObservationalMemory(config) = &mut approach else {
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
    if config.observation_buffer_interval_tokens() >= config.observation_message_tokens {
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
    "<rlm|standard>"
}

pub(crate) fn ensure_supported_execution_mode(
    mode: ExecutionMode,
) -> Result<ExecutionMode, String> {
    Ok(mode)
}

pub(crate) fn execution_mode_label(mode: &ExecutionMode) -> &str {
    mode.as_str()
}

pub(crate) fn standard_context_approach_label(approach: &StandardContextApproach) -> &'static str {
    match approach.kind() {
        StandardContextApproachKind::RollingHistory => "rolling_history",
        StandardContextApproachKind::ObservationalMemory => "observational_memory",
    }
}

pub(crate) fn validate_model_selection(
    provider: &ProviderHandle,
    selection: &ModelSelection,
) -> Result<(), String> {
    provider.validate_model_name(&selection.model)
}

pub(crate) fn resolve_model_selection(
    provider: &ProviderHandle,
    selection: &ModelSelection,
    catalog: &CachedModelCatalog,
) -> Result<ResolvedModelSpec, String> {
    provider.validate_model_name(&selection.model)?;
    let configured_model = selection.model.trim();
    let catalog_model_id =
        crate::provider_metadata::provider_catalog_model_id(provider.kind(), configured_model);
    let Some(info) = catalog.get(&catalog_model_id) else {
        let normalized =
            crate::provider_metadata::provider_wire_model_id(provider.kind(), configured_model);
        let mut message = format!(
            "model `{}` has no context-window entry in the models.dev catalog for {}. Choose a cataloged model.",
            configured_model,
            provider.kind(),
        );
        if normalized != configured_model {
            message.push_str(&format!("\nResolved provider model ID: `{normalized}`"));
        }
        return Err(message);
    };
    Ok(ResolvedModelSpec {
        configured_model: configured_model.to_string(),
        provider_model: crate::provider_metadata::provider_wire_model_id(
            provider.kind(),
            configured_model,
        ),
        catalog_model_id,
        info,
    })
}

pub(crate) fn resolve_model_variant(
    provider: &ProviderHandle,
    model: &str,
    requested: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(raw) = requested else {
        return Ok(
            crate::provider_metadata::default_model_variant_for_provider(
                provider.kind(),
                model,
                provider.supported_variants(model),
            )
            .map(str::to_string),
        );
    };
    let variant = parse_variant_input(raw)?;
    if variant == "default" {
        return Ok(
            crate::provider_metadata::default_model_variant_for_provider(
                provider.kind(),
                model,
                provider.supported_variants(model),
            )
            .map(str::to_string),
        );
    }
    provider.validate_variant(model, &variant)?;
    Ok(Some(variant))
}

pub(crate) fn variant_lines(
    provider: &ProviderHandle,
    model: &str,
    current_variant: Option<&str>,
) -> Vec<String> {
    let supported = provider.supported_variants(model);
    let mut lines = Vec::new();
    if supported.is_empty() {
        lines.push(format!(
            "`{}` on {} does not expose configurable variants.",
            model,
            provider_display_label(provider)
        ));
        return lines;
    }
    lines.push(format!(
        "Current variant: `{}`",
        current_variant.unwrap_or("(none)")
    ));
    if let Some(default_variant) = crate::provider_metadata::default_model_variant_for_provider(
        provider.kind(),
        model,
        supported,
    ) {
        lines.push(format!("Recommended default: `{}`", default_variant));
    }
    lines.push(format!("Available variants: {}", supported.join(", ")));
    lines.push("Usage: `/variant <name>` or `/variant default`".to_string());
    lines
}

pub(crate) fn provider_display_label(provider: &ProviderHandle) -> &'static str {
    crate::provider_metadata::provider_cli_label(provider.kind())
}

pub(crate) fn expose_provider_thinking(provider: &mut ProviderHandle) {
    let mut options = provider.options();
    options.thinking.expose = true;
    provider.set_options(options);
}

pub(crate) fn hash12(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{:x}", digest)[..12].to_string()
}

pub(crate) fn push_system_message(app: &mut App, msg: impl Into<String>) {
    let msg = msg.into();
    let duplicate = matches!(
        app.timeline.last(),
        Some(UiTimelineItem::SystemMessage(existing)) if existing == &msg
    );
    if duplicate {
        return;
    }
    crate::ui_trace::record_system_message_aux(&msg);
    app.timeline.push(UiTimelineItem::SystemMessage(msg));
    app.invalidate_height_cache();
    app.scroll_to_bottom();
}

pub(crate) fn version_text() -> String {
    format!(
        "lash-cli {}
lash-sansio {}",
        crate::APP_VERSION,
        lash_core::SANSIO_VERSION
    )
}

pub(crate) fn info_text_unconfigured(execution_mode: &ExecutionMode, cwd: &str) -> String {
    [
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash_core::SANSIO_VERSION),
        "provider: (not configured)".to_string(),
        "configured model: (not configured)".to_string(),
        "resolved model: (not configured)".to_string(),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        "context window: unknown".to_string(),
        format!("cwd: {}", cwd),
        "session: (not started)".to_string(),
    ]
    .join("\n")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn info_text(
    provider: &ProviderHandle,
    configured_model: &str,
    model_variant: Option<&str>,
    execution_mode: &ExecutionMode,
    standard_context_approach: Option<&StandardContextApproach>,
    context_window: Option<u64>,
    tool_summary: Option<(usize, &str)>,
    cwd: &str,
    session_name: Option<&str>,
    session_id: Option<&str>,
    session_db_path: Option<&str>,
) -> String {
    let resolved_model =
        crate::provider_metadata::provider_wire_model_id(provider.kind(), configured_model);
    let mut lines = vec![
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash_core::SANSIO_VERSION),
        format!(
            "provider: {} ({})",
            provider_display_label(provider),
            provider.kind()
        ),
        format!("configured model: {}", configured_model),
        format!("resolved model: {}", resolved_model),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
    ];
    if *execution_mode == ExecutionMode::standard()
        && let Some(standard_context_approach) = standard_context_approach
    {
        lines.push(format!(
            "context approach: {}",
            standard_context_approach_label(standard_context_approach)
        ));
    }

    if let Some(variant) = model_variant {
        lines.push(format!("variant: {}", variant));
    }
    if let Some(window) = context_window {
        lines.push(format!("context window: {}", window));
    } else {
        lines.push("context window: unknown".to_string());
    }

    if let Some((tool_count, toolset_hash)) = tool_summary {
        lines.push(format!("tools: {} (hash {})", tool_count, toolset_hash));
    } else {
        lines.push("tools: (session not started)".to_string());
    }
    lines.extend([
        format!("cwd: {}", cwd),
        format!("session: {}", session_name.unwrap_or("(not started)")),
    ]);
    if let Some(session_id) = session_id {
        lines.push(format!("session id: {}", session_id));
    }
    if let Some(session_db_path) = session_db_path {
        lines.push(format!("session db: {}", session_db_path));
    }

    lines.join("\n")
}

pub(crate) fn help_text(skills: &SkillCatalog, ui_extensions: &TuiExtensions) -> String {
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
        "  /tools rm <name>".to_string(),
        "  /tools update <name> key=value ...".to_string(),
        "  /tools enable <name> | /tools disable <name>".to_string(),
        "  /reconfigure status|apply|clear".to_string(),
    ]);

    lines.push(String::new());
    lines.push("Shortcuts:".to_string());
    lines.extend(render_shortcut_lines(ui_extensions, false));

    lines.join("\n")
}

pub(crate) fn apply_ui_host_effects(app: &mut App, effects: Vec<TuiHostEffect>) {
    for effect in effects {
        match effect {
            TuiHostEffect::PushSystemMessage(message) => push_system_message(app, message),
            TuiHostEffect::DesktopNotification {
                title,
                body,
                only_when_unfocused,
            } => {
                if !only_when_unfocused || !app.focused {
                    crate::interactive::notify_desktop(&title, &body);
                }
            }
            TuiHostEffect::UpsertModeIndicator { key, label } => {
                app.upsert_mode_indicator(key, label);
            }
            TuiHostEffect::ClearModeIndicator { key } => {
                app.clear_mode_indicator(&key);
            }
            TuiHostEffect::UpsertPanel {
                plugin_id,
                key,
                title,
                content,
            } => {
                let target = crate::plugin_surface::surface_key(&plugin_id, &key);
                if let Some(existing) = app.timeline.iter_mut().find_map(|block| match block {
                    UiTimelineItem::PluginPanel(panel)
                        if crate::plugin_surface::surface_key(&panel.plugin_id, &panel.key)
                            == target =>
                    {
                        Some(panel)
                    }
                    _ => None,
                }) {
                    existing.title = title;
                    existing.content = content;
                } else {
                    app.timeline
                        .push(UiTimelineItem::PluginPanel(PluginPanelBlock {
                            plugin_id,
                            key,
                            title,
                            content,
                        }));
                }
                app.invalidate_height_cache();
                app.scroll_to_bottom();
                app.dirty = true;
            }
            TuiHostEffect::ClearPanel { plugin_id, key } => {
                let target = crate::plugin_surface::surface_key(&plugin_id, &key);
                let original_len = app.timeline.len();
                app.timeline.retain(|block| match block {
                    UiTimelineItem::PluginPanel(panel) => {
                        crate::plugin_surface::surface_key(&panel.plugin_id, &panel.key) != target
                    }
                    _ => true,
                });
                if app.timeline.len() != original_len {
                    app.invalidate_height_cache();
                    app.dirty = true;
                }
            }
            TuiHostEffect::QueueTurn { input } => {
                app.queue_turn(PreparedTurn::prepare(input, Vec::new(), &app.skills));
                app.dirty = true;
            }
            TuiHostEffect::QueuePreparedTurn {
                display_text,
                effective_text,
            } => {
                app.queue_turn(PreparedTurn::prepare_with_effective_text(
                    display_text,
                    effective_text,
                    Vec::new(),
                ));
                app.dirty = true;
            }
            TuiHostEffect::WakeSession { input } => {
                app.queue_process_wake(input);
            }
            TuiHostEffect::MountSurface { .. }
            | TuiHostEffect::UpdateSurface { .. }
            | TuiHostEffect::UnmountSurface { .. }
            | TuiHostEffect::FocusSurface { .. }
            | TuiHostEffect::BlurSurface { .. } => {
                app.dirty = true;
            }
        }
    }
}

pub(crate) async fn collect_ui_snapshot(
    session: lash::LashSession,
    ui_extensions: &TuiExtensions,
) -> crate::event::UiSnapshotResult {
    let started = std::time::Instant::now();
    let mut diagnostics = Vec::new();
    let processes = match session.process_control().list().await {
        Ok(tasks) => Some(tasks),
        Err(err) => {
            diagnostics.push(format!("process snapshot failed: {err}"));
            None
        }
    };
    let effects = match ui_extensions
        .snapshot_all(TuiExtensionContext {
            actions: &session.plugin_actions(),
        })
        .await
    {
        Ok(effects) => effects,
        Err(err) => {
            diagnostics.push(format!("Failed to snapshot UI extensions: {err}"));
            Vec::new()
        }
    };
    crate::event::UiSnapshotResult {
        effects,
        processes,
        duration: started.elapsed(),
        timed_out: false,
        diagnostics,
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
    use lash_core::SessionMeta;

    #[test]
    fn desktop_notification_effect_respects_focus() {
        let mut app = App::new("test-model".into(), "test".into(), "test-session-id".into());
        app.focused = true;

        apply_ui_host_effects(
            &mut app,
            vec![TuiHostEffect::DesktopNotification {
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
            relation: lash_core::SessionRelation::Root,
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
    fn observational_memory_overrides_apply_to_standard_context_approach() {
        let approach = apply_standard_context_approach_overrides(
            StandardContextApproach::ObservationalMemory(Default::default()),
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
        let StandardContextApproach::ObservationalMemory(config) = approach else {
            panic!("expected observational_memory");
        };
        assert_eq!(config.observation_message_tokens, 45_000);
        assert_eq!(config.observation_buffer_tokens, 8_000);
        assert_eq!(config.observation_max_tokens_per_batch, 12_000);
        assert_eq!(config.previous_observer_tokens, 3_000);
        assert_eq!(config.reflection_buffer_activation_bps, 6_000);
    }

    #[test]
    fn observational_memory_overrides_require_om_standard_context_approach() {
        let err = apply_standard_context_approach_overrides(
            StandardContextApproach::RollingHistory(Default::default()),
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

    #[test]
    fn info_text_includes_session_id_and_db_path() {
        let provider = ProviderHandle::new(
            lash_provider_openai::OpenAiCompatibleProvider::new(
                "test",
                "https://openrouter.ai/api/v1",
            )
            .into_components(),
        );
        let text = info_text(
            &provider,
            "google/gemini-3-flash-preview",
            None,
            &ExecutionMode::rlm(),
            None,
            Some(123_000),
            Some((7, "abcd1234")),
            "/tmp/demo",
            Some("demo-session"),
            Some("sess-123"),
            Some("/tmp/demo/session.db"),
        );
        assert!(text.contains("session: demo-session"));
        assert!(text.contains("session id: sess-123"));
        assert!(text.contains("session db: /tmp/demo/session.db"));
    }
}
