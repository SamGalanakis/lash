use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lash::provider::LashConfig;
use lash::*;
use lash_tui::Terminal;

use crate::app::App;
use crate::setup;
use crate::{
    ensure_supported_execution_mode, execution_mode_label, execution_mode_usage,
    parse_execution_mode, parse_model_selection, provider_display_label, push_system_message,
    resolve_model_selection, resolve_model_variant, validate_model_selection, variant_lines,
};

pub(super) async fn handle_model(
    new_model: Option<String>,
    app: &mut App,
    runtime: &mut Option<LashRuntime>,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    model_catalog: &CachedModelCatalog,
) -> anyhow::Result<bool> {
    let Some(new_model) = new_model else {
        let mut lines = vec![
            format!("Current model: `{}`", app.model),
            format!("Provider: {}", provider_display_label(provider)),
        ];
        lines.extend(variant_lines(
            provider,
            &app.model,
            current_model_variant.as_deref(),
        ));
        if let Some(window) = app.context_window {
            lines.push(format!("Context window: {}", window));
        }
        lines.push("Usage: `/model <name>`".to_string());
        lines.push("Use `/variant` to inspect or change the active variant.".to_string());
        push_system_message(app, lines.join("\n"));
        return Ok(false);
    };
    let selection = match parse_model_selection(&new_model) {
        Ok(s) => s,
        Err(e) => {
            push_system_message(app, format!("Invalid model input: {}", e));
            return Ok(false);
        }
    };
    if let Err(e) = validate_model_selection(provider, &selection) {
        push_system_message(app, format!("Model rejected: {}", e));
        return Ok(false);
    }
    let resolved_model_spec = match resolve_model_selection(provider, &selection, model_catalog) {
        Ok(spec) => spec,
        Err(err) => {
            push_system_message(app, format!("Model rejected: {}", err));
            return Ok(false);
        }
    };
    let model_variant = provider
        .default_model_variant(&selection.model)
        .map(str::to_string);
    if let Some(rt) = runtime.as_mut() {
        rt.update_session_config(
            None,
            Some(selection.model.clone()),
            Some(model_variant.clone()),
            Some(resolved_model_spec.context_window() as usize),
        )
        .await;
    }
    *current_model_variant = model_variant;
    app.context_window = Some(resolved_model_spec.context_window());
    app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    app.model = selection.model.clone();
    app.set_model_variant(current_model_variant.clone());
    let mut msg = format!("Model set to `{}`", app.model);
    if let Some(variant) = current_model_variant.as_deref() {
        msg.push_str(&format!("\nVariant reset to `{}`", variant));
        msg.push_str("\nUse `/variant` to pick a different provider-native preset.");
    } else {
        msg.push_str("\nThis model does not expose configurable variants.");
    }
    if let Some(window) = app.context_window {
        msg.push_str(&format!("\nContext window: {}", window));
    }
    push_system_message(app, msg);
    Ok(false)
}

pub(super) async fn handle_variant(
    new_variant: Option<String>,
    app: &mut App,
    runtime: &mut Option<LashRuntime>,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
) -> anyhow::Result<bool> {
    let Some(new_variant) = new_variant else {
        let mut lines = vec![
            format!("Current model: `{}`", app.model),
            format!("Provider: {}", provider_display_label(provider)),
        ];
        lines.extend(variant_lines(
            provider,
            &app.model,
            current_model_variant.as_deref(),
        ));
        push_system_message(app, lines.join("\n"));
        return Ok(false);
    };
    let variant = match resolve_model_variant(provider, &app.model, Some(new_variant.as_str())) {
        Ok(variant) => variant,
        Err(err) => {
            push_system_message(app, format!("Variant rejected: {}", err));
            return Ok(false);
        }
    };
    if let Some(rt) = runtime.as_mut() {
        rt.update_session_config(None, None, Some(variant.clone()), None)
            .await;
    }
    *current_model_variant = variant;
    app.set_model_variant(current_model_variant.clone());
    let mut lines = vec![format!("Model: `{}`", app.model)];
    if let Some(variant) = current_model_variant.as_deref() {
        lines.push(format!("Variant set to `{}`", variant));
    } else {
        lines.push("Variant reset to provider default `(none)`.".to_string());
    }
    lines.extend(variant_lines(
        provider,
        &app.model,
        current_model_variant.as_deref(),
    ));
    push_system_message(app, lines.join("\n"));
    Ok(false)
}

pub(super) fn handle_mode(
    new_mode: Option<String>,
    app: &mut App,
    current_execution_mode: &ExecutionMode,
) -> anyhow::Result<bool> {
    let Some(new_mode) = new_mode else {
        push_system_message(
            app,
            format!(
                "Current execution mode: `{}`\nThis is locked for the current session.\nStart a new session to use a different mode.\nUsage: `/mode {}`",
                execution_mode_label(*current_execution_mode),
                execution_mode_usage()
            ),
        );
        return Ok(false);
    };
    let new_mode = match parse_execution_mode(&new_mode).and_then(ensure_supported_execution_mode) {
        Ok(mode) => mode,
        Err(err) => {
            push_system_message(app, format!("Invalid execution mode: {}", err));
            return Ok(false);
        }
    };
    if new_mode == *current_execution_mode {
        push_system_message(
            app,
            format!(
                "Execution mode is already `{}`.\nThis is locked for the current session.",
                execution_mode_label(*current_execution_mode)
            ),
        );
    } else {
        push_system_message(
            app,
            format!(
                "Execution mode is locked for the current session (`{}`).\nStart a new session with `--mode {}` to use `{}`.",
                execution_mode_label(*current_execution_mode),
                execution_mode_label(new_mode),
                execution_mode_label(new_mode)
            ),
        );
    }
    Ok(false)
}

pub(super) async fn handle_change_provider(
    terminal: &mut Terminal,
    app: &mut App,
    paused: &Arc<AtomicBool>,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    model_catalog: &CachedModelCatalog,
    runtime: &mut Option<LashRuntime>,
) -> anyhow::Result<bool> {
    paused.store(true, Ordering::Relaxed);
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableFocusChange);
    let previous_kind = provider.kind();
    let previous_provider = provider.clone();
    let previous_model = app.model.clone();
    let previous_context_window = app.context_window;
    let previous_context_usage = app.context_usage_excludes_cached_input;
    let previous_variant = current_model_variant.clone();

    terminal.restore();
    let existing_cfg = LashConfig::load(&crate::paths::config_file());
    let setup_result = setup::run_setup_with_existing(existing_cfg.as_ref()).await;
    *terminal = Terminal::enter()?;
    paused.store(false, Ordering::Relaxed);

    match setup_result {
        Ok(new_cfg) => {
            let mut new_provider = match new_cfg.build_active_provider() {
                Ok(p) => p,
                Err(err) => {
                    push_system_message(
                        app,
                        format!(
                            "Provider setup completed, but materializing failed: {}",
                            err
                        ),
                    );
                    *provider = previous_provider;
                    app.model = previous_model;
                    app.context_window = previous_context_window;
                    app.context_usage_excludes_cached_input = previous_context_usage;
                    *current_model_variant = previous_variant;
                    app.set_model_variant(current_model_variant.clone());
                    return Ok(false);
                }
            };
            if let Err(e) = new_provider.ensure_fresh().await {
                push_system_message(
                    app,
                    format!("Provider setup completed, but token refresh failed: {}", e),
                );
                *provider = previous_provider;
                app.model = previous_model;
                app.context_window = previous_context_window;
                app.context_usage_excludes_cached_input = previous_context_usage;
                *current_model_variant = previous_variant;
                app.set_model_variant(current_model_variant.clone());
                return Ok(false);
            }
            let mut new_cfg = new_cfg;
            new_cfg.upsert_provider(&new_provider);
            if let Err(e) = new_cfg.save(&crate::paths::config_file()) {
                push_system_message(
                    app,
                    format!("Provider updated, but saving config failed: {}", e),
                );
            }
            *provider = new_provider;
            if let Err(err) = model_catalog
                .refresh_if_stale(lash::model_info::DEFAULT_REFRESH_INTERVAL)
                .await
            {
                push_system_message(
                    app,
                    format!("Warning: failed to refresh models.dev catalog: {}", err),
                );
            }
            let new_model = provider.default_model().to_string();
            let selection = match parse_model_selection(&new_model) {
                Ok(s) => s,
                Err(e) => {
                    push_system_message(app, format!("Provider default model is invalid: {}", e));
                    *provider = previous_provider;
                    app.model = previous_model;
                    app.context_window = previous_context_window;
                    app.context_usage_excludes_cached_input = previous_context_usage;
                    *current_model_variant = previous_variant;
                    app.set_model_variant(current_model_variant.clone());
                    return Ok(false);
                }
            };
            if let Err(e) = validate_model_selection(provider, &selection) {
                push_system_message(
                    app,
                    format!("Provider default model failed validation: {}", e),
                );
                *provider = previous_provider;
                app.model = previous_model;
                app.context_window = previous_context_window;
                app.context_usage_excludes_cached_input = previous_context_usage;
                *current_model_variant = previous_variant;
                app.set_model_variant(current_model_variant.clone());
                return Ok(false);
            }
            let resolved_model_spec =
                match resolve_model_selection(provider, &selection, model_catalog) {
                    Ok(spec) => spec,
                    Err(err) => {
                        push_system_message(
                            app,
                            format!("Provider default model failed validation: {}", err),
                        );
                        *provider = previous_provider;
                        app.model = previous_model;
                        app.context_window = previous_context_window;
                        app.context_usage_excludes_cached_input = previous_context_usage;
                        *current_model_variant = previous_variant;
                        app.set_model_variant(current_model_variant.clone());
                        return Ok(false);
                    }
                };
            let model_variant = provider
                .default_model_variant(&selection.model)
                .map(str::to_string);
            if let Some(rt) = runtime.as_mut() {
                rt.update_session_config(
                    Some(provider.clone()),
                    Some(selection.model.clone()),
                    Some(model_variant.clone()),
                    Some(resolved_model_spec.context_window() as usize),
                )
                .await;
            }
            *current_model_variant = model_variant;
            app.context_window = Some(resolved_model_spec.context_window());
            app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
            app.model = selection.model.clone();
            app.set_model_variant(current_model_variant.clone());
            let saved_kinds = new_cfg.provider_kinds().join(", ");
            push_system_message(
                app,
                format!(
                    "Provider {}: {}\nSaved providers: {}\nModel set to default: `{}`",
                    if provider.kind() == previous_kind {
                        "reauthenticated"
                    } else {
                        "switched"
                    },
                    provider_display_label(provider),
                    saved_kinds,
                    selection.model
                ),
            );
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("Setup cancelled") {
                push_system_message(app, "Provider setup cancelled. Current provider unchanged.");
            } else {
                push_system_message(
                    app,
                    format!(
                        "Provider setup failed: {}. Current provider unchanged.",
                        msg
                    ),
                );
            }
        }
    }
    Ok(false)
}

fn kind_cli_label(kind: &str) -> &'static str {
    lash::provider_factory(kind)
        .map(|f| f.cli_label())
        .unwrap_or("Provider")
}

pub(super) fn handle_logout(app: &mut App, provider: &ProviderHandle) -> anyhow::Result<bool> {
    let active_kind = provider.kind();
    match LashConfig::load(&crate::paths::config_file()) {
        Some(mut cfg) => {
            if !cfg.has_provider(active_kind) {
                push_system_message(app, "The active provider is not stored on disk.");
                return Ok(false);
            }
            if cfg.provider_count() == 1 {
                match LashConfig::clear(&crate::paths::config_file()) {
                    Ok(()) => push_system_message(
                        app,
                        format!(
                            "Removed stored credentials for {}.\n\nThis running session may continue using in-memory credentials.\nUse `/provider` or `/login` to sign in again without restarting.",
                            kind_cli_label(active_kind)
                        ),
                    ),
                    Err(e) => {
                        push_system_message(app, format!("Failed to remove credentials: {}", e))
                    }
                }
            } else {
                cfg.remove_provider(active_kind);
                let next_kind = cfg.active_provider_kind().to_string();
                match cfg.save(&crate::paths::config_file()) {
                    Ok(()) => push_system_message(
                        app,
                        format!(
                            "Removed stored credentials for {}.\nNew sessions will default to {}.\n\nThis running session may continue using in-memory credentials.",
                            kind_cli_label(active_kind),
                            kind_cli_label(&next_kind)
                        ),
                    ),
                    Err(e) => push_system_message(
                        app,
                        format!("Failed to save updated provider registry: {}", e),
                    ),
                }
            }
        }
        None => push_system_message(app, "No stored provider credentials found."),
    }
    Ok(false)
}
