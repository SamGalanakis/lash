mod surface;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash::{PluginHost, SessionEvent, SessionManager, ToolResult};
use lash_tui::{Frame, InputEvent, TermCapabilities};

pub use surface::{
    UiMountedSurface, UiSurfaceScene, UiSurfaceSize, UiSurfaceSlot, UiSurfaceSpec, UiSurfaceUpdate,
    global_surface_id,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyModifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
}

impl KeyModifiers {
    pub const NONE: Self = Self {
        shift: false,
        control: false,
        alt: false,
    };

    pub const SHIFT: Self = Self {
        shift: true,
        control: false,
        alt: false,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyCode {
    Tab,
    Enter,
    Esc,
    Up,
    Down,
    PageUp,
    PageDown,
    Char(char),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyChord {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

impl KeyChord {
    pub const SHIFT_TAB: Self = Self {
        code: KeyCode::Tab,
        modifiers: KeyModifiers::SHIFT,
    };

    pub fn display(self) -> String {
        let mut parts = Vec::new();
        if self.modifiers.control {
            parts.push("Ctrl");
        }
        if self.modifiers.alt {
            parts.push("Alt");
        }
        if self.modifiers.shift {
            parts.push("Shift");
        }
        let key = match self.code {
            KeyCode::Tab => "Tab".to_string(),
            KeyCode::Enter => "Enter".to_string(),
            KeyCode::Esc => "Esc".to_string(),
            KeyCode::Up => "Up".to_string(),
            KeyCode::Down => "Down".to_string(),
            KeyCode::PageUp => "PgUp".to_string(),
            KeyCode::PageDown => "PgDn".to_string(),
            KeyCode::Char(ch) => ch.to_ascii_uppercase().to_string(),
        };
        parts.push(&key);
        parts.join("+")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlashCommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub usage: &'static str,
    pub description: &'static str,
    pub takes_argument: bool,
    pub allow_while_running: bool,
    pub action: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortcutSpec {
    pub chord: KeyChord,
    pub description: &'static str,
    pub action: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UiHostEffect {
    PushSystemMessage(String),
    DesktopNotification {
        title: String,
        body: String,
        only_when_unfocused: bool,
    },
    UpsertModeIndicator {
        key: String,
        label: String,
    },
    ClearModeIndicator {
        key: String,
    },
    UpsertPanel {
        plugin_id: String,
        key: String,
        title: String,
        content: String,
    },
    ClearPanel {
        plugin_id: String,
        key: String,
    },
    QueueTurn {
        input: String,
    },
    SwitchToNewSession {
        session_id: String,
    },
    MountSurface {
        spec: UiSurfaceSpec,
    },
    UpdateSurface {
        key: String,
        update: UiSurfaceUpdate,
    },
    UnmountSurface {
        key: String,
    },
    FocusSurface {
        key: String,
    },
    BlurSurface {
        key: String,
    },
}

pub struct UiContext<'a> {
    pub plugin_host: &'a PluginHost,
    pub session_id: &'a str,
    pub session_manager: Arc<dyn SessionManager>,
}

#[derive(Clone, Copy, Debug)]
pub struct UiRenderContext<'a> {
    pub session_id: &'a str,
    pub capabilities: TermCapabilities,
    pub surface_id: &'a str,
    pub focused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UiInputOutcome {
    Ignored,
    Handled(Vec<UiHostEffect>),
}

#[async_trait]
pub trait UiExtension: Send + Sync {
    fn id(&self) -> &'static str;

    fn commands(&self) -> &'static [SlashCommandSpec] {
        &[]
    }

    fn shortcuts(&self) -> &'static [ShortcutSpec] {
        &[]
    }

    fn render_surface(
        &self,
        _surface_key: &str,
        _ctx: UiRenderContext<'_>,
        _frame: &mut Frame<'_>,
    ) {
    }

    fn handle_surface_input(
        &self,
        _surface_key: &str,
        _event: &InputEvent,
        _ctx: UiContext<'_>,
    ) -> UiInputOutcome {
        UiInputOutcome::Ignored
    }

    async fn sync(&self, _ctx: UiContext<'_>) -> Result<Vec<UiHostEffect>, String> {
        Ok(Vec::new())
    }

    async fn invoke_action(
        &self,
        action: &str,
        arg: Option<&str>,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String>;

    fn handle_session_event(&self, _event: &SessionEvent) -> Vec<UiHostEffect> {
        Vec::new()
    }
}

#[derive(Clone)]
struct RegisteredCommand {
    extension: Arc<dyn UiExtension>,
    spec: SlashCommandSpec,
}

#[derive(Clone)]
struct RegisteredShortcut {
    extension: Arc<dyn UiExtension>,
    spec: ShortcutSpec,
}

#[derive(Clone)]
pub struct UiCommandInvocation {
    extension: Arc<dyn UiExtension>,
    spec: SlashCommandSpec,
    arg: Option<String>,
}

impl UiCommandInvocation {
    pub fn allow_while_running(&self) -> bool {
        self.spec.allow_while_running
    }
}

#[derive(Clone)]
pub struct UiShortcutInvocation {
    extension: Arc<dyn UiExtension>,
    spec: ShortcutSpec,
}

#[derive(Clone, Default)]
pub struct UiExtensions {
    extensions: Vec<Arc<dyn UiExtension>>,
    extension_map: BTreeMap<String, Arc<dyn UiExtension>>,
    commands: Vec<RegisteredCommand>,
    shortcuts: Vec<RegisteredShortcut>,
    surfaces: Arc<Mutex<surface::SurfaceRegistry>>,
}

impl UiExtensions {
    pub fn with_builtins(mut extra: Vec<Arc<dyn UiExtension>>) -> Result<Self, String> {
        let mut extensions: Vec<Arc<dyn UiExtension>> = vec![Arc::new(PlanModeUiExtension)];
        extensions.append(&mut extra);
        Self::new(extensions)
    }

    pub fn new(extensions: Vec<Arc<dyn UiExtension>>) -> Result<Self, String> {
        let mut command_names = BTreeMap::<String, &'static str>::new();
        let mut extension_map = BTreeMap::<String, Arc<dyn UiExtension>>::new();
        let mut shortcuts = BTreeMap::<KeyChord, &'static str>::new();
        let mut registered_commands = Vec::new();
        let mut registered_shortcuts = Vec::new();

        for extension in &extensions {
            if extension_map
                .insert(extension.id().to_string(), Arc::clone(extension))
                .is_some()
            {
                return Err(format!(
                    "duplicate UI extension id `{}` registered",
                    extension.id()
                ));
            }
            for spec in extension.commands() {
                for token in std::iter::once(spec.name).chain(spec.aliases.iter().copied()) {
                    if let Some(existing) = command_names.insert(token.to_string(), extension.id())
                    {
                        return Err(format!(
                            "duplicate UI slash command `{token}` registered by `{existing}` and `{}`",
                            extension.id()
                        ));
                    }
                }
                registered_commands.push(RegisteredCommand {
                    extension: Arc::clone(extension),
                    spec: *spec,
                });
            }
            for spec in extension.shortcuts() {
                if let Some(existing) = shortcuts.insert(spec.chord, extension.id()) {
                    return Err(format!(
                        "duplicate UI shortcut `{}` registered by `{existing}` and `{}`",
                        spec.chord.display(),
                        extension.id()
                    ));
                }
                registered_shortcuts.push(RegisteredShortcut {
                    extension: Arc::clone(extension),
                    spec: *spec,
                });
            }
        }

        Ok(Self {
            extensions,
            extension_map,
            commands: registered_commands,
            shortcuts: registered_shortcuts,
            surfaces: Arc::new(Mutex::new(surface::SurfaceRegistry::default())),
        })
    }

    pub fn builtin() -> Result<Self, String> {
        Self::with_builtins(Vec::new())
    }

    pub fn command_specs(&self) -> Vec<SlashCommandSpec> {
        self.commands.iter().map(|entry| entry.spec).collect()
    }

    pub fn shortcut_specs(&self) -> Vec<ShortcutSpec> {
        self.shortcuts.iter().map(|entry| entry.spec).collect()
    }

    pub fn completions(&self, prefix: &str) -> Vec<(String, String)> {
        if !prefix.starts_with('/') {
            return Vec::new();
        }
        self.commands
            .iter()
            .filter(|entry| entry.spec.name.starts_with(prefix))
            .map(|entry| {
                (
                    entry.spec.name.to_string(),
                    entry.spec.description.to_string(),
                )
            })
            .collect()
    }

    pub fn command_takes_argument(&self, cmd: &str) -> Option<bool> {
        self.commands
            .iter()
            .find(|entry| command_matches(cmd, entry.spec))
            .map(|entry| entry.spec.takes_argument)
    }

    pub fn parse_command(&self, input: &str) -> Option<UiCommandInvocation> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let rest = &trimmed[1..];
        let (cmd, arg) = match rest.split_once(' ') {
            Some((command, remainder)) => (format!("/{command}"), Some(remainder.trim())),
            None => (format!("/{rest}"), None),
        };
        let entry = self
            .commands
            .iter()
            .find(|entry| command_matches(&cmd, entry.spec))?;
        Some(UiCommandInvocation {
            extension: Arc::clone(&entry.extension),
            spec: entry.spec,
            arg: arg.filter(|value| !value.is_empty()).map(str::to_string),
        })
    }

    pub fn shortcut_for(&self, chord: KeyChord) -> Option<UiShortcutInvocation> {
        let entry = self
            .shortcuts
            .iter()
            .find(|entry| entry.spec.chord == chord)?;
        Some(UiShortcutInvocation {
            extension: Arc::clone(&entry.extension),
            spec: entry.spec,
        })
    }

    pub async fn sync_all(&self, ctx: UiContext<'_>) -> Result<Vec<UiHostEffect>, String> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            let extension_effects = extension
                .sync(UiContext {
                    plugin_host: ctx.plugin_host,
                    session_id: ctx.session_id,
                    session_manager: Arc::clone(&ctx.session_manager),
                })
                .await?;
            effects.extend(self.process_effects(extension.id(), extension_effects));
        }
        Ok(effects)
    }

    pub async fn invoke_command(
        &self,
        invocation: &UiCommandInvocation,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        let effects = invocation
            .extension
            .invoke_action(invocation.spec.action, invocation.arg.as_deref(), ctx)
            .await?;
        Ok(self.process_effects(invocation.extension.id(), effects))
    }

    pub async fn invoke_shortcut(
        &self,
        invocation: &UiShortcutInvocation,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        let effects = invocation
            .extension
            .invoke_action(invocation.spec.action, None, ctx)
            .await?;
        Ok(self.process_effects(invocation.extension.id(), effects))
    }

    pub fn effects_for_session_event(&self, event: &SessionEvent) -> Vec<UiHostEffect> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            effects.extend(
                self.process_effects(extension.id(), extension.handle_session_event(event)),
            );
        }
        effects
    }

    pub fn clear_surface_areas(&self) {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .clear_areas();
    }

    pub fn has_surface_in_slot(&self, slot: UiSurfaceSlot) -> bool {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .has_surface_in_slot(slot)
    }

    pub fn mounted_surfaces(&self, slot: UiSurfaceSlot) -> Vec<UiMountedSurface> {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .surfaces_in_slot(slot)
    }

    pub fn surface_scene(&self) -> UiSurfaceScene {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .scene()
    }

    pub fn stack_height(&self, slot: UiSurfaceSlot, max_height: u16) -> u16 {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .stack_height(slot, max_height)
    }

    pub fn focused_surface(&self) -> Option<String> {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .focused_surface()
    }

    pub fn set_surface_area(&self, id: &str, area: Option<lash_tui::Rect>) {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .set_area(id, area);
    }

    pub fn sync_surface_areas<I>(&self, areas: I)
    where
        I: IntoIterator<Item = (String, lash_tui::Rect)>,
    {
        let mut surfaces = self.surfaces.lock().expect("surface registry poisoned");
        surfaces.clear_areas();
        for (id, area) in areas {
            surfaces.set_area(&id, Some(area));
        }
    }

    pub fn render_surface(&self, id: &str, ctx: UiRenderContext<'_>, frame: &mut Frame<'_>) {
        let surface = self
            .surfaces
            .lock()
            .expect("surface registry poisoned")
            .surface(id);
        let Some(surface) = surface else {
            return;
        };
        self.render_mounted_surface(
            &surface,
            UiRenderContext {
                focused: self.focused_surface().as_deref() == Some(surface.id.as_str()),
                ..ctx
            },
            frame,
        );
    }

    pub fn render_mounted_surface(
        &self,
        surface: &UiMountedSurface,
        ctx: UiRenderContext<'_>,
        frame: &mut Frame<'_>,
    ) {
        let Some(extension) = self.extension_map.get(&surface.owner_id) else {
            return;
        };
        extension.render_surface(
            &surface.key,
            UiRenderContext {
                surface_id: &surface.id,
                ..ctx
            },
            frame,
        );
    }

    pub fn handle_input(&self, event: &InputEvent, ctx: UiContext<'_>) -> UiInputOutcome {
        let target = self
            .surfaces
            .lock()
            .expect("surface registry poisoned")
            .target_for_input(event);
        let Some(surface) = target else {
            return UiInputOutcome::Ignored;
        };
        let Some(extension) = self.extension_map.get(&surface.owner_id) else {
            return UiInputOutcome::Ignored;
        };
        match extension.handle_surface_input(&surface.key, event, ctx) {
            UiInputOutcome::Ignored => UiInputOutcome::Ignored,
            UiInputOutcome::Handled(effects) => {
                UiInputOutcome::Handled(self.process_effects(extension.id(), effects))
            }
        }
    }

    fn process_effects(&self, owner_id: &str, effects: Vec<UiHostEffect>) -> Vec<UiHostEffect> {
        let mut passthrough = Vec::new();
        let mut surfaces = self.surfaces.lock().expect("surface registry poisoned");
        for effect in effects {
            match effect {
                UiHostEffect::MountSurface { spec } => surfaces.mount(owner_id, spec),
                UiHostEffect::UpdateSurface { key, update } => {
                    surfaces.update(owner_id, &key, update)
                }
                UiHostEffect::UnmountSurface { key } => surfaces.unmount(owner_id, &key),
                UiHostEffect::FocusSurface { key } => surfaces.focus(owner_id, &key),
                UiHostEffect::BlurSurface { key } => surfaces.blur(owner_id, &key),
                other => passthrough.push(other),
            }
        }
        passthrough
    }
}

fn command_matches(name: &str, spec: SlashCommandSpec) -> bool {
    name == spec.name || spec.aliases.contains(&name)
}

fn surface_key(plugin_id: &str, key: &str) -> String {
    format!("{plugin_id}:{key}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PlanModeStatus {
    enabled: bool,
    panel_title: Option<String>,
    panel_content: Option<String>,
}

fn plan_mode_status_from_result(
    result: ToolResult,
    op_name: &str,
) -> Result<PlanModeStatus, String> {
    if !result.success {
        return Err(format!("{op_name} failed: {}", result.result));
    }
    Ok(PlanModeStatus {
        enabled: result
            .result
            .get("enabled")
            .and_then(|value| value.as_bool())
            .ok_or_else(|| format!("{op_name} response missing `enabled`"))?,
        panel_title: result
            .result
            .get("panel_title")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        panel_content: result
            .result
            .get("panel_content")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

async fn plan_mode_status(ctx: UiContext<'_>, op_name: &str) -> Result<PlanModeStatus, String> {
    let result = ctx
        .plugin_host
        .invoke_external_for_session(
            ctx.session_id,
            op_name,
            serde_json::json!({}),
            ctx.session_manager,
        )
        .await
        .map_err(|err| err.to_string())?;
    plan_mode_status_from_result(result, op_name)
}

fn plan_mode_effects(status: &PlanModeStatus) -> Vec<UiHostEffect> {
    let key = surface_key("plan_mode", "mode");
    let mut effects = if status.enabled {
        vec![UiHostEffect::UpsertModeIndicator {
            key,
            label: "plan".to_string(),
        }]
    } else {
        vec![UiHostEffect::ClearModeIndicator { key }]
    };
    let panel_key = "panel".to_string();
    match (
        status.enabled,
        status.panel_title.as_deref(),
        status.panel_content.as_deref(),
    ) {
        (true, Some(title), Some(content)) => effects.push(UiHostEffect::UpsertPanel {
            plugin_id: "plan_mode".to_string(),
            key: panel_key,
            title: title.to_string(),
            content: content.to_string(),
        }),
        _ => effects.push(UiHostEffect::ClearPanel {
            plugin_id: "plan_mode".to_string(),
            key: panel_key,
        }),
    }
    effects
}

struct PlanModeUiExtension;

const PLAN_MODE_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
    name: "/plan",
    aliases: &[],
    usage: "/plan",
    description: "Toggle plan mode",
    takes_argument: false,
    allow_while_running: true,
    action: "toggle",
}];

const PLAN_MODE_SHORTCUTS: &[ShortcutSpec] = &[ShortcutSpec {
    chord: KeyChord::SHIFT_TAB,
    description: "Toggle plan mode",
    action: "toggle",
}];

#[async_trait]
impl UiExtension for PlanModeUiExtension {
    fn id(&self) -> &'static str {
        "plan_mode_ui"
    }

    fn commands(&self) -> &'static [SlashCommandSpec] {
        PLAN_MODE_COMMANDS
    }

    fn shortcuts(&self) -> &'static [ShortcutSpec] {
        PLAN_MODE_SHORTCUTS
    }

    async fn sync(&self, ctx: UiContext<'_>) -> Result<Vec<UiHostEffect>, String> {
        let status = plan_mode_status(ctx, "plan_mode.status")
            .await
            .map_err(|err| format!("failed to sync plan mode: {err}"))?;
        Ok(plan_mode_effects(&status))
    }

    async fn invoke_action(
        &self,
        action: &str,
        _arg: Option<&str>,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        match action {
            "toggle" => {
                let status = plan_mode_status(ctx, "plan_mode.toggle")
                    .await
                    .map_err(|err| format!("failed to toggle plan mode: {err}"))?;
                let mut effects = plan_mode_effects(&status);
                effects.push(UiHostEffect::PushSystemMessage(if status.enabled {
                    "Plan mode on.".to_string()
                } else {
                    "Plan mode off.".to_string()
                }));
                Ok(effects)
            }
            other => Err(format!("unknown plan-mode UI action `{other}`")),
        }
    }

    fn handle_session_event(&self, event: &SessionEvent) -> Vec<UiHostEffect> {
        let SessionEvent::ToolCall {
            name,
            result,
            success,
            ..
        } = event
        else {
            return Vec::new();
        };
        if name != "plan_exit" || !success {
            return Vec::new();
        }
        let approved = result
            .get("approved")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if approved {
            let mut effects = vec![
                UiHostEffect::ClearModeIndicator {
                    key: surface_key("plan_mode", "mode"),
                },
                UiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string(),
                },
            ];
            if result
                .get("execution_mode")
                .and_then(|value| value.as_str())
                == Some("fresh_context")
                && let Some(session_id) = result.get("session_id").and_then(|value| value.as_str())
            {
                effects.push(UiHostEffect::SwitchToNewSession {
                    session_id: session_id.to_string(),
                });
            }
            effects
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use super::*;
    use lash::SessionEvent;
    use lash_tui::{Rect, Style};
    use serde_json::json;

    #[test]
    fn builtin_extensions_register_plan_command_and_shortcut() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");

        assert!(extensions.parse_command("/plan").is_some());
        assert!(extensions.shortcut_for(KeyChord::SHIFT_TAB).is_some());
        assert_eq!(
            extensions.completions("/pl"),
            vec![("/plan".to_string(), "Toggle plan mode".to_string())]
        );
    }

    #[test]
    fn plan_mode_effects_show_panel_when_enabled() {
        assert_eq!(
            plan_mode_effects(&PlanModeStatus {
                enabled: true,
                panel_title: Some("PLAN".to_string()),
                panel_content: Some("- Path: `.lash/plans/root.md`".to_string()),
            }),
            vec![
                UiHostEffect::UpsertModeIndicator {
                    key: "plan_mode:mode".to_string(),
                    label: "plan".to_string(),
                },
                UiHostEffect::UpsertPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string(),
                    title: "PLAN".to_string(),
                    content: "- Path: `.lash/plans/root.md`".to_string(),
                }
            ]
        );
    }

    #[test]
    fn plan_mode_effects_clear_panel_when_disabled() {
        assert_eq!(
            plan_mode_effects(&PlanModeStatus {
                enabled: false,
                panel_title: Some("PLAN".to_string()),
                panel_content: Some("stale".to_string()),
            }),
            vec![
                UiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string(),
                },
                UiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string(),
                }
            ]
        );
    }

    #[test]
    fn duplicate_commands_are_rejected() {
        struct Duplicate;

        #[async_trait]
        impl UiExtension for Duplicate {
            fn id(&self) -> &'static str {
                "duplicate"
            }

            fn commands(&self) -> &'static [SlashCommandSpec] {
                PLAN_MODE_COMMANDS
            }

            async fn invoke_action(
                &self,
                _action: &str,
                _arg: Option<&str>,
                _ctx: UiContext<'_>,
            ) -> Result<Vec<UiHostEffect>, String> {
                Ok(Vec::new())
            }
        }

        let err = UiExtensions::new(vec![Arc::new(PlanModeUiExtension), Arc::new(Duplicate)])
            .err()
            .expect("duplicate commands should fail");
        assert!(err.contains("/plan"));
    }

    #[test]
    fn plan_exit_event_clears_plan_ui_without_queueing_turn() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");

        let effects = extensions.effects_for_session_event(&SessionEvent::ToolCall {
            call_id: None,
            name: "plan_exit".to_string(),
            args: json!({}),
            result: json!({
                "approved": true,
                "execution_mode": "current_session",
                "next_turn_input": "Execute the approved plan."
            }),
            success: true,
            duration_ms: 12,
        });

        assert_eq!(
            effects,
            vec![
                UiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string()
                },
                UiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string()
                }
            ]
        );
    }

    #[test]
    fn plan_exit_event_can_switch_to_fresh_context_session() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");

        let effects = extensions.effects_for_session_event(&SessionEvent::ToolCall {
            call_id: None,
            name: "plan_exit".to_string(),
            args: json!({}),
            result: json!({
                "approved": true,
                "execution_mode": "fresh_context",
                "session_id": "new-plan-session"
            }),
            success: true,
            duration_ms: 12,
        });

        assert_eq!(
            effects,
            vec![
                UiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string()
                },
                UiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string()
                },
                UiHostEffect::SwitchToNewSession {
                    session_id: "new-plan-session".to_string()
                }
            ]
        );
    }

    #[test]
    fn command_tokens_and_shortcuts_are_unique() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");
        let command_names = extensions
            .command_specs()
            .into_iter()
            .map(|spec| spec.name.to_string())
            .collect::<BTreeSet<_>>();
        let shortcuts = extensions
            .shortcut_specs()
            .into_iter()
            .map(|spec| spec.chord)
            .collect::<BTreeSet<_>>();

        assert_eq!(command_names.len(), 1);
        assert_eq!(shortcuts.len(), 1);
    }

    #[derive(Default)]
    struct SurfaceHarness {
        renders: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl UiExtension for SurfaceHarness {
        fn id(&self) -> &'static str {
            "surface_harness"
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: UiContext<'_>,
        ) -> Result<Vec<UiHostEffect>, String> {
            Ok(Vec::new())
        }

        fn render_surface(
            &self,
            surface_key: &str,
            _ctx: UiRenderContext<'_>,
            frame: &mut Frame<'_>,
        ) {
            self.renders
                .lock()
                .expect("renders lock poisoned")
                .push(surface_key.to_string());
            frame.write_text(0, 0, surface_key, Style::default(), frame.area().width);
        }

        fn handle_session_event(&self, event: &SessionEvent) -> Vec<UiHostEffect> {
            match event {
                SessionEvent::TextDelta { content } if content == "mount" => vec![
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "workspace".to_string(),
                            slot: UiSurfaceSlot::Workspace,
                            size: UiSurfaceSize::Auto,
                            order: 0,
                            focusable: true,
                            visible: true,
                            modal: false,
                        },
                    },
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "footer".to_string(),
                            slot: UiSurfaceSlot::Footer,
                            size: UiSurfaceSize::Lines(1),
                            order: 0,
                            focusable: false,
                            visible: true,
                            modal: false,
                        },
                    },
                    UiHostEffect::FocusSurface {
                        key: "workspace".to_string(),
                    },
                ],
                SessionEvent::TextDelta { content } if content == "overlay" => vec![
                    UiHostEffect::MountSurface {
                        spec: UiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: UiSurfaceSlot::Overlay,
                            size: UiSurfaceSize::Fixed {
                                width: 12,
                                height: 3,
                            },
                            order: 10,
                            focusable: true,
                            visible: true,
                            modal: true,
                        },
                    },
                    UiHostEffect::FocusSurface {
                        key: "overlay".to_string(),
                    },
                ],
                SessionEvent::TextDelta { content } if content == "close" => vec![
                    UiHostEffect::BlurSurface {
                        key: "overlay".to_string(),
                    },
                    UiHostEffect::UnmountSurface {
                        key: "overlay".to_string(),
                    },
                ],
                _ => Vec::new(),
            }
        }
    }

    #[test]
    fn surface_effects_mount_and_restore_focus() {
        let harness = Arc::new(SurfaceHarness::default());
        let extensions = UiExtensions::new(vec![harness]).expect("surface harness");

        extensions.effects_for_session_event(&SessionEvent::TextDelta {
            content: "mount".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:workspace")
        );
        assert_eq!(
            extensions.mounted_surfaces(UiSurfaceSlot::Workspace).len(),
            1
        );
        assert_eq!(extensions.mounted_surfaces(UiSurfaceSlot::Footer).len(), 1);

        extensions.effects_for_session_event(&SessionEvent::TextDelta {
            content: "overlay".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:overlay")
        );

        extensions.effects_for_session_event(&SessionEvent::TextDelta {
            content: "close".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:workspace")
        );
    }

    #[test]
    fn render_surface_dispatches_to_owner_extension() {
        let harness = Arc::new(SurfaceHarness::default());
        let extensions = UiExtensions::new(vec![Arc::clone(&harness) as Arc<dyn UiExtension>])
            .expect("surface harness");
        extensions.effects_for_session_event(&SessionEvent::TextDelta {
            content: "mount".to_string(),
        });
        let workspace = extensions
            .mounted_surfaces(UiSurfaceSlot::Workspace)
            .into_iter()
            .next()
            .expect("workspace surface");

        let snapshot = lash_tui::render_snapshot(20, 4, |frame| {
            let area = Rect::new(0, 0, 20, 4);
            extensions.set_surface_area(&workspace.id, Some(area));
            let mut viewport = frame.viewport(area);
            extensions.render_surface(
                &workspace.id,
                UiRenderContext {
                    session_id: "root",
                    capabilities: TermCapabilities::default(),
                    surface_id: "",
                    focused: false,
                },
                &mut viewport,
            );
        });

        assert_eq!(snapshot.visible_line_trimmed(0), "workspace");
        assert_eq!(
            harness
                .renders
                .lock()
                .expect("renders lock poisoned")
                .as_slice(),
            ["workspace"]
        );
    }
}
