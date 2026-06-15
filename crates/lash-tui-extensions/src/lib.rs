mod surface;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash::admin::PluginAction;
use lash::{PluginActions, TurnEvent};
use lash_tui::{Frame, InputEvent, TermCapabilities};

pub use surface::{
    TuiMountedSurface, TuiSurfaceScene, TuiSurfaceSize, TuiSurfaceSlot, TuiSurfaceSpec,
    TuiSurfaceUpdate, global_surface_id,
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
    pub argument_hint: Option<&'static str>,
    pub argument_options: &'static [&'static str],
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
pub enum TuiHostEffect {
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
    QueuePreparedTurn {
        display_text: String,
        effective_text: String,
    },
    MountSurface {
        spec: TuiSurfaceSpec,
    },
    UpdateSurface {
        key: String,
        update: TuiSurfaceUpdate,
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

pub async fn call_plugin_action<Op>(
    actions: &PluginActions,
    args: Op::Args,
) -> Result<Op::Output, String>
where
    Op: PluginAction,
{
    actions
        .call::<Op>(args)
        .await
        .map_err(|err| err.to_string())
}

pub struct TuiExtensionContext<'a> {
    pub actions: &'a PluginActions,
}

#[derive(Clone, Copy, Debug)]
pub struct TuiRenderContext<'a> {
    pub session_id: &'a str,
    pub capabilities: TermCapabilities,
    pub surface_id: &'a str,
    pub focused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TuiInputOutcome {
    Ignored,
    Handled(Vec<TuiHostEffect>),
}

#[async_trait]
pub trait TuiExtension: Send + Sync {
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
        _ctx: TuiRenderContext<'_>,
        _frame: &mut Frame<'_>,
    ) {
    }

    fn handle_surface_input(
        &self,
        _surface_key: &str,
        _event: &InputEvent,
        _ctx: TuiExtensionContext<'_>,
    ) -> TuiInputOutcome {
        TuiInputOutcome::Ignored
    }

    async fn snapshot(&self, _ctx: TuiExtensionContext<'_>) -> Result<Vec<TuiHostEffect>, String> {
        Ok(Vec::new())
    }

    async fn invoke_action(
        &self,
        action: &str,
        arg: Option<&str>,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String>;

    fn handle_turn_event(&self, _event: &TurnEvent) -> Vec<TuiHostEffect> {
        Vec::new()
    }

    fn handle_turn_finished(&self) -> Vec<TuiHostEffect> {
        Vec::new()
    }
}

#[derive(Clone)]
struct RegisteredCommand {
    extension: Arc<dyn TuiExtension>,
    spec: SlashCommandSpec,
}

#[derive(Clone)]
struct RegisteredShortcut {
    extension: Arc<dyn TuiExtension>,
    spec: ShortcutSpec,
}

#[derive(Clone)]
pub struct TuiSlashInvocation {
    extension: Arc<dyn TuiExtension>,
    spec: SlashCommandSpec,
    arg: Option<String>,
}

impl TuiSlashInvocation {
    pub fn allow_while_running(&self) -> bool {
        self.spec.allow_while_running
    }
}

#[derive(Clone)]
pub struct TuiShortcutInvocation {
    extension: Arc<dyn TuiExtension>,
    spec: ShortcutSpec,
}

#[derive(Clone, Default)]
pub struct TuiExtensions {
    extensions: Vec<Arc<dyn TuiExtension>>,
    extension_map: BTreeMap<String, Arc<dyn TuiExtension>>,
    commands: Vec<RegisteredCommand>,
    shortcuts: Vec<RegisteredShortcut>,
    surfaces: Arc<Mutex<surface::SurfaceRegistry>>,
}

impl TuiExtensions {
    pub fn with_builtins(mut extra: Vec<Arc<dyn TuiExtension>>) -> Result<Self, String> {
        let mut extensions: Vec<Arc<dyn TuiExtension>> = vec![Arc::new(PlanModeTuiExtension)];
        extensions.append(&mut extra);
        Self::new(extensions)
    }

    pub fn new(extensions: Vec<Arc<dyn TuiExtension>>) -> Result<Self, String> {
        let mut command_names = BTreeMap::<String, &'static str>::new();
        let mut extension_map = BTreeMap::<String, Arc<dyn TuiExtension>>::new();
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

    pub fn command_spec(&self, cmd: &str) -> Option<SlashCommandSpec> {
        self.commands
            .iter()
            .find(|entry| command_matches(cmd, entry.spec))
            .map(|entry| entry.spec)
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

    pub fn parse_command(&self, input: &str) -> Option<TuiSlashInvocation> {
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
        Some(TuiSlashInvocation {
            extension: Arc::clone(&entry.extension),
            spec: entry.spec,
            arg: arg.filter(|value| !value.is_empty()).map(str::to_string),
        })
    }

    pub fn shortcut_for(&self, chord: KeyChord) -> Option<TuiShortcutInvocation> {
        let entry = self
            .shortcuts
            .iter()
            .find(|entry| entry.spec.chord == chord)?;
        Some(TuiShortcutInvocation {
            extension: Arc::clone(&entry.extension),
            spec: entry.spec,
        })
    }

    pub async fn snapshot_all(
        &self,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            let extension_effects = extension
                .snapshot(TuiExtensionContext {
                    actions: ctx.actions,
                })
                .await?;
            effects.extend(self.process_effects(extension.id(), extension_effects));
        }
        Ok(effects)
    }

    pub async fn invoke_parsed_command(
        &self,
        invocation: &TuiSlashInvocation,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        let effects = invocation
            .extension
            .invoke_action(invocation.spec.action, invocation.arg.as_deref(), ctx)
            .await?;
        Ok(self.process_effects(invocation.extension.id(), effects))
    }

    pub async fn invoke_shortcut(
        &self,
        invocation: &TuiShortcutInvocation,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        let effects = invocation
            .extension
            .invoke_action(invocation.spec.action, None, ctx)
            .await?;
        Ok(self.process_effects(invocation.extension.id(), effects))
    }

    pub fn effects_for_turn_event(&self, event: &TurnEvent) -> Vec<TuiHostEffect> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            effects
                .extend(self.process_effects(extension.id(), extension.handle_turn_event(event)));
        }
        effects
    }

    pub fn effects_for_turn_finished(&self) -> Vec<TuiHostEffect> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            effects.extend(self.process_effects(extension.id(), extension.handle_turn_finished()));
        }
        effects
    }

    pub fn clear_surface_areas(&self) {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .clear_areas();
    }

    pub fn has_surface_in_slot(&self, slot: TuiSurfaceSlot) -> bool {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .has_surface_in_slot(slot)
    }

    pub fn mounted_surfaces(&self, slot: TuiSurfaceSlot) -> Vec<TuiMountedSurface> {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .surfaces_in_slot(slot)
    }

    pub fn surface_scene(&self) -> TuiSurfaceScene {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .scene()
    }

    pub fn stack_height(&self, slot: TuiSurfaceSlot, max_height: u16) -> u16 {
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

    pub fn mount_surface(&self, owner_id: &str, spec: TuiSurfaceSpec) {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .mount(owner_id, spec);
    }

    pub fn unmount_surface(&self, owner_id: &str, key: &str) {
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .unmount(owner_id, key);
    }

    pub fn surface_is_mounted(&self, owner_id: &str, key: &str) -> bool {
        let id = surface::global_surface_id(owner_id, key);
        self.surfaces
            .lock()
            .expect("surface registry poisoned")
            .surface(&id)
            .is_some()
    }

    pub fn render_surface(&self, id: &str, ctx: TuiRenderContext<'_>, frame: &mut Frame<'_>) {
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
            TuiRenderContext {
                focused: self.focused_surface().as_deref() == Some(surface.id.as_str()),
                ..ctx
            },
            frame,
        );
    }

    pub fn render_mounted_surface(
        &self,
        surface: &TuiMountedSurface,
        ctx: TuiRenderContext<'_>,
        frame: &mut Frame<'_>,
    ) {
        let Some(extension) = self.extension_map.get(&surface.owner_id) else {
            return;
        };
        extension.render_surface(
            &surface.key,
            TuiRenderContext {
                surface_id: &surface.id,
                ..ctx
            },
            frame,
        );
    }

    pub fn handle_input(
        &self,
        event: &InputEvent,
        ctx: TuiExtensionContext<'_>,
    ) -> TuiInputOutcome {
        let target = self
            .surfaces
            .lock()
            .expect("surface registry poisoned")
            .target_for_input(event);
        let Some(surface) = target else {
            return TuiInputOutcome::Ignored;
        };
        let Some(extension) = self.extension_map.get(&surface.owner_id) else {
            return TuiInputOutcome::Ignored;
        };
        match extension.handle_surface_input(&surface.key, event, ctx) {
            TuiInputOutcome::Ignored => TuiInputOutcome::Ignored,
            TuiInputOutcome::Handled(effects) => {
                TuiInputOutcome::Handled(self.process_effects(extension.id(), effects))
            }
        }
    }

    fn process_effects(&self, owner_id: &str, effects: Vec<TuiHostEffect>) -> Vec<TuiHostEffect> {
        let mut passthrough = Vec::new();
        let mut surfaces = self.surfaces.lock().expect("surface registry poisoned");
        for effect in effects {
            match effect {
                TuiHostEffect::MountSurface { spec } => surfaces.mount(owner_id, spec),
                TuiHostEffect::UpdateSurface { key, update } => {
                    surfaces.update(owner_id, &key, update)
                }
                TuiHostEffect::UnmountSurface { key } => surfaces.unmount(owner_id, &key),
                TuiHostEffect::FocusSurface { key } => surfaces.focus(owner_id, &key),
                TuiHostEffect::BlurSurface { key } => surfaces.blur(owner_id, &key),
                other => passthrough.push(other),
            }
        }
        passthrough
    }
}

fn command_matches(name: &str, spec: SlashCommandSpec) -> bool {
    name == spec.name || spec.aliases.contains(&name)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PlanModeStatus {
    enabled: bool,
    plan_path: Option<String>,
}

async fn invoke_plan_mode_action(
    ctx: TuiExtensionContext<'_>,
    op_name: &str,
) -> Result<PlanModeStatus, String> {
    let status = match op_name {
        "plan_mode.toggle" => {
            call_plugin_action::<lash_plugin_plan_mode::PlanModeToggleOp>(
                ctx.actions,
                lash_plugin_plan_mode::PlanModeExternalArgs {},
            )
            .await?
        }
        "plan_mode.enable" => {
            call_plugin_action::<lash_plugin_plan_mode::PlanModeEnableOp>(
                ctx.actions,
                lash_plugin_plan_mode::PlanModeExternalArgs {},
            )
            .await?
        }
        "plan_mode.disable" => {
            call_plugin_action::<lash_plugin_plan_mode::PlanModeDisableOp>(
                ctx.actions,
                lash_plugin_plan_mode::PlanModeExternalArgs {},
            )
            .await?
        }
        other => return Err(format!("unknown plan mode op `{other}`")),
    };
    Ok(PlanModeStatus {
        enabled: status.enabled,
        plan_path: status.plan_path,
    })
}

fn plan_mode_effects(status: &PlanModeStatus) -> Vec<TuiHostEffect> {
    let key = global_surface_id("plan_mode", "mode");
    let mut effects = if status.enabled {
        vec![TuiHostEffect::UpsertModeIndicator {
            key,
            label: "plan".to_string(),
        }]
    } else {
        vec![TuiHostEffect::ClearModeIndicator { key }]
    };
    let panel_key = "panel".to_string();
    if status.enabled {
        if let Some(path) = status.plan_path.as_deref() {
            effects.push(TuiHostEffect::UpsertPanel {
                plugin_id: "plan_mode".to_string(),
                key: panel_key,
                title: "PLAN".to_string(),
                content: format!("Path: `{path}`"),
            });
        } else {
            effects.push(TuiHostEffect::ClearPanel {
                plugin_id: "plan_mode".to_string(),
                key: panel_key,
            });
        }
    } else {
        effects.push(TuiHostEffect::ClearPanel {
            plugin_id: "plan_mode".to_string(),
            key: panel_key,
        });
    }
    effects
}

struct PlanModeTuiExtension;

const PLAN_MODE_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
    name: "/plan",
    aliases: &[],
    usage: "/plan",
    description: "Open file-backed planning mode",
    argument_hint: None,
    argument_options: &[],
    takes_argument: false,
    allow_while_running: true,
    action: "toggle",
}];

const PLAN_MODE_SHORTCUTS: &[ShortcutSpec] = &[ShortcutSpec {
    chord: KeyChord::SHIFT_TAB,
    description: "Open file-backed planning mode",
    action: "toggle",
}];

#[async_trait]
impl TuiExtension for PlanModeTuiExtension {
    fn id(&self) -> &'static str {
        "plan_mode_ui"
    }

    fn commands(&self) -> &'static [SlashCommandSpec] {
        PLAN_MODE_COMMANDS
    }

    fn shortcuts(&self) -> &'static [ShortcutSpec] {
        PLAN_MODE_SHORTCUTS
    }

    async fn invoke_action(
        &self,
        action: &str,
        _arg: Option<&str>,
        ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        match action {
            "toggle" => {
                let status = invoke_plan_mode_action(ctx, "plan_mode.toggle")
                    .await
                    .map_err(|err| format!("failed to toggle plan mode: {err}"))?;
                let mut effects = plan_mode_effects(&status);
                effects.push(TuiHostEffect::PushSystemMessage(if status.enabled {
                    "Plan mode on.".to_string()
                } else {
                    "Plan mode off.".to_string()
                }));
                Ok(effects)
            }
            other => Err(format!("unknown plan-mode UI action `{other}`")),
        }
    }

    fn handle_turn_event(&self, event: &TurnEvent) -> Vec<TuiHostEffect> {
        let TurnEvent::ToolCallCompleted { name, output, .. } = event else {
            return Vec::new();
        };
        if name != "plan_exit" || !output.is_success() {
            return Vec::new();
        }
        let result = output.value_for_projection();
        let approved = result
            .get("approved")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if approved {
            let mut effects = vec![
                TuiHostEffect::ClearModeIndicator {
                    key: global_surface_id("plan_mode", "mode"),
                },
                TuiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string(),
                },
            ];
            if result
                .get("execution_mode")
                .and_then(|value| value.as_str())
                == Some("current_session")
                && let Some(input) = result
                    .get("next_turn_input")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.trim().is_empty())
            {
                let display_text = result
                    .get("confirmation_display")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or(input);
                effects.push(TuiHostEffect::QueuePreparedTurn {
                    display_text: display_text.to_string(),
                    effective_text: input.to_string(),
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
    use lash_tui::{Rect, Style};
    use serde_json::json;

    fn completed_tool_event(name: &str, result: serde_json::Value) -> TurnEvent {
        TurnEvent::ToolCallCompleted {
            call_id: None,
            name: name.to_string(),
            args: json!({}),
            output: lash::tools::ToolCallOutput::success(result),
            duration_ms: 12,
        }
    }

    #[test]
    fn builtin_extensions_register_plan_command_and_shortcut() {
        let extensions = TuiExtensions::builtin().expect("builtin extensions");

        assert!(extensions.parse_command("/plan").is_some());
        assert!(extensions.shortcut_for(KeyChord::SHIFT_TAB).is_some());
        assert_eq!(
            extensions.completions("/pl"),
            vec![(
                "/plan".to_string(),
                "Open file-backed planning mode".to_string(),
            )]
        );
    }

    #[test]
    fn plan_mode_effects_show_panel_when_enabled() {
        assert_eq!(
            plan_mode_effects(&PlanModeStatus {
                enabled: true,
                plan_path: Some(".lash/plans/root.md".to_string()),
            }),
            vec![
                TuiHostEffect::UpsertModeIndicator {
                    key: "plan_mode:mode".to_string(),
                    label: "plan".to_string(),
                },
                TuiHostEffect::UpsertPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string(),
                    title: "PLAN".to_string(),
                    content: "Path: `.lash/plans/root.md`".to_string(),
                }
            ]
        );
    }

    #[test]
    fn plan_mode_effects_clear_panel_when_disabled() {
        assert_eq!(
            plan_mode_effects(&PlanModeStatus {
                enabled: false,
                plan_path: Some(".lash/plans/root.md".to_string()),
            }),
            vec![
                TuiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string(),
                },
                TuiHostEffect::ClearPanel {
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
        impl TuiExtension for Duplicate {
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
                _ctx: TuiExtensionContext<'_>,
            ) -> Result<Vec<TuiHostEffect>, String> {
                Ok(Vec::new())
            }
        }

        let err = TuiExtensions::new(vec![Arc::new(PlanModeTuiExtension), Arc::new(Duplicate)])
            .err()
            .expect("duplicate commands should fail");
        assert!(err.contains("/plan"));
    }

    #[test]
    fn plan_exit_event_queues_current_session_follow_up() {
        let extensions = TuiExtensions::builtin().expect("builtin extensions");

        let effects = extensions.effects_for_turn_event(&completed_tool_event(
            "plan_exit",
            json!({
                "approved": true,
                "execution_mode": "current_session",
                "confirmation_display": "Start implementing now",
                "next_turn_input": "Execute the approved plan."
            }),
        ));

        assert_eq!(
            effects,
            vec![
                TuiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string()
                },
                TuiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string()
                },
                TuiHostEffect::QueuePreparedTurn {
                    display_text: "Start implementing now".to_string(),
                    effective_text: "Execute the approved plan.".to_string()
                }
            ]
        );
    }

    #[test]
    fn plan_exit_event_clears_ui_for_fresh_context_frame_switch() {
        let extensions = TuiExtensions::builtin().expect("builtin extensions");

        let effects = extensions.effects_for_turn_event(&completed_tool_event(
            "plan_exit",
            json!({
                "approved": true,
                "execution_mode": "fresh_context",
                "session_id": "new-plan-session",
            }),
        ));

        assert_eq!(
            effects,
            vec![
                TuiHostEffect::ClearModeIndicator {
                    key: "plan_mode:mode".to_string()
                },
                TuiHostEffect::ClearPanel {
                    plugin_id: "plan_mode".to_string(),
                    key: "panel".to_string()
                },
            ]
        );
    }

    #[test]
    fn command_tokens_and_shortcuts_are_unique() {
        let extensions = TuiExtensions::builtin().expect("builtin extensions");
        let command_specs = extensions.command_specs();
        let command_names = command_specs
            .iter()
            .map(|spec| spec.name.to_string())
            .collect::<BTreeSet<_>>();
        let shortcut_specs = extensions.shortcut_specs();
        let shortcuts = shortcut_specs
            .iter()
            .map(|spec| spec.chord)
            .collect::<BTreeSet<_>>();

        assert_eq!(command_names.len(), command_specs.len());
        assert_eq!(shortcuts.len(), shortcut_specs.len());
    }

    #[derive(Default)]
    struct SurfaceHarness {
        renders: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl TuiExtension for SurfaceHarness {
        fn id(&self) -> &'static str {
            "surface_harness"
        }

        async fn invoke_action(
            &self,
            _action: &str,
            _arg: Option<&str>,
            _ctx: TuiExtensionContext<'_>,
        ) -> Result<Vec<TuiHostEffect>, String> {
            Ok(Vec::new())
        }

        fn render_surface(
            &self,
            surface_key: &str,
            _ctx: TuiRenderContext<'_>,
            frame: &mut Frame<'_>,
        ) {
            self.renders
                .lock()
                .expect("renders lock poisoned")
                .push(surface_key.to_string());
            frame.write_text(0, 0, surface_key, Style::default(), frame.area().width);
        }

        fn handle_turn_event(&self, event: &TurnEvent) -> Vec<TuiHostEffect> {
            match event {
                TurnEvent::AssistantProseDelta { text } if text == "mount" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "workspace".to_string(),
                            slot: TuiSurfaceSlot::Workspace,
                            size: TuiSurfaceSize::Auto,
                            order: 0,
                            focusable: true,
                            visible: true,
                            modal: false,
                        },
                    },
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "footer".to_string(),
                            slot: TuiSurfaceSlot::Footer,
                            size: TuiSurfaceSize::Lines(1),
                            order: 0,
                            focusable: false,
                            visible: true,
                            modal: false,
                        },
                    },
                    TuiHostEffect::FocusSurface {
                        key: "workspace".to_string(),
                    },
                ],
                TurnEvent::AssistantProseDelta { text } if text == "overlay" => vec![
                    TuiHostEffect::MountSurface {
                        spec: TuiSurfaceSpec {
                            key: "overlay".to_string(),
                            slot: TuiSurfaceSlot::Overlay,
                            size: TuiSurfaceSize::Fixed {
                                width: 12,
                                height: 3,
                            },
                            order: 10,
                            focusable: true,
                            visible: true,
                            modal: true,
                        },
                    },
                    TuiHostEffect::FocusSurface {
                        key: "overlay".to_string(),
                    },
                ],
                TurnEvent::AssistantProseDelta { text } if text == "close" => vec![
                    TuiHostEffect::BlurSurface {
                        key: "overlay".to_string(),
                    },
                    TuiHostEffect::UnmountSurface {
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
        let extensions = TuiExtensions::new(vec![harness]).expect("surface harness");

        extensions.effects_for_turn_event(&TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:workspace")
        );
        assert_eq!(
            extensions.mounted_surfaces(TuiSurfaceSlot::Workspace).len(),
            1
        );
        assert_eq!(extensions.mounted_surfaces(TuiSurfaceSlot::Footer).len(), 1);

        extensions.effects_for_turn_event(&TurnEvent::AssistantProseDelta {
            text: "overlay".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:overlay")
        );

        extensions.effects_for_turn_event(&TurnEvent::AssistantProseDelta {
            text: "close".to_string(),
        });
        assert_eq!(
            extensions.focused_surface().as_deref(),
            Some("surface_harness:workspace")
        );
    }

    #[test]
    fn render_surface_dispatches_to_owner_extension() {
        let harness = Arc::new(SurfaceHarness::default());
        let extensions = TuiExtensions::new(vec![Arc::clone(&harness) as Arc<dyn TuiExtension>])
            .expect("surface harness");
        extensions.effects_for_turn_event(&TurnEvent::AssistantProseDelta {
            text: "mount".to_string(),
        });
        let workspace = extensions
            .mounted_surfaces(TuiSurfaceSlot::Workspace)
            .into_iter()
            .next()
            .expect("workspace surface");

        let snapshot = lash_tui::render_snapshot(20, 4, |frame| {
            let area = Rect::new(0, 0, 20, 4);
            extensions.set_surface_area(&workspace.id, Some(area));
            let mut viewport = frame.viewport(area);
            extensions.render_surface(
                &workspace.id,
                TuiRenderContext {
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
