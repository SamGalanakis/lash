use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use lash::{PluginHost, SessionEvent, SessionManager, ToolResult};

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
    UpsertModeIndicator { key: String, label: String },
    ClearModeIndicator { key: String },
    QueueTurn { input: String },
}

pub struct UiContext<'a> {
    pub plugin_host: &'a PluginHost,
    pub session_id: &'a str,
    pub session_manager: Arc<dyn SessionManager>,
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
    commands: Vec<RegisteredCommand>,
    shortcuts: Vec<RegisteredShortcut>,
}

impl UiExtensions {
    pub fn new(extensions: Vec<Arc<dyn UiExtension>>) -> Result<Self, String> {
        let mut command_names = BTreeMap::<String, &'static str>::new();
        let mut shortcuts = BTreeMap::<KeyChord, &'static str>::new();
        let mut registered_commands = Vec::new();
        let mut registered_shortcuts = Vec::new();

        for extension in &extensions {
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
            commands: registered_commands,
            shortcuts: registered_shortcuts,
        })
    }

    pub fn builtin() -> Result<Self, String> {
        Self::new(vec![Arc::new(PlanModeUiExtension)])
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
            effects.extend(
                extension
                    .sync(UiContext {
                        plugin_host: ctx.plugin_host,
                        session_id: ctx.session_id,
                        session_manager: Arc::clone(&ctx.session_manager),
                    })
                    .await?,
            );
        }
        Ok(effects)
    }

    pub async fn invoke_command(
        &self,
        invocation: &UiCommandInvocation,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        invocation
            .extension
            .invoke_action(invocation.spec.action, invocation.arg.as_deref(), ctx)
            .await
    }

    pub async fn invoke_shortcut(
        &self,
        invocation: &UiShortcutInvocation,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        invocation
            .extension
            .invoke_action(invocation.spec.action, None, ctx)
            .await
    }

    pub fn effects_for_session_event(&self, event: &SessionEvent) -> Vec<UiHostEffect> {
        let mut effects = Vec::new();
        for extension in &self.extensions {
            effects.extend(extension.handle_session_event(event));
        }
        effects
    }
}

fn command_matches(name: &str, spec: SlashCommandSpec) -> bool {
    name == spec.name || spec.aliases.contains(&name)
}

fn surface_key(plugin_id: &str, key: &str) -> String {
    format!("{plugin_id}:{key}")
}

fn bool_from_result(result: ToolResult, op_name: &str) -> Result<bool, String> {
    if !result.success {
        return Err(format!("{op_name} failed: {}", result.result));
    }
    result
        .result
        .get("enabled")
        .and_then(|value| value.as_bool())
        .ok_or_else(|| format!("{op_name} response missing `enabled`"))
}

async fn plan_mode_enabled(ctx: UiContext<'_>, op_name: &str) -> Result<bool, String> {
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
    bool_from_result(result, op_name)
}

fn plan_mode_indicator_effects(enabled: bool) -> Vec<UiHostEffect> {
    let key = surface_key("plan_mode", "mode");
    if enabled {
        vec![UiHostEffect::UpsertModeIndicator {
            key,
            label: "plan".to_string(),
        }]
    } else {
        vec![UiHostEffect::ClearModeIndicator { key }]
    }
}

struct PlanModeUiExtension;

const PLAN_MODE_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
    name: "/plan",
    aliases: &[],
    usage: "/plan",
    description: "Toggle persistent plan mode",
    takes_argument: false,
    allow_while_running: true,
    action: "toggle",
}];

const PLAN_MODE_SHORTCUTS: &[ShortcutSpec] = &[ShortcutSpec {
    chord: KeyChord::SHIFT_TAB,
    description: "Toggle persistent plan mode",
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
        let enabled = plan_mode_enabled(ctx, "plan_mode.status")
            .await
            .map_err(|err| format!("failed to sync plan mode: {err}"))?;
        Ok(plan_mode_indicator_effects(enabled))
    }

    async fn invoke_action(
        &self,
        action: &str,
        _arg: Option<&str>,
        ctx: UiContext<'_>,
    ) -> Result<Vec<UiHostEffect>, String> {
        match action {
            "toggle" => {
                let enabled = plan_mode_enabled(ctx, "plan_mode.toggle")
                    .await
                    .map_err(|err| format!("failed to toggle plan mode: {err}"))?;
                let mut effects = plan_mode_indicator_effects(enabled);
                effects.push(UiHostEffect::PushSystemMessage(if enabled {
                    "Plan mode enabled.".to_string()
                } else {
                    "Plan mode disabled.".to_string()
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
        let next_turn_input = result
            .get("next_turn_input")
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty());
        match (approved, next_turn_input) {
            (true, Some(input)) => vec![
                UiHostEffect::ClearModeIndicator {
                    key: surface_key("plan_mode", "mode"),
                },
                UiHostEffect::QueueTurn {
                    input: input.to_string(),
                },
            ],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use lash::SessionEvent;
    use serde_json::json;

    #[test]
    fn builtin_extensions_register_plan_command_and_shortcut() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");

        assert!(extensions.parse_command("/plan").is_some());
        assert!(extensions.shortcut_for(KeyChord::SHIFT_TAB).is_some());
        assert_eq!(
            extensions.completions("/pl"),
            vec![(
                "/plan".to_string(),
                "Toggle persistent plan mode".to_string()
            )]
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
    fn plan_exit_event_queues_follow_up_turn() {
        let extensions = UiExtensions::builtin().expect("builtin extensions");

        let effects = extensions.effects_for_session_event(&SessionEvent::ToolCall {
            call_id: None,
            name: "plan_exit".to_string(),
            args: json!({}),
            result: json!({
                "approved": true,
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
                UiHostEffect::QueueTurn {
                    input: "Execute the approved plan.".to_string()
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
}
