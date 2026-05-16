use crate::SkillCatalog;
use lash_tui_extensions::TuiExtensions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub usage: &'static str,
    pub description: &'static str,
    pub argument_hint: Option<&'static str>,
    pub argument_options: &'static [&'static str],
    pub takes_argument: bool,
    /// When true, the dispatch loop may invoke this command while a
    /// turn is streaming instead of queueing it. Reserved for handlers
    /// that don't mutate runtime / dynamic-tools / provider state.
    pub runs_out_of_band: bool,
}

/// Builtin slash-command catalog used for parse, autocomplete, and help.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/clear",
        aliases: &["/new"],
        usage: "/clear",
        description: "Reset conversation",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/compact",
        aliases: &[],
        usage: "/compact [focus instructions]",
        description: "Summarize older messages to free up context",
        argument_hint: Some("[focus instructions]"),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/controls",
        aliases: &[],
        usage: "/controls",
        description: "Show keyboard shortcuts",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/fork",
        aliases: &[],
        usage: "/fork",
        description: "Open a forked session in a new terminal",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/tree",
        aliases: &[],
        usage: "/tree",
        description: "Browse and switch branches in the current session",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/version",
        aliases: &[],
        usage: "/version",
        description: "Show lash-cli and lash-sansio versions",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/info",
        aliases: &[],
        usage: "/info",
        description: "Show current session/runtime info",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/model",
        aliases: &[],
        usage: "/model [name]",
        description: "Show or switch LLM model",
        argument_hint: Some("[name]"),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/variant",
        aliases: &[],
        usage: "/variant [name]",
        description: "Show or switch model variant",
        argument_hint: Some("[name]"),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/mode",
        aliases: &[],
        usage: "/mode [name]",
        description: "Show current execution mode",
        argument_hint: Some("[name]"),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/provider",
        aliases: &["/login"],
        usage: "/provider",
        description: "Switch, add, or re-authenticate providers",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/logout",
        aliases: &[],
        usage: "/logout",
        description: "Remove stored credentials for active provider",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/retry",
        aliases: &[],
        usage: "/retry",
        description: "Replay the previous turn payload",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/resume",
        aliases: &["/continue"],
        usage: "/resume [name]",
        description: "Browse or load a previous session",
        argument_hint: Some("[name]"),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/skills",
        aliases: &[],
        usage: "/skills",
        description: "Browse loaded skills",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/tools",
        aliases: &[],
        usage: "/tools ...",
        description: "Inspect or edit tool registry",
        argument_hint: Some("..."),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/reconfigure",
        aliases: &[],
        usage: "/reconfigure ...",
        description: "Apply or inspect pending runtime reconfigure",
        argument_hint: Some("..."),
        argument_options: &[],
        takes_argument: true,
        runs_out_of_band: false,
    },
    CommandSpec {
        name: "/help",
        aliases: &["/?"],
        usage: "/help",
        description: "Show commands and shortcuts",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
    CommandSpec {
        name: "/exit",
        aliases: &["/quit"],
        usage: "/exit",
        description: "Quit",
        argument_hint: None,
        argument_options: &[],
        takes_argument: false,
        runs_out_of_band: true,
    },
];

pub fn catalog() -> &'static [CommandSpec] {
    COMMANDS
}

fn builtin_command_spec(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS
        .iter()
        .find(|spec| spec.name == name || spec.aliases.contains(&name))
}

/// Return commands matching the given prefix.
pub fn completions(prefix: &str, skills: &SkillCatalog) -> Vec<(String, String)> {
    let mut results = COMMANDS
        .iter()
        .filter(|spec| spec.name.starts_with(prefix))
        .map(|spec| (spec.name.to_string(), spec.description.to_string()))
        .collect::<Vec<_>>();

    if prefix.starts_with('/') {
        for skill in skills.iter() {
            let cmd = format!("/{}", skill.name);
            if cmd.starts_with(prefix) && !results.iter().any(|(existing, _)| existing == &cmd) {
                let desc = if skill.description.is_empty() {
                    "Invoke skill".to_string()
                } else {
                    format!("Invoke skill: {}", skill.description)
                };
                results.push((cmd, desc));
            }
        }
    }

    results
}

pub fn argument_hint(
    name: &str,
    skills: &SkillCatalog,
    ui_extensions: &TuiExtensions,
) -> Option<String> {
    if let Some(spec) = builtin_command_spec(name) {
        return spec.argument_hint.map(str::to_string);
    }
    if let Some(spec) = ui_extensions.command_spec(name) {
        return spec.argument_hint.map(str::to_string);
    }
    let skill_name = name.strip_prefix('/')?;
    skills.argument_hint(skill_name).map(str::to_string)
}

pub fn argument_completions(
    name: &str,
    prefix: &str,
    skills: &SkillCatalog,
    ui_extensions: &TuiExtensions,
) -> Vec<(String, String)> {
    let (options, description) = if let Some(spec) = builtin_command_spec(name) {
        (spec.argument_options, spec.description)
    } else if let Some(spec) = ui_extensions.command_spec(name) {
        (spec.argument_options, spec.description)
    } else if let Some(skill_name) = name.strip_prefix('/') {
        let options = skills.argument_options(skill_name);
        return options
            .iter()
            .filter(|option| option.starts_with(prefix))
            .map(|option| (option.clone(), format!("Argument for {name}")))
            .collect();
    } else {
        return Vec::new();
    };

    options
        .iter()
        .filter(|option| option.starts_with(prefix))
        .map(|option| ((*option).to_string(), description.to_string()))
        .collect()
}

/// Whether accepting autocomplete should append a trailing space.
pub fn completion_inserts_space(cmd: &str, skills: &SkillCatalog) -> bool {
    if let Some(spec) = builtin_command_spec(cmd) {
        return spec.takes_argument;
    }
    slash_skill_prompt(cmd, skills).is_some()
}

/// Whether the dispatch loop is allowed to fire `cmd` while a turn is
/// streaming (instead of queueing it).
pub fn runs_out_of_band_while_running(cmd: &Command) -> bool {
    let name = match cmd {
        Command::Clear => "/clear",
        Command::Compact(_) => "/compact",
        Command::Controls => "/controls",
        Command::Fork => "/fork",
        Command::Tree => "/tree",
        Command::Version => "/version",
        Command::Info => "/info",
        Command::Model(_) => "/model",
        Command::Variant(_) => "/variant",
        Command::Mode(_) => "/mode",
        Command::ChangeProvider => "/provider",
        Command::Logout => "/logout",
        Command::Retry => "/retry",
        Command::Help => "/help",
        Command::Exit => "/exit",
        Command::Resume(_) => "/resume",
        Command::Skills => "/skills",
        Command::Tools(_) => "/tools",
        Command::Reconfigure(_) => "/reconfigure",
    };
    COMMANDS
        .iter()
        .find(|spec| spec.name == name)
        .is_some_and(|spec| spec.runs_out_of_band)
}

/// Slash commands recognized by the TUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Clear,
    Compact(Option<String>),
    Controls,
    Fork,
    Tree,
    Version,
    Info,
    Model(Option<String>),
    Variant(Option<String>),
    Mode(Option<String>),
    ChangeProvider,
    Logout,
    Retry,
    Help,
    Exit,
    Resume(Option<String>),
    Skills,
    Tools(Option<String>),
    Reconfigure(Option<String>),
}

pub fn slash_skill_prompt(input: &str, skills: &SkillCatalog) -> Option<String> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let rest = &trimmed[1..];
    let (cmd, arg) = match rest.split_once(' ') {
        Some((c, a)) => (c, Some(a.trim())),
        None => (rest, None),
    };
    skills.get(cmd)?;
    Some(match arg.filter(|a| !a.is_empty()) {
        Some(arg) => format!("/{cmd} {arg}"),
        None => format!("/{cmd}"),
    })
}

/// Try to parse a slash command from user input.
pub fn parse(input: &str, _skills: &SkillCatalog) -> Option<Command> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let rest = &trimmed[1..];
    let (cmd, arg) = match rest.split_once(' ') {
        Some((c, a)) => (c, Some(a.trim())),
        None => (rest, None),
    };
    match cmd {
        "clear" | "new" => Some(Command::Clear),
        "compact" => Some(Command::Compact(
            arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        )),
        "controls" => Some(Command::Controls),
        "fork" => Some(Command::Fork),
        "tree" => Some(Command::Tree),
        "version" => Some(Command::Version),
        "info" => Some(Command::Info),
        "model" => Some(Command::Model(
            arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        )),
        "variant" => Some(Command::Variant(
            arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        )),
        "mode" => Some(Command::Mode(
            arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        )),
        "provider" | "login" => Some(Command::ChangeProvider),
        "logout" => Some(Command::Logout),
        "retry" => Some(Command::Retry),
        "help" | "?" => Some(Command::Help),
        "exit" | "quit" => Some(Command::Exit),
        "resume" | "continue" => Some(Command::Resume(arg.map(|a| a.to_string()))),
        "skills" => Some(Command::Skills),
        "tools" => Some(Command::Tools(arg.map(|a| a.to_string()))),
        "reconfigure" => Some(Command::Reconfigure(arg.map(|a| a.to_string()))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lash_tui_extensions::{
        SlashCommandSpec, TuiExtension, TuiExtensionContext, TuiExtensions, TuiHostEffect,
    };
    use std::path::PathBuf;
    use std::sync::Arc;

    fn skill_catalog_with(names: &[(&str, &str)]) -> SkillCatalog {
        let root =
            std::env::temp_dir().join(format!("lash-command-skills-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        for (name, description) in names {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("skill dir");
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {description}\n---\n\nbody\n"),
            )
            .expect("skill file");
        }
        let catalog = SkillCatalog::from_dirs(&[PathBuf::from(&root)]);
        let _ = std::fs::remove_dir_all(root);
        catalog
    }

    fn skill_catalog_with_hints(entries: &[(&str, &str, Option<&str>)]) -> SkillCatalog {
        let root = std::env::temp_dir().join(format!(
            "lash-command-skills-hints-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).expect("temp root");
        for (name, description, argument_hint) in entries {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("skill dir");
            let hint_line = argument_hint
                .map(|hint| format!("argument-hint: \"{hint}\"\n"))
                .unwrap_or_default();
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {description}\n{hint_line}---\n\nbody\n"),
            )
            .expect("skill file");
        }
        let catalog = SkillCatalog::from_dirs(&[PathBuf::from(&root)]);
        let _ = std::fs::remove_dir_all(root);
        catalog
    }

    #[test]
    fn parses_all_primary_commands() {
        let skills = SkillCatalog::from_dirs(&crate::paths::default_skill_dirs());
        for spec in COMMANDS {
            assert!(
                parse(spec.name, &skills).is_some(),
                "displayed command should parse: {}",
                spec.name
            );
        }
    }

    #[test]
    fn parses_aliases_and_arguments() {
        let skills = SkillCatalog::from_dirs(&crate::paths::default_skill_dirs());
        assert!(matches!(parse("/new", &skills), Some(Command::Clear)));
        assert!(matches!(parse("/fork", &skills), Some(Command::Fork)));
        assert!(matches!(
            parse("/fork draft a reply", &skills),
            Some(Command::Fork)
        ));
        assert!(matches!(parse("/tree", &skills), Some(Command::Tree)));
        assert!(matches!(
            parse("/provider", &skills),
            Some(Command::ChangeProvider)
        ));
        assert!(matches!(
            parse("/login", &skills),
            Some(Command::ChangeProvider)
        ));
        assert!(matches!(parse("/retry", &skills), Some(Command::Retry)));
        assert!(matches!(parse("/version", &skills), Some(Command::Version)));
        assert!(matches!(parse("/info", &skills), Some(Command::Info)));
        assert!(matches!(parse("/quit", &skills), Some(Command::Exit)));
        assert!(matches!(parse("/?", &skills), Some(Command::Help)));
        assert!(matches!(
            parse("/model gpt-5.4", &skills),
            Some(Command::Model(Some(_)))
        ));
        assert!(matches!(
            parse("/variant high", &skills),
            Some(Command::Variant(Some(_)))
        ));
        assert!(matches!(
            parse("/mode standard", &skills),
            Some(Command::Mode(Some(_)))
        ));
        assert!(matches!(
            parse("/resume", &skills),
            Some(Command::Resume(None))
        ));
        assert!(matches!(parse("/tools", &skills), Some(Command::Tools(_))));
        assert!(matches!(
            parse("/compact focus on X", &skills),
            Some(Command::Compact(Some(ref arg))) if arg == "focus on X"
        ));
        assert!(matches!(
            parse("/reconfigure apply", &skills),
            Some(Command::Reconfigure(Some(_)))
        ));
        assert!(parse("/not-a-command", &skills).is_none());
    }

    #[test]
    fn completion_spacing_matches_argument_commands() {
        let skills = SkillCatalog::from_dirs(&crate::paths::default_skill_dirs());
        for cmd in [
            "/compact",
            "/model",
            "/variant",
            "/mode",
            "/resume",
            "/tools",
            "/reconfigure",
        ] {
            assert!(completion_inserts_space(cmd, &skills));
        }

        for cmd in ["/clear", "/fork", "/tree", "/skills", "/help", "/exit"] {
            assert!(!completion_inserts_space(cmd, &skills));
        }
    }

    #[test]
    fn read_only_commands_run_out_of_band() {
        for cmd in [
            Command::Fork,
            Command::Help,
            Command::Version,
            Command::Info,
            Command::Skills,
            Command::Controls,
            Command::Exit,
        ] {
            assert!(
                runs_out_of_band_while_running(&cmd),
                "expected {:?} to run out-of-band",
                cmd
            );
        }
    }

    #[test]
    fn mutating_commands_must_wait_while_running() {
        for cmd in [
            Command::Clear,
            Command::Compact(None),
            Command::Tree,
            Command::Model(None),
            Command::Variant(None),
            Command::Mode(None),
            Command::Resume(None),
            Command::Tools(None),
            Command::Reconfigure(None),
            Command::ChangeProvider,
            Command::Logout,
            Command::Retry,
        ] {
            assert!(
                !runs_out_of_band_while_running(&cmd),
                "expected {:?} to wait until the turn finishes",
                cmd
            );
        }
    }

    #[test]
    fn completions_include_matching_skills() {
        let skills =
            skill_catalog_with(&[("yolopush", "ship changes"), ("spring-cleaning", "cleanup")]);
        let results = completions("/s", &skills);
        assert!(results.iter().any(|(cmd, _)| cmd == "/skills"));
        assert!(results.iter().any(|(cmd, _)| cmd == "/spring-cleaning"));
        assert!(!results.iter().any(|(cmd, _)| cmd == "/yolopush"));
    }

    #[test]
    fn completions_include_compact_builtin() {
        let skills = SkillCatalog::from_dirs(&crate::paths::default_skill_dirs());
        let results = completions("/c", &skills);
        assert!(results.iter().any(|(cmd, _)| cmd == "/compact"));
        assert!(results.iter().any(|(cmd, _)| cmd == "/clear"));
    }

    #[test]
    fn slash_skill_prompts_preserve_slash_mentions() {
        let skills = skill_catalog_with(&[("yolopush", "ship changes")]);
        assert_eq!(
            slash_skill_prompt("/yolopush", &skills).as_deref(),
            Some("/yolopush")
        );
        assert_eq!(
            slash_skill_prompt("/yolopush merge staging", &skills).as_deref(),
            Some("/yolopush merge staging")
        );
        assert!(slash_skill_prompt("/skills", &skills).is_none());
    }

    #[test]
    fn argument_completions_include_ui_command_options() {
        struct DemoTuiExtension;

        const DEMO_COMMANDS: &[SlashCommandSpec] = &[SlashCommandSpec {
            name: "/demo",
            aliases: &[],
            usage: "/demo [alpha|beta]",
            description: "Demo command",
            argument_hint: Some("[alpha|beta]"),
            argument_options: &["alpha", "beta"],
            takes_argument: true,
            allow_while_running: true,
            action: "demo",
        }];

        #[async_trait]
        impl TuiExtension for DemoTuiExtension {
            fn id(&self) -> &'static str {
                "demo_ui"
            }

            fn commands(&self) -> &'static [SlashCommandSpec] {
                DEMO_COMMANDS
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

        let skills = SkillCatalog::default();
        let ui_extensions = TuiExtensions::new(vec![Arc::new(DemoTuiExtension)]).expect("ui");
        let results = argument_completions("/demo", "a", &skills, &ui_extensions);
        assert_eq!(
            results,
            vec![("alpha".to_string(), "Demo command".to_string())]
        );
        assert_eq!(
            argument_hint("/demo", &skills, &ui_extensions).as_deref(),
            Some("[alpha|beta]")
        );
    }

    #[test]
    fn argument_completions_include_skill_options_from_hint() {
        let skills = skill_catalog_with_hints(&[(
            "impeccable",
            "design helper",
            Some("[craft|teach|extract]"),
        )]);
        let ui_extensions = TuiExtensions::builtin().expect("builtin ui");
        let results = argument_completions("/impeccable", "te", &skills, &ui_extensions);
        assert_eq!(
            results,
            vec![("teach".to_string(), "Argument for /impeccable".to_string())]
        );
    }
}
