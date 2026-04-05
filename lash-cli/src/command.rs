use lash::SkillCatalog;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub usage: &'static str,
    pub description: &'static str,
    pub takes_argument: bool,
}

/// Builtin slash-command catalog used for parse, autocomplete, and help.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/clear",
        aliases: &["/new"],
        usage: "/clear",
        description: "Reset conversation",
        takes_argument: false,
    },
    CommandSpec {
        name: "/controls",
        aliases: &[],
        usage: "/controls",
        description: "Show keyboard shortcuts",
        takes_argument: false,
    },
    CommandSpec {
        name: "/fork",
        aliases: &[],
        usage: "/fork [prompt]",
        description: "Open a forked session in a new terminal",
        takes_argument: true,
    },
    CommandSpec {
        name: "/version",
        aliases: &[],
        usage: "/version",
        description: "Show lash-cli and lash-sansio versions",
        takes_argument: false,
    },
    CommandSpec {
        name: "/info",
        aliases: &[],
        usage: "/info",
        description: "Show current session/runtime info",
        takes_argument: false,
    },
    CommandSpec {
        name: "/model",
        aliases: &[],
        usage: "/model [name]",
        description: "Show or switch LLM model",
        takes_argument: true,
    },
    CommandSpec {
        name: "/variant",
        aliases: &[],
        usage: "/variant [name]",
        description: "Show or switch model variant",
        takes_argument: true,
    },
    CommandSpec {
        name: "/mode",
        aliases: &[],
        usage: "/mode [name]",
        description: "Show current execution mode",
        takes_argument: true,
    },
    CommandSpec {
        name: "/provider",
        aliases: &["/login"],
        usage: "/provider",
        description: "Switch, add, or re-authenticate providers",
        takes_argument: false,
    },
    CommandSpec {
        name: "/logout",
        aliases: &[],
        usage: "/logout",
        description: "Remove stored credentials for active provider",
        takes_argument: false,
    },
    CommandSpec {
        name: "/retry",
        aliases: &[],
        usage: "/retry",
        description: "Replay the previous turn payload",
        takes_argument: false,
    },
    CommandSpec {
        name: "/resume",
        aliases: &["/continue"],
        usage: "/resume [name]",
        description: "Browse or load a previous session",
        takes_argument: true,
    },
    CommandSpec {
        name: "/skills",
        aliases: &[],
        usage: "/skills",
        description: "Browse loaded skills",
        takes_argument: false,
    },
    CommandSpec {
        name: "/tools",
        aliases: &[],
        usage: "/tools ...",
        description: "Inspect or edit dynamic tools",
        takes_argument: true,
    },
    CommandSpec {
        name: "/reconfigure",
        aliases: &[],
        usage: "/reconfigure ...",
        description: "Apply or inspect pending runtime reconfigure",
        takes_argument: true,
    },
    CommandSpec {
        name: "/help",
        aliases: &["/?"],
        usage: "/help",
        description: "Show commands and shortcuts",
        takes_argument: false,
    },
    CommandSpec {
        name: "/exit",
        aliases: &["/quit"],
        usage: "/exit",
        description: "Quit",
        takes_argument: false,
    },
];

pub fn catalog() -> &'static [CommandSpec] {
    COMMANDS
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

/// Whether accepting autocomplete should append a trailing space.
pub fn completion_inserts_space(cmd: &str, skills: &SkillCatalog) -> bool {
    if let Some(spec) = COMMANDS.iter().find(|spec| spec.name == cmd) {
        return spec.takes_argument;
    }
    slash_skill_prompt(cmd, skills).is_some()
}

pub fn runs_out_of_band_while_running(cmd: &Command) -> bool {
    matches!(cmd, Command::Fork(_))
}

/// Slash commands recognized by the TUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Clear,
    Controls,
    Fork(Option<String>),
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
        "controls" => Some(Command::Controls),
        "fork" => Some(Command::Fork(
            arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        )),
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
    use std::path::PathBuf;

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

    #[test]
    fn parses_all_primary_commands() {
        let skills = SkillCatalog::load();
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
        let skills = SkillCatalog::load();
        assert!(matches!(parse("/new", &skills), Some(Command::Clear)));
        assert!(matches!(parse("/fork", &skills), Some(Command::Fork(None))));
        assert!(matches!(
            parse("/fork draft a reply", &skills),
            Some(Command::Fork(Some(_)))
        ));
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
            parse("/reconfigure apply", &skills),
            Some(Command::Reconfigure(Some(_)))
        ));
        assert!(parse("/not-a-command", &skills).is_none());
    }

    #[test]
    fn completion_spacing_matches_argument_commands() {
        let skills = SkillCatalog::load();
        for cmd in [
            "/fork",
            "/model",
            "/variant",
            "/mode",
            "/resume",
            "/tools",
            "/reconfigure",
        ] {
            assert!(completion_inserts_space(cmd, &skills));
        }

        for cmd in ["/clear", "/skills", "/help", "/exit"] {
            assert!(!completion_inserts_space(cmd, &skills));
        }
    }

    #[test]
    fn fork_runs_out_of_band_while_running() {
        assert!(runs_out_of_band_while_running(&Command::Fork(None)));
        assert!(!runs_out_of_band_while_running(&Command::Help));
        assert!(!runs_out_of_band_while_running(&Command::Model(None)));
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
}
