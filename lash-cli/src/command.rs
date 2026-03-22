use lash::SkillCatalog;

/// Primary commands shown in autocomplete (command, description).
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "Reset conversation"),
    ("/controls", "Show keyboard shortcuts"),
    ("/fork", "Open a forked session in a new terminal"),
    ("/version", "Show Lash and lash-sansio versions"),
    ("/info", "Show current session/runtime info"),
    ("/model", "Show or switch LLM model"),
    ("/variant", "Show or switch model variant"),
    ("/mode", "Show current execution mode"),
    ("/provider", "Switch, add, or re-authenticate providers"),
    ("/login", "Sign in or reconfigure provider"),
    ("/logout", "Remove stored credentials for active provider"),
    ("/retry", "Replay the previous turn payload"),
    ("/resume", "Resume a previous session"),
    ("/skills", "Browse loaded skills"),
    ("/tools", "Inspect or edit dynamic tools"),
    (
        "/reconfigure",
        "Apply or inspect pending runtime reconfigure",
    ),
    ("/help", "Show commands and shortcuts"),
    ("/exit", "Quit"),
];

/// Return commands matching the given prefix.
pub fn completions(prefix: &str, skills: &SkillCatalog) -> Vec<(String, String)> {
    let mut results = COMMANDS
        .iter()
        .filter(|(cmd, _)| cmd.starts_with(prefix))
        .map(|(cmd, desc)| (cmd.to_string(), desc.to_string()))
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
    if matches!(
        cmd,
        "/fork" | "/model" | "/variant" | "/mode" | "/resume" | "/tools" | "/reconfigure"
    ) {
        return true;
    }
    if COMMANDS.iter().any(|(builtin, _)| *builtin == cmd) {
        return false;
    }
    slash_skill_prompt(cmd, skills).is_some()
}

/// Slash commands recognized by the TUI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Reset conversation to splash screen
    Clear,
    /// Show keyboard shortcuts
    Controls,
    /// Fork the current session into a new terminal (optional initial prompt)
    Fork(Option<String>),
    /// Show Lash and lash-sansio versions
    Version,
    /// Show session/runtime metadata
    Info,
    /// Show or switch LLM model
    Model(Option<String>),
    /// Show or switch the provider-native variant for the active model
    Variant(Option<String>),
    /// Show current execution mode or request a new-session change
    Mode(Option<String>),
    /// Show provider status and switch instructions
    ChangeProvider,
    /// Remove stored credentials for the active provider
    Logout,
    /// Replay the previous turn payload
    Retry,
    /// Show available commands and shortcuts
    Help,
    /// Quit the application
    Exit,
    /// Resume a previous session (optional filename)
    Resume(Option<String>),
    /// Browse loaded skills
    Skills,
    /// Dynamic tool commands (raw args)
    Tools(Option<String>),
    /// Reconfigure control commands (raw args)
    Reconfigure(Option<String>),
}

/// Convert `/skill ...` into the ordinary `$skill ...` prompt form.
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
        Some(arg) => format!("${cmd} {arg}"),
        None => format!("${cmd}"),
    })
}

/// Try to parse a slash command from user input.
/// Returns `None` if the input is not a recognized command.
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
        for (cmd, _) in COMMANDS {
            assert!(
                parse(cmd, &skills).is_some(),
                "displayed command should parse: {cmd}"
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
    fn completions_include_matching_skills() {
        let skills =
            skill_catalog_with(&[("yolopush", "ship changes"), ("spring-cleaning", "cleanup")]);
        let results = completions("/s", &skills);
        assert!(results.iter().any(|(cmd, _)| cmd == "/skills"));
        assert!(results.iter().any(|(cmd, _)| cmd == "/spring-cleaning"));
        assert!(!results.iter().any(|(cmd, _)| cmd == "/yolopush"));
    }

    #[test]
    fn slash_skill_prompts_convert_to_dollar_mentions() {
        let skills = skill_catalog_with(&[("yolopush", "ship changes")]);
        assert_eq!(
            slash_skill_prompt("/yolopush", &skills).as_deref(),
            Some("$yolopush")
        );
        assert_eq!(
            slash_skill_prompt("/yolopush merge staging", &skills).as_deref(),
            Some("$yolopush merge staging")
        );
        assert!(slash_skill_prompt("/skills", &skills).is_none());
    }
}
