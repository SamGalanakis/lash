use crate::skill::SkillRegistry;

/// Primary commands shown in autocomplete (command, description).
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "Reset conversation"),
    ("/controls", "Show keyboard shortcuts"),
    ("/model", "Show or switch LLM model"),
    ("/mode", "Show or switch execution mode"),
    ("/provider", "Open provider setup (in-app)"),
    ("/login", "Sign in or reconfigure provider"),
    ("/logout", "Remove stored credentials"),
    ("/retry", "Replay the previous turn payload"),
    ("/resume", "Resume a previous session"),
    ("/skills", "Browse loaded skills"),
    ("/tools", "Inspect or edit dynamic tools"),
    ("/caps", "Inspect or edit dynamic capabilities"),
    (
        "/reconfigure",
        "Apply or inspect pending runtime reconfigure",
    ),
    ("/help", "Show commands and shortcuts"),
    ("/exit", "Quit"),
];

/// Return commands matching the given prefix, including skills.
pub fn completions(prefix: &str, skills: &SkillRegistry) -> Vec<(String, String)> {
    let mut results: Vec<(String, String)> = COMMANDS
        .iter()
        .filter(|(cmd, _)| cmd.starts_with(prefix))
        .map(|(cmd, desc)| (cmd.to_string(), desc.to_string()))
        .collect();
    results.extend(skills.completions(prefix));
    results
}

/// Whether accepting autocomplete should append a trailing space.
pub fn completion_inserts_space(cmd: &str, skills: &SkillRegistry) -> bool {
    matches!(
        cmd,
        "/model" | "/mode" | "/resume" | "/tools" | "/caps" | "/reconfigure"
    ) || skills.get(cmd.trim_start_matches('/')).is_some()
}

/// Slash commands recognized by the TUI.
pub enum Command {
    /// Reset conversation to splash screen
    Clear,
    /// Show keyboard shortcuts
    Controls,
    /// Show or switch LLM model
    Model(Option<String>),
    /// Show or switch execution mode
    Mode(Option<String>),
    /// Show provider status and switch instructions
    ChangeProvider,
    /// Remove stored credentials
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
    /// Dynamic capability commands (raw args)
    Caps(Option<String>),
    /// Reconfigure control commands (raw args)
    Reconfigure(Option<String>),
    /// Invoke a skill (name, optional args)
    Skill(String, Option<String>),
}

/// Try to parse a slash command from user input.
/// Returns `None` if the input is not a recognized command.
pub fn parse(input: &str, skills: &SkillRegistry) -> Option<Command> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let rest = &trimmed[1..];
    let (cmd, arg) = match rest.split_once(' ') {
        Some((c, a)) => (c, Some(a.trim())),
        None => (rest, None),
    };
    // Built-in commands always win
    match cmd {
        "clear" | "new" => Some(Command::Clear),
        "controls" => Some(Command::Controls),
        "model" => Some(Command::Model(
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
        "caps" => Some(Command::Caps(arg.map(|a| a.to_string()))),
        "reconfigure" => Some(Command::Reconfigure(arg.map(|a| a.to_string()))),
        _ => {
            // Check skills
            if skills.get(cmd).is_some() {
                Some(Command::Skill(
                    cmd.to_string(),
                    arg.filter(|a| !a.is_empty()).map(|a| a.to_string()),
                ))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_primary_commands() {
        let skills = SkillRegistry::load();
        for (cmd, _) in COMMANDS {
            assert!(
                parse(cmd, &skills).is_some(),
                "displayed command should parse: {cmd}"
            );
        }
    }

    #[test]
    fn parses_aliases_and_arguments() {
        let skills = SkillRegistry::load();
        assert!(matches!(parse("/new", &skills), Some(Command::Clear)));
        assert!(matches!(
            parse("/provider", &skills),
            Some(Command::ChangeProvider)
        ));
        assert!(matches!(
            parse("/login", &skills),
            Some(Command::ChangeProvider)
        ));
        assert!(matches!(parse("/retry", &skills), Some(Command::Retry)));
        assert!(matches!(parse("/quit", &skills), Some(Command::Exit)));
        assert!(matches!(parse("/?", &skills), Some(Command::Help)));
        assert!(matches!(
            parse("/model gpt-5.4 high", &skills),
            Some(Command::Model(Some(_)))
        ));
        assert!(matches!(
            parse("/mode native-tools", &skills),
            Some(Command::Mode(Some(_)))
        ));
        assert!(matches!(
            parse("/resume", &skills),
            Some(Command::Resume(None))
        ));
        assert!(matches!(parse("/tools", &skills), Some(Command::Tools(_))));
        assert!(matches!(
            parse("/caps enable web", &skills),
            Some(Command::Caps(_))
        ));
        assert!(matches!(
            parse("/reconfigure apply", &skills),
            Some(Command::Reconfigure(_))
        ));
    }

    #[test]
    fn completion_spacing_matches_arg_taking_commands() {
        let skills = SkillRegistry::load();
        for cmd in [
            "/model",
            "/mode",
            "/resume",
            "/tools",
            "/caps",
            "/reconfigure",
        ] {
            assert!(completion_inserts_space(cmd, &skills));
        }
        for cmd in [
            "/clear",
            "/controls",
            "/provider",
            "/login",
            "/logout",
            "/retry",
            "/skills",
            "/help",
            "/exit",
        ] {
            assert!(!completion_inserts_space(cmd, &skills));
        }
    }
}
