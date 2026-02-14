use crate::skill::SkillRegistry;

/// Primary commands shown in autocomplete (command, description).
pub const COMMANDS: &[(&str, &str)] = &[
    ("/clear", "Reset conversation"),
    ("/model", "Switch LLM model"),
    ("/provider", "Change LLM provider"),
    ("/logout", "Remove stored credentials"),
    ("/resume", "Resume a previous session"),
    ("/skills", "Browse loaded skills"),
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

/// Slash commands recognized by the TUI.
pub enum Command {
    /// Reset conversation to splash screen
    Clear,
    /// Switch LLM model
    Model(String),
    /// Show message to restart with --provider
    ChangeProvider,
    /// Remove stored credentials
    Logout,
    /// Show available commands and shortcuts
    Help,
    /// Quit the application
    Exit,
    /// Resume a previous session (optional filename)
    Resume(Option<String>),
    /// Browse loaded skills
    Skills,
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
        "model" => arg
            .filter(|a| !a.is_empty())
            .map(|a| Command::Model(a.to_string())),
        "provider" => Some(Command::ChangeProvider),
        "logout" => Some(Command::Logout),
        "help" | "?" => Some(Command::Help),
        "exit" | "quit" => Some(Command::Exit),
        "resume" | "continue" => Some(Command::Resume(arg.map(|a| a.to_string()))),
        "skills" => Some(Command::Skills),
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
