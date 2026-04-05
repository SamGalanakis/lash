mod activity;
mod app;
mod assistant_text;
mod autonomous;
mod bootstrap;
mod cli_support;
mod command;
mod delegate_tools;
mod diff;
mod editor;
mod event;
mod fork;
mod input_items;
mod interactive;
mod markdown;
mod overlay;
mod plugin_surface;
mod prompt_overrides;
mod repo_status;
mod resume;
mod session_log;
mod setup;
#[cfg(test)]
mod test_support;
mod text_display;
mod theme;
mod ui;
mod ui_resume;
mod update;
mod util;

use clap::Parser;
use crossterm::cursor::SetCursorStyle;
use lash::*;

#[cfg(test)]
use app::PreparedTurn;
#[cfg(test)]
use autonomous::AutonomousRenderer;
#[cfg(test)]
use input_items::insert_inline_marker;
#[cfg(test)]
use input_items::{build_items_from_editor_input, image_marker_ranges};
use prompt_overrides::resolve_prompt_overrides;

pub(crate) use cli_support::*;
pub(crate) use interactive::generate_session_name;
#[cfg(test)]
pub(crate) use interactive::{injected_image_part_indices, make_injected_plugin_message};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const BUILD_GIT_HEAD: &str = env!("LASH_BUILD_GIT_HEAD");
const ROOT_SESSION_ID: &str = "root";
const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n",
    "lash-sansio ",
    env!("CARGO_PKG_VERSION")
);

fn turn_has_visible_output(turn: &AssembledTurn) -> bool {
    !turn.assistant_output.safe_text.trim().is_empty()
        || !turn.code_outputs.is_empty()
        || !turn.errors.is_empty()
        || turn.has_plugin_visible_output
}

fn autonomous_prompt_overrides() -> Vec<PromptSectionOverride> {
    vec![
        PromptSectionOverride {
            section: PromptSectionName::Intro,
            mode: PromptOverrideMode::Prepend,
            content: "You are an autonomous AI coding agent running without a human in the loop.\nComplete the task end-to-end without asking for user input.".to_string(),
        },
        PromptSectionOverride {
            section: PromptSectionName::Execution,
            mode: PromptOverrideMode::Append,
            content: "- No user is available during this run. Do not rely on `ask`; make the best reasonable decision from local context and continue.".to_string(),
        },
    ]
}

#[derive(Parser)]
#[command(name = "lash-cli", bin_name = "lash", version = APP_VERSION, long_version = LONG_VERSION)]
struct Args {
    /// OpenAI-compatible API key (optional — use --provider to configure interactively)
    #[arg(long, env = "OPENAI_COMPATIBLE_API_KEY")]
    api_key: Option<String>,

    /// Tavily API key for web search
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_api_key: Option<String>,

    /// Model name (defaults per provider: Codex/OpenAI-compatible/Google OAuth)
    #[arg(long)]
    model: Option<String>,

    /// Provider-native model variant (for example: high, max, xhigh)
    #[arg(long)]
    variant: Option<String>,

    /// Execution backend (`repl` or `standard`, default: `standard`)
    #[arg(long = "execution-mode")]
    execution_mode: Option<String>,

    /// Context strategy (`rolling_context`)
    #[arg(long = "context-strategy")]
    context_strategy: Option<String>,

    /// Base URL for the LLM API
    #[arg(long, default_value = "")]
    base_url: String,

    /// Enable detailed lifecycle/debug logs and per-session LLM traces
    #[arg(long)]
    debug: bool,

    /// Resume an existing session file on startup
    #[arg(long, value_name = "SESSION.db")]
    resume: Option<String>,

    /// Queue and immediately send a prompt after startup resume
    #[arg(long, value_name = "PROMPT")]
    resume_prompt: Option<String>,

    /// Force re-run provider setup
    #[arg(long)]
    provider: bool,

    /// Delete all lash data (~/.lash/ and ~/.cache/lash/) and exit
    #[arg(long)]
    reset: bool,

    /// Print current runtime/config info and exit
    #[arg(long)]
    info: bool,

    /// Check GitHub releases for a newer lash version and exit
    #[arg(long, conflicts_with = "update")]
    check_update: bool,

    /// Install the latest lash release and exit
    #[arg(long, conflicts_with = "check_update")]
    update: bool,

    /// Run autonomously: execute prompt, print response to stdout, exit
    #[arg(short = 'p', long = "print")]
    print_prompt: Option<String>,

    /// Replace a prompt section: --prompt-replace section=text
    #[arg(long = "prompt-replace", value_name = "SECTION=TEXT")]
    prompt_replace: Vec<String>,

    /// Replace a prompt section from file: --prompt-replace-file section=path
    #[arg(long = "prompt-replace-file", value_name = "SECTION=PATH")]
    prompt_replace_file: Vec<String>,

    /// Prepend text to a prompt section: --prompt-prepend section=text
    #[arg(long = "prompt-prepend", value_name = "SECTION=TEXT")]
    prompt_prepend: Vec<String>,

    /// Prepend text to a prompt section from file: --prompt-prepend-file section=path
    #[arg(long = "prompt-prepend-file", value_name = "SECTION=PATH")]
    prompt_prepend_file: Vec<String>,

    /// Append text to a prompt section: --prompt-append section=text
    #[arg(long = "prompt-append", value_name = "SECTION=TEXT")]
    prompt_append: Vec<String>,

    /// Append text to a prompt section from file: --prompt-append-file section=path
    #[arg(long = "prompt-append-file", value_name = "SECTION=PATH")]
    prompt_append_file: Vec<String>,

    /// Disable a prompt section entirely.
    #[arg(long = "prompt-disable", value_name = "SECTION")]
    prompt_disable: Vec<String>,
}

fn cleanup_terminal() {
    // Pop kitty keyboard protocol, restore background color and cursor style
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PopKeyboardEnhancementFlags
    );
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste);
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]111\x1b\\"),
        SetCursorStyle::DefaultUserShape
    );
    ratatui::restore();
}

fn configure_terminal_ui() -> anyhow::Result<()> {
    crossterm::execute!(
        std::io::stdout(),
        crossterm::style::Print("\x1b]11;rgb:0e/0d/0b\x1b\\"),
        SetCursorStyle::SteadyBar
    )?;
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        )
    );
    // Bracketed paste tells the terminal we can accept paste payloads as literal text.
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableBracketedPaste)?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableFocusChange)?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Set up file-based structured tracing (JSON logs at $LASH_HOME/lash.log)
    {
        let log_dir = lash::lash_home();
        std::fs::create_dir_all(&log_dir).ok();
        let log_file = std::fs::File::create(log_dir.join("lash.log"))?;
        let filter_text = effective_lash_log_filter(args.debug);

        use tracing_subscriber::EnvFilter;
        let fallback = if args.debug { "debug" } else { "warn" };
        let filter = EnvFilter::try_new(&filter_text).unwrap_or_else(|_| EnvFilter::new(fallback));
        tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_env_filter(filter)
            .with_writer(log_file)
            .with_ansi(false)
            .init();

        tracing::debug!(
            current_exe = ?std::env::current_exe().ok(),
            cwd = ?std::env::current_dir().ok(),
            build_git_head = BUILD_GIT_HEAD,
            cli_debug = args.debug,
            filter = %filter_text,
            "initialized lash tracing"
        );
    }
    let prompt_overrides = resolve_prompt_overrides(&args)?;
    bootstrap::run(args, prompt_overrides).await
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use lash::agent::MessageRole;

    use crate::app::App;

    fn skill_catalog_with(names: &[(&str, &str)]) -> SkillCatalog {
        let root = std::env::temp_dir().join(format!("lash-main-skills-{}", uuid::Uuid::new_v4()));
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
    fn insert_inline_marker_adds_spaces_when_touching_text() {
        let mut app = App::new("model".into(), "session".into());
        app.set_input("hello world".into());
        app.editor.cursor_pos = 5;
        insert_inline_marker(&mut app, "[Image #1]");
        assert_eq!(app.input(), "hello [Image #1] world");
    }

    #[test]
    fn insert_inline_marker_keeps_existing_spacing() {
        let mut app = App::new("model".into(), "session".into());
        app.set_input("hello ".into());
        app.editor.cursor_pos = app.input().len();
        insert_inline_marker(&mut app, "[Image #1]");
        assert_eq!(app.input(), "hello [Image #1]");
    }

    #[test]
    fn parse_image_marker_rejects_zero_index() {
        assert_eq!(input_items::parse_image_marker_at("[Image #0]", 0), None);
    }

    #[test]
    fn build_items_preserves_interleaving_for_images_and_paths() {
        let unique = format!(
            "lash-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        let tmp_path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&tmp_path).expect("mkdir temp test dir");
        let file_path = tmp_path.join("a.txt");
        std::fs::write(&file_path, "x").expect("write temp file");
        let original_cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&tmp_path).expect("chdir");
        let (items, image_blobs) = build_items_from_editor_input(
            "before [Image #1] @a.txt after",
            vec![app::PendingImage {
                id: 1,
                png_bytes: vec![1, 2, 3],
            }],
        );
        std::env::set_current_dir(original_cwd).expect("restore cwd");
        let _ = std::fs::remove_dir_all(&tmp_path);

        let kinds: Vec<&'static str> = items
            .iter()
            .map(|item| match item {
                InputItem::Text { .. } => "text",
                InputItem::ImageRef { .. } => "image",
                InputItem::FileRef { .. } => "file",
                InputItem::DirRef { .. } => "dir",
            })
            .collect();
        assert_eq!(kinds, vec!["text", "image", "text", "file", "text"]);
        assert_eq!(image_blobs.len(), 1);
    }

    #[test]
    fn build_items_drops_removed_image_markers() {
        let (items, image_blobs) = build_items_from_editor_input(
            "before after",
            vec![app::PendingImage {
                id: 1,
                png_bytes: vec![1, 2, 3],
            }],
        );

        assert!(
            items
                .iter()
                .all(|item| !matches!(item, InputItem::ImageRef { .. }))
        );
        assert!(image_blobs.is_empty());
    }

    #[test]
    fn image_marker_ranges_find_multiple_markers() {
        let ranges = image_marker_ranges("x [Image #2] y [Image #5]");

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].1, 2);
        assert_eq!(ranges[1].1, 5);
    }

    #[test]
    fn controls_text_mentions_alt_up_queue_edit_binding() {
        let controls = controls_text();
        assert!(controls.contains("Up (empty draft)   Edit last queued turn"));
        assert!(!controls.contains("Backspace          Restore last next-turn draft"));
        assert!(!controls.contains("Shift+Drag"));
        assert!(!controls.contains("F6"));
    }

    #[test]
    fn autonomous_renderer_prints_missing_final_tail_after_streamed_prefix() {
        let mut renderer = AutonomousRenderer::new();
        renderer.handle(AgentEvent::TextDelta {
            content: "Inspected files.\n".to_string(),
        });

        renderer.finish_output("Inspected files.\nCompleted successfully.");

        assert_eq!(
            renderer.stdout_text,
            "Inspected files.\nCompleted successfully."
        );
    }

    #[test]
    fn autonomous_renderer_collects_plugin_panel_output() {
        let mut renderer = AutonomousRenderer::new();
        renderer.handle(AgentEvent::PluginEvent {
            plugin_id: "demo".to_string(),
            event: PluginSurfaceEvent::PanelUpsert {
                key: "panel:1".to_string(),
                title: "TASK BOARD".to_string(),
                content: "1. Inspect\n2. Patch".to_string(),
            },
        });

        assert_eq!(
            renderer.rendered_plugin_output().as_deref(),
            Some("TASK BOARD\n1. Inspect\n2. Patch")
        );
    }

    #[test]
    fn turn_has_visible_output_accepts_plugin_rendered_turns() {
        let turn = AssembledTurn {
            state: AgentStateEnvelope::default(),
            status: TurnStatus::Completed,
            assistant_output: AssistantOutput {
                safe_text: String::new(),
                raw_text: String::new(),
                state: OutputState::Sanitized,
            },
            has_plugin_visible_output: true,
            done_reason: DoneReason::ModelStop,
            execution: ExecutionSummary {
                mode: ExecutionMode::Standard,
                had_tool_calls: false,
                had_code_execution: false,
            },
            token_usage: TokenUsage::default(),
            tool_calls: Vec::new(),
            code_outputs: Vec::new(),
            errors: Vec::new(),
        };

        assert!(turn_has_visible_output(&turn));
    }

    #[test]
    fn normalize_prepared_turn_keeps_plain_slash_text() {
        let skills = skill_catalog_with(&[("yolopush", "ship changes")]);
        let turn = PreparedTurn::new("/not-a-command details".into(), Vec::new());

        let normalized = normalize_prepared_turn_for_dispatch(turn, &skills);

        assert_eq!(normalized.display_text, "/not-a-command details");
        assert_eq!(normalized.effective_text, "/not-a-command details");
        assert!(command::parse(&normalized.display_text, &skills).is_none());
    }

    #[test]
    fn normalize_prepared_turn_rewrites_slash_skill_prompts() {
        let skills = skill_catalog_with(&[("yolopush", "ship changes")]);
        let turn = PreparedTurn::prepare("/yolopush merge staging".into(), Vec::new(), &skills);

        let normalized = normalize_prepared_turn_for_dispatch(turn, &skills);

        assert_eq!(normalized.display_text, "/yolopush merge staging");
        assert!(
            normalized
                .effective_text
                .starts_with("/yolopush merge staging")
        );
        assert_eq!(normalized.transform_labels, vec!["yolopush".to_string()]);
        assert!(normalized.effective_text.contains("<skill>"));
    }

    #[test]
    fn prepared_turn_collects_inline_slash_skills_once() {
        let skills = skill_catalog_with(&[("localref", "fetch local references")]);
        let turn = PreparedTurn::prepare(
            "Compare with /localref opencode and mention /localref again".into(),
            Vec::new(),
            &skills,
        );

        assert_eq!(
            turn.display_text,
            "Compare with /localref opencode and mention /localref again"
        );
        assert_eq!(turn.transform_labels, vec!["localref".to_string()]);
        assert_eq!(turn.effective_text.matches("<skill>").count(), 1);
        assert!(turn.effective_text.contains("<name>localref</name>"));
    }

    #[test]
    fn injected_plugin_message_preserves_images_and_paths() {
        let turn = PreparedTurn::new(
            "before [Image #1] @README.md after [Image #2]".into(),
            vec![vec![1, 2, 3], vec![4, 5, 6]],
        );

        let message = make_injected_plugin_message(&turn);

        assert_eq!(message.role, MessageRole::User);
        assert!(message.images.is_empty());
        assert_eq!(injected_image_part_indices(&message).len(), 2);
        assert!(
            message
                .parts
                .iter()
                .any(|part| part.content.contains("before"))
        );
        assert!(
            message
                .parts
                .iter()
                .any(|part| part.content.contains("after"))
        );
        assert!(
            message
                .parts
                .iter()
                .any(|part| part.content.contains("README.md"))
        );
    }
}
