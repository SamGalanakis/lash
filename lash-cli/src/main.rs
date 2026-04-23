mod activity;
mod app;
mod assistant_text;
mod autonomous;
mod bootstrap;
mod cli_support;
mod clipboard;
mod command;
mod diff;
mod editor;
mod event;
mod fork;
mod host_docs;
mod input_items;
mod interactive;
mod markdown;
mod overlay;
mod paths;
mod persistence;
mod plugin_surface;
mod render;
mod repo_status;
mod resume;
mod runtime_perf;
mod scratch_tui;
mod session_bootstrap;
mod session_log;
mod setup;
mod stream_markdown;
#[cfg(test)]
mod test_support;
mod text_display;
mod text_layout;
mod theme;
mod tree;
mod turn_runner;
mod ui_action;
mod ui_perf;
mod ui_trace;
mod update;
mod util;

use clap::Parser;
#[cfg(feature = "dhat-heap")]
use dhat::Alloc as DhatAlloc;
use lash::*;
#[cfg(not(feature = "dhat-heap"))]
use stats_alloc::{INSTRUMENTED_SYSTEM, StatsAlloc};
#[cfg(not(feature = "dhat-heap"))]
use std::alloc::System;

#[cfg(test)]
use app::PreparedTurn;
#[cfg(test)]
use autonomous::AutonomousRenderer;
pub(crate) use cli_support::*;
#[cfg(test)]
use input_items::insert_inline_marker;
#[cfg(test)]
use input_items::{build_items_from_editor_input, image_marker_ranges};
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

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static GLOBAL_ALLOCATOR: DhatAlloc = DhatAlloc;

#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn turn_has_visible_output(turn: &AssembledTurn) -> bool {
    !turn.assistant_output.safe_text.trim().is_empty()
        || !turn.errors.is_empty()
        || turn.has_plugin_visible_output
}

fn autonomous_prompt_template() -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![
            PromptTemplateEntry::text(
                "You are an autonomous AI coding assistant running without a human in the loop.\nComplete the task end-to-end without asking for user input.\nIf the task is incomplete and concrete next actions are available, continue executing them instead of stopping to summarize incompletion.",
            ),
            PromptTemplateEntry::builtin(PromptBuiltin::MainAgentIntro),
            PromptTemplateEntry::slot(PromptSlot::Intro),
        ]),
        PromptTemplateSection::titled(
            "Execution",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
                PromptTemplateEntry::text(
                    "- No user is available during this run. Default to acting without asking. Ask only when progress is blocked and user intervention is strictly required; otherwise make the best reasonable decision from local context and continue.\n- Do not stop merely to report that work remains. If concrete actions are still available, keep executing them.\n- Only summarize remaining work when you are blocked, need a decision, or have exhausted feasible actions for this turn.\n- Do not claim completion unless you have actually verified the required end state.",
                ),
                PromptTemplateEntry::slot(PromptSlot::Execution),
            ],
        ),
        PromptTemplateSection::titled(
            "Guidance",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::CoreGuidance),
                PromptTemplateEntry::slot(PromptSlot::ProjectInstructions),
                PromptTemplateEntry::slot(PromptSlot::Guidance),
            ],
        ),
        PromptTemplateSection::titled(
            "Environment",
            vec![
                PromptTemplateEntry::slot(PromptSlot::RuntimeContext),
                PromptTemplateEntry::slot(PromptSlot::Environment),
            ],
        ),
    ])
}

fn normalized_cli_args() -> Vec<std::ffi::OsString> {
    let mut out = Vec::new();
    let mut iter = std::env::args_os();
    if let Some(bin) = iter.next() {
        out.push(bin);
    }
    for arg in iter {
        if let Some(raw) = arg.to_str() {
            if raw == "-ca" {
                out.push("--context-approach".into());
                continue;
            }
            if let Some(value) = raw.strip_prefix("-ca=") {
                out.push(format!("--context-approach={value}").into());
                continue;
            }
            if raw == "-em" {
                out.push("--execution-mode".into());
                continue;
            }
            if let Some(value) = raw.strip_prefix("-em=") {
                out.push(format!("--execution-mode={value}").into());
                continue;
            }
        }
        out.push(arg);
    }
    out
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

    /// Execution backend (`rlm` or `standard`, default: `standard`)
    #[arg(long = "execution-mode")]
    execution_mode: Option<String>,

    /// Context approach (`rolling_history` or `observational_memory`)
    #[arg(short = 'c', long = "context-approach", value_name = "APPROACH")]
    context_approach: Option<String>,

    /// OM: observe once recent raw history reaches this many tokens
    #[arg(long = "om-observation-message-tokens", value_name = "TOKENS")]
    om_observation_message_tokens: Option<usize>,

    /// OM: keep this many recent raw-history tokens unobserved on the active tail
    #[arg(long = "om-observation-buffer-tokens", value_name = "TOKENS")]
    om_observation_buffer_tokens: Option<usize>,

    /// OM: hard ceiling for unobserved raw-history tokens before the tail is fully eligible
    #[arg(long = "om-observation-block-after-tokens", value_name = "TOKENS")]
    om_observation_block_after_tokens: Option<usize>,

    /// OM: maximum tokens per observer batch sent to the background worker
    #[arg(long = "om-observation-max-tokens-per-batch", value_name = "TOKENS")]
    om_observation_max_tokens_per_batch: Option<usize>,

    /// OM: how much prior memory text to carry into observer runs
    #[arg(long = "om-previous-observer-tokens", value_name = "TOKENS")]
    om_previous_observer_tokens: Option<usize>,

    /// OM: reflect once accumulated memory reaches this many tokens
    #[arg(long = "om-reflection-observation-tokens", value_name = "TOKENS")]
    om_reflection_observation_tokens: Option<usize>,

    /// OM: start reflection buffering at this percent of reflection tokens
    #[arg(
        long = "om-reflection-buffer-activation-percent",
        value_name = "PERCENT"
    )]
    om_reflection_buffer_activation_percent: Option<u16>,

    /// OM: hard ceiling for reflection memory tokens
    #[arg(long = "om-reflection-block-after-tokens", value_name = "TOKENS")]
    om_reflection_block_after_tokens: Option<usize>,

    /// RLM only: upsert a bound variable from JSON, for example `--rlm-var input='{\"path\":\"src\"}'`
    #[arg(long = "rlm-var", value_name = "NAME=JSON")]
    rlm_var: Vec<String>,

    /// RLM only: load a JSON object of bound variables from a file and upsert them before the next turn
    #[arg(long = "rlm-vars-file", value_name = "PATH")]
    rlm_vars_file: Option<std::path::PathBuf>,

    /// RLM only: remove a previously bound variable before the next turn
    #[arg(long = "rlm-unset", value_name = "NAME")]
    rlm_unset: Vec<String>,

    /// Base URL for the LLM API
    #[arg(long, default_value = "")]
    base_url: String,

    /// Enable detailed lifecycle/debug logs and per-session LLM traces
    #[arg(long)]
    debug: bool,

    /// Record the live TUI session as a replayable UI trace JSON plus final .snap
    #[arg(long, value_name = "TRACE.json")]
    debug_ui_trace: Option<std::path::PathBuf>,

    /// When recording a UI trace, also capture numbered checkpoint snapshots every N ms
    #[arg(long, value_name = "MS")]
    debug_ui_trace_interval_ms: Option<u64>,

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

    /// In autonomous mode, wait for plugin background work for this session before exiting
    #[arg(long)]
    await_background_work: bool,

    /// In autonomous mode, write token-usage delta and cumulative session usage for the turn to this JSON file
    #[arg(long, value_name = "PATH")]
    turn_usage_json: Option<std::path::PathBuf>,

    /// Run the synthetic non-provider UI performance benchmark and exit
    #[arg(long, hide = true)]
    ui_perf_benchmark: bool,

    /// Write the UI benchmark JSON report to this file
    #[arg(long, hide = true, value_name = "OUT.json")]
    ui_perf_out: Option<std::path::PathBuf>,

    /// Number of measured runs for the UI benchmark
    #[arg(long, hide = true, default_value_t = 5)]
    ui_perf_runs: usize,

    /// Number of warmup runs for the UI benchmark
    #[arg(long, hide = true, default_value_t = 1)]
    ui_perf_warmups: usize,

    /// Run the synthetic non-inference runtime performance benchmark and exit
    #[arg(long, hide = true)]
    runtime_perf_benchmark: bool,

    /// Write the runtime benchmark JSON report to this file
    #[arg(long, hide = true, value_name = "OUT.json")]
    runtime_perf_out: Option<std::path::PathBuf>,

    /// Write a dhat heap profile for the measured runtime benchmark window
    #[arg(long, hide = true)]
    runtime_perf_dhat: bool,

    /// Write a dhat heap profile for the measured runtime benchmark window
    #[arg(long, hide = true, value_name = "OUT.json")]
    runtime_perf_dhat_out: Option<std::path::PathBuf>,

    /// Trim dhat backtraces to this many frames
    #[arg(long, hide = true, value_name = "FRAMES")]
    runtime_perf_dhat_frames: Option<usize>,

    /// Number of measured runs for the runtime benchmark
    #[arg(long, hide = true, default_value_t = 5)]
    runtime_perf_runs: usize,

    /// Number of warmup runs for the runtime benchmark
    #[arg(long, hide = true, default_value_t = 1)]
    runtime_perf_warmups: usize,

    /// Limit the runtime benchmark to one or more named scenarios
    #[arg(long, hide = true, value_name = "SCENARIO")]
    runtime_perf_scenario: Vec<String>,

    /// Number of committed turns to run inside each measured runtime session
    #[arg(long, hide = true, default_value_t = 12)]
    runtime_perf_turns: usize,

    /// Export a persisted session by id or `.db` path and exit
    #[arg(long, value_name = "SESSION")]
    export: Option<String>,

    /// Export format: `html` (default) or `json`. Only meaningful with --export.
    #[arg(long = "export-format", value_name = "FORMAT", default_value = "html")]
    export_format: String,

    /// Destination file for --export. If omitted, the rendered document is written to stdout.
    #[arg(long = "export-out", value_name = "PATH")]
    export_out: Option<std::path::PathBuf>,
}

fn run_export(session: &str, format: &str, out: Option<&std::path::Path>) -> anyhow::Result<()> {
    let format = lash_export::ExportFormat::parse(format)?;
    let selector_path = std::path::PathBuf::from(session);
    let selector = if selector_path.is_file() {
        lash_export::SessionSelector::Path(selector_path.as_path())
    } else {
        lash_export::SessionSelector::Id(session)
    };
    let sessions_dir = crate::paths::lash_home().join("sessions");
    let rendered = lash_export::export(selector, &sessions_dir, format, out)?;
    if out.is_none() {
        print!("{rendered}");
    }
    Ok(())
}

fn cleanup_terminal() {
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = crossterm::execute!(
        std::io::stdout(),
        crossterm::cursor::MoveTo(0, 0),
        crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
        crossterm::style::ResetColor,
        crossterm::style::SetAttribute(crossterm::style::Attribute::Reset),
        crossterm::cursor::Show,
        crossterm::event::PopKeyboardEnhancementFlags,
        crossterm::event::DisableMouseCapture,
        crossterm::event::DisableBracketedPaste,
        crossterm::event::DisableFocusChange,
        crossterm::terminal::LeaveAlternateScreen
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse_from(normalized_cli_args());
    if args.ui_perf_benchmark {
        return ui_perf::run_cli(
            args.ui_perf_out,
            args.ui_perf_runs,
            args.ui_perf_warmups,
            APP_VERSION,
        );
    }
    if args.runtime_perf_benchmark {
        return runtime_perf::run_cli(
            args.runtime_perf_out,
            args.runtime_perf_dhat,
            args.runtime_perf_dhat_out,
            args.runtime_perf_dhat_frames,
            args.runtime_perf_runs,
            args.runtime_perf_warmups,
            args.runtime_perf_scenario,
            args.runtime_perf_turns,
            APP_VERSION,
        )
        .await;
    }
    if let Some(session) = args.export.as_deref() {
        return run_export(session, &args.export_format, args.export_out.as_deref());
    }
    // Set up file-based structured tracing (JSON logs at $LASH_HOME/lash.log)
    {
        let log_dir = crate::paths::lash_home();
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
    bootstrap::run(args, default_prompt_template()).await
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use lash::session_model::MessageRole;

    use crate::app::App;
    use crate::cli_support::{CopyBinding, copy_binding_from_env};

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
    fn autonomous_renderer_prints_missing_final_tail_after_streamed_prefix() {
        let mut renderer = AutonomousRenderer::new();
        let _ = renderer.handle(SessionEvent::TextDelta {
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
        let _ = renderer.handle(SessionEvent::PluginEvent {
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
            state: SessionStateEnvelope::default(),
            status: TurnStatus::Completed,
            assistant_output: AssistantOutput {
                safe_text: String::new(),
                raw_text: String::new(),
                state: OutputState::Usable,
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
            errors: Vec::new(),
            typed_finish: None,
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
        assert!(command::parse(&normalized.display_text, &skills, &[]).is_none());
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
        assert_eq!(normalized.input_provenance.transforms.len(), 1);
        assert!(matches!(
            normalized.input_provenance.transforms.first(),
            Some(lash::UserInputTransform::SkillBlockAppend { skill_name, .. })
                if skill_name == "yolopush"
        ));
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
        assert_eq!(turn.input_provenance.transforms.len(), 1);
        assert!(matches!(
            turn.input_provenance.transforms.first(),
            Some(lash::UserInputTransform::SkillBlockAppend { skill_name, .. })
                if skill_name == "localref"
        ));
        assert_eq!(turn.effective_text.matches("<skill>").count(), 1);
        assert!(turn.effective_text.contains("<name>localref</name>"));
    }

    #[test]
    fn copy_binding_defaults_to_ctrl_c() {
        assert_eq!(copy_binding_from_env(None), CopyBinding::CtrlC);
    }

    #[test]
    fn copy_binding_accepts_configured_variants() {
        assert_eq!(
            copy_binding_from_env(Some("ctrl-shift-c")),
            CopyBinding::CtrlShiftC
        );
        assert_eq!(copy_binding_from_env(Some("ctrl-y")), CopyBinding::CtrlY);
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
