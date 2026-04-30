use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lash::plugin::{
    ExternalInvokeContext, ExternalOpDef, ExternalOpKind, PluginDirective, PluginError,
    PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta, PromptHookContext,
    SessionParam, SessionPlugin, SnapshotReader, SnapshotWriter, ToolCallHookContext,
    ToolResultHookContext, TurnResultHookContext,
};
use lash::{
    MessageRole, PluginMessage, PluginSurfaceEvent, PromptContribution, ToolDefinition,
    ToolExecutionContext, ToolExecutionMode, ToolParam, ToolProvider, ToolResult, TurnStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::model::{
    ConfigEntry, Direction, EXPORT_FILE, ExperimentStatus, JOURNAL_FILE, JournalEntry,
    LastRunSummary, MARKDOWN_FILE, ModeSnapshot, PLUGIN_ID, ResultEntry, RunningStatus,
    StatusSummary, append_journal_entry, compute_summary, confidence_for_candidate,
    delete_session_files, format_confidence, load_journal, now_ms, rewrite_markdown,
    write_export_html,
};

const OUTPUT_MAX_BYTES: usize = 4 * 1024;
const OUTPUT_MAX_LINES: usize = 12;
const DEFAULT_TIMEOUT_SECONDS: u64 = 600;
const DEFAULT_CHECKS_TIMEOUT_SECONDS: u64 = 300;
const AUTORESEARCH_TOOL_NAMES: [&str; 3] = ["init_experiment", "run_experiment", "log_experiment"];

pub struct AutoresearchPluginFactory;

impl PluginFactory for AutoresearchPluginFactory {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        let workdir = std::env::current_dir()
            .map_err(|err| PluginError::Session(format!("failed to determine cwd: {err}")))?;
        Ok(Arc::new(AutoresearchPlugin::new(workdir)))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedPluginState {
    #[serde(default)]
    touched: bool,
    mode: ModeSnapshot,
    last_run: Option<LastRunSummary>,
}

#[derive(Clone, Debug, Default)]
struct RuntimeState {
    touched: bool,
    mode: ModeSnapshot,
    running: Option<RunningStatus>,
    last_run: Option<LastRunSummary>,
}

#[derive(Clone, Debug)]
struct SummaryState {
    touched: bool,
    mode: ModeSnapshot,
    running: Option<RunningStatus>,
    last_run: Option<LastRunSummary>,
}

struct AutoresearchPlugin {
    workdir: PathBuf,
    state: Arc<Mutex<RuntimeState>>,
    provider: Arc<AutoresearchTools>,
}

impl AutoresearchPlugin {
    fn new(workdir: PathBuf) -> Self {
        let state = Arc::new(Mutex::new(RuntimeState::default()));
        Self {
            workdir: workdir.clone(),
            state: Arc::clone(&state),
            provider: Arc::new(AutoresearchTools { workdir, state }),
        }
    }
}

impl SessionPlugin for AutoresearchPlugin {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn version(&self) -> &'static str {
        "1"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;

        let prompt_root = self.workdir.clone();
        let prompt_state = Arc::clone(&self.state);
        reg.prompt()
            .contribute(Arc::new(move |_ctx: PromptHookContext| {
                let prompt_root = prompt_root.clone();
                let prompt_state = Arc::clone(&prompt_state);
                Box::pin(async move {
                    let summary = session_summary_from_runtime(&prompt_root, &prompt_state)?;
                    if !summary.active {
                        return Ok(Vec::new());
                    }
                    Ok(vec![
                        PromptContribution::guidance("Autoresearch", prompt_text(&summary))
                            .with_priority(20),
                    ])
                })
            }));

        let before_root = self.workdir.clone();
        let before_state = Arc::clone(&self.state);
        reg.tool_calls()
            .before(Arc::new(move |ctx: ToolCallHookContext| {
                let before_root = before_root.clone();
                let before_state = Arc::clone(&before_state);
                Box::pin(async move {
                    if ctx.tool_name != "run_experiment" {
                        return Ok(Vec::new());
                    }
                    if let Some(command) = ctx.args.get("command").and_then(Value::as_str) {
                        let mut state = before_state.lock().map_err(|_| {
                            PluginError::Session("autoresearch state poisoned".to_string())
                        })?;
                        state.touched = true;
                        state.running = Some(RunningStatus {
                            command: command.to_string(),
                            started_at_ms: now_ms(),
                        });
                        let summary = compute_summary(
                            &state.mode,
                            &load_journal(&before_root).map_err(PluginError::Session)?,
                            state.running.clone(),
                            state.last_run.clone(),
                        );
                        return Ok(vec![PluginDirective::emit_events(vec![status_event(
                            &summary,
                        )?])]);
                    }
                    Ok(Vec::new())
                })
            }));

        let after_root = self.workdir.clone();
        let after_state = Arc::clone(&self.state);
        reg.tool_calls()
            .after(Arc::new(move |ctx: ToolResultHookContext| {
                let after_root = after_root.clone();
                let after_state = Arc::clone(&after_state);
                Box::pin(async move {
                    if !matches!(
                        ctx.tool_name.as_str(),
                        "run_experiment" | "init_experiment" | "log_experiment"
                    ) {
                        return Ok(Vec::new());
                    }
                    let summary = full_summary_from_runtime(&after_root, &after_state)?;
                    Ok(vec![PluginDirective::emit_events(vec![status_event(
                        &summary,
                    )?])])
                })
            }));

        let turn_root = self.workdir.clone();
        let turn_state = Arc::clone(&self.state);
        reg.turn().after(Arc::new(move |ctx: TurnResultHookContext| {
            let turn_root = turn_root.clone();
            let turn_state = Arc::clone(&turn_state);
            Box::pin(async move {
                let state = turn_state
                    .lock()
                    .map_err(|_| PluginError::Session("autoresearch state poisoned".to_string()))?;
                if !state.mode.active || ctx.turn.status != TurnStatus::Completed {
                    return Ok(Vec::new());
                }
                let summary = compute_summary(
                    &state.mode,
                    &load_journal(&turn_root).map_err(PluginError::Session)?,
                    state.running.clone(),
                    state.last_run.clone(),
                );
                let mut message = String::from(
                    "Continue autoresearch. Make one concrete improvement, measure it, log it, keep wins, discard regressions, and continue.",
                );
                if let Some(objective) = summary.objective.as_deref() {
                    message.push_str("\nObjective: ");
                    message.push_str(objective);
                }
                if let Some(metric_name) = summary.metric_name.as_deref() {
                    let _ = std::fmt::Write::write_fmt(
                        &mut message,
                        format_args!(
                            "\nMetric: {} ({})",
                            metric_name,
                            summary.direction.map(Direction::as_str).unwrap_or("lower")
                        ),
                    );
                }
                Ok(vec![PluginDirective::EnqueueMessages {
                    messages: vec![PluginMessage::text(MessageRole::System, message)],
                }])
            })
        }));

        reg.external().op(
            status_external_def(),
            Arc::new({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |_ctx: ExternalInvokeContext, _args: Value| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    Box::pin(async move { status_tool_result(&root, &state) })
                }
            }),
        )?;
        reg.external().op(
            start_external_def(),
            Arc::new({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |ctx: ExternalInvokeContext, args: Value| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    Box::pin(async move { start_mode_command(ctx, &root, &state, args).await })
                }
            }),
        )?;
        reg.external().op(
            stop_external_def(),
            Arc::new({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |ctx: ExternalInvokeContext, _args: Value| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    Box::pin(async move { stop_mode_command(ctx, &root, &state).await })
                }
            }),
        )?;
        reg.external().op(
            clear_external_def(),
            Arc::new({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |ctx: ExternalInvokeContext, _args: Value| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    Box::pin(async move { clear_mode_command(ctx, &root, &state).await })
                }
            }),
        )?;
        reg.external().op(
            export_external_def(),
            Arc::new({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |_ctx: ExternalInvokeContext, _args: Value| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    Box::pin(async move { export_summary(&root, &state) })
                }
            }),
        )?;

        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("autoresearch state poisoned".to_string()))?;
        let persisted = PersistedPluginState {
            touched: state.touched,
            mode: state.mode.clone(),
            last_run: state.last_run.clone(),
        };
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            revision: self.snapshot_revision(),
            state: Some(serde_json::to_value(persisted).map_err(|err| {
                PluginError::Snapshot(format!("failed to snapshot autoresearch: {err}"))
            })?),
        })
    }

    fn snapshot_revision(&self) -> u64 {
        1
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let Some(state_value) = meta.state.clone() else {
            return Ok(());
        };
        let restored =
            serde_json::from_value::<PersistedPluginState>(state_value).map_err(|err| {
                PluginError::Snapshot(format!("failed to restore autoresearch: {err}"))
            })?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("autoresearch state poisoned".to_string()))?;
        state.touched = restored.touched
            || restored.mode.active
            || restored.mode.objective.is_some()
            || restored.last_run.is_some();
        state.mode = restored.mode;
        state.last_run = restored.last_run;
        state.running = None;
        Ok(())
    }
}

struct AutoresearchTools {
    workdir: PathBuf,
    state: Arc<Mutex<RuntimeState>>,
}

#[async_trait]
impl ToolProvider for AutoresearchTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "init_experiment".into(),
                description: format!(
                    "Initialize an autoresearch segment in `{}` / `{}` with the primary metric and direction.",
                    JOURNAL_FILE, MARKDOWN_FILE
                ),
                params: vec![
                    ToolParam::typed("name", "str"),
                    ToolParam::typed("metric_name", "str"),
                    ToolParam {
                        name: "metric_unit".into(),
                        r#type: "str".into(),
                        description: "Display unit for the metric, e.g. ms, s, kb, or empty.".into(),
                        default_value: Some(json!("")),
                        required: false,
                    },
                    ToolParam {
                        name: "direction".into(),
                        r#type: "str".into(),
                        description: "Whether lower or higher values are better.".into(),
                        default_value: Some(json!("lower")),
                        required: false,
                    },
                ],
                returns: "json".into(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::hidden(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "run_experiment".into(),
                description:
                    "Run a benchmark command, capture wall-clock timing, parse `METRIC name=value` lines, and optionally run `autoresearch.checks.sh`.".into(),
                params: vec![
                    ToolParam::typed("command", "str"),
                    ToolParam {
                        name: "timeout_seconds".into(),
                        r#type: "int".into(),
                        description: "Kill the benchmark after this many seconds.".into(),
                        default_value: Some(json!(DEFAULT_TIMEOUT_SECONDS)),
                        required: false,
                    },
                    ToolParam {
                        name: "checks_timeout_seconds".into(),
                        r#type: "int".into(),
                        description: "Kill autoresearch.checks.sh after this many seconds.".into(),
                        default_value: Some(json!(DEFAULT_CHECKS_TIMEOUT_SECONDS)),
                        required: false,
                    },
                ],
                returns: "json".into(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::hidden(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "log_experiment".into(),
                description: format!(
                    "Append a result to `{}` and refresh `{}`.",
                    JOURNAL_FILE, MARKDOWN_FILE
                ),
                params: vec![
                    ToolParam::typed("commit", "str"),
                    ToolParam {
                        name: "metric".into(),
                        r#type: "float".into(),
                        description: "Primary metric value for this iteration.".into(),
                        default_value: None,
                        required: true,
                    },
                    ToolParam {
                        name: "status".into(),
                        r#type: "str".into(),
                        description: "keep, discard, crash, or checks_failed.".into(),
                        default_value: None,
                        required: true,
                    },
                    ToolParam::typed("description", "str"),
                    ToolParam {
                        name: "metrics".into(),
                        r#type: "json".into(),
                        description: "Additional named metrics to track.".into(),
                        default_value: Some(json!({})),
                        required: false,
                    },
                ],
                returns: "json".into(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::hidden(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: ToolExecutionMode::Parallel,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &Value) -> ToolResult {
        match name {
            "init_experiment" => self.execute_init(args),
            "log_experiment" => self.execute_log(args),
            "run_experiment" => self.execute_run(args, None, None).await,
            other => ToolResult::err_fmt(format_args!("unknown autoresearch tool `{other}`")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
        progress: Option<&lash::ProgressSender>,
    ) -> ToolResult {
        match name {
            "run_experiment" => self.execute_run(args, Some(context), progress).await,
            _ => self.execute(name, args).await,
        }
    }
}

impl AutoresearchTools {
    fn execute_init(&self, args: &Value) -> ToolResult {
        let name = match require_string(args, "name") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let metric_name = match require_string(args, "metric_name") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let metric_unit = args
            .get("metric_unit")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let direction = match parse_direction(args.get("direction").and_then(Value::as_str)) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        let entries = match load_journal(&self.workdir) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        let next_segment = entries
            .iter()
            .filter_map(|entry| match entry {
                JournalEntry::Config(config) => Some(config.segment),
                JournalEntry::Result(_) => None,
            })
            .max()
            .unwrap_or(0)
            + 1;
        let entry = JournalEntry::Config(ConfigEntry {
            segment: next_segment,
            created_at_ms: now_ms(),
            name,
            metric_name,
            metric_unit,
            direction,
        });
        if let Err(err) = append_journal_entry(&self.workdir, &entry) {
            return ToolResult::err_fmt(err);
        }
        if let Ok(mut state) = self.state.lock() {
            state.touched = true;
        }
        let summary = match self.summary() {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        if let Err(err) = rewrite_markdown(&self.workdir, &summary) {
            return ToolResult::err_fmt(err);
        }
        ToolResult::ok(json!({
            "segment": next_segment,
            "status": summary,
        }))
    }

    fn execute_log(&self, args: &Value) -> ToolResult {
        let commit = match require_string(args, "commit") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let metric = match require_f64(args, "metric") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let status = match parse_status(args.get("status").and_then(Value::as_str)) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        let description = match require_string(args, "description") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let metrics = match parse_metrics_object(args.get("metrics")) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        let mut entries = match load_journal(&self.workdir) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        let Some(config) = entries.iter().rev().find_map(|entry| match entry {
            JournalEntry::Config(config) => Some(config.clone()),
            JournalEntry::Result(_) => None,
        }) else {
            return ToolResult::err_fmt("init_experiment must be called before log_experiment");
        };
        let current_results = entries
            .iter()
            .filter_map(|entry| match entry {
                JournalEntry::Result(result) if result.segment == config.segment => {
                    Some(result.clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let confidence = confidence_for_candidate(&config, &current_results, metric);
        let state = match self.state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        let last_run = state.last_run.clone();
        drop(state);
        let entry = JournalEntry::Result(ResultEntry {
            segment: config.segment,
            timestamp_ms: now_ms(),
            commit,
            metric,
            metrics,
            status,
            description,
            duration_seconds: last_run.as_ref().map(|value| value.duration_seconds),
            exit_code: last_run.as_ref().and_then(|value| value.exit_code),
            checks_pass: last_run.as_ref().and_then(|value| value.checks_pass),
            command: last_run.as_ref().map(|value| value.command.clone()),
            confidence,
        });
        if let Err(err) = append_journal_entry(&self.workdir, &entry) {
            return ToolResult::err_fmt(err);
        }
        if let Ok(mut state) = self.state.lock() {
            state.touched = true;
        }
        entries.push(entry);
        let summary = {
            let state = match self.state.lock() {
                Ok(value) => value,
                Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
            };
            compute_summary(
                &state.mode,
                &entries,
                state.running.clone(),
                state.last_run.clone(),
            )
        };
        if let Err(err) = rewrite_markdown(&self.workdir, &summary) {
            return ToolResult::err_fmt(err);
        }
        ToolResult::ok(json!({
            "status": summary,
        }))
    }

    async fn execute_run(
        &self,
        args: &Value,
        context: Option<&ToolExecutionContext>,
        progress: Option<&lash::ProgressSender>,
    ) -> ToolResult {
        let command = match require_string(args, "command") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let timeout_seconds = args
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS);
        let checks_timeout_seconds = args
            .get("checks_timeout_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_CHECKS_TIMEOUT_SECONDS);
        let config = load_journal(&self.workdir).ok().and_then(|entries| {
            entries.iter().rev().find_map(|entry| match entry {
                JournalEntry::Config(config) => Some(config.clone()),
                JournalEntry::Result(_) => None,
            })
        });

        let run = match execute_shell_command(
            &self.workdir,
            &command,
            timeout_seconds,
            context.and_then(|value| value.cancellation_token.clone()),
            progress,
        )
        .await
        {
            Ok(value) => value,
            Err(err) => {
                self.finish_run(None);
                return ToolResult::err_fmt(err);
            }
        };

        let parsed_metrics = parse_metric_lines(&run.output);
        let parsed_primary = config
            .as_ref()
            .and_then(|value| parsed_metrics.get(&value.metric_name).copied());
        let checks = if run.passed && self.workdir.join("autoresearch.checks.sh").exists() {
            match execute_shell_command(
                &self.workdir,
                "bash autoresearch.checks.sh",
                checks_timeout_seconds,
                context.and_then(|value| value.cancellation_token.clone()),
                None,
            )
            .await
            {
                Ok(value) => Some(value),
                Err(err) => {
                    self.finish_run(None);
                    return ToolResult::err_fmt(err);
                }
            }
        } else {
            None
        };

        let last_run = LastRunSummary {
            command: command.clone(),
            duration_seconds: run.duration_seconds,
            exit_code: run.exit_code,
            passed: run.passed,
            crashed: run.crashed,
            timed_out: run.timed_out,
            checks_pass: checks.as_ref().map(|value| value.passed),
            checks_timed_out: checks
                .as_ref()
                .map(|value| value.timed_out)
                .unwrap_or(false),
            parsed_primary,
            parsed_metrics: parsed_metrics.clone(),
            tail_output: truncate_tail(&run.output, OUTPUT_MAX_LINES, OUTPUT_MAX_BYTES),
        };
        self.finish_run(Some(last_run.clone()));

        ToolResult::ok(json!({
            "command": command,
            "exit_code": run.exit_code,
            "duration_seconds": run.duration_seconds,
            "passed": run.passed,
            "crashed": run.crashed,
            "timed_out": run.timed_out,
            "tail_output": last_run.tail_output,
            "checks_pass": checks.as_ref().map(|value| value.passed),
            "checks_timed_out": checks.as_ref().map(|value| value.timed_out),
            "checks_output": checks
                .as_ref()
                .map(|value| truncate_tail(&value.output, OUTPUT_MAX_LINES, OUTPUT_MAX_BYTES)),
            "checks_duration_seconds": checks.as_ref().map(|value| value.duration_seconds),
            "parsed_metrics": parsed_metrics,
            "parsed_primary": parsed_primary,
            "metric_name": config.as_ref().map(|value| value.metric_name.clone()),
            "metric_unit": config.as_ref().map(|value| value.metric_unit.clone()).unwrap_or_default(),
        }))
    }

    fn finish_run(&self, last_run: Option<LastRunSummary>) {
        if let Ok(mut state) = self.state.lock() {
            state.touched = true;
            state.running = None;
            if let Some(last_run) = last_run {
                state.last_run = Some(last_run);
            }
        }
    }

    fn summary(&self) -> Result<StatusSummary, String> {
        full_summary_from_runtime(&self.workdir, &self.state).map_err(|err| err.to_string())
    }
}

#[derive(Clone, Debug)]
struct CommandRun {
    output: String,
    exit_code: Option<i32>,
    duration_seconds: f64,
    passed: bool,
    crashed: bool,
    timed_out: bool,
}

async fn execute_shell_command(
    workdir: &Path,
    command: &str,
    timeout_seconds: u64,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
    progress: Option<&lash::ProgressSender>,
) -> Result<CommandRun, String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "bash".to_string());
    let mut child = Command::new(shell);
    child
        .arg("-lc")
        .arg(format!("{command} 2>&1"))
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = child
        .spawn()
        .map_err(|err| format!("failed to spawn experiment command: {err}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture experiment stdout".to_string())?;
    let progress = progress.cloned();
    let read_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let read = stdout
                .read(&mut chunk)
                .await
                .map_err(|err| format!("failed to read experiment output: {err}"))?;
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
            if let Some(progress) = progress.as_ref() {
                let _ = progress.send(lash::SandboxMessage {
                    text: String::from_utf8_lossy(&chunk[..read]).into_owned(),
                    kind: "tool_output".to_string(),
                });
            }
        }
        Ok::<Vec<u8>, String>(bytes)
    });

    let started = Instant::now();
    let timeout = tokio::time::sleep(Duration::from_secs(timeout_seconds));
    tokio::pin!(timeout);
    let status = if let Some(token) = cancellation_token {
        tokio::select! {
            _ = token.cancelled() => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err("experiment cancelled".to_string());
            }
            _ = &mut timeout => {
                let _ = child.kill().await;
                child.wait().await.map_err(|err| format!("failed to wait on timed out experiment: {err}"))?
            }
            status = child.wait() => status.map_err(|err| format!("failed to wait on experiment: {err}"))?,
        }
    } else {
        tokio::select! {
            _ = &mut timeout => {
                let _ = child.kill().await;
                child.wait().await.map_err(|err| format!("failed to wait on timed out experiment: {err}"))?
            }
            status = child.wait() => status.map_err(|err| format!("failed to wait on experiment: {err}"))?,
        }
    };
    let timed_out = started.elapsed().as_secs() >= timeout_seconds && !status.success();
    let output = read_task
        .await
        .map_err(|err| format!("experiment output task failed: {err}"))??;
    let output = String::from_utf8_lossy(&output).into_owned();
    Ok(CommandRun {
        duration_seconds: started.elapsed().as_secs_f64(),
        exit_code: status.code(),
        passed: status.success() && !timed_out,
        crashed: !status.success() && !timed_out,
        timed_out,
        output,
    })
}

fn prompt_text(summary: &StatusSummary) -> String {
    let objective = summary
        .objective
        .as_deref()
        .unwrap_or("No objective recorded.");
    let metric_name = summary.metric_name.as_deref().unwrap_or("metric");
    let direction = summary.direction.map(Direction::as_str).unwrap_or("lower");
    let best = summary
        .best_metric
        .map(|value| {
            format!(
                "{}{}",
                crate::model::format_metric(value),
                summary.metric_unit
            )
        })
        .unwrap_or_else(|| "—".to_string());
    let confidence = summary
        .confidence
        .map(format_confidence)
        .unwrap_or_else(|| "—".to_string());
    format!(
        "Autoresearch mode is active.\nObjective: {objective}\nCurrent metric: {metric_name} ({direction} is better).\nBest observed: {best}.\nConfidence: {confidence}.\nUse `init_experiment` once per segment, `run_experiment` for measurements, and `log_experiment` after every run. Keep improvements, revert regressions, and continue autonomously until interrupted. Keep `{}` and `{}` up to date.",
        JOURNAL_FILE, MARKDOWN_FILE
    )
}

fn status_event(summary: &StatusSummary) -> Result<PluginSurfaceEvent, PluginError> {
    Ok(PluginSurfaceEvent::Custom {
        name: "autoresearch.status".to_string(),
        payload: serde_json::to_value(summary).map_err(|err| {
            PluginError::Session(format!("failed to encode autoresearch status: {err}"))
        })?,
    })
}

fn summary_state(state: &Arc<Mutex<RuntimeState>>) -> Result<SummaryState, PluginError> {
    let state = state
        .lock()
        .map_err(|_| PluginError::Session("autoresearch state poisoned".to_string()))?;
    Ok(SummaryState {
        touched: state.touched,
        mode: state.mode.clone(),
        running: state.running.clone(),
        last_run: state.last_run.clone(),
    })
}

fn compute_summary_from_state(
    root: &Path,
    state: SummaryState,
) -> Result<StatusSummary, PluginError> {
    let entries = load_journal(root).map_err(PluginError::Session)?;
    Ok(compute_summary(
        &state.mode,
        &entries,
        state.running,
        state.last_run,
    ))
}

fn full_summary_from_runtime(
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> Result<StatusSummary, PluginError> {
    compute_summary_from_state(root, summary_state(state)?)
}

fn session_summary_from_runtime(
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> Result<StatusSummary, PluginError> {
    let state = summary_state(state)?;
    if !state.touched && !state.mode.active {
        return Ok(StatusSummary::default());
    }
    compute_summary_from_state(root, state)
}

fn status_tool_result(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    match session_summary_from_runtime(root, state) {
        Ok(summary) => ToolResult::ok(json!(summary)),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn autoresearch_tool_names() -> Vec<String> {
    AUTORESEARCH_TOOL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

async fn set_autoresearch_tools_enabled(
    ctx: &ExternalInvokeContext,
    enabled: bool,
) -> Result<(), ToolResult> {
    let Some(session_id) = ctx.session_id.as_deref() else {
        return Err(ToolResult::err_fmt(
            "autoresearch commands require a session-scoped invocation",
        ));
    };
    let availability = if enabled {
        Some(lash::ToolAvailability::Documented)
    } else {
        Some(lash::ToolAvailability::Hidden)
    };
    ctx.host
        .set_tools_availability(session_id, &autoresearch_tool_names(), availability)
        .await
        .map_err(|err| {
            let action = if enabled { "enable" } else { "disable" };
            ToolResult::err_fmt(format_args!("failed to {action} autoresearch tools: {err}"))
        })?;
    Ok(())
}

async fn start_mode_command(
    ctx: ExternalInvokeContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
    args: Value,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, true).await {
        return result;
    }
    let result = start_mode(root, state, args);
    if !result.success {
        let _ = set_autoresearch_tools_enabled(&ctx, false).await;
    }
    result
}

async fn stop_mode_command(
    ctx: ExternalInvokeContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, false).await {
        return result;
    }
    let result = stop_mode(root, state);
    if !result.success {
        let _ = set_autoresearch_tools_enabled(&ctx, true).await;
    }
    result
}

async fn clear_mode_command(
    ctx: ExternalInvokeContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
) -> ToolResult {
    if let Err(result) = set_autoresearch_tools_enabled(&ctx, false).await {
        return result;
    }
    let result = clear_mode(root, state);
    if !result.success {
        let _ = set_autoresearch_tools_enabled(&ctx, true).await;
    }
    result
}

fn start_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>, args: Value) -> ToolResult {
    let objective = args
        .get("objective")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.touched = true;
        state.mode.active = true;
        if let Some(objective) = objective.clone() {
            state.mode.objective = Some(objective);
        }
        let entries = match load_journal(root) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        compute_summary(
            &state.mode,
            &entries,
            state.running.clone(),
            state.last_run.clone(),
        )
    };
    if let Err(err) = rewrite_markdown(root, &summary) {
        return ToolResult::err_fmt(err);
    }
    let queued_input = objective.as_ref().map(|objective| {
        format!(
            "Start autoresearch.\nObjective: {objective}\nIf there is no active experiment segment yet, initialize one. Then make one concrete change, run a measurement, log the result, keep wins, discard regressions, and continue."
        )
    });
    ToolResult::ok(json!({
        "status": summary,
        "queued_input": queued_input,
        "message": "Autoresearch mode on."
    }))
}

fn stop_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.mode.active = false;
        state.running = None;
        let entries = match load_journal(root) {
            Ok(value) => value,
            Err(err) => return ToolResult::err_fmt(err),
        };
        compute_summary(&state.mode, &entries, None, state.last_run.clone())
    };
    if let Err(err) = rewrite_markdown(root, &summary) {
        return ToolResult::err_fmt(err);
    }
    ToolResult::ok(json!({
        "status": summary,
        "message": "Autoresearch mode off."
    }))
}

fn clear_mode(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    if let Err(err) = delete_session_files(root) {
        return ToolResult::err_fmt(err);
    }
    let summary = {
        let mut state = match state.lock() {
            Ok(value) => value,
            Err(_) => return ToolResult::err_fmt("autoresearch state poisoned"),
        };
        state.touched = false;
        state.mode = ModeSnapshot::default();
        state.running = None;
        state.last_run = None;
        StatusSummary::default()
    };
    ToolResult::ok(json!({
        "status": summary,
        "message": "Cleared autoresearch session files."
    }))
}

fn export_summary(root: &Path, state: &Arc<Mutex<RuntimeState>>) -> ToolResult {
    let summary = match full_summary_from_runtime(root, state) {
        Ok(value) => value,
        Err(err) => return ToolResult::err_fmt(err.to_string()),
    };
    match write_export_html(root, &summary) {
        Ok(path) => ToolResult::ok(json!({
            "status": summary,
            "path": path.display().to_string(),
            "message": format!("Wrote {}.", EXPORT_FILE),
        })),
        Err(err) => ToolResult::err_fmt(err),
    }
}

fn status_external_def() -> ExternalOpDef {
    ExternalOpDef {
        name: "autoresearch.status".to_string(),
        description: "Return the current autoresearch runtime and journal summary.".to_string(),
        kind: ExternalOpKind::Query,
        session_param: SessionParam::Required,
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "object" }),
    }
}

fn start_external_def() -> ExternalOpDef {
    ExternalOpDef {
        name: "autoresearch.start".to_string(),
        description: "Enable autoresearch mode and optionally seed a new objective.".to_string(),
        kind: ExternalOpKind::Command,
        session_param: SessionParam::Required,
        input_schema: json!({
            "type": "object",
            "properties": { "objective": { "type": "string" } }
        }),
        output_schema: json!({ "type": "object" }),
    }
}

fn stop_external_def() -> ExternalOpDef {
    ExternalOpDef {
        name: "autoresearch.stop".to_string(),
        description: "Disable autoresearch mode for the current session.".to_string(),
        kind: ExternalOpKind::Command,
        session_param: SessionParam::Required,
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "object" }),
    }
}

fn clear_external_def() -> ExternalOpDef {
    ExternalOpDef {
        name: "autoresearch.clear".to_string(),
        description: "Delete autoresearch session files and clear runtime state.".to_string(),
        kind: ExternalOpKind::Command,
        session_param: SessionParam::Required,
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "object" }),
    }
}

fn export_external_def() -> ExternalOpDef {
    ExternalOpDef {
        name: "autoresearch.export".to_string(),
        description: "Write an HTML export of the current autoresearch summary.".to_string(),
        kind: ExternalOpKind::Task,
        session_param: SessionParam::Required,
        input_schema: json!({ "type": "object" }),
        output_schema: json!({ "type": "object" }),
    }
}

fn require_string(args: &Value, key: &str) -> Result<String, ToolResult> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ToolResult::err_fmt(format_args!("missing required string `{key}`")))
}

fn require_f64(args: &Value, key: &str) -> Result<f64, ToolResult> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| ToolResult::err_fmt(format_args!("missing required number `{key}`")))
}

fn parse_direction(value: Option<&str>) -> Result<Direction, String> {
    match value
        .unwrap_or("lower")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "lower" | "" => Ok(Direction::Lower),
        "higher" => Ok(Direction::Higher),
        other => Err(format!(
            "invalid direction `{other}`; expected `lower` or `higher`"
        )),
    }
}

fn parse_status(value: Option<&str>) -> Result<ExperimentStatus, String> {
    match value
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "keep" => Ok(ExperimentStatus::Keep),
        "discard" => Ok(ExperimentStatus::Discard),
        "crash" => Ok(ExperimentStatus::Crash),
        "checks_failed" => Ok(ExperimentStatus::ChecksFailed),
        other => Err(format!(
            "invalid experiment status `{other}`; expected keep, discard, crash, or checks_failed"
        )),
    }
}

fn parse_metrics_object(value: Option<&Value>) -> Result<BTreeMap<String, f64>, String> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let Some(object) = value.as_object() else {
        return Err("`metrics` must be a JSON object".to_string());
    };
    let mut metrics = BTreeMap::new();
    for (name, value) in object {
        let Some(number) = value.as_f64() else {
            return Err(format!("metric `{name}` must be numeric"));
        };
        metrics.insert(name.clone(), number);
    }
    Ok(metrics)
}

fn parse_metric_lines(output: &str) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("METRIC ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let Ok(number) = value.trim().parse::<f64>() else {
            continue;
        };
        if number.is_finite() && !name.trim().is_empty() {
            metrics.insert(name.trim().to_string(), number);
        }
    }
    metrics
}

fn truncate_tail(text: &str, max_lines: usize, max_bytes: usize) -> String {
    let bytes = text.as_bytes();
    let start = bytes.len().saturating_sub(max_bytes);
    let mut tail = String::from_utf8_lossy(&bytes[start..]).into_owned();
    let lines = tail.lines().collect::<Vec<_>>();
    if lines.len() > max_lines {
        tail = lines[lines.len() - max_lines..].join("\n");
    }
    tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_metrics_reads_metric_lines() {
        let metrics = parse_metric_lines("hello\nMETRIC total_ms=12.5\nMETRIC score=99\n");
        assert_eq!(metrics.get("total_ms"), Some(&12.5));
        assert_eq!(metrics.get("score"), Some(&99.0));
    }

    #[test]
    fn session_summary_short_circuits_when_untouched() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join(JOURNAL_FILE), "{not valid json").expect("write journal");

        let summary = session_summary_from_runtime(
            dir.path(),
            &Arc::new(Mutex::new(RuntimeState::default())),
        )
        .expect("untouched session should not read journal");

        assert_eq!(summary, StatusSummary::default());
    }

    #[test]
    fn full_summary_reads_journal_when_explicitly_requested() {
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join(JOURNAL_FILE), "{not valid json").expect("write journal");

        let err =
            full_summary_from_runtime(dir.path(), &Arc::new(Mutex::new(RuntimeState::default())))
                .expect_err("explicit summary should read journal");

        assert!(err.to_string().contains("failed to parse"));
    }

    #[test]
    fn autoresearch_tools_start_disabled() {
        let dir = tempdir().expect("tempdir");
        let tools = AutoresearchTools {
            workdir: dir.path().to_path_buf(),
            state: Arc::new(Mutex::new(RuntimeState::default())),
        };
        assert!(tools.definitions().into_iter().all(|tool| {
            tool.effective_availability(&lash::ExecutionMode::standard())
                == lash::ToolAvailability::Hidden
        }));
    }
}
