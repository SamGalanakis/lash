//! Autoresearch tool catalog: tool provider, definitions, and shell execution.

use super::*;
use lash_lashlang_runtime::ToolDefinitionLashlangExt;

pub(crate) struct AutoresearchTools {
    pub(crate) workdir: PathBuf,
    pub(crate) state: Arc<Mutex<RuntimeState>>,
}

pub(crate) fn autoresearch_tool(
    name: &str,
    description: impl Into<String>,
    input_schema: Value,
    output_schema: Value,
) -> ToolDefinition {
    let operation = match name {
        "init_experiment" => "init",
        "run_experiment" => "run",
        "log_experiment" => "log",
        other => panic!("unknown autoresearch Lashlang tool binding for `{other}`"),
    };
    ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        description,
        input_schema,
        output_schema,
    )
    .with_availability(lash_core::ToolAvailabilityConfig::off())
    .with_lashlang_binding(lash_lashlang_runtime::LashlangToolBinding::new(
        ["research"],
        operation,
    ))
    .with_scheduling(ToolScheduling::Parallel)
}

#[async_trait]
impl ToolProvider for AutoresearchTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.tool_definitions()
            .into_iter()
            .find(|tool| tool.name() == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "init_experiment" => self.execute_init(call.args),
            "log_experiment" => self.execute_log(call.args),
            "run_experiment" => {
                self.execute_run(call.args, Some(call.context), call.progress)
                    .await
            }
            other => ToolResult::err_fmt(format_args!("unknown autoresearch tool `{other}`")),
        }
    }
}

impl AutoresearchTools {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            autoresearch_tool(
                "init_experiment",
                format!(
                    "Initialize an autoresearch segment in `{}` / `{}` with the primary metric and direction.",
                    JOURNAL_FILE, MARKDOWN_FILE
                ),
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "metric_name": { "type": "string" },
                        "metric_unit": {
                            "type": "string",
                            "default": "",
                            "description": "Display unit for the metric, e.g. ms, s, kb, or empty."
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["lower", "higher"],
                            "default": "lower",
                            "description": "Whether lower or higher values are better."
                        }
                    },
                    "required": ["name", "metric_name"],
                    "additionalProperties": false
                }),
                init_experiment_output_schema(),
            ),
            autoresearch_tool(
                "run_experiment",
                "Run a benchmark command, capture wall-clock timing, parse `METRIC name=value` lines, and optionally run `autoresearch.checks.sh`.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "default": DEFAULT_TIMEOUT_SECONDS,
                            "description": "Kill the benchmark after this many seconds."
                        },
                        "checks_timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "default": DEFAULT_CHECKS_TIMEOUT_SECONDS,
                            "description": "Kill autoresearch.checks.sh after this many seconds."
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }),
                run_experiment_output_schema(),
            ),
            autoresearch_tool(
                "log_experiment",
                format!(
                    "Append a result to `{}` and refresh `{}`.",
                    JOURNAL_FILE, MARKDOWN_FILE
                ),
                json!({
                    "type": "object",
                    "properties": {
                        "commit": { "type": "string" },
                        "metric": {
                            "type": "number",
                            "description": "Primary metric value for this iteration."
                        },
                        "status": {
                            "type": "string",
                            "enum": ["keep", "discard", "crash", "checks_failed"],
                            "description": "keep, discard, crash, or checks_failed."
                        },
                        "description": { "type": "string" },
                        "metrics": {
                            "type": "object",
                            "additionalProperties": true,
                            "default": {},
                            "description": "Additional named metrics to track."
                        }
                    },
                    "required": ["commit", "metric", "status", "description"],
                    "additionalProperties": false
                }),
                log_experiment_output_schema(),
            ),
        ]
    }

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
        context: Option<&ToolContext<'_>>,
        progress: Option<&lash_core::ProgressSender>,
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
            context.and_then(|value| value.cancellation_token().cloned()),
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
                context.and_then(|value| value.cancellation_token().cloned()),
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

fn init_experiment_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "segment": { "type": "integer", "minimum": 0 },
            "status": status_summary_output_schema()
        },
        "required": ["segment", "status"],
        "additionalProperties": false
    })
}

fn log_experiment_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "status": status_summary_output_schema()
        },
        "required": ["status"],
        "additionalProperties": false
    })
}

fn run_experiment_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": { "type": "string" },
            "exit_code": nullable_schema(json!({ "type": "integer" })),
            "duration_seconds": { "type": "number", "minimum": 0 },
            "passed": { "type": "boolean" },
            "crashed": { "type": "boolean" },
            "timed_out": { "type": "boolean" },
            "tail_output": { "type": "string" },
            "checks_pass": nullable_schema(json!({ "type": "boolean" })),
            "checks_timed_out": nullable_schema(json!({ "type": "boolean" })),
            "checks_output": nullable_schema(json!({ "type": "string" })),
            "checks_duration_seconds": nullable_schema(json!({ "type": "number", "minimum": 0 })),
            "parsed_metrics": {
                "type": "object",
                "additionalProperties": { "type": "number" }
            },
            "parsed_primary": nullable_schema(json!({ "type": "number" })),
            "metric_name": nullable_schema(json!({ "type": "string" })),
            "metric_unit": { "type": "string" }
        },
        "required": [
            "command",
            "exit_code",
            "duration_seconds",
            "passed",
            "crashed",
            "timed_out",
            "tail_output",
            "checks_pass",
            "checks_timed_out",
            "checks_output",
            "checks_duration_seconds",
            "parsed_metrics",
            "parsed_primary",
            "metric_name",
            "metric_unit"
        ],
        "additionalProperties": false
    })
}

fn status_summary_output_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(StatusSummary)).unwrap_or_else(|_| {
        json!({
            "type": "object",
            "additionalProperties": true
        })
    })
}

fn nullable_schema(schema: Value) -> Value {
    json!({ "anyOf": [schema, { "type": "null" }] })
}

#[derive(Clone, Debug)]
pub(crate) struct CommandRun {
    output: String,
    exit_code: Option<i32>,
    duration_seconds: f64,
    passed: bool,
    crashed: bool,
    timed_out: bool,
}

pub(crate) async fn execute_shell_command(
    workdir: &Path,
    command: &str,
    timeout_seconds: u64,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
    progress: Option<&lash_core::ProgressSender>,
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
                let _ = progress.send(lash_core::SandboxMessage {
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
