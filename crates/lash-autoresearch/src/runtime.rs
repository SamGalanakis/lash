use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lash_core::plugin::{
    PluginAction, PluginActionContext, PluginActionFailure, PluginActionKind, PluginDirective,
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta,
    PromptHookContext, SessionParam, SessionPlugin, SnapshotReader, SnapshotWriter,
    ToolCallHookContext, ToolResultHookContext, TurnResultHookContext,
};
use lash_core::{
    MessageRole, PluginMessage, PluginRuntimeEvent, PromptContribution, ToolCall, ToolContext,
    ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolResult, ToolScheduling,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct AutoresearchEmptyArgs {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct AutoresearchStartArgs {
    pub objective: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct AutoresearchCommandOutput {
    pub status: StatusSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued_input: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct AutoresearchExportOutput {
    pub status: StatusSummary,
    pub path: String,
    pub message: String,
}

pub struct AutoresearchStatusOp;
pub struct AutoresearchStartOp;
pub struct AutoresearchStopOp;
pub struct AutoresearchClearOp;
pub struct AutoresearchExportOp;

impl PluginAction for AutoresearchStatusOp {
    const NAME: &'static str = "autoresearch.status";
    const DESCRIPTION: &'static str =
        "Return the current autoresearch runtime and journal summary.";
    const KIND: PluginActionKind = PluginActionKind::Query;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchEmptyArgs;
    type Output = StatusSummary;
}

impl PluginAction for AutoresearchStartOp {
    const NAME: &'static str = "autoresearch.start";
    const DESCRIPTION: &'static str =
        "Enable autoresearch mode and optionally seed a new objective.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchStartArgs;
    type Output = AutoresearchCommandOutput;
}

impl PluginAction for AutoresearchStopOp {
    const NAME: &'static str = "autoresearch.stop";
    const DESCRIPTION: &'static str = "Disable autoresearch mode for the current session.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchEmptyArgs;
    type Output = AutoresearchCommandOutput;
}

impl PluginAction for AutoresearchClearOp {
    const NAME: &'static str = "autoresearch.clear";
    const DESCRIPTION: &'static str = "Delete autoresearch session files and clear runtime state.";
    const KIND: PluginActionKind = PluginActionKind::Command;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchEmptyArgs;
    type Output = AutoresearchCommandOutput;
}

impl PluginAction for AutoresearchExportOp {
    const NAME: &'static str = "autoresearch.export";
    const DESCRIPTION: &'static str = "Write an HTML export of the current autoresearch summary.";
    const KIND: PluginActionKind = PluginActionKind::Task;
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchEmptyArgs;
    type Output = AutoresearchExportOutput;
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
                        return Ok(vec![PluginDirective::emit_runtime_events(vec![
                            status_event(&summary)?,
                        ])]);
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
                    Ok(vec![PluginDirective::emit_runtime_events(vec![
                        status_event(&summary)?,
                    ])])
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
                if !state.mode.active
                    || !matches!(
                        &ctx.turn.outcome,
                        lash_core::TurnOutcome::Finished(_) | lash_core::TurnOutcome::AgentFrameSwitch { .. }
                    )
                {
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

        reg.actions().typed::<AutoresearchStatusOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |_ctx: PluginActionContext, _args: AutoresearchEmptyArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move { tool_result_output(status_tool_result(&root, &state)) }
            }
        })?;
        reg.actions().typed::<AutoresearchStartOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |ctx: PluginActionContext, args: AutoresearchStartArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move {
                    tool_result_output(
                        start_mode_command(
                            ctx,
                            &root,
                            &state,
                            serde_json::to_value(args).unwrap_or_default(),
                        )
                        .await,
                    )
                }
            }
        })?;
        reg.actions().typed::<AutoresearchStopOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |ctx: PluginActionContext, _args: AutoresearchEmptyArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move { tool_result_output(stop_mode_command(ctx, &root, &state).await) }
            }
        })?;
        reg.actions().typed::<AutoresearchClearOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |ctx: PluginActionContext, _args: AutoresearchEmptyArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move { tool_result_output(clear_mode_command(ctx, &root, &state).await) }
            }
        })?;
        reg.actions().typed::<AutoresearchExportOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |_ctx: PluginActionContext, _args: AutoresearchEmptyArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move { tool_result_output(export_summary(&root, &state)) }
            }
        })?;

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

mod commands;
mod summary;
mod tools;

use commands::*;
use summary::*;
use tools::*;

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
        assert!(
            tools
                .tool_manifests()
                .into_iter()
                .all(|tool| { tool.effective_availability() == lash_core::ToolAvailability::Off })
        );
    }
}
