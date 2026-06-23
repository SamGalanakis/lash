use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lash_core::plugin::{
    PluginCommand, PluginCommandContext, PluginCommandOutcome, PluginDirective, PluginError,
    PluginFactory, PluginOperation, PluginOperationFailure, PluginRegistrar,
    PluginRuntimeDirective, PluginSessionContext, PluginSnapshotMeta, PluginTask,
    PluginTaskContext, PluginTaskOutcome, PromptHookContext, SessionParam, SessionPlugin,
    SnapshotReader, SnapshotWriter, ToolCallHookContext, ToolCatalogContext, ToolResultHookContext,
    TurnResultHookContext,
};
use lash_core::{
    MessageRole, PluginMessage, PluginRuntimeEvent, PromptContribution, ToolCall,
    ToolCatalogContribution, ToolContext, ToolContract, ToolDefinition, ToolManifest, ToolProvider,
    ToolResult, ToolScheduling,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, lash_core::JsonSchema)]
pub struct AutoresearchCommandArgs {
    pub raw: Option<String>,
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

pub struct AutoresearchCommandOp;
pub struct AutoresearchExportOp;

impl PluginOperation for AutoresearchCommandOp {
    const NAME: &'static str = "autoresearch.command";
    const DESCRIPTION: &'static str = "Parse and apply an autoresearch slash command.";
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchCommandArgs;
    type Output = AutoresearchCommandOutput;
}

impl PluginCommand for AutoresearchCommandOp {}

impl PluginOperation for AutoresearchExportOp {
    const NAME: &'static str = "autoresearch.export";
    const DESCRIPTION: &'static str = "Write an HTML export of the current autoresearch summary.";
    const SESSION_PARAM: SessionParam = SessionParam::Required;
    type Args = AutoresearchEmptyArgs;
    type Output = AutoresearchExportOutput;
}

impl PluginTask for AutoresearchExportOp {}

struct AutoresearchPlugin {
    workdir: PathBuf,
    state: Arc<Mutex<RuntimeState>>,
    provider: Arc<AutoresearchTools>,
}

impl AutoresearchPlugin {
    fn new(workdir: PathBuf) -> Self {
        Self::with_state(workdir, Arc::new(Mutex::new(RuntimeState::default())))
    }

    fn with_state(workdir: PathBuf, state: Arc<Mutex<RuntimeState>>) -> Self {
        Self {
            workdir: workdir.clone(),
            state: Arc::clone(&state),
            provider: Arc::new(AutoresearchTools { workdir, state }),
        }
    }
}

async fn run_autoresearch_command(
    ctx: PluginCommandContext,
    root: &Path,
    state: &Arc<Mutex<RuntimeState>>,
    args: AutoresearchCommandArgs,
) -> Result<PluginCommandOutcome<AutoresearchCommandOutput>, PluginOperationFailure> {
    let raw = args.raw.as_deref().map(str::trim).unwrap_or_default();
    if raw.eq_ignore_ascii_case("help") {
        let status: StatusSummary = tool_result_output(status_tool_result(root, state))?;
        return Ok(PluginCommandOutcome::new(AutoresearchCommandOutput {
            status,
            queued_input: None,
            message: autoresearch_help_text(),
        }));
    }

    let result = if raw.eq_ignore_ascii_case("off") {
        stop_mode_command(ctx, root, state).await
    } else if raw.eq_ignore_ascii_case("clear") {
        clear_mode_command(ctx, root, state).await
    } else {
        let objective = (!raw.is_empty()).then(|| raw.to_string());
        start_mode_command(
            ctx,
            root,
            state,
            serde_json::to_value(AutoresearchStartArgs { objective }).unwrap_or_default(),
        )
        .await
    };
    let output: AutoresearchCommandOutput = tool_result_output(result)?;
    let mut directives = Vec::new();
    if let Some(input) = output.queued_input.clone() {
        directives.push(PluginRuntimeDirective::queue_turn(
            lash_core::TurnInput::text(input),
        ));
    }
    Ok(PluginCommandOutcome::new(output.clone())
        .with_events(vec![status_event(&output.status)?])
        .with_directives(directives))
}

fn autoresearch_help_text() -> String {
    [
        "Autoresearch",
        "",
        "Commands:",
        "  /autoresearch <objective>  Start autoresearch or update the objective",
        "  /autoresearch help         Show this help",
        "  /autoresearch off          Stop autoresearch mode",
        "  /autoresearch clear        Clear autoresearch state and UI",
        "  /autoresearch export       Export the current summary",
    ]
    .join("\n")
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

        // Autoresearch tools are catalog members only while the mode is active.
        // Inactive sessions remove them from the catalog, so the model never
        // sees them.
        let catalog_state = Arc::clone(&self.state);
        reg.tool_catalog()
            .contribute(Arc::new(move |_ctx: ToolCatalogContext| {
                let active = catalog_state
                    .lock()
                    .map_err(|_| PluginError::Session("autoresearch state poisoned".to_string()))?
                    .mode
                    .active;
                if active {
                    Ok(ToolCatalogContribution::default())
                } else {
                    Ok(ToolCatalogContribution::remove_tools(
                        autoresearch_tool_names(),
                    ))
                }
            }));

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

        reg.operations()
            .typed_command::<AutoresearchCommandOp, _, _>({
                let root = self.workdir.clone();
                let state = Arc::clone(&self.state);
                move |ctx: PluginCommandContext, args: AutoresearchCommandArgs| {
                    let root = root.clone();
                    let state = Arc::clone(&state);
                    async move { run_autoresearch_command(ctx, &root, &state, args).await }
                }
            })?;
        reg.operations().typed_task::<AutoresearchExportOp, _, _>({
            let root = self.workdir.clone();
            let state = Arc::clone(&self.state);
            move |_ctx: PluginTaskContext, _args: AutoresearchEmptyArgs| {
                let root = root.clone();
                let state = Arc::clone(&state);
                async move {
                    let output: AutoresearchExportOutput =
                        tool_result_output(export_summary(&root, &state))?;
                    Ok(PluginTaskOutcome::new(output.clone())
                        .with_events(vec![status_event(&output.status)?]))
                }
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
    use lash_core::InputItem;
    use lash_core::plugin::{PluginHost, PluginRuntimeDirective};
    use lash_core::testing::MockSessionManager;
    use tempfile::tempdir;

    fn command_context(
        manager: Arc<MockSessionManager>,
    ) -> lash_core::plugin::PluginCommandContext {
        lash_core::plugin::PluginCommandContext {
            session_id: Some("root".to_string()),
            sessions: manager.clone(),
            session_lifecycle: manager.clone(),
            session_graph: manager.clone(),
            processes: manager,
        }
    }

    fn manager_with_autoresearch_tools(
        root: &Path,
        state: Arc<Mutex<RuntimeState>>,
    ) -> Arc<MockSessionManager> {
        let provider = Arc::new(AutoresearchTools {
            workdir: root.to_path_buf(),
            state,
        }) as Arc<dyn ToolProvider>;
        let registry =
            lash_core::ToolRegistry::from_tool_provider(provider).expect("tool registry");
        Arc::new(MockSessionManager::default().with_tool_registry(registry))
    }

    fn queued_directive_text(directive: &PluginRuntimeDirective) -> Option<&str> {
        match directive {
            PluginRuntimeDirective::QueueTurn { input, .. } => {
                input.items.iter().find_map(|item| match item {
                    InputItem::Text { text } => Some(text.as_str()),
                    InputItem::ImageRef { .. } => None,
                })
            }
        }
    }

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
    fn autoresearch_tools_advertised_as_plain_members() {
        // The provider advertises its tools as plain catalog members; their
        // visibility is gated by the per-turn catalog contribution keyed on
        // mode state, not by a manifest tier.
        let dir = tempdir().expect("tempdir");
        let tools = AutoresearchTools {
            workdir: dir.path().to_path_buf(),
            state: Arc::new(Mutex::new(RuntimeState::default())),
        };
        let names = tools
            .tool_manifests()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<std::collections::BTreeSet<_>>();
        for name in autoresearch_tool_names() {
            assert!(names.contains(&name), "provider advertises `{name}`");
        }
    }

    #[test]
    fn catalog_membership_follows_autoresearch_mode_state() {
        // Membership is gated by the per-turn catalog contribution: inactive
        // removes the autoresearch tools; active restores them.
        struct TestFactory {
            workdir: PathBuf,
            state: Arc<Mutex<RuntimeState>>,
        }
        impl PluginFactory for TestFactory {
            fn id(&self) -> &'static str {
                PLUGIN_ID
            }
            fn build(
                &self,
                _ctx: &PluginSessionContext,
            ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
                Ok(Arc::new(AutoresearchPlugin::with_state(
                    self.workdir.clone(),
                    Arc::clone(&self.state),
                )))
            }
        }

        let dir = tempdir().expect("tempdir");
        let state = Arc::new(Mutex::new(RuntimeState::default()));
        let mut factories: Vec<Arc<dyn PluginFactory>> = vec![Arc::new(TestFactory {
            workdir: dir.path().to_path_buf(),
            state: Arc::clone(&state),
        })];
        factories.extend(lash_core::testing::test_code_protocol_factories());
        let host = PluginHost::new(factories);
        let session = host.build_session("root", None).expect("session");

        let members = |session: &Arc<lash_core::plugin::PluginSession>| {
            session
                .resolved_tool_catalog("root")
                .expect("catalog")
                .callable_tools()
                .into_iter()
                .map(|manifest| manifest.name)
                .collect::<std::collections::BTreeSet<_>>()
        };

        // Inactive: removed from the catalog.
        let inactive = members(&session);
        for name in autoresearch_tool_names() {
            assert!(!inactive.contains(&name), "inactive removes `{name}`");
        }

        // Active: restored as members.
        state.lock().expect("state").mode.active = true;
        let active = members(&session);
        for name in autoresearch_tool_names() {
            assert!(active.contains(&name), "active restores `{name}`");
        }
    }

    #[tokio::test]
    async fn autoresearch_command_start_emits_status_and_queue_directive() {
        let dir = tempdir().expect("tempdir");
        let state = Arc::new(Mutex::new(RuntimeState::default()));
        let manager = manager_with_autoresearch_tools(dir.path(), Arc::clone(&state));

        let outcome = run_autoresearch_command(
            command_context(Arc::clone(&manager)),
            dir.path(),
            &state,
            AutoresearchCommandArgs {
                raw: Some("improve latency".to_string()),
            },
        )
        .await
        .expect("autoresearch command");

        assert!(outcome.output.status.active);
        assert_eq!(
            outcome.output.status.objective.as_deref(),
            Some("improve latency")
        );
        assert!(
            outcome
                .output
                .queued_input
                .as_deref()
                .is_some_and(|input| input.contains("Objective: improve latency"))
        );
        assert_eq!(outcome.events.len(), 1);
        let PluginRuntimeEvent::Custom { name, payload } = &outcome.events[0] else {
            panic!("expected custom status event");
        };
        assert_eq!(name, "autoresearch.status");
        let status: StatusSummary = serde_json::from_value(payload.clone()).expect("status");
        assert!(status.active);
        assert_eq!(status.objective.as_deref(), Some("improve latency"));
        assert_eq!(outcome.directives.len(), 1);
        assert!(
            queued_directive_text(&outcome.directives[0])
                .is_some_and(|text| text.contains("Objective: improve latency"))
        );
        // Membership is gated by mode state, which is now active.
        assert!(state.lock().expect("state").mode.active);
    }

    #[tokio::test]
    async fn autoresearch_command_off_emits_inactive_status() {
        let dir = tempdir().expect("tempdir");
        let state = Arc::new(Mutex::new(RuntimeState::default()));
        let manager = manager_with_autoresearch_tools(dir.path(), Arc::clone(&state));
        run_autoresearch_command(
            command_context(Arc::clone(&manager)),
            dir.path(),
            &state,
            AutoresearchCommandArgs {
                raw: Some("improve latency".to_string()),
            },
        )
        .await
        .expect("start command");

        let outcome = run_autoresearch_command(
            command_context(Arc::clone(&manager)),
            dir.path(),
            &state,
            AutoresearchCommandArgs {
                raw: Some("off".to_string()),
            },
        )
        .await
        .expect("off command");

        assert!(!outcome.output.status.active);
        assert!(outcome.directives.is_empty());
        assert_eq!(outcome.events.len(), 1);
        let PluginRuntimeEvent::Custom { name, payload } = &outcome.events[0] else {
            panic!("expected custom status event");
        };
        assert_eq!(name, "autoresearch.status");
        let status: StatusSummary = serde_json::from_value(payload.clone()).expect("status");
        assert!(!status.active);
        // Membership follows mode state, which is now inactive.
        assert!(!state.lock().expect("state").mode.active);
    }

    #[test]
    fn autoresearch_export_writes_artifact_without_ui_status_rpc() {
        let dir = tempdir().expect("tempdir");
        let state = Arc::new(Mutex::new(RuntimeState::default()));
        let start: AutoresearchCommandOutput = tool_result_output(start_mode(
            dir.path(),
            &state,
            serde_json::json!({ "objective": "improve latency" }),
        ))
        .expect("start mode");
        assert!(start.status.active);

        let output: AutoresearchExportOutput =
            tool_result_output(export_summary(dir.path(), &state)).expect("export");

        assert!(output.status.active);
        assert!(Path::new(&output.path).exists());
        assert!(output.message.contains(EXPORT_FILE));
    }
}
