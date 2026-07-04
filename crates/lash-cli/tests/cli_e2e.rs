#![cfg(feature = "test-provider")]

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use lash_debug_cli_harness::{
    ExecutionMode, HarnessConfig, LiveHarness, MouseHarnessEvent, repo_root_from_manifest_dir,
};

#[test]
fn cli_short_rlm_mode_runs_subagent_spawn_with_test_provider() {
    let temp = tempfile::tempdir().expect("temp lash home");
    write_test_provider_config(temp.path(), "rlm-subagent-smoke");
    write_test_model_catalog(temp.path());

    let output = run_lash_with_timeout(
        Command::new(lash_bin())
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("LASH_HOME", temp.path())
            .env("LASH_LOG", "warn")
            .args([
                "-em",
                "rlm",
                "--model",
                "test/cli-e2e-model",
                "--print",
                "Does your subagent tool work",
            ]),
        Duration::from_secs(30),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        output.status.success(),
        "lash CLI failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );
    assert!(
        combined.contains("[tool] spawn_agent"),
        "spawn_agent tool was not exercised\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    assert!(
        combined.contains("subagent-ok"),
        "subagent result was not printed\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
}

#[test]
fn cli_json_mode_stamps_protocol_version_in_turn_start() {
    let temp = tempfile::tempdir().expect("temp lash home");
    write_test_provider_config(temp.path(), "rlm-subagent-smoke");
    write_test_model_catalog(temp.path());

    let output = run_lash_with_timeout(
        Command::new(lash_bin())
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("LASH_HOME", temp.path())
            .env("LASH_LOG", "warn")
            .args([
                "-em",
                "rlm",
                "--model",
                "test/cli-e2e-model",
                "--mode",
                "json",
                "--print",
                "Does your subagent tool work",
            ]),
        Duration::from_secs(30),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "lash CLI json mode failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );

    // The first NDJSON record is the versioned `turn_start` frame.
    let first_line = stdout
        .lines()
        .next()
        .unwrap_or_else(|| panic!("json mode produced no stdout\nstderr:\n{stderr}"));
    let turn_start: serde_json::Value = serde_json::from_str(first_line)
        .unwrap_or_else(|err| panic!("parse turn_start line `{first_line}`: {err}"));
    assert_eq!(turn_start["type"], "turn_start");
    assert_eq!(
        turn_start["protocol_version"], 1,
        "json-mode turn_start must carry protocol_version\nstdout:\n{stdout}"
    );
}

#[test]
fn cli_interactive_pty_smoke_runs_turn_and_exits() {
    let lash_home = test_lash_home("standard-echo");
    let mut harness = start_interactive_harness(&lash_home, ExecutionMode::Standard, None);
    harness.send_line("hello from pty").expect("send prompt");
    harness
        .wait_for_text(
            "test-provider echo: hello from pty",
            Duration::from_secs(30),
        )
        .expect("wait for provider response");
    let run = harness.finish_cleanly().expect("finish harness");
    let trace = std::fs::read_to_string(&run.artifacts.ui_trace_json).unwrap_or_else(|err| {
        panic!(
            "failed to read UI trace {}: {err}\nvisible screen:\n{}",
            run.artifacts.ui_trace_json.display(),
            run.screen_text
        )
    });
    assert!(
        run.screen_text.contains("hello from pty"),
        "final screen did not contain submitted prompt\nvisible screen:\n{}",
        run.screen_text
    );
    assert!(
        run.screen_text
            .contains("test-provider echo: hello from pty"),
        "final screen did not contain provider response\nvisible screen:\n{}",
        run.screen_text
    );
    assert!(
        trace.contains("\"user_turn\"") && trace.contains("hello from pty"),
        "UI trace did not record the submitted turn\ntrace:\n{trace}\nvisible screen:\n{}",
        run.screen_text
    );
}

#[test]
fn cli_interactive_pty_active_steer_escape_replays_once() {
    let lash_home = test_lash_home("standard-gated-escape");
    let mut harness = start_interactive_harness(&lash_home, ExecutionMode::Standard, None);

    harness
        .send_line("gated initial prompt")
        .expect("send gated prompt");
    wait_for_provider_marker(
        lash_home.path(),
        "gated-initial-started",
        Duration::from_secs(10),
    );
    harness
        .type_text("queued after escape")
        .expect("type active steer");
    harness.press_key("Enter").expect("submit active steer");
    harness
        .wait_for_text("queued after escape", Duration::from_secs(10))
        .expect("wait for active steer preview");
    harness.press_key("Esc").expect("interrupt active turn");
    harness
        .wait_for_text("Manually interrupted.", Duration::from_secs(30))
        .expect("wait for manual interrupt marker");
    harness
        .wait_for_text(
            "test-provider echo: queued after escape",
            Duration::from_secs(45),
        )
        .expect("wait for deferred active steer response");

    let run = harness.finish_cleanly().expect("finish harness");
    assert!(
        run.screen_text.contains("Manually interrupted.")
            && run
                .screen_text
                .contains("test-provider echo: queued after escape"),
        "final screen should show the interrupted first turn and exactly-once replayed steer\nscreen:\n{}\nartifacts: {}",
        run.screen_text,
        run.artifacts.output_dir.display()
    );

    let trace = std::fs::read_to_string(&run.artifacts.ui_trace_json).unwrap_or_else(|err| {
        panic!(
            "failed to read UI trace {}: {err}\nvisible screen:\n{}",
            run.artifacts.ui_trace_json.display(),
            run.screen_text
        )
    });
    let trace: serde_json::Value = serde_json::from_str(&trace).expect("parse UI trace");
    assert_eq!(
        trace_op_count(&trace, "queue_current_turn_input", "queued after escape"),
        1,
        "Enter during the active turn should enqueue one current-turn steer\ntrace:\n{trace:#}"
    );
    assert_eq!(
        trace_op_count(&trace, "queue_turn", "queued after escape"),
        0,
        "Enter during the active turn must not also queue the same text as an idle next turn\ntrace:\n{trace:#}"
    );

    let provider_requests = provider_request_visible_user_texts(lash_home.path());
    assert_eq!(
        provider_requests
            .iter()
            .flatten()
            .filter(|text| text.as_str() == "queued after escape")
            .count(),
        1,
        "deferred steer should appear in provider requests exactly once\nrequests:\n{provider_requests:#?}"
    );
}

#[test]
fn cli_interactive_pty_mouse_selection_copies_and_escape_keeps_draft() {
    let lash_home = test_lash_home("standard-echo");
    let mut harness = start_interactive_harness(&lash_home, ExecutionMode::Standard, None);
    let input = "abc123 selectable text";
    harness.type_text(input).expect("type draft input");
    let visible = harness
        .wait_for_text(input, Duration::from_secs(10))
        .expect("wait for draft input");
    let (text_col, text_row) =
        screen_cell_for(&visible, input).expect("draft input is visible on screen");
    let selection_start = "abc123 ".len() as u16;
    let selection_len = "selectab".len() as u16;
    let start_col = text_col + selection_start;
    let end_col = start_col + selection_len;
    harness
        .send_mouse(MouseHarnessEvent::Down {
            col: start_col,
            row: text_row,
        })
        .expect("mouse down on draft input");
    harness
        .send_mouse(MouseHarnessEvent::Drag {
            col: end_col,
            row: text_row,
        })
        .expect("mouse drag on draft input");
    harness
        .send_mouse(MouseHarnessEvent::Up {
            col: end_col,
            row: text_row,
        })
        .expect("mouse up on draft input");
    harness
        .wait_for_text("Copied to clipboard", Duration::from_secs(10))
        .expect("wait for selection copy toast");
    let selected_snapshot = harness
        .screenshot("input-selection-copied")
        .expect("selection copied screenshot");
    assert!(
        selected_snapshot.text.exists()
            && selected_snapshot.svg.exists()
            && selected_snapshot.png.exists(),
        "selection screenshot artifacts were not written: {selected_snapshot:?}"
    );

    harness
        .press_key("Esc")
        .expect("clear selection with Escape");
    std::thread::sleep(Duration::from_millis(200));
    harness
        .type_text("X")
        .expect("type after clearing selection");
    std::thread::sleep(Duration::from_millis(100));
    let after_escape = harness.screen_text().expect("screen after typing");
    let draft_line = after_escape
        .lines()
        .find(|line| line.contains("abc123"))
        .unwrap_or_else(|| panic!("draft input disappeared\nscreen:\n{after_escape}"));
    assert!(
        draft_line.contains("selectable") && draft_line.contains('X'),
        "Escape should clear the selection so typing inserts without deleting selected text\nscreen:\n{after_escape}"
    );
    assert!(
        !draft_line.contains("abc123 selectable textX"),
        "mouse selection should move the cursor before Escape clears it, not append at the end\nscreen:\n{after_escape}"
    );
    let cleared_snapshot = harness
        .screenshot("selection-cleared")
        .expect("selection-cleared screenshot");
    assert!(
        cleared_snapshot.text.exists()
            && cleared_snapshot.svg.exists()
            && cleared_snapshot.png.exists(),
        "selection-cleared screenshot artifacts were not written: {cleared_snapshot:?}"
    );
    harness.press_key("Ctrl-C").expect("clear draft input");
    harness.finish_cleanly().expect("finish harness");
}

#[test]
fn cli_interactive_pty_ctrl_p_opens_command_palette() {
    let lash_home = test_lash_home("standard-echo");
    let mut harness = start_interactive_harness(&lash_home, ExecutionMode::Standard, None);

    harness.press_key("Ctrl-P").expect("open command palette");
    let screen = harness
        .wait_for_text("Theme: Lash", Duration::from_secs(10))
        .expect("wait for command palette");

    assert!(
        screen.contains("Commands") && screen.contains("Model") && screen.contains("Provider"),
        "command palette did not show expected settings\nscreen:\n{screen}"
    );
    harness.type_text("system").expect("filter command palette");
    harness
        .press_key("Enter")
        .expect("select system theme from command palette");
    harness
        .wait_for_text("Theme set to System", Duration::from_secs(10))
        .expect("wait for theme toast");

    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(lash_home.path().join("config.json")).expect("read config"),
    )
    .expect("parse config");
    assert_eq!(config["theme"], serde_json::json!("system"));
    harness.finish_cleanly().expect("finish harness");
}

#[test]
fn cli_interactive_pty_rlm_subagent_smoke_runs_turn_and_exits() {
    let lash_home = test_lash_home("rlm-subagent-smoke");
    let mut harness = start_interactive_harness(&lash_home, ExecutionMode::Rlm, None);
    harness
        .send_line("Does your subagent tool work")
        .expect("send prompt");
    harness
        .wait_for_text("■ subagent-ok", Duration::from_secs(45))
        .expect("wait for subagent response");
    let run = harness.finish_cleanly().expect("finish harness");
    assert!(
        run.screen_text.contains("■ subagent-ok"),
        "interactive RLM screen did not show submitted subagent value\nscreen:\n{}\nartifacts: {}",
        run.screen_text,
        run.artifacts.output_dir.display()
    );
    let trace = std::fs::read_to_string(&run.artifacts.ui_trace_json).unwrap_or_else(|err| {
        panic!(
            "failed to read UI trace {}: {err}\nvisible screen:\n{}",
            run.artifacts.ui_trace_json.display(),
            run.screen_text
        )
    });
    assert!(
        trace.contains("code_block_started") && trace.contains("spawn_agent"),
        "interactive RLM trace did not include lashlang/subagent execution\ntrace:\n{trace}\nvisible screen:\n{}",
        run.screen_text
    );
}

#[test]
fn cli_interactive_pty_rlm_uses_temporary_working_directory() {
    let workspace = tempfile::tempdir().expect("temp workspace");
    let lash_home = test_lash_home("rlm-workspace-smoke");
    let mut harness =
        start_interactive_harness(&lash_home, ExecutionMode::Rlm, Some(workspace.path()));
    harness
        .send_line("Exercise the temporary workspace")
        .expect("send prompt");
    harness
        .wait_for_text("■ workspace-smoke-ok", Duration::from_secs(45))
        .expect("wait for workspace response");
    let run = harness.finish_cleanly().expect("finish harness");
    assert!(
        run.screen_text.contains("■ workspace-smoke-ok"),
        "interactive RLM screen did not show workspace smoke result\nscreen:\n{}\nartifacts: {}",
        run.screen_text,
        run.artifacts.output_dir.display()
    );
    assert!(
        run.screen_text
            .contains(&workspace.path().display().to_string()),
        "workspace smoke did not run from the requested cwd\nscreen:\n{}\nworkspace: {}",
        run.screen_text,
        workspace.path().display()
    );
    let written = std::fs::read_to_string(workspace.path().join("qc-workspace.txt"))
        .expect("workspace smoke file");
    assert_eq!(written, "workspace-smoke-ok\n");
}

#[test]
fn cli_interactive_pty_rlm_treats_nonzero_shell_exit_as_data() {
    let workspace = tempfile::tempdir().expect("temp workspace");
    let lash_home = test_lash_home("rlm-nonzero-exit-smoke");
    let mut harness =
        start_interactive_harness(&lash_home, ExecutionMode::Rlm, Some(workspace.path()));
    harness
        .send_line("Exercise nonzero shell exit")
        .expect("send prompt");
    harness
        .wait_for_text("■ nonzero-smoke-ok exit=7", Duration::from_secs(45))
        .expect("wait for nonzero response");
    let run = harness.finish_cleanly().expect("finish harness");
    assert!(
        run.screen_text.contains("■ nonzero-smoke-ok exit=7"),
        "interactive RLM screen did not show nonzero exit as data\nscreen:\n{}\nartifacts: {}",
        run.screen_text,
        run.artifacts.output_dir.display()
    );
    let trace = std::fs::read_to_string(&run.artifacts.ui_trace_json).unwrap_or_else(|err| {
        panic!(
            "failed to read UI trace {}: {err}\nvisible screen:\n{}",
            run.artifacts.ui_trace_json.display(),
            run.screen_text
        )
    });
    assert!(
        !trace.contains("unwrapped failed module operation"),
        "nonzero shell exit should not become a failed module operation\ntrace:\n{trace}"
    );
}

fn lash_bin() -> std::path::PathBuf {
    std::env::var_os("CARGO_BIN_EXE_lash")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/lash")
        })
}

fn start_interactive_harness(
    lash_home: &tempfile::TempDir,
    execution_mode: ExecutionMode,
    working_dir: Option<&std::path::Path>,
) -> LiveHarness {
    let repo_root =
        repo_root_from_manifest_dir(env!("CARGO_MANIFEST_DIR")).expect("derive repo root");
    let mut config = HarnessConfig::new(repo_root);
    config.lash_bin = Some(lash_bin());
    config.lash_home = Some(lash_home.path().to_path_buf());
    config.model = Some("test/cli-e2e-model".to_string());
    config.execution_mode = execution_mode;
    config.working_dir = working_dir.map(std::path::Path::to_path_buf);
    config.build_lash = false;
    config.timeout = Duration::from_secs(45);
    LiveHarness::start(config).expect("start interactive PTY harness")
}

fn test_lash_home(scenario: &str) -> tempfile::TempDir {
    let temp = tempfile::tempdir().expect("temp lash home");
    write_test_provider_config(temp.path(), scenario);
    write_test_model_catalog(temp.path());
    temp
}

fn write_test_provider_config(lash_home: &std::path::Path, scenario: &str) {
    let config = serde_json::json!({
        "active_provider": "test",
        "providers": {
            "test": {
                "type": "test",
                "scenario": scenario
            }
        }
    });
    std::fs::create_dir_all(lash_home).expect("create lash home");
    std::fs::write(
        lash_home.join("config.json"),
        serde_json::to_vec_pretty(&config).expect("config json"),
    )
    .expect("write config");
}

fn write_test_model_catalog(lash_home: &std::path::Path) {
    let cache_dir = lash_home.join("cache");
    std::fs::create_dir_all(&cache_dir).expect("create model cache");
    let catalog = serde_json::json!({
        "test": {
            "models": {
                "cli-e2e-model": {
                    "limit": {
                        "context": 64000,
                        "output": 4096
                    }
                }
            }
        }
    });
    std::fs::write(
        cache_dir.join("models.json"),
        serde_json::to_vec_pretty(&catalog).expect("catalog json"),
    )
    .expect("write model catalog");
}

fn trace_op_count(trace: &serde_json::Value, op: &str, text: &str) -> usize {
    trace
        .get("ops")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("op").and_then(serde_json::Value::as_str) == Some(op))
        .filter(|entry| entry.get("text").and_then(serde_json::Value::as_str) == Some(text))
        .count()
}

fn provider_request_visible_user_texts(lash_home: &std::path::Path) -> Vec<Vec<String>> {
    let path = lash_home.join("test-provider-requests.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read provider request log {}: {err}", path.display()));
    raw.lines()
        .map(|line| {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("parse provider request log line");
            value
                .get("user_texts")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn wait_for_provider_marker(lash_home: &std::path::Path, marker: &str, timeout: Duration) {
    let path = lash_home.join(marker);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting for provider marker {}", path.display());
}

fn screen_cell_for(screen: &str, needle: &str) -> Option<(u16, u16)> {
    screen
        .lines()
        .enumerate()
        .find_map(|(row, line)| line.find(needle).map(|col| (col as u16, row as u16)))
}

fn run_lash_with_timeout(command: &mut Command, timeout: Duration) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lash CLI");
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait().expect("poll lash CLI").is_some() {
            return child.wait_with_output().expect("lash CLI output");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("timed-out lash CLI output");
            panic!(
                "lash CLI timed out after {:?}\nstdout:\n{}\nstderr:\n{}",
                timeout,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
