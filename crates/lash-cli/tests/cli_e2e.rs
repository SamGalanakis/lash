#![cfg(feature = "test-provider")]

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use lash_debug_cli_harness::{
    ExecutionMode, HarnessConfig, LiveHarness, repo_root_from_manifest_dir,
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
        trace.contains("lashlang_code") && trace.contains("spawn_agent"),
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
