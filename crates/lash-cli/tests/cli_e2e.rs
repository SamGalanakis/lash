#![cfg(feature = "test-provider")]

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

#[test]
fn cli_short_rlm_mode_runs_subagent_spawn_with_test_provider() {
    let temp = tempfile::tempdir().expect("temp lash home");
    write_test_provider_config(temp.path());
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

fn lash_bin() -> std::path::PathBuf {
    std::env::var_os("CARGO_BIN_EXE_lash")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/lash")
        })
}

fn write_test_provider_config(lash_home: &std::path::Path) {
    let config = serde_json::json!({
        "active_provider": "test",
        "providers": {
            "test": {
                "type": "test",
                "scenario": "rlm-subagent-smoke"
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
