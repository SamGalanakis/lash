#![cfg(feature = "test-provider")]

use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{io::Read, io::Write};

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
    let temp = tempfile::tempdir().expect("temp lash home");
    write_test_provider_config(temp.path(), "standard-echo");
    write_test_model_catalog(temp.path());
    let trace_path = temp.path().join("interactive-trace.json");
    let snapshot_path = temp.path().join("interactive-trace.snap");

    let mut session = PtySession::spawn(
        &[
            "--model",
            "test/cli-e2e-model",
            "--debug-ui-trace",
            trace_path.to_str().expect("trace path is utf8"),
        ],
        temp.path(),
        env!("CARGO_MANIFEST_DIR"),
    );
    session.wait_for("Idle", Duration::from_secs(10));
    session.write("hello from pty\r");
    session.wait_for(
        "test-provider echo: hello from pty",
        Duration::from_secs(30),
    );
    session.write("/exit\r");
    let (status, output) = session.finish(Duration::from_secs(10));

    assert!(
        status.success(),
        "interactive lash exited unsuccessfully: {status:?}\npty output:\n{output}"
    );
    let snapshot = std::fs::read_to_string(&snapshot_path).unwrap_or_else(|err| {
        panic!(
            "failed to read final UI snapshot {}: {err}\npty output:\n{output}",
            snapshot_path.display()
        )
    });
    assert!(
        snapshot.contains("hello from pty"),
        "final UI snapshot did not contain submitted prompt\nsnapshot:\n{snapshot}\npty output:\n{output}"
    );
    assert!(
        snapshot.contains("test-provider echo: hello from pty"),
        "final UI snapshot did not contain provider response\nsnapshot:\n{snapshot}\npty output:\n{output}"
    );
    let trace = std::fs::read_to_string(&trace_path).unwrap_or_else(|err| {
        panic!(
            "failed to read UI trace {}: {err}\npty output:\n{output}",
            trace_path.display()
        )
    });
    assert!(
        trace.contains("\"user_turn\"") && trace.contains("hello from pty"),
        "UI trace did not record the submitted turn\ntrace:\n{trace}\npty output:\n{output}"
    );
}

fn lash_bin() -> std::path::PathBuf {
    std::env::var_os("CARGO_BIN_EXE_lash")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/lash")
        })
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

struct PtySession {
    child: Box<dyn portable_pty::Child + Send>,
    writer: Box<dyn Write + Send>,
    output: Arc<Mutex<Vec<u8>>>,
    reader_thread: Option<std::thread::JoinHandle<()>>,
}

impl PtySession {
    fn spawn(args: &[&str], lash_home: &std::path::Path, cwd: &str) -> Self {
        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(portable_pty::PtySize {
                rows: 24,
                cols: 100,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("open pty");

        let mut cmd = portable_pty::CommandBuilder::new(lash_bin());
        cmd.args(args);
        cmd.cwd(cwd);
        cmd.env("LASH_HOME", lash_home.as_os_str());
        cmd.env("LASH_LOG", "warn");
        cmd.env("TERM", "xterm-256color");
        cmd.env("NO_COLOR", "1");

        let child = pair.slave.spawn_command(cmd).expect("spawn lash in pty");
        let mut reader = pair.master.try_clone_reader().expect("clone pty reader");
        let writer = pair.master.take_writer().expect("take pty writer");
        drop(pair.slave);

        let output = Arc::new(Mutex::new(Vec::new()));
        let reader_output = Arc::clone(&output);
        let reader_thread = std::thread::spawn(move || {
            let mut buf = [0_u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => reader_output
                        .lock()
                        .expect("pty output lock")
                        .extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
            }
        });

        Self {
            child,
            writer,
            output,
            reader_thread: Some(reader_thread),
        }
    }

    fn write(&mut self, input: &str) {
        self.writer
            .write_all(input.as_bytes())
            .expect("write pty input");
        self.writer.flush().expect("flush pty input");
    }

    fn wait_for(&mut self, needle: &str, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            let output = self.output_string();
            if output.contains(needle) {
                return;
            }
            if let Some(status) = self.child.try_wait().expect("poll pty child") {
                panic!("lash exited before `{needle}` appeared: {status:?}\npty output:\n{output}");
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                panic!("timed out waiting for `{needle}`\npty output:\n{output}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn finish(mut self, timeout: Duration) -> (portable_pty::ExitStatus, String) {
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("poll pty child") {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let output = self.output_string();
                panic!("timed out waiting for lash to exit\npty output:\n{output}");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let output = Arc::clone(&self.output);
        drop(self.writer);
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
        (
            status,
            String::from_utf8_lossy(&output.lock().expect("pty output lock")).to_string(),
        )
    }

    fn output_string(&self) -> String {
        String::from_utf8_lossy(&self.output.lock().expect("pty output lock")).to_string()
    }
}
