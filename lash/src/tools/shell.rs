use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// Run shell commands via bash.
pub struct Shell {
    timeout: std::time::Duration,
}

impl Shell {
    pub fn new() -> Self {
        Self {
            timeout: std::time::Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_OUTPUT: usize = 50_000;

#[async_trait::async_trait]
impl ToolProvider for Shell {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "bash".into(),
            description: "Run a bash command and return stdout+stderr (30s timeout, 50KB output limit).".into(),
            params: vec![ToolParam::typed("command", "str")],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if command.is_empty() {
            return ToolResult::err(json!("Missing required parameter: command"));
        }

        let child = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let child = match child {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(json!(format!("Failed to spawn: {e}")))
            }
        };

        let output = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return ToolResult::err(json!(format!("Process error: {e}")))
            }
            Err(_) => {
                // child was moved into wait_with_output, so we can't kill it here.
                // The process will be cleaned up when the Child is dropped.
                return ToolResult::err(json!(format!(
                        "Command timed out after {}s",
                        self.timeout.as_secs()
                    )));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);

        let mut combined = String::new();
        if !stdout.is_empty() {
            combined.push_str(&stdout[..stdout.len().min(MAX_OUTPUT)]);
        }
        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            let remaining = MAX_OUTPUT.saturating_sub(combined.len());
            combined.push_str(&stderr[..stderr.len().min(remaining)]);
        }

        let truncated = stdout.len() + stderr.len() > MAX_OUTPUT;
        if truncated {
            combined.push_str("\n[truncated]");
        }

        if code != 0 {
            combined.push_str(&format!("\n[exit code: {}]", code));
        }

        ToolResult {
            success: code == 0,
            result: json!(combined),
            images: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;

    #[tokio::test]
    async fn test_echo() {
        let tool = Shell::default();
        let result = tool
            .execute("bash", &json!({"command": "echo hello"}))
            .await;
        assert!(result.success);
        assert!(result.result.as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn test_nonzero_exit() {
        let tool = Shell::default();
        let result = tool
            .execute("bash", &json!({"command": "exit 1"}))
            .await;
        assert!(!result.success);
        assert!(result.result.as_str().unwrap().contains("exit code: 1"));
    }

    #[tokio::test]
    async fn test_timeout() {
        let tool = Shell::new().with_timeout(std::time::Duration::from_millis(100));
        let result = tool
            .execute("bash", &json!({"command": "sleep 10"}))
            .await;
        assert!(!result.success);
        assert!(result.result.as_str().unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn test_missing_command() {
        let tool = Shell::default();
        let result = tool.execute("bash", &json!({})).await;
        assert!(!result.success);
    }
}
