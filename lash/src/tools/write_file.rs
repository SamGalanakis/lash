use serde_json::json;
use std::path::Path;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

/// Write complete file contents, creating parent directories if needed.
#[derive(Default)]
pub struct WriteFile;

#[async_trait::async_trait]
impl ToolProvider for WriteFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file, creating parent directories if needed.".into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam::typed("content", "str"),
            ],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = match require_str(args, "path") {
            Ok(s) => s,
            Err(e) => return e,
        };
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let path = Path::new(path_str);

        // Create parent directories
        if let Some(parent) = path.parent()
            && !parent.exists()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return ToolResult::err_fmt(format_args!("Failed to create directories: {e}"));
        }

        match std::fs::write(path, content) {
            Ok(()) => {
                let bytes = content.len();
                ToolResult::ok(json!(format!("Wrote {} bytes to {}", bytes, path_str)))
            }
            Err(e) => ToolResult::err_fmt(format_args!("Failed to write file: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFile::default();
        let path = dir.path().join("test.txt");
        let result = tool
            .execute(
                "write_file",
                &json!({"path": path.to_str().unwrap(), "content": "hello world"}),
            )
            .await;
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_write_nested_path() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFile::default();
        let path = dir.path().join("a/b/c/test.txt");
        let result = tool
            .execute(
                "write_file",
                &json!({"path": path.to_str().unwrap(), "content": "nested"}),
            )
            .await;
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
    }

    #[tokio::test]
    async fn test_overwrite_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFile::default();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "old content").unwrap();
        let result = tool
            .execute(
                "write_file",
                &json!({"path": path.to_str().unwrap(), "content": "new content"}),
            )
            .await;
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_missing_path() {
        let tool = WriteFile::default();
        let result = tool
            .execute("write_file", &json!({"content": "hello"}))
            .await;
        assert!(!result.success);
    }
}
