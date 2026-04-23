use std::path::{Path, PathBuf};

use serde_json::json;

use crate::{ToolDefinition, ToolExecutionMode, ToolParam, ToolProvider, ToolResult};

use super::{require_str, run_blocking};

const MAX_SNIPPET_LINES: usize = 400;

#[derive(Default)]
pub struct ShowSnippetToUser;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnippetRenderMode {
    Markdown,
    Code,
    Text,
}

impl SnippetRenderMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Code => "code",
            Self::Text => "text",
        }
    }
}

impl ShowSnippetToUser {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ToolProvider for ShowSnippetToUser {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "show_snippet_to_user".into(),
            description: "Show a specific line range from a file as a framed snippet to the user. Only for sharing a snippet with the user — not for your own viewing or model-context ingestion. On success the tool returns `{ displayed: true, path, start_line, end_line, ... }`; that `displayed: true` flag is the confirmation that the snippet card was dispatched to the UI. There is no separate user-acknowledgement signal."
                .into(),
            params: vec![
                ToolParam::typed("path", "str"),
                ToolParam::typed("start_line", "int"),
                ToolParam::typed("end_line", "int"),
                ToolParam::optional("title", "str"),
            ],
            returns: "dict".into(),
            examples: vec![
                "show_snippet_to_user(path=\"lash-cli/src/render/mod.rs\", start_line=120, end_line=180)"
                    .into(),
            ],
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let path_str = match require_str(args, "path") {
            Ok(value) => value.to_string(),
            Err(err) => return err,
        };
        let start_line = match parse_required_line_arg(args, "start_line") {
            Ok(value) => value,
            Err(err) => return err,
        };
        let end_line = match parse_required_line_arg(args, "end_line") {
            Ok(value) => value,
            Err(err) => return err,
        };
        if end_line < start_line {
            return ToolResult::err_fmt(format_args!("`end_line` must be >= `start_line`"));
        }
        let line_count = end_line - start_line + 1;
        if line_count > MAX_SNIPPET_LINES {
            return ToolResult::err_fmt(format_args!(
                "Snippet range too large: {} lines requested (max {}).",
                line_count, MAX_SNIPPET_LINES
            ));
        }

        let title = args
            .get("title")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);

        run_blocking(move || execute_show_snippet_to_user(&path_str, start_line, end_line, title))
            .await
    }
}

fn parse_required_line_arg(args: &serde_json::Value, key: &str) -> Result<usize, ToolResult> {
    let Some(raw) = args.get(key) else {
        return Err(ToolResult::err_fmt(format_args!(
            "Missing required parameter: {key}"
        )));
    };
    let Some(value) = raw.as_u64() else {
        return Err(ToolResult::err_fmt(format_args!(
            "Invalid {key}: expected int >= 1"
        )));
    };
    let value = value as usize;
    if value == 0 {
        return Err(ToolResult::err_fmt(format_args!(
            "Invalid {key}: expected int >= 1"
        )));
    }
    Ok(value)
}

fn execute_show_snippet_to_user(
    path_str: &str,
    start_line: usize,
    end_line: usize,
    title: Option<String>,
) -> ToolResult {
    let path = Path::new(path_str);
    if !path.exists() {
        return ToolResult::err_fmt(format_args!("Path does not exist: {path_str}"));
    }
    if path.is_dir() {
        return ToolResult::err_fmt(format_args!("Cannot show a directory: {path_str}"));
    }

    let content = match std::fs::read_to_string(path) {
        Ok(value) => value,
        Err(err) => {
            return ToolResult::err_fmt(format_args!("Failed to read `{path_str}`: {err}"));
        }
    };
    let lines = content.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return ToolResult::err_fmt(format_args!("File is empty: {path_str}"));
    }
    if start_line > lines.len() {
        return ToolResult::err_fmt(format_args!(
            "`start_line` {} is past end of file ({} lines)",
            start_line,
            lines.len()
        ));
    }
    let end_line = end_line.min(lines.len());
    let snippet = lines[start_line - 1..end_line].join("\n");
    let display_path = display_path(path);
    let (render_mode, language) = classify_path(path, &content);

    ToolResult::ok(json!({
        "displayed": true,
        "path": display_path,
        "start_line": start_line,
        "end_line": end_line,
        "title": title,
        "content": snippet,
        "render_mode": render_mode.as_str(),
        "language": language,
    }))
}

fn display_path(path: &Path) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| path.strip_prefix(&cwd).ok().map(PathBuf::from))
        .unwrap_or_else(|| path.to_path_buf())
        .display()
        .to_string()
        .replace('\\', "/")
}

fn classify_path(path: &Path, content: &str) -> (SnippetRenderMode, Option<String>) {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match file_name {
        "Dockerfile" => return (SnippetRenderMode::Code, Some("dockerfile".into())),
        "Makefile" | "GNUmakefile" => return (SnippetRenderMode::Code, Some("make".into())),
        "justfile" | "Justfile" => return (SnippetRenderMode::Code, Some("just".into())),
        "README" | "CHANGELOG" | "CONTRIBUTING" | "AGENTS" | "CLAUDE" => {
            return (SnippetRenderMode::Markdown, Some("markdown".into()));
        }
        _ => {}
    }
    match ext.as_str() {
        "md" | "markdown" | "mdx" => (SnippetRenderMode::Markdown, Some("markdown".into())),
        "rs" | "py" | "js" | "jsx" | "ts" | "tsx" | "go" | "java" | "c" | "h" | "cc" | "cpp"
        | "cxx" | "hpp" | "hxx" | "cs" | "rb" | "php" | "swift" | "kt" | "kts" | "scala" | "sh"
        | "bash" | "zsh" | "fish" | "toml" | "yaml" | "yml" | "json" | "html" | "css" | "scss"
        | "sql" | "nix" | "lua" | "zig" => {
            let language = match ext.as_str() {
                "yml" => Some("yaml".to_string()),
                _ => (!ext.is_empty()).then_some(ext),
            };
            (SnippetRenderMode::Code, language)
        }
        _ => classify_from_shebang(content),
    }
}

fn classify_from_shebang(content: &str) -> (SnippetRenderMode, Option<String>) {
    let Some(first_line) = content.lines().next().map(str::trim) else {
        return (SnippetRenderMode::Text, None);
    };
    let Some(shebang) = first_line.strip_prefix("#!") else {
        return (SnippetRenderMode::Text, None);
    };
    let language = shebang
        .split_whitespace()
        .last()
        .and_then(|value| value.rsplit('/').next())
        .map(|value| value.to_ascii_lowercase());
    match language.as_deref() {
        Some("python") | Some("python3") => (SnippetRenderMode::Code, Some("py".into())),
        Some("bash") => (SnippetRenderMode::Code, Some("bash".into())),
        Some("sh") => (SnippetRenderMode::Code, Some("sh".into())),
        Some("zsh") => (SnippetRenderMode::Code, Some("zsh".into())),
        Some("fish") => (SnippetRenderMode::Code, Some("fish".into())),
        Some("node") => (SnippetRenderMode::Code, Some("js".into())),
        Some("ruby") => (SnippetRenderMode::Code, Some("rb".into())),
        Some(other) if !other.is_empty() => (SnippetRenderMode::Code, Some(other.to_string())),
        _ => (SnippetRenderMode::Code, Some("sh".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn show_snippet_to_user_returns_code_metadata_and_content() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("sample.rs");
        std::fs::write(&path, "fn one() {}\nfn two() {}\nfn three() {}\n").expect("write file");

        let tool = ShowSnippetToUser::new();
        let result = tool
            .execute(
                "show_snippet_to_user",
                &json!({
                    "path": path,
                    "start_line": 2,
                    "end_line": 3,
                    "title": "Focus"
                }),
            )
            .await;

        assert!(result.success);
        assert_eq!(
            result.result["displayed"],
            json!(true),
            "tool must explicitly signal the snippet was dispatched to the UI"
        );
        assert!(
            result.result["path"]
                .as_str()
                .is_some_and(|value| value.ends_with("sample.rs"))
        );
        assert_eq!(result.result["start_line"], json!(2));
        assert_eq!(result.result["end_line"], json!(3));
        assert_eq!(result.result["title"], json!("Focus"));
        assert_eq!(result.result["render_mode"], json!("code"));
        assert_eq!(result.result["language"], json!("rs"));
        assert_eq!(
            result.result["content"],
            json!("fn two() {}\nfn three() {}")
        );
    }

    #[tokio::test]
    async fn show_snippet_to_user_detects_markdown() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("README.md");
        std::fs::write(&path, "# Title\n\n- item\n").expect("write file");

        let tool = ShowSnippetToUser::new();
        let result = tool
            .execute(
                "show_snippet_to_user",
                &json!({
                    "path": path,
                    "start_line": 1,
                    "end_line": 3
                }),
            )
            .await;

        assert!(result.success);
        assert_eq!(result.result["render_mode"], json!("markdown"));
        assert_eq!(result.result["language"], json!("markdown"));
    }

    #[tokio::test]
    async fn show_snippet_to_user_detects_shebang_script_as_code() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("script");
        std::fs::write(&path, "#!/usr/bin/env python3\nprint('hi')\n").expect("write file");

        let tool = ShowSnippetToUser::new();
        let result = tool
            .execute(
                "show_snippet_to_user",
                &json!({
                    "path": path,
                    "start_line": 1,
                    "end_line": 2
                }),
            )
            .await;

        assert!(result.success);
        assert_eq!(result.result["render_mode"], json!("code"));
        assert_eq!(result.result["language"], json!("py"));
    }

    #[tokio::test]
    async fn show_snippet_to_user_rejects_invalid_range() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("notes.txt");
        std::fs::write(&path, "one\ntwo\n").expect("write file");

        let tool = ShowSnippetToUser::new();
        let result = tool
            .execute(
                "show_snippet_to_user",
                &json!({
                    "path": path,
                    "start_line": 3,
                    "end_line": 2
                }),
            )
            .await;

        assert!(!result.success);
        assert!(
            result
                .result
                .as_str()
                .is_some_and(|message| message.contains("`end_line` must be >= `start_line`"))
        );
    }
}
