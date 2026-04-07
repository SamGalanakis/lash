use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::ToolResult;
use crate::lash_cache_dir;
use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolResultProjectionContext, ToolResultProjectionHook,
};

const APPROX_BYTES_PER_TOKEN: usize = 4;
pub const DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES: usize = 16 * 1024;
pub const DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultProjectionMode {
    Bytes,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ToolResultProjectionPluginConfig {
    pub mode: ToolResultProjectionMode,
    pub limit: usize,
    pub max_lines: usize,
}

impl Default for ToolResultProjectionPluginConfig {
    fn default() -> Self {
        Self {
            mode: ToolResultProjectionMode::Bytes,
            limit: DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES,
            max_lines: DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionDirection {
    Head,
    Tail,
}

pub struct BuiltinToolResultProjectionPluginFactory {
    config: ToolResultProjectionPluginConfig,
}

impl BuiltinToolResultProjectionPluginFactory {
    pub fn new(config: ToolResultProjectionPluginConfig) -> Self {
        Self { config }
    }
}

impl Default for BuiltinToolResultProjectionPluginFactory {
    fn default() -> Self {
        Self::new(ToolResultProjectionPluginConfig::default())
    }
}

impl PluginFactory for BuiltinToolResultProjectionPluginFactory {
    fn id(&self) -> &'static str {
        "tool_result_projection"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ToolResultProjectionPlugin {
            config: self.config.clone(),
        }))
    }
}

struct ToolResultProjectionPlugin {
    config: ToolResultProjectionPluginConfig,
}

impl SessionPlugin for ToolResultProjectionPlugin {
    fn id(&self) -> &'static str {
        "tool_result_projection"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        register_projector(reg, ToolResultProjectionHook::BeforeState, &self.config)?;
        register_projector(reg, ToolResultProjectionHook::BeforeModel, &self.config)?;
        register_projector(reg, ToolResultProjectionHook::BeforeHistory, &self.config)?;
        Ok(())
    }
}

fn register_projector(
    reg: &mut PluginRegistrar,
    hook: ToolResultProjectionHook,
    config: &ToolResultProjectionPluginConfig,
) -> Result<(), PluginError> {
    let config = config.clone();
    reg.tool_results().projector(
        hook,
        Arc::new(move |ctx| {
            let config = config.clone();
            Box::pin(async move { Ok(project_tool_result(&config, ctx)) })
        }),
    )
}

fn project_tool_result(
    config: &ToolResultProjectionPluginConfig,
    ctx: ToolResultProjectionContext,
) -> ToolResult {
    let result = match ctx.hook {
        ToolResultProjectionHook::BeforeState => project_stateful_value(config, &ctx),
        ToolResultProjectionHook::BeforeModel => project_model_value(config, &ctx),
        ToolResultProjectionHook::BeforeHistory => project_stateful_value(config, &ctx),
    };
    ToolResult {
        success: ctx.result.success,
        result,
        images: ctx.result.images,
    }
}

fn project_model_value(
    config: &ToolResultProjectionPluginConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    if ctx.tool_name == "batch" {
        return serde_json::Value::String(render_batch_model_summary(&ctx.result.result));
    }
    let rendered = render_tool_result_payload(ctx.result.success, &ctx.result.result);
    if !needs_truncation(&rendered, config) {
        return project_json_value(&ctx.result.result, config, ctx);
    }
    serde_json::Value::String(formatted_truncate_text(
        &rendered,
        config,
        tool_projection_direction(&ctx.tool_name),
        Some(ctx),
    ))
}

fn project_stateful_value(
    config: &ToolResultProjectionPluginConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    if ctx.tool_name == "batch" {
        return project_batch_history_value(&ctx.result.result, config, ctx);
    }
    project_json_value(&ctx.result.result, config, ctx)
}

fn render_tool_result_payload(success: bool, value: &serde_json::Value) -> String {
    if success {
        match value {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        }
    } else {
        match value {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(&serde_json::json!({ "error": other }))
                .unwrap_or_else(|_| "{\"error\":\"tool execution failed\"}".to_string()),
        }
    }
}

fn project_json_value(
    value: &serde_json::Value,
    config: &ToolResultProjectionPluginConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    match value {
        serde_json::Value::String(text) => {
            serde_json::Value::String(project_text(text, config, ctx))
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| project_json_value(item, config, ctx))
                .collect(),
        ),
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), project_json_value(value, config, ctx)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn project_text(
    text: &str,
    config: &ToolResultProjectionPluginConfig,
    ctx: &ToolResultProjectionContext,
) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    truncate_text(
        text,
        config,
        tool_projection_direction(&ctx.tool_name),
        Some(ctx),
    )
}

fn needs_truncation(text: &str, config: &ToolResultProjectionPluginConfig) -> bool {
    if text.lines().count() > config.max_lines {
        return true;
    }
    match config.mode {
        ToolResultProjectionMode::Bytes => text.len() > config.limit,
        ToolResultProjectionMode::Tokens => approx_token_count(text) > config.limit,
    }
}

fn formatted_truncate_text(
    text: &str,
    config: &ToolResultProjectionPluginConfig,
    direction: ProjectionDirection,
    ctx: Option<&ToolResultProjectionContext>,
) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    truncate_text(text, config, direction, ctx)
}

fn truncate_text(
    text: &str,
    config: &ToolResultProjectionPluginConfig,
    direction: ProjectionDirection,
    ctx: Option<&ToolResultProjectionContext>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    let max_bytes = match config.mode {
        ToolResultProjectionMode::Bytes => config.limit,
        ToolResultProjectionMode::Tokens => approx_bytes_for_tokens(config.limit),
    };
    if max_bytes == 0 {
        return format_truncation_marker(
            config.mode,
            removed_units(config.mode, text.len(), text.chars().count()),
        );
    }
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut preview_lines = Vec::new();
    let mut bytes = 0usize;
    let mut hit_budget = false;

    match direction {
        ProjectionDirection::Head => {
            for (idx, line) in lines.iter().enumerate().take(config.max_lines) {
                let size = line.len() + usize::from(idx > 0);
                if bytes + size > max_bytes {
                    hit_budget = true;
                    break;
                }
                preview_lines.push(*line);
                bytes += size;
            }
        }
        ProjectionDirection::Tail => {
            for (idx, line) in lines.iter().rev().take(config.max_lines).enumerate() {
                let size = line.len() + usize::from(idx > 0);
                if bytes + size > max_bytes {
                    hit_budget = true;
                    break;
                }
                preview_lines.push(*line);
                bytes += size;
            }
            preview_lines.reverse();
        }
    }

    let preview = preview_lines.join("\n");
    let removed = if hit_budget {
        removed_units(
            config.mode,
            text.len().saturating_sub(bytes),
            text.chars().count().saturating_sub(preview.chars().count()),
        )
    } else {
        u64::try_from(lines.len().saturating_sub(preview_lines.len())).unwrap_or(u64::MAX)
    };
    let unit = if hit_budget {
        match config.mode {
            ToolResultProjectionMode::Bytes => "bytes",
            ToolResultProjectionMode::Tokens => "tokens",
        }
    } else {
        "lines"
    };
    let hint = truncation_hint(ctx, text);
    match direction {
        ProjectionDirection::Head => {
            format!("{preview}\n\n...{removed} {unit} truncated...\n\n{hint}")
        }
        ProjectionDirection::Tail => {
            format!("...{removed} {unit} truncated...\n\n{hint}\n\n{preview}")
        }
    }
}

fn format_truncation_marker(mode: ToolResultProjectionMode, removed: u64) -> String {
    match mode {
        ToolResultProjectionMode::Bytes => format!("…{removed} chars truncated…"),
        ToolResultProjectionMode::Tokens => format!("…{removed} tokens truncated…"),
    }
}

fn removed_units(
    mode: ToolResultProjectionMode,
    removed_bytes: usize,
    removed_chars: usize,
) -> u64 {
    match mode {
        ToolResultProjectionMode::Bytes => u64::try_from(removed_chars).unwrap_or(u64::MAX),
        ToolResultProjectionMode::Tokens => approx_tokens_from_byte_count(removed_bytes),
    }
}

fn approx_token_count(text: &str) -> usize {
    text.len()
        .saturating_add(APPROX_BYTES_PER_TOKEN.saturating_sub(1))
        / APPROX_BYTES_PER_TOKEN
}

fn approx_bytes_for_tokens(tokens: usize) -> usize {
    tokens.saturating_mul(APPROX_BYTES_PER_TOKEN)
}

fn approx_tokens_from_byte_count(bytes: usize) -> u64 {
    let bytes = bytes as u64;
    bytes.saturating_add((APPROX_BYTES_PER_TOKEN as u64).saturating_sub(1))
        / (APPROX_BYTES_PER_TOKEN as u64)
}

fn tool_projection_direction(tool_name: &str) -> ProjectionDirection {
    match tool_name {
        "exec_command" | "write_stdin" => ProjectionDirection::Tail,
        _ => ProjectionDirection::Head,
    }
}

fn truncation_hint(ctx: Option<&ToolResultProjectionContext>, text: &str) -> String {
    let output_path =
        ctx.and_then(|ctx| spill_tool_output(&ctx.session_id, &ctx.tool_name, &ctx.args, text));
    match output_path {
        Some(path) => format!(
            "The tool output was truncated. Full output saved to: {}\nUse `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.",
            path.display()
        ),
        None => "The tool output was truncated. Use `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.".to_string(),
    }
}

fn spill_tool_output(
    session_id: &str,
    tool_name: &str,
    args: &serde_json::Value,
    full_output: &str,
) -> Option<PathBuf> {
    let dir = lash_cache_dir().join("tool-output").join(session_id);
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }

    let mut hasher = Sha256::new();
    hasher.update(tool_name.as_bytes());
    hasher.update(args.to_string().as_bytes());
    hasher.update(full_output.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    let stem = tool_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let path = dir.join(format!("{stem}-{}.txt", &digest[..12]));
    if write_if_changed(&path, full_output).is_err() {
        return None;
    }
    Some(path)
}

fn write_if_changed(path: &Path, content: &str) -> std::io::Result<()> {
    let should_write = match fs::read_to_string(path) {
        Ok(existing) => existing != content,
        Err(_) => true,
    };
    if should_write {
        fs::write(path, content)?;
    }
    Ok(())
}

fn render_batch_model_summary(value: &serde_json::Value) -> String {
    let summary = value
        .get("summary")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            let successful = value
                .get("successful")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let total = value
                .get("total")
                .and_then(|value| value.as_u64())
                .unwrap_or(successful);
            format!("Executed {successful}/{total} tools successfully.")
        });

    let mut lines = vec![summary];
    if let Some(results) = value.get("results").and_then(|value| value.as_array()) {
        let details = results
            .iter()
            .filter_map(|item| {
                let tool = item.get("tool").and_then(|value| value.as_str())?;
                let status = if item.get("success").and_then(|value| value.as_bool()) == Some(true)
                {
                    "ok"
                } else {
                    "failed"
                };
                Some(format!("- {tool}: {status}"))
            })
            .collect::<Vec<_>>();
        if !details.is_empty() {
            lines.push(String::new());
            lines.extend(details);
        }
    }
    lines.join("\n")
}

fn project_batch_history_value(
    value: &serde_json::Value,
    config: &ToolResultProjectionPluginConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    let Some(map) = value.as_object() else {
        return project_json_value(value, config, ctx);
    };
    let mut projected = serde_json::Map::new();
    for key in ["summary", "total", "successful", "failed"] {
        if let Some(value) = map.get(key) {
            projected.insert(key.to_string(), value.clone());
        }
    }
    let details = map
        .get("results")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    let mut detail = serde_json::Map::new();
                    if let Some(tool) = item.get("tool") {
                        detail.insert("tool".to_string(), tool.clone());
                    }
                    if let Some(success) = item.get("success") {
                        detail.insert("success".to_string(), success.clone());
                    }
                    if let Some(duration_ms) = item.get("duration_ms") {
                        detail.insert("duration_ms".to_string(), duration_ms.clone());
                    }
                    if let Some(error) = item.get("error") {
                        detail.insert("error".to_string(), project_json_value(error, config, ctx));
                    }
                    serde_json::Value::Object(detail)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    projected.insert("details".to_string(), serde_json::Value::Array(details));
    serde_json::Value::Object(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::runtime_impl::NoopSessionManager;
    use serde_json::json;

    #[test]
    fn truncates_strings_with_terminal_style_marker() {
        let config = ToolResultProjectionPluginConfig {
            mode: ToolResultProjectionMode::Tokens,
            limit: 5,
            max_lines: DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
        };
        let got = project_text(
            "this is an example of a long output that should be truncated",
            &config,
            &ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeHistory,
                session_id: "root".to_string(),
                tool_name: "grep".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!("unused")),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        assert!(got.contains("tokens truncated"));
        assert!(got.contains("Full output saved to:"));
    }

    #[test]
    fn history_projection_preserves_shape() {
        let config = ToolResultProjectionPluginConfig {
            limit: 512,
            ..ToolResultProjectionPluginConfig::default()
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeHistory,
                session_id: "root".to_string(),
                tool_name: "exec_command".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "output": "x".repeat(20_000),
                    "nested": { "stderr": "y".repeat(20_000) }
                })),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        assert!(projected.result.is_object());
        let output = projected
            .result
            .get("output")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        assert!(output.contains("bytes truncated"));
        assert!(output.contains("Full output saved to:"));
    }

    #[test]
    fn state_projection_preserves_shape() {
        let config = ToolResultProjectionPluginConfig {
            limit: 512,
            ..ToolResultProjectionPluginConfig::default()
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeState,
                session_id: "root".to_string(),
                tool_name: "exec_command".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "output": "x".repeat(20_000),
                    "nested": { "stderr": "y".repeat(20_000) }
                })),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        assert!(projected.result.is_object());
        let output = projected
            .result
            .get("output")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        assert!(output.contains("bytes truncated"));
        assert!(output.contains("Full output saved to:"));
    }

    #[test]
    fn model_projection_can_collapse_large_structured_payload_to_string() {
        let config = ToolResultProjectionPluginConfig {
            mode: ToolResultProjectionMode::Bytes,
            limit: 40,
            max_lines: DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeModel,
                session_id: "root".to_string(),
                tool_name: "search_tools".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "results": [{"output": "x".repeat(200)}]
                })),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        assert!(projected.result.is_string());
        assert!(
            projected
                .result
                .as_str()
                .unwrap_or_default()
                .contains("bytes truncated")
        );
    }

    #[test]
    fn batch_model_projection_drops_nested_child_payloads() {
        let projected = project_tool_result(
            &ToolResultProjectionPluginConfig::default(),
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeModel,
                session_id: "root".to_string(),
                tool_name: "batch".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "summary": "Executed 1/2 tools successfully. 1 failed.",
                    "total": 2,
                    "successful": 1,
                    "failed": 1,
                    "results": [
                        {"tool": "read_file", "success": true, "duration_ms": 1, "result": "very long child payload"},
                        {"tool": "grep", "success": false, "duration_ms": 1, "error": "boom"}
                    ]
                })),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        let text = projected.result.as_str().unwrap_or_default();
        assert!(text.contains("Executed 1/2 tools successfully. 1 failed."));
        assert!(text.contains("- read_file: ok"));
        assert!(!text.contains("very long child payload"));
    }

    #[test]
    fn batch_history_projection_keeps_only_metadata() {
        let projected = project_tool_result(
            &ToolResultProjectionPluginConfig::default(),
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeHistory,
                session_id: "root".to_string(),
                tool_name: "batch".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "summary": "Executed 1/2 tools successfully. 1 failed.",
                    "total": 2,
                    "successful": 1,
                    "failed": 1,
                    "results": [
                        {"tool": "read_file", "success": true, "duration_ms": 1, "result": "child payload"},
                        {"tool": "grep", "success": false, "duration_ms": 1, "error": "boom"}
                    ]
                })),
                duration_ms: 1,
                host: Arc::new(NoopSessionManager),
            },
        );
        let details = projected
            .result
            .get("details")
            .and_then(|value| value.as_array())
            .expect("details");
        assert_eq!(details.len(), 2);
        assert!(details[0].get("result").is_none());
        assert_eq!(details[1].get("error"), Some(&json!("boom")));
    }
}
