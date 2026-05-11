use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::ToolResult;
use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolResultProjectionContext,
};

const APPROX_BYTES_PER_TOKEN: usize = 4;
pub const DEFAULT_TOOL_OUTPUT_BUDGET_LIMIT_BYTES: usize = 16 * 1024;
pub const DEFAULT_TOOL_OUTPUT_BUDGET_MAX_LINES: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutputBudgetMode {
    Bytes,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ToolOutputBudgetConfig {
    pub mode: ToolOutputBudgetMode,
    pub limit: usize,
    pub max_lines: usize,
}

impl Default for ToolOutputBudgetConfig {
    fn default() -> Self {
        Self {
            mode: ToolOutputBudgetMode::Bytes,
            limit: DEFAULT_TOOL_OUTPUT_BUDGET_LIMIT_BYTES,
            max_lines: DEFAULT_TOOL_OUTPUT_BUDGET_MAX_LINES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionDirection {
    Head,
    Tail,
}

pub struct ToolOutputBudgetPluginFactory {
    config: ToolOutputBudgetConfig,
}

impl ToolOutputBudgetPluginFactory {
    pub fn new(config: ToolOutputBudgetConfig) -> Self {
        Self { config }
    }
}

impl Default for ToolOutputBudgetPluginFactory {
    fn default() -> Self {
        Self::new(ToolOutputBudgetConfig::default())
    }
}

impl PluginFactory for ToolOutputBudgetPluginFactory {
    fn id(&self) -> &'static str {
        "tool_output_budget"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ToolOutputBudgetPlugin {
            config: self.config.clone(),
        }))
    }
}

struct ToolOutputBudgetPlugin {
    config: ToolOutputBudgetConfig,
}

impl SessionPlugin for ToolOutputBudgetPlugin {
    fn id(&self) -> &'static str {
        "tool_output_budget"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        register_projector(reg, &self.config)
    }
}

fn register_projector(
    reg: &mut PluginRegistrar,
    config: &ToolOutputBudgetConfig,
) -> Result<(), PluginError> {
    let config = config.clone();
    reg.tool_results().projector(Arc::new(move |ctx| {
        let config = config.clone();
        Box::pin(async move { Ok(project_tool_result(&config, ctx)) })
    }))
}

fn project_tool_result(
    config: &ToolOutputBudgetConfig,
    ctx: ToolResultProjectionContext,
) -> ToolResult {
    let result = project_model_value(config, &ctx);
    ToolResult {
        success: ctx.result.success,
        result,
        images: ctx.result.images,
        control: ctx.result.control,
    }
}

fn project_model_value(
    config: &ToolOutputBudgetConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    if ctx.tool_name == "batch" {
        return project_batch_value(config, ctx);
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
    config: &ToolOutputBudgetConfig,
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
    config: &ToolOutputBudgetConfig,
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

fn needs_truncation(text: &str, config: &ToolOutputBudgetConfig) -> bool {
    if text.lines().count() > config.max_lines {
        return true;
    }
    match config.mode {
        ToolOutputBudgetMode::Bytes => text.len() > config.limit,
        ToolOutputBudgetMode::Tokens => approx_token_count(text) > config.limit,
    }
}

fn formatted_truncate_text(
    text: &str,
    config: &ToolOutputBudgetConfig,
    direction: ProjectionDirection,
    ctx: Option<&ToolResultProjectionContext>,
) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    truncate_text_with_hint(text, config, direction, truncation_hint(ctx, text))
}

pub fn truncate_observation_text(text: &str, config: &ToolOutputBudgetConfig) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    truncate_text_with_hint(
        text,
        config,
        ProjectionDirection::Head,
        observation_truncation_hint(text, config),
    )
}

pub fn project_observation_text(
    text: &str,
    config: &ToolOutputBudgetConfig,
) -> (String, crate::TextProjectionMetadata) {
    let projected = truncate_observation_text(text, config);
    let metadata = observation_projection_metadata(text, &projected, config);
    (projected, metadata)
}

pub fn observation_projection_metadata(
    original: &str,
    projected: &str,
    config: &ToolOutputBudgetConfig,
) -> crate::TextProjectionMetadata {
    let limit_mode = match config.mode {
        ToolOutputBudgetMode::Bytes => "bytes",
        ToolOutputBudgetMode::Tokens => "tokens",
    };
    crate::TextProjectionMetadata {
        truncated: original != projected,
        original_chars: original.chars().count(),
        projected_chars: projected.chars().count(),
        original_lines: original.lines().count(),
        projected_lines: projected.lines().count(),
        limit: config.limit,
        limit_mode: limit_mode.to_string(),
        max_lines: config.max_lines,
    }
}

fn truncate_text(
    text: &str,
    config: &ToolOutputBudgetConfig,
    direction: ProjectionDirection,
    ctx: Option<&ToolResultProjectionContext>,
) -> String {
    truncate_text_with_hint(text, config, direction, truncation_hint(ctx, text))
}

fn truncate_text_with_hint(
    text: &str,
    config: &ToolOutputBudgetConfig,
    direction: ProjectionDirection,
    hint: String,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    let max_bytes = match config.mode {
        ToolOutputBudgetMode::Bytes => config.limit,
        ToolOutputBudgetMode::Tokens => approx_bytes_for_tokens(config.limit),
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
            ToolOutputBudgetMode::Bytes => "bytes",
            ToolOutputBudgetMode::Tokens => "tokens",
        }
    } else {
        "lines"
    };
    match direction {
        ProjectionDirection::Head => {
            format!("{preview}\n\n...{removed} {unit} truncated...\n\n{hint}")
        }
        ProjectionDirection::Tail => {
            format!("...{removed} {unit} truncated...\n\n{hint}\n\n{preview}")
        }
    }
}

fn format_truncation_marker(mode: ToolOutputBudgetMode, removed: u64) -> String {
    match mode {
        ToolOutputBudgetMode::Bytes => format!("…{removed} chars truncated…"),
        ToolOutputBudgetMode::Tokens => format!("…{removed} tokens truncated…"),
    }
}

fn removed_units(mode: ToolOutputBudgetMode, removed_bytes: usize, removed_chars: usize) -> u64 {
    match mode {
        ToolOutputBudgetMode::Bytes => u64::try_from(removed_chars).unwrap_or(u64::MAX),
        ToolOutputBudgetMode::Tokens => approx_tokens_from_byte_count(removed_bytes),
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
    let output_path = ctx
        .and_then(existing_tool_output_path)
        .or_else(|| ctx.and_then(|ctx| spill_tool_output(&ctx.tool_name, &ctx.args, text)));
    match output_path {
        Some(path) => format!(
            "The tool output was truncated. Full output saved to: {}\nUse `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.",
            path.display()
        ),
        None => "The tool output was truncated. Use `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.".to_string(),
    }
}

fn observation_truncation_hint(text: &str, config: &ToolOutputBudgetConfig) -> String {
    let limit_unit = match config.mode {
        ToolOutputBudgetMode::Bytes => "bytes",
        ToolOutputBudgetMode::Tokens => "tokens",
    };
    let total_units = match config.mode {
        ToolOutputBudgetMode::Bytes => text.len(),
        ToolOutputBudgetMode::Tokens => approx_token_count(text),
    };
    let total_lines = text.lines().count();
    format!(
        "The print output was capped at {} {} and {} lines max; original size was {} {} across {} lines. Use a narrower `print` expression to inspect specific fields or slices instead of dumping the whole value at once.",
        config.limit, limit_unit, config.max_lines, total_units, limit_unit, total_lines
    )
}

fn existing_tool_output_path(ctx: &ToolResultProjectionContext) -> Option<PathBuf> {
    ctx.result
        .result
        .get("full_output_path")
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}

fn spill_tool_output(
    tool_name: &str,
    args: &serde_json::Value,
    full_output: &str,
) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join("lash-tool-output");
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

fn project_batch_value(
    config: &ToolOutputBudgetConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    let Some(map) = ctx.result.result.as_object() else {
        return project_json_value(&ctx.result.result, config, ctx);
    };

    let mut projected = serde_json::Map::new();

    let results = map
        .get("results")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .enumerate()
                .map(|(index, item)| project_batch_child_value(index, item, config, ctx))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    projected.insert("results".to_string(), serde_json::Value::Array(results));
    serde_json::Value::Object(projected)
}

fn project_batch_child_value(
    index: usize,
    item: &serde_json::Value,
    config: &ToolOutputBudgetConfig,
    ctx: &ToolResultProjectionContext,
) -> serde_json::Value {
    let Some(map) = item.as_object() else {
        return project_json_value(item, config, ctx);
    };

    let tool_name = map
        .get("tool")
        .and_then(|value| value.as_str())
        .or_else(|| batch_child_tool_name(&ctx.args, index))
        .unwrap_or("tool")
        .to_string();
    let success = map
        .get("success")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let duration_ms = map
        .get("duration_ms")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let child_value = if success {
        map.get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    } else {
        map.get("error").cloned().unwrap_or(serde_json::Value::Null)
    };
    let child_args = batch_child_args(&ctx.args, index);

    let projected_child = if tool_name == "batch" {
        ToolResult {
            success,
            result: project_json_value(&child_value, config, ctx),
            images: Vec::new(),
            control: None,
        }
    } else {
        project_tool_result(
            config,
            ToolResultProjectionContext {
                session_id: ctx.session_id.clone(),
                tool_name: tool_name.clone(),
                args: child_args,
                result: ToolResult {
                    success,
                    result: child_value,
                    images: Vec::new(),
                    control: None,
                },
                duration_ms,
            },
        )
    };

    let mut projected = serde_json::Map::new();
    if let Some(value) = map.get("index") {
        projected.insert("index".to_string(), value.clone());
    }
    projected.insert("tool".to_string(), serde_json::json!(tool_name));
    projected.insert("success".to_string(), serde_json::json!(success));
    projected.insert("duration_ms".to_string(), serde_json::json!(duration_ms));
    projected.insert(
        if success {
            "result".to_string()
        } else {
            "error".to_string()
        },
        projected_child.result,
    );
    serde_json::Value::Object(projected)
}

fn batch_child_tool_name(batch_args: &serde_json::Value, index: usize) -> Option<&str> {
    batch_args
        .get("tool_calls")
        .and_then(|value| value.as_array())
        .and_then(|items| items.get(index))
        .and_then(|value| value.get("tool"))
        .and_then(|value| value.as_str())
}

fn batch_child_args(batch_args: &serde_json::Value, index: usize) -> serde_json::Value {
    batch_args
        .get("tool_calls")
        .and_then(|value| value.as_array())
        .and_then(|items| items.get(index))
        .and_then(|value| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truncates_strings_with_terminal_style_marker() {
        let config = ToolOutputBudgetConfig {
            mode: ToolOutputBudgetMode::Tokens,
            limit: 5,
            max_lines: DEFAULT_TOOL_OUTPUT_BUDGET_MAX_LINES,
        };
        let got = project_text(
            "this is an example of a long output that should be truncated",
            &config,
            &ToolResultProjectionContext {
                session_id: "root".to_string(),
                tool_name: "grep".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!("unused")),
                duration_ms: 1,
            },
        );
        assert!(got.contains("tokens truncated"));
        assert!(got.contains("Full output saved to:"));
    }

    #[test]
    fn truncation_hint_reuses_existing_full_output_path() {
        let config = ToolOutputBudgetConfig {
            limit: 512,
            ..ToolOutputBudgetConfig::default()
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                session_id: "root".to_string(),
                tool_name: "exec_command".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "output": "x".repeat(20_000),
                    "full_output_path": "/tmp/existing-shell-output.log",
                })),
                duration_ms: 1,
            },
        );
        let output = projected.result.as_str().unwrap_or_default();
        assert!(output.contains("Full output saved to: /tmp/existing-shell-output.log"));
    }

    #[test]
    fn model_projection_can_collapse_large_structured_payload_to_string() {
        let config = ToolOutputBudgetConfig {
            mode: ToolOutputBudgetMode::Bytes,
            limit: 40,
            max_lines: DEFAULT_TOOL_OUTPUT_BUDGET_MAX_LINES,
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                session_id: "root".to_string(),
                tool_name: "search_tools".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "results": [{"output": "x".repeat(200)}]
                })),
                duration_ms: 1,
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
    fn batch_model_projection_preserves_projected_child_payloads() {
        let projected = project_tool_result(
            &ToolOutputBudgetConfig::default(),
            ToolResultProjectionContext {
                session_id: "root".to_string(),
                tool_name: "batch".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "results": [
                        {"tool": "read_file", "success": true, "duration_ms": 1, "result": "very long child payload"},
                        {"tool": "grep", "success": false, "duration_ms": 1, "error": "boom"}
                    ]
                })),
                duration_ms: 1,
            },
        );
        let results = projected
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].get("result"),
            Some(&json!("very long child payload"))
        );
        assert_eq!(results[1].get("error"), Some(&json!("boom")));
    }

    #[test]
    fn batch_history_projection_recursively_projects_child_payloads() {
        let projected = project_tool_result(
            &ToolOutputBudgetConfig {
                limit: 8,
                ..ToolOutputBudgetConfig::default()
            },
            ToolResultProjectionContext {
                session_id: "root".to_string(),
                tool_name: "batch".to_string(),
                args: json!({}),
                result: ToolResult::ok(json!({
                    "results": [
                        {"tool": "read_file", "success": true, "duration_ms": 1, "result": "child payload"},
                        {"tool": "grep", "success": false, "duration_ms": 1, "error": "boom"}
                    ]
                })),
                duration_ms: 1,
            },
        );
        let details = projected
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(details.len(), 2);
        let child_result = details[0]
            .get("result")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        assert!(child_result.contains("truncated"));
        assert_eq!(details[1].get("error"), Some(&json!("boom")));
    }

    #[test]
    fn observation_truncation_uses_shared_limits_and_hint() {
        let config = ToolOutputBudgetConfig {
            mode: ToolOutputBudgetMode::Bytes,
            limit: 12,
            max_lines: 2,
        };
        let projected = truncate_observation_text("line one\nline two\nline three", &config);
        assert!(projected.contains("truncated"));
        assert!(projected.contains("print output was capped at 12 bytes and 2 lines max"));
        assert!(projected.contains("Use a narrower `print` expression"));

        let (_projected, metadata) =
            project_observation_text("line one\nline two\nline three", &config);
        assert!(metadata.truncated);
        assert_eq!(metadata.original_lines, 3);
        assert_eq!(metadata.limit, 12);
        assert_eq!(metadata.limit_mode, "bytes");
    }
}
