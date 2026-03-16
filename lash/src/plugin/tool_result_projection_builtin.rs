use std::sync::Arc;

use crate::ToolResult;
use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolResultProjectionContext, ToolResultProjectionHook,
};

const APPROX_BYTES_PER_TOKEN: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultProjectionMode {
    Bytes,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolResultProjectionPluginConfig {
    pub mode: ToolResultProjectionMode,
    pub limit: usize,
}

impl Default for ToolResultProjectionPluginConfig {
    fn default() -> Self {
        Self {
            mode: ToolResultProjectionMode::Bytes,
            limit: 10_000,
        }
    }
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
    let projected_value = project_json_value(&ctx.result.result, config, false);
    let result = match ctx.hook {
        ToolResultProjectionHook::BeforeModel => {
            project_model_value(projected_value, ctx.result.success, config)
        }
        ToolResultProjectionHook::BeforeHistory => projected_value,
    };
    ToolResult {
        success: ctx.result.success,
        result,
        images: ctx.result.images,
    }
}

fn project_model_value(
    value: serde_json::Value,
    success: bool,
    config: &ToolResultProjectionPluginConfig,
) -> serde_json::Value {
    let rendered = render_tool_result_payload(success, &value);
    if !needs_truncation(&rendered, config) {
        return value;
    }
    serde_json::Value::String(formatted_truncate_text(&rendered, config))
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
    formatted: bool,
) -> serde_json::Value {
    match value {
        serde_json::Value::String(text) => {
            serde_json::Value::String(project_text(text, config, formatted))
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| project_json_value(item, config, formatted))
                .collect(),
        ),
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), project_json_value(value, config, formatted)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn project_text(text: &str, config: &ToolResultProjectionPluginConfig, formatted: bool) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    if formatted {
        formatted_truncate_text(text, config)
    } else {
        truncate_text(text, config)
    }
}

fn needs_truncation(text: &str, config: &ToolResultProjectionPluginConfig) -> bool {
    match config.mode {
        ToolResultProjectionMode::Bytes => text.len() > config.limit,
        ToolResultProjectionMode::Tokens => approx_token_count(text) > config.limit,
    }
}

fn formatted_truncate_text(text: &str, config: &ToolResultProjectionPluginConfig) -> String {
    if !needs_truncation(text, config) {
        return text.to_string();
    }
    let total_lines = text.lines().count();
    format!(
        "Total output lines: {total_lines}\n\n{}",
        truncate_text(text, config)
    )
}

fn truncate_text(text: &str, config: &ToolResultProjectionPluginConfig) -> String {
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
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let (left_budget, right_budget) = split_budget(max_bytes);
    let (removed_chars, prefix, suffix) = split_string(text, left_budget, right_budget);
    let removed_bytes = text.len().saturating_sub(prefix.len() + suffix.len());
    let marker = format_truncation_marker(
        config.mode,
        removed_units(config.mode, removed_bytes, removed_chars),
    );
    let mut out = String::with_capacity(prefix.len() + marker.len() + suffix.len());
    out.push_str(prefix);
    out.push_str(&marker);
    out.push_str(suffix);
    out
}

fn split_budget(budget: usize) -> (usize, usize) {
    let left = budget / 2;
    (left, budget - left)
}

fn split_string(s: &str, prefix_bytes: usize, suffix_bytes: usize) -> (usize, &str, &str) {
    if s.is_empty() {
        return (0, "", "");
    }
    let len = s.len();
    let tail_start_target = len.saturating_sub(suffix_bytes);
    let mut prefix_end = 0usize;
    let mut suffix_start = len;
    let mut removed_chars = 0usize;
    let mut suffix_started = false;

    for (idx, ch) in s.char_indices() {
        let char_end = idx + ch.len_utf8();
        if char_end <= prefix_bytes {
            prefix_end = char_end;
            continue;
        }
        if idx >= tail_start_target {
            if !suffix_started {
                suffix_start = idx;
                suffix_started = true;
            }
            continue;
        }
        removed_chars = removed_chars.saturating_add(1);
    }

    if suffix_start < prefix_end {
        suffix_start = prefix_end;
    }

    (removed_chars, &s[..prefix_end], &s[suffix_start..])
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
        };
        let got = project_text(
            "this is an example of a long output that should be truncated",
            &config,
            true,
        );
        assert!(got.contains("Total output lines: 1"));
        assert!(got.contains("tokens truncated"));
    }

    #[test]
    fn history_projection_preserves_shape() {
        let config = ToolResultProjectionPluginConfig::default();
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
        assert!(output.contains("chars truncated"));
        assert!(!output.contains("Total output lines"));
    }

    #[test]
    fn model_projection_can_collapse_large_structured_payload_to_string() {
        let config = ToolResultProjectionPluginConfig {
            mode: ToolResultProjectionMode::Bytes,
            limit: 40,
        };
        let projected = project_tool_result(
            &config,
            ToolResultProjectionContext {
                hook: ToolResultProjectionHook::BeforeModel,
                session_id: "root".to_string(),
                tool_name: "search_history".to_string(),
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
                .contains("chars truncated")
        );
    }
}
