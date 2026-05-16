use lash_core::session_model::{Message, PartKind};

use crate::trace::LlmCallUsage;

use super::stats::SessionStats;

pub(crate) struct RenderCtx<'a> {
    next_index: usize,
    pub(crate) stats: &'a SessionStats,
}

impl<'a> RenderCtx<'a> {
    pub(crate) fn new(stats: &'a SessionStats) -> Self {
        Self {
            next_index: 0,
            stats,
        }
    }
    pub(crate) fn next_id(&mut self) -> String {
        let n = self.next_index;
        self.next_index += 1;
        format!("e{n}")
    }
}

pub(crate) fn submit_value_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub(crate) fn message_matches_text(message: &Message, expected: &str) -> bool {
    let expected = expected.trim();
    if expected.is_empty() {
        return false;
    }
    let collected: String = message
        .parts
        .iter()
        .filter(|p| matches!(p.kind, PartKind::Text | PartKind::Prose))
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    collected.trim() == expected
}

pub(crate) fn compact_usage_label(
    usage: &LlmCallUsage,
    context_window_tokens: Option<u64>,
) -> String {
    let context = context_percent_label(usage, context_window_tokens)
        .unwrap_or_else(|| "ctx n/a".to_string());
    let cached = percent_of(usage.cached_input_tokens, usage.input_tokens)
        .map(|pct| format!("{pct:.1}% cached"))
        .unwrap_or_else(|| "cache n/a".to_string());
    format!("{context} · {cached}")
}

pub(crate) fn context_percent_label(
    usage: &LlmCallUsage,
    context_window_tokens: Option<u64>,
) -> Option<String> {
    context_percent(usage, context_window_tokens).map(|pct| format!("ctx {pct:.1}%"))
}

pub(crate) fn usage_title(
    usage: Option<&LlmCallUsage>,
    context_window_tokens: Option<u64>,
) -> String {
    let Some(usage) = usage else {
        return "token usage not recorded".to_string();
    };
    let context = match (
        context_percent(usage, context_window_tokens),
        context_window_tokens,
    ) {
        (Some(pct), Some(window)) => format!(
            "context {pct:.3}% ({input} / {window})",
            input = usage.input_tokens.max(0),
        ),
        _ => format!(
            "context n/a ({input} input)",
            input = usage.input_tokens.max(0)
        ),
    };
    let cached = percent_of(usage.cached_input_tokens, usage.input_tokens)
        .map(|pct| {
            format!(
                "cached {pct:.3}% ({cached} / {input})",
                cached = usage.cached_input_tokens.max(0),
                input = usage.input_tokens.max(0),
            )
        })
        .unwrap_or_else(|| "cached n/a".to_string());
    let duration = usage
        .duration_ms
        .map(|ms| format!(" · duration {}", format_duration(ms)))
        .unwrap_or_default();
    format!(
        "{context} · {cached} · input {input} · output {output} · reasoning {reasoning}{duration}",
        input = usage.input_tokens.max(0),
        output = usage.output_tokens.max(0),
        reasoning = usage.reasoning_tokens.max(0),
    )
}

pub(crate) fn context_percent(
    usage: &LlmCallUsage,
    context_window_tokens: Option<u64>,
) -> Option<f64> {
    let window = context_window_tokens?;
    if window == 0 {
        return None;
    }
    Some(usage.input_tokens.max(0) as f64 * 100.0 / window as f64)
}

pub(crate) fn percent_of(numerator: i64, denominator: i64) -> Option<f64> {
    if denominator <= 0 {
        return None;
    }
    Some(numerator.max(0) as f64 * 100.0 / denominator as f64)
}

pub(crate) fn one_line_summary(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "(empty)".to_string();
    }
    let first_line: String = trimmed
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(trimmed)
        .trim()
        .to_string();
    truncate(&first_line, max_chars)
}

pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{head}…")
}

pub(crate) fn summarize_args(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::String(s) => format!("\"{}\"", truncate(s, 160)),
        Value::Null => "—".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(arr) => {
            let preview: Vec<String> = arr.iter().take(3).map(|v| short_value(v, 40)).collect();
            let more = if arr.len() > 3 {
                format!(", … +{}", arr.len() - 3)
            } else {
                String::new()
            };
            format!("[{}{}]", preview.join(", "), more)
        }
        Value::Object(map) => {
            // Priority keys to surface first — action-shaped keys (the
            // WHAT) outrank location keys (the WHERE) so a grep call shows
            // its query instead of its path. Tools without an action key
            // fall through to path naturally.
            const PRIORITY: &[&str] = &[
                "query", "q", "command", "cmd", "shell", "url", "uri", "prompt", "path", "file",
                "filename", "filepath", "name", "title", "id", "key",
            ];
            for k in PRIORITY {
                if let Some(v) = map.get(*k) {
                    return format!("{}={}", k, short_value(v, 160));
                }
            }
            // fall back to first key=value
            if let Some((k, v)) = map.iter().next() {
                if map.len() == 1 {
                    return format!("{}={}", k, short_value(v, 160));
                }
                let extra = map.len() - 1;
                return format!("{}={} +{} more", k, short_value(v, 100), extra);
            }
            "(empty)".to_string()
        }
    }
}

fn short_value(v: &serde_json::Value, max_chars: usize) -> String {
    use serde_json::Value;
    let raw = match v {
        Value::String(s) => format!("\"{s}\""),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => format!("[…{} items]", a.len()),
        Value::Object(o) => format!("{{…{} keys}}", o.len()),
    };
    truncate(&raw.replace('\n', " "), max_chars)
}

pub(crate) fn json_byte_size(value: &serde_json::Value) -> usize {
    serde_json::to_string(value).map(|s| s.len()).unwrap_or(0)
}

pub(crate) fn pretty_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        _ => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
    }
}

pub(crate) fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        let total_s = ms / 1000;
        let m = total_s / 60;
        let s = total_s % 60;
        format!("{m}m{s:02}s")
    }
}

pub(crate) fn format_count(n: u64) -> String {
    if n < 1024 {
        format!("{n}b")
    } else if n < 1024 * 1024 {
        format!("{:.1}kb", n as f64 / 1024.0)
    } else if n < 1024 * 1024 * 1024 {
        format!("{:.1}mb", n as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2}gb", n as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub(crate) fn format_tokens(n: i64) -> String {
    let n = n.max(0) as f64;
    if n < 1_000.0 {
        format!("{}", n as u64)
    } else if n < 1_000_000.0 {
        format!("{:.1}k", n / 1_000.0)
    } else {
        format!("{:.2}m", n / 1_000_000.0)
    }
}

pub(crate) fn strip_first_lashlang_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    let opener_len = text.as_bytes()[open_rel..]
        .iter()
        .take_while(|&&b| b == b'`')
        .count();
    let after_open = open_rel + opener_len;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let body_bytes = &text.as_bytes()[body_start..];
    let mut close = text.len();
    let mut consumed = 0usize;
    let mut i = 0;
    while i < body_bytes.len() {
        if body_bytes[i] == b'`' {
            let start = i;
            while i < body_bytes.len() && body_bytes[i] == b'`' {
                i += 1;
            }
            if i - start >= opener_len {
                close = body_start + start;
                consumed = opener_len;
                break;
            }
        } else {
            i += 1;
        }
    }
    let after_close = (close + consumed).min(text.len());
    let mut out = String::new();
    out.push_str(text[..open_rel].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
}
