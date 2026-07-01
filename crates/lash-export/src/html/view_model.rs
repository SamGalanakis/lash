use crate::trace::LlmCallUsage;

pub(crate) struct RenderCtx {
    next_index: usize,
}

impl RenderCtx {
    pub(crate) fn new() -> Self {
        Self { next_index: 0 }
    }

    pub(crate) fn next_id(&mut self) -> String {
        let n = self.next_index;
        self.next_index += 1;
        format!("e{n}")
    }
}

pub(crate) fn compact_usage_label(
    usage: &LlmCallUsage,
    context_window_tokens: Option<u64>,
) -> String {
    let context = context_percent_label(usage, context_window_tokens)
        .unwrap_or_else(|| "ctx n/a".to_string());
    let cached = percent_of(usage.cache_read_input_tokens, prompt_total_tokens(usage))
        .map(|pct| format!("{pct:.1}% cache read"))
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
            input = prompt_total_tokens(usage).max(0),
        ),
        _ => format!(
            "context n/a ({input} input)",
            input = prompt_total_tokens(usage).max(0)
        ),
    };
    let prompt_total = prompt_total_tokens(usage);
    let cached = percent_of(usage.cache_read_input_tokens, prompt_total)
        .map(|pct| {
            format!(
                "cached {pct:.3}% ({cached} / {input})",
                cached = usage.cache_read_input_tokens.max(0),
                input = prompt_total.max(0),
            )
        })
        .unwrap_or_else(|| "cached n/a".to_string());
    let duration = usage
        .duration_ms
        .map(|ms| format!(" · duration {}", format_duration(ms)))
        .unwrap_or_default();
    format!(
        "{context} · {cached} · input {input} · cache write {cache_write} · output {output} · reasoning {reasoning}{duration}",
        input = usage.input_tokens.max(0),
        cache_write = usage.cache_write_input_tokens.max(0),
        output = usage.output_tokens.max(0),
        reasoning = usage.reasoning_output_tokens.max(0),
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
    Some(prompt_total_tokens(usage).max(0) as f64 * 100.0 / window as f64)
}

pub(crate) fn prompt_total_tokens(usage: &LlmCallUsage) -> i64 {
    usage
        .input_tokens
        .saturating_add(usage.cache_read_input_tokens)
        .saturating_add(usage.cache_write_input_tokens)
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
