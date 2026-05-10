use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const PLUGIN_ID: &str = "autoresearch";
pub const JOURNAL_FILE: &str = "autoresearch.jsonl";
pub const MARKDOWN_FILE: &str = "autoresearch.md";
pub const EXPORT_FILE: &str = "autoresearch.html";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    #[default]
    Lower,
    Higher,
}

impl Direction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::Higher => "higher",
        }
    }

    pub fn is_better(self, candidate: f64, best: f64) -> bool {
        match self {
            Self::Lower => candidate < best,
            Self::Higher => candidate > best,
        }
    }

    pub fn delta_percent(self, baseline: f64, candidate: f64) -> Option<f64> {
        if !baseline.is_finite() || baseline.abs() <= f64::EPSILON {
            return None;
        }
        let delta = match self {
            Self::Lower => (baseline - candidate) / baseline,
            Self::Higher => (candidate - baseline) / baseline,
        };
        Some(delta * 100.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Keep,
    Discard,
    Crash,
    ChecksFailed,
}

impl ExperimentStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Discard => "discard",
            Self::Crash => "crash",
            Self::ChecksFailed => "checks_failed",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModeSnapshot {
    pub active: bool,
    pub objective: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunningStatus {
    pub command: String,
    pub started_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LastRunSummary {
    pub command: String,
    pub duration_seconds: f64,
    pub exit_code: Option<i32>,
    pub passed: bool,
    pub crashed: bool,
    pub timed_out: bool,
    pub checks_pass: Option<bool>,
    pub checks_timed_out: bool,
    pub parsed_primary: Option<f64>,
    pub parsed_metrics: BTreeMap<String, f64>,
    pub tail_output: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConfigEntry {
    pub segment: u64,
    pub created_at_ms: u64,
    pub name: String,
    pub metric_name: String,
    #[serde(default)]
    pub metric_unit: String,
    pub direction: Direction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResultEntry {
    pub segment: u64,
    pub timestamp_ms: u64,
    pub commit: String,
    pub metric: f64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    pub status: ExperimentStatus,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checks_pass: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JournalEntry {
    Config(ConfigEntry),
    Result(ResultEntry),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResultRow {
    pub run: usize,
    pub commit: String,
    pub metric: f64,
    pub delta_percent: Option<f64>,
    pub status: ExperimentStatus,
    pub description: String,
    pub duration_seconds: Option<f64>,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StatusSummary {
    pub active: bool,
    pub objective: Option<String>,
    pub name: Option<String>,
    pub metric_name: Option<String>,
    #[serde(default)]
    pub metric_unit: String,
    pub direction: Option<Direction>,
    pub current_segment: Option<u64>,
    pub run_count: usize,
    pub kept_count: usize,
    pub best_metric: Option<f64>,
    pub baseline_metric: Option<f64>,
    pub best_delta_percent: Option<f64>,
    pub confidence: Option<f64>,
    pub running: Option<RunningStatus>,
    pub last_run: Option<LastRunSummary>,
    #[serde(default)]
    pub results: Vec<ResultRow>,
}

pub fn journal_path(root: &Path) -> PathBuf {
    root.join(JOURNAL_FILE)
}

pub fn markdown_path(root: &Path) -> PathBuf {
    root.join(MARKDOWN_FILE)
}

pub fn export_path(root: &Path) -> PathBuf {
    root.join(EXPORT_FILE)
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis() as u64)
        .unwrap_or(0)
}

pub fn load_journal(root: &Path) -> Result<Vec<JournalEntry>, String> {
    let path = journal_path(root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut entries = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<JournalEntry>(trimmed).map_err(|err| {
            format!(
                "failed to parse {} line {}: {err}",
                path.display(),
                line_no + 1
            )
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

pub fn append_journal_entry(root: &Path, entry: &JournalEntry) -> Result<(), String> {
    let path = journal_path(root);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed to serialize journal entry: {err}"))?;
    use std::io::Write;
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub fn rewrite_markdown(root: &Path, summary: &StatusSummary) -> Result<(), String> {
    let path = markdown_path(root);
    fs::write(&path, render_markdown(summary))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub fn delete_session_files(root: &Path) -> Result<(), String> {
    for path in [journal_path(root), markdown_path(root), export_path(root)] {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(format!("failed to remove {}: {err}", path.display())),
        }
    }
    Ok(())
}

pub fn compute_summary(
    mode: &ModeSnapshot,
    entries: &[JournalEntry],
    running: Option<RunningStatus>,
    last_run: Option<LastRunSummary>,
) -> StatusSummary {
    let config = current_config(entries);
    let results = current_results(entries, config.as_ref().map(|entry| entry.segment));
    let baseline = results.first().map(|entry| entry.metric);
    let best_metric = best_metric(config.as_ref(), &results);
    let confidence = compute_confidence(config.as_ref(), &results);
    let rows = results
        .iter()
        .enumerate()
        .map(|(index, entry)| ResultRow {
            run: index + 1,
            commit: entry.commit.clone(),
            metric: entry.metric,
            delta_percent: baseline.and_then(|value| {
                config
                    .as_ref()
                    .and_then(|cfg| cfg.direction.delta_percent(value, entry.metric))
            }),
            status: entry.status,
            description: entry.description.clone(),
            duration_seconds: entry.duration_seconds,
            confidence: entry.confidence,
        })
        .collect::<Vec<_>>();
    let kept_count = results
        .iter()
        .filter(|entry| entry.status == ExperimentStatus::Keep)
        .count();
    let best_delta_percent = match (config.as_ref(), baseline, best_metric) {
        (Some(cfg), Some(base), Some(best)) if (best - base).abs() > f64::EPSILON => {
            cfg.direction.delta_percent(base, best)
        }
        _ => None,
    };

    StatusSummary {
        active: mode.active,
        objective: mode.objective.clone(),
        name: config.as_ref().map(|entry| entry.name.clone()),
        metric_name: config.as_ref().map(|entry| entry.metric_name.clone()),
        metric_unit: config
            .as_ref()
            .map(|entry| entry.metric_unit.clone())
            .unwrap_or_default(),
        direction: config.as_ref().map(|entry| entry.direction),
        current_segment: config.as_ref().map(|entry| entry.segment),
        run_count: results.len(),
        kept_count,
        best_metric,
        baseline_metric: baseline,
        best_delta_percent,
        confidence,
        running,
        last_run,
        results: rows,
    }
}

pub fn render_markdown(summary: &StatusSummary) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# autoresearch");
    let _ = writeln!(
        out,
        "\n- Mode: {}",
        if summary.active { "active" } else { "inactive" }
    );
    if let Some(objective) = summary.objective.as_deref() {
        let _ = writeln!(out, "- Objective: {objective}");
    }
    if let Some(name) = summary.name.as_deref() {
        let _ = writeln!(out, "- Session: {name}");
    }
    if let Some(metric_name) = summary.metric_name.as_deref() {
        let direction = summary.direction.map(Direction::as_str).unwrap_or("lower");
        let _ = writeln!(out, "- Metric: {metric_name} ({direction} is better)");
    }
    let _ = writeln!(
        out,
        "- Runs: {} total, {} kept",
        summary.run_count, summary.kept_count
    );
    if let Some(best) = summary.best_metric {
        let metric_name = summary.metric_name.as_deref().unwrap_or("metric");
        let _ = writeln!(
            out,
            "- Best: {metric_name} = {}{}{}",
            format_metric(best),
            summary.metric_unit,
            summary
                .best_delta_percent
                .map(|delta| format!(" ({})", format_delta(delta)))
                .unwrap_or_default()
        );
    }
    if let Some(confidence) = summary.confidence {
        let _ = writeln!(out, "- Confidence: {}", format_confidence(confidence));
    }
    if let Some(running) = summary.running.as_ref() {
        let _ = writeln!(out, "- Running: `{}`", running.command);
    } else if let Some(last_run) = summary.last_run.as_ref() {
        let _ = writeln!(
            out,
            "- Last run: `{}` in {}",
            last_run.command,
            format_seconds(last_run.duration_seconds)
        );
    }

    if !summary.results.is_empty() {
        let _ = writeln!(
            out,
            "\n## Current Segment\n\n| Run | Commit | Metric | Status | Description |\n| --- | --- | --- | --- | --- |"
        );
        for row in &summary.results {
            let metric = if let Some(delta) = row.delta_percent {
                format!(
                    "{}{} ({})",
                    format_metric(row.metric),
                    summary.metric_unit,
                    format_delta(delta)
                )
            } else {
                format!("{}{}", format_metric(row.metric), summary.metric_unit)
            };
            let _ = writeln!(
                out,
                "| {} | `{}` | {} | {} | {} |",
                row.run,
                row.commit,
                metric,
                row.status.label(),
                row.description.replace('|', "\\|")
            );
        }
    }

    out
}

pub fn render_export_html(summary: &StatusSummary) -> String {
    let title = summary.name.as_deref().unwrap_or("autoresearch");
    let metric_name = summary.metric_name.as_deref().unwrap_or("metric");
    let objective = summary.objective.as_deref().unwrap_or("No objective set.");
    let best = summary
        .best_metric
        .map(|value| format!("{}{}", format_metric(value), summary.metric_unit))
        .unwrap_or_else(|| "—".to_string());
    let delta = summary
        .best_delta_percent
        .map(format_delta)
        .unwrap_or_else(|| "—".to_string());
    let confidence = summary
        .confidence
        .map(format_confidence)
        .unwrap_or_else(|| "—".to_string());
    let mut rows = String::new();
    for row in &summary.results {
        let _ = writeln!(
            rows,
            "<tr><td>{}</td><td><code>{}</code></td><td>{}{}</td><td>{}</td><td>{}</td></tr>",
            row.run,
            escape_html(&row.commit),
            escape_html(&format_metric(row.metric)),
            escape_html(&summary.metric_unit),
            escape_html(row.status.label()),
            escape_html(&row.description)
        );
    }
    format!(
        "<!doctype html>\
<html><head><meta charset=\"utf-8\"><title>{}</title>\
<style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1117;color:#e6edf3;padding:32px;}}\
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:24px 0;}}\
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;}}\
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:off;}}\
th,td{{padding:10px 12px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top;}}\
h1{{margin:0 0 12px 0;}}\
p{{color:#8b949e;max-width:72ch;}}\
code{{color:#a5d6ff;}}\
</style></head><body>\
<h1>{}</h1>\
<p>{}</p>\
<div class=\"grid\">\
<div class=\"card\"><strong>Runs</strong><div>{}</div></div>\
<div class=\"card\"><strong>Best {}</strong><div>{}</div></div>\
<div class=\"card\"><strong>Best Delta</strong><div>{}</div></div>\
<div class=\"card\"><strong>Confidence</strong><div>{}</div></div>\
</div>\
<table><thead><tr><th>Run</th><th>Commit</th><th>Metric</th><th>Status</th><th>Description</th></tr></thead><tbody>{}</tbody></table>\
</body></html>",
        escape_html(title),
        escape_html(title),
        escape_html(objective),
        summary.run_count,
        escape_html(metric_name),
        escape_html(&best),
        escape_html(&delta),
        escape_html(&confidence),
        rows
    )
}

pub fn write_export_html(root: &Path, summary: &StatusSummary) -> Result<PathBuf, String> {
    let path = export_path(root);
    fs::write(&path, render_export_html(summary))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    Ok(path)
}

pub fn format_metric(value: f64) -> String {
    if (value.fract()).abs() <= f64::EPSILON {
        format!("{value:.0}")
    } else if value.abs() >= 100.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

pub fn format_delta(value: f64) -> String {
    if value >= 0.0 {
        format!("+{value:.1}%")
    } else {
        format!("{value:.1}%")
    }
}

pub fn format_confidence(value: f64) -> String {
    format!("{value:.1}x")
}

pub fn format_seconds(value: f64) -> String {
    if value >= 60.0 {
        let minutes = (value / 60.0).floor();
        let seconds = value - minutes * 60.0;
        format!("{minutes:.0}m {seconds:04.1}s")
    } else {
        format!("{value:.2}s")
    }
}

pub(crate) fn confidence_for_candidate(
    config: &ConfigEntry,
    results: &[ResultEntry],
    metric: f64,
) -> Option<f64> {
    let baseline = results.first()?.metric;
    let mut values = finite_metrics(results);
    if metric.is_finite() {
        values.push(metric);
    }
    confidence_from_values(config.direction, baseline, &values)
}

fn current_config(entries: &[JournalEntry]) -> Option<ConfigEntry> {
    entries.iter().rev().find_map(|entry| match entry {
        JournalEntry::Config(config) => Some(config.clone()),
        JournalEntry::Result(_) => None,
    })
}

fn current_results(entries: &[JournalEntry], segment: Option<u64>) -> Vec<ResultEntry> {
    let Some(segment) = segment else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| match entry {
            JournalEntry::Result(result) if result.segment == segment => Some(result.clone()),
            _ => None,
        })
        .collect()
}

fn best_metric(config: Option<&ConfigEntry>, results: &[ResultEntry]) -> Option<f64> {
    let config = config?;
    let baseline = results
        .iter()
        .find(|entry| entry.metric.is_finite())
        .map(|entry| entry.metric)?;
    let mut best = None;
    for result in results
        .iter()
        .filter(|entry| entry.status == ExperimentStatus::Keep && entry.metric.is_finite())
    {
        best = Some(match best {
            Some(current) if config.direction.is_better(result.metric, current) => result.metric,
            Some(current) => current,
            None => result.metric,
        });
    }
    best.or(Some(baseline))
}

fn compute_confidence(config: Option<&ConfigEntry>, results: &[ResultEntry]) -> Option<f64> {
    let config = config?;
    let baseline = results.first()?.metric;
    let values = finite_metrics(results);
    confidence_from_values(config.direction, baseline, &values)
}

fn finite_metrics(results: &[ResultEntry]) -> Vec<f64> {
    results
        .iter()
        .map(|result| result.metric)
        .filter(|value| value.is_finite())
        .collect()
}

fn confidence_from_values(direction: Direction, baseline: f64, values: &[f64]) -> Option<f64> {
    if values.len() < 3 || !baseline.is_finite() || baseline.abs() <= f64::EPSILON {
        return None;
    }
    let median_value = median(values.to_vec());
    let mad = median(
        values
            .iter()
            .map(|value| (value - median_value).abs())
            .collect(),
    );
    if mad <= f64::EPSILON {
        return None;
    }
    let best = values.iter().copied().reduce(|current, candidate| {
        if direction.is_better(candidate, current) {
            candidate
        } else {
            current
        }
    })?;
    let delta = (best - baseline).abs();
    (delta > f64::EPSILON).then_some(delta / mad)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_uses_latest_segment() {
        let summary = compute_summary(
            &ModeSnapshot {
                active: true,
                objective: Some("speed up tests".to_string()),
            },
            &[
                JournalEntry::Config(ConfigEntry {
                    segment: 1,
                    created_at_ms: 1,
                    name: "old".to_string(),
                    metric_name: "ms".to_string(),
                    metric_unit: "ms".to_string(),
                    direction: Direction::Lower,
                }),
                JournalEntry::Result(ResultEntry {
                    segment: 1,
                    timestamp_ms: 2,
                    commit: "aaaaaaa".to_string(),
                    metric: 10.0,
                    metrics: BTreeMap::new(),
                    status: ExperimentStatus::Keep,
                    description: "old".to_string(),
                    duration_seconds: None,
                    exit_code: None,
                    checks_pass: None,
                    command: None,
                    confidence: None,
                }),
                JournalEntry::Config(ConfigEntry {
                    segment: 2,
                    created_at_ms: 3,
                    name: "new".to_string(),
                    metric_name: "total_ms".to_string(),
                    metric_unit: "ms".to_string(),
                    direction: Direction::Lower,
                }),
                JournalEntry::Result(ResultEntry {
                    segment: 2,
                    timestamp_ms: 4,
                    commit: "bbbbbbb".to_string(),
                    metric: 8.0,
                    metrics: BTreeMap::new(),
                    status: ExperimentStatus::Keep,
                    description: "new".to_string(),
                    duration_seconds: Some(1.2),
                    exit_code: Some(0),
                    checks_pass: Some(true),
                    command: Some("cargo test".to_string()),
                    confidence: Some(2.0),
                }),
            ],
            None,
            None,
        );

        assert_eq!(summary.name.as_deref(), Some("new"));
        assert_eq!(summary.current_segment, Some(2));
        assert_eq!(summary.run_count, 1);
        assert_eq!(summary.best_metric, Some(8.0));
    }

    #[test]
    fn markdown_mentions_best_and_objective() {
        let text = render_markdown(&StatusSummary {
            active: true,
            objective: Some("speed up tests".to_string()),
            name: Some("unit tests".to_string()),
            metric_name: Some("total_ms".to_string()),
            metric_unit: "ms".to_string(),
            direction: Some(Direction::Lower),
            current_segment: Some(1),
            run_count: 2,
            kept_count: 1,
            best_metric: Some(8.0),
            baseline_metric: Some(10.0),
            best_delta_percent: Some(20.0),
            confidence: Some(2.3),
            running: None,
            last_run: None,
            results: vec![],
        });
        assert!(text.contains("speed up tests"));
        assert!(text.contains("20.0%"));
        assert!(text.contains("Confidence: 2.3x"));
    }

    #[test]
    fn summary_uses_baseline_until_a_keep_exists() {
        let summary = compute_summary(
            &ModeSnapshot {
                active: true,
                objective: Some("speed up tests".to_string()),
            },
            &[
                JournalEntry::Config(ConfigEntry {
                    segment: 1,
                    created_at_ms: 1,
                    name: "segment".to_string(),
                    metric_name: "total_ms".to_string(),
                    metric_unit: "ms".to_string(),
                    direction: Direction::Lower,
                }),
                JournalEntry::Result(ResultEntry {
                    segment: 1,
                    timestamp_ms: 2,
                    commit: "aaaaaaa".to_string(),
                    metric: 10.0,
                    metrics: BTreeMap::new(),
                    status: ExperimentStatus::Discard,
                    description: "baseline".to_string(),
                    duration_seconds: Some(1.0),
                    exit_code: Some(0),
                    checks_pass: Some(true),
                    command: Some("bench".to_string()),
                    confidence: None,
                }),
                JournalEntry::Result(ResultEntry {
                    segment: 1,
                    timestamp_ms: 3,
                    commit: "bbbbbbb".to_string(),
                    metric: 8.0,
                    metrics: BTreeMap::new(),
                    status: ExperimentStatus::Discard,
                    description: "faster but discarded".to_string(),
                    duration_seconds: Some(1.0),
                    exit_code: Some(0),
                    checks_pass: Some(true),
                    command: Some("bench".to_string()),
                    confidence: None,
                }),
            ],
            None,
            None,
        );

        assert_eq!(summary.best_metric, Some(10.0));
        assert_eq!(summary.best_delta_percent, None);
    }
}
