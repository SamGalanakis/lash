//! `lash-trace-viewer` renders Lash trace JSONL into a self-contained HTML
//! debugging surface, preserving unknown future events as raw records.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use lash_trace::TraceRecord;
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "lash-trace-viewer",
    about = "Render a Lash .trace.jsonl file as a self-contained HTML trace browser"
)]
struct Cli {
    /// Path to a Lash *.trace.jsonl file.
    trace: PathBuf,

    /// Output HTML path. Defaults to <trace-stem>.html beside the trace.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Write HTML to stdout instead of a file.
    #[arg(long)]
    stdout: bool,

    /// Page title shown in the viewer.
    #[arg(long)]
    title: Option<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lash-trace-viewer: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let trace = load_trace(&cli.trace)?;
    let title = cli.title.unwrap_or_else(|| {
        cli.trace
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Lash trace")
            .to_string()
    });
    let html = render_html(&title, &cli.trace, &trace)?;

    if cli.stdout {
        io::stdout()
            .write_all(html.as_bytes())
            .context("write HTML to stdout")?;
        return Ok(());
    }

    let out = cli.out.unwrap_or_else(|| default_output_path(&cli.trace));
    fs::write(&out, html).with_context(|| format!("write {}", out.display()))?;
    println!("{}", out.display());
    Ok(())
}

#[derive(Debug)]
struct TraceEntry {
    raw: Value,
    typed: Option<TraceRecord>,
}

#[derive(Debug)]
struct LoadedTrace {
    records: Vec<TraceEntry>,
}

fn load_trace(path: &Path) -> Result<LoadedTrace> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let raw = serde_json::from_str::<Value>(trimmed)
            .with_context(|| format!("parse trace JSONL line {}", index + 1))?;
        let typed = serde_json::from_value::<TraceRecord>(raw.clone()).ok();
        records.push(TraceEntry { raw, typed });
    }
    if records.is_empty() {
        return Err(anyhow!(
            "trace file contains no records: {}",
            path.display()
        ));
    }
    Ok(LoadedTrace { records })
}

fn default_output_path(trace: &Path) -> PathBuf {
    let mut out = trace.to_path_buf();
    out.set_extension("html");
    out
}

fn render_html(title: &str, source_path: &Path, trace: &LoadedTrace) -> Result<String> {
    let raw_records = trace.records.iter().map(|record| &record.raw);
    let records_json = escape_script_json(
        &serde_json::to_string(&raw_records.collect::<Vec<_>>())
            .context("serialize trace records")?,
    );
    let stats = TraceStats::from_trace(trace);
    let source = source_path.display().to_string();
    Ok(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<header class="shell-header">
  <div>
    <p class="eyebrow">Lash Trace Viewer</p>
    <h1>{title}</h1>
    <p class="source">{source}</p>
  </div>
  <div class="summary" aria-label="Trace summary">
    <div><strong>{total}</strong><span>events</span></div>
    <div><strong>{llm_calls}</strong><span>LLM calls</span></div>
    <div><strong>{failures}</strong><span>failures</span></div>
    <div><strong>{tokens}</strong><span>tokens</span></div>
  </div>
</header>

<main class="layout">
  <aside class="sidebar">
    <label class="search-label" for="search">Search trace</label>
    <input id="search" class="search" type="search" placeholder="model, tool, text, id">
    <div id="filters" class="filters" aria-label="Event type filters"></div>
    <section class="sidebar-panel">
      <h2>Current Selection</h2>
      <div id="selectionMeta" class="selection-meta">Select an event.</div>
    </section>
  </aside>

  <section class="content">
    <nav class="tabs" aria-label="Trace views">
      <button class="tab active" data-view="timeline">Timeline</button>
      <button class="tab" data-view="llm">LLM Calls</button>
      <button class="tab" data-view="streams">Streams</button>
      <button class="tab" data-view="raw">Raw</button>
    </nav>
    <section id="timeline" class="view active"></section>
    <section id="llm" class="view"></section>
    <section id="streams" class="view"></section>
    <section id="raw" class="view"></section>
  </section>
</main>

<script id="trace-data" type="application/json">{records_json}</script>
<script>
{js}
</script>
</body>
</html>"#,
        title = escape_html(title),
        source = escape_html(&source),
        total = trace.records.len(),
        llm_calls = stats.llm_calls,
        failures = stats.failures,
        tokens = stats.total_tokens,
        css = CSS,
        js = JS,
    ))
}

#[derive(Default)]
struct TraceStats {
    llm_calls: usize,
    failures: usize,
    total_tokens: i64,
}

impl TraceStats {
    fn from_trace(trace: &LoadedTrace) -> Self {
        let mut stats = Self::default();
        for record in trace
            .records
            .iter()
            .filter_map(|record| record.typed.as_ref())
        {
            match &record.event {
                lash_trace::TraceEvent::LlmCallStarted { .. } => stats.llm_calls += 1,
                lash_trace::TraceEvent::LlmCallFailed { .. } => stats.failures += 1,
                lash_trace::TraceEvent::ToolCallCompleted { output, .. }
                    if !output.is_success() =>
                {
                    stats.failures += 1;
                }
                lash_trace::TraceEvent::TurnCompleted { status, .. } if status == "failed" => {
                    stats.failures += 1;
                }
                lash_trace::TraceEvent::LlmCallCompleted {
                    usage: Some(usage), ..
                }
                | lash_trace::TraceEvent::TokenUsage { usage, .. } => {
                    stats.total_tokens += usage.input_tokens
                        + usage.output_tokens
                        + usage.cache_read_input_tokens
                        + usage.cache_write_input_tokens;
                }
                lash_trace::TraceEvent::LashlangExecution {
                    event:
                        lash_trace::TraceLashlangExecutionEvent::NodeFailed { .. }
                        | lash_trace::TraceLashlangExecutionEvent::ExecutionFinished {
                            status: lash_trace::TraceLashlangStatus::Failed,
                            ..
                        },
                } => {
                    stats.failures += 1;
                }
                _ => {}
            }
        }
        stats
    }
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn escape_script_json(value: &str) -> String {
    value
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

const CSS: &str = r#"
:root {
  color-scheme: dark;
  --bg: #11110f;
  --panel: #191815;
  --panel-2: #222018;
  --line: #39352b;
  --text: #f2efe6;
  --muted: #aaa28e;
  --dim: #756f61;
  --amber: #e7bd5a;
  --cyan: #81d6d0;
  --red: #ff6f61;
  --green: #93d977;
  --blue: #8fb8ff;
  --shadow: 0 22px 70px rgba(0, 0, 0, .35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    linear-gradient(90deg, rgba(231, 189, 90, .055) 1px, transparent 1px),
    linear-gradient(180deg, rgba(231, 189, 90, .04) 1px, transparent 1px),
    var(--bg);
  background-size: 28px 28px;
  color: var(--text);
  font: 14px/1.5 ui-monospace, SFMono-Regular, "Cascadia Mono", "Liberation Mono", Menlo, monospace;
}
.shell-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 28px 32px 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(17, 17, 15, .92);
  position: sticky;
  top: 0;
  z-index: 3;
  backdrop-filter: blur(14px);
}
.eyebrow {
  margin: 0 0 5px;
  color: var(--amber);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-size: 12px;
}
h1 {
  margin: 0;
  font: 700 28px/1.15 ui-serif, Georgia, Cambria, serif;
  letter-spacing: 0;
}
.source {
  margin: 7px 0 0;
  color: var(--muted);
  word-break: break-all;
}
.summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(82px, 1fr));
  gap: 10px;
  min-width: 420px;
}
.summary div {
  padding: 12px 14px;
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
}
.summary strong {
  display: block;
  font-size: 21px;
  line-height: 1;
}
.summary span {
  color: var(--muted);
  font-size: 12px;
}
.layout {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  min-height: calc(100vh - 117px);
}
.sidebar {
  border-right: 1px solid var(--line);
  padding: 22px;
  position: sticky;
  top: 118px;
  height: calc(100vh - 118px);
  overflow: auto;
  background: rgba(17, 17, 15, .86);
}
.search-label {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}
.search {
  width: 100%;
  min-height: 42px;
  color: var(--text);
  background: #0d0d0b;
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 10px 12px;
  font: inherit;
}
.filters {
  display: grid;
  gap: 7px;
  margin: 18px 0;
}
.filter {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 34px;
  padding: 6px 9px;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  border-radius: 4px;
  cursor: pointer;
  text-align: left;
  font: inherit;
}
.filter.off { opacity: .42; }
.count { color: var(--dim); }
.sidebar-panel {
  border-top: 1px solid var(--line);
  padding-top: 16px;
}
.sidebar-panel h2 {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--amber);
}
.selection-meta {
  color: var(--muted);
  word-break: break-word;
}
.content {
  min-width: 0;
  padding: 22px 28px 40px;
}
.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
}
.tab {
  border: 1px solid var(--line);
  border-bottom: 0;
  background: var(--panel);
  color: var(--muted);
  padding: 9px 13px;
  cursor: pointer;
  font: inherit;
  border-radius: 4px 4px 0 0;
}
.tab.active {
  color: var(--text);
  background: var(--panel-2);
}
.view { display: none; }
.view.active { display: block; }
.event, .llm-card, .stream-card, .raw-card {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr);
  gap: 18px;
  border: 1px solid var(--line);
  background: rgba(25, 24, 21, .96);
  margin-bottom: 10px;
  box-shadow: var(--shadow);
}
.event.selected, .llm-card.selected, .stream-card.selected {
  outline: 2px solid var(--amber);
}
.event-rail {
  padding: 14px;
  border-right: 1px solid var(--line);
  color: var(--muted);
  background: rgba(0, 0, 0, .14);
}
.event-body {
  min-width: 0;
  padding: 14px 16px 16px 0;
}
.kind {
  display: inline-block;
  color: #11110f;
  background: var(--amber);
  padding: 2px 7px;
  border-radius: 3px;
  font-weight: 700;
}
.kind.fail { background: var(--red); }
.kind.llm { background: var(--cyan); }
.kind.tool { background: var(--green); }
.kind.stream { background: var(--blue); }
.meta-line {
  margin-top: 8px;
  color: var(--dim);
  font-size: 12px;
}
.title-line {
  font-weight: 700;
  margin-bottom: 6px;
}
.summary-line {
  color: var(--muted);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
details {
  margin-top: 10px;
}
summary {
  cursor: pointer;
  color: var(--amber);
}
pre {
  overflow: auto;
  max-height: 460px;
  margin: 10px 0 0;
  padding: 12px;
  color: #e9e1d0;
  background: #0b0b09;
  border: 1px solid #2b281f;
  border-radius: 4px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.pill {
  border: 1px solid var(--line);
  color: var(--muted);
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 12px;
}
.empty {
  padding: 24px;
  border: 1px dashed var(--line);
  color: var(--muted);
}
@media (max-width: 900px) {
  .shell-header { position: static; display: block; }
  .summary { min-width: 0; grid-template-columns: repeat(2, 1fr); margin-top: 16px; }
  .layout { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .event, .llm-card, .stream-card, .raw-card { grid-template-columns: 1fr; }
  .event-rail { border-right: 0; border-bottom: 1px solid var(--line); }
  .event-body { padding: 14px; }
}
"#;

const JS: &str = r#"
const records = JSON.parse(document.getElementById('trace-data').textContent);
const state = { query: '', enabled: new Set(records.map(eventKind)), selected: null };

const eventKinds = [...records.reduce((set, record) => set.add(eventKind(record)), new Set())].sort();
const counts = records.reduce((map, record) => {
  const kind = eventKind(record);
  map[kind] = (map[kind] || 0) + 1;
  return map;
}, {});

const filters = document.getElementById('filters');
for (const kind of eventKinds) {
  const button = document.createElement('button');
  button.className = 'filter';
  button.dataset.kind = kind;
  button.innerHTML = `<span>${escapeHtml(kind)}</span><span class="count">${counts[kind]}</span>`;
  button.addEventListener('click', () => {
    if (state.enabled.has(kind)) state.enabled.delete(kind);
    else state.enabled.add(kind);
    button.classList.toggle('off', !state.enabled.has(kind));
    renderAll();
  });
  filters.appendChild(button);
}

document.getElementById('search').addEventListener('input', event => {
  state.query = event.target.value.trim().toLowerCase();
  renderAll();
});

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.view).classList.add('active');
  });
});

function filteredRecords() {
  return records.filter(record => {
    if (!state.enabled.has(eventKind(record))) return false;
    if (!state.query) return true;
    return JSON.stringify(record).toLowerCase().includes(state.query);
  });
}

function renderAll() {
  renderTimeline();
  renderLlm();
  renderStreams();
  renderRaw();
}

function renderTimeline() {
  const root = document.getElementById('timeline');
  const visible = filteredRecords();
  root.innerHTML = visible.length ? '' : `<div class="empty">No events match the current filters.</div>`;
  visible.forEach((record, index) => root.appendChild(eventCard(record, index)));
}

function renderLlm() {
  const root = document.getElementById('llm');
  const calls = filteredRecords().filter(r => eventKind(r).startsWith('llm_call_'));
  root.innerHTML = calls.length ? '' : `<div class="empty">No LLM call events match.</div>`;
  calls.forEach((record, index) => root.appendChild(eventCard(record, index, 'llm-card')));
}

function renderStreams() {
  const root = document.getElementById('streams');
  const streams = filteredRecords().filter(r => eventKind(r).endsWith('stream_event'));
  root.innerHTML = streams.length ? '' : `<div class="empty">No stream events match. Run with --trace-level extended to capture provider/runtime stream records.</div>`;
  streams.forEach((record, index) => root.appendChild(eventCard(record, index, 'stream-card')));
}

function renderRaw() {
  const root = document.getElementById('raw');
  const visible = filteredRecords();
  root.innerHTML = visible.length ? '' : `<div class="empty">No raw records match.</div>`;
  visible.forEach((record, index) => {
    const card = document.createElement('article');
    card.className = 'raw-card';
    card.innerHTML = `<div class="event-rail"><span class="kind">${escapeHtml(eventKind(record))}</span><div class="meta-line">#${index + 1}</div></div><div class="event-body"><pre>${escapeHtml(JSON.stringify(record, null, 2))}</pre></div>`;
    root.appendChild(card);
  });
}

function eventCard(record, index, className = 'event') {
  const kind = eventKind(record);
  const card = document.createElement('article');
  card.className = className;
  card.dataset.id = record.id;
  card.addEventListener('click', () => selectRecord(record, card));
  const badgeClass = kind.includes('failed') || isFailed(record) ? 'fail'
    : kind.startsWith('llm') ? 'llm'
    : kind.startsWith('tool') ? 'tool'
    : kind === 'lashlang_execution' ? 'stream'
    : kind.endsWith('stream_event') ? 'stream'
    : '';
  card.innerHTML = `
    <div class="event-rail">
      <span class="kind ${badgeClass}">${escapeHtml(kind)}</span>
      <div class="meta-line">${escapeHtml(shortTime(record.timestamp))}</div>
      <div class="meta-line">turn ${record.context?.turn_index ?? 'na'} · step ${record.context?.protocol_iteration ?? 'na'}</div>
    </div>
    <div class="event-body">
      <div class="title-line">${escapeHtml(eventTitle(record))}</div>
      <div class="summary-line">${escapeHtml(eventSummary(record))}</div>
      ${pillRow(record)}
      <details><summary>Raw JSON</summary><pre>${escapeHtml(JSON.stringify(record, null, 2))}</pre></details>
    </div>`;
  return card;
}

function selectRecord(record, element) {
  document.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
  element.classList.add('selected');
  state.selected = record.id;
  document.getElementById('selectionMeta').innerHTML = `
    <div><strong>${escapeHtml(eventKind(record))}</strong></div>
    <div>${escapeHtml(record.timestamp || '')}</div>
    <div>${escapeHtml(record.id || '')}</div>
    <div>llm: ${escapeHtml(record.context?.llm_call_id || 'none')}</div>
    <div>effect: ${escapeHtml(record.context?.effect_id || 'none')}</div>`;
}

function eventKind(record) {
  return record.type || 'unknown';
}

function eventTitle(record) {
  switch (eventKind(record)) {
    case 'llm_call_started':
      return `${record.request.model}${record.request.model_variant ? ' / ' + record.request.model_variant : ''}`;
    case 'llm_call_completed':
      return `completed in ${record.response.duration_ms} ms`;
    case 'llm_call_failed':
      return record.error.message;
    case 'tool_call_started':
    case 'tool_call_completed':
      return record.name || 'tool call';
    case 'provider_stream_event':
      return `${record.event.provider}: ${record.event.event_name}`;
    case 'runtime_stream_event':
      return record.event.event_name;
    case 'turn_completed':
      return `${record.status}: ${record.done_reason}`;
    case 'lashlang_execution':
      return lashlangExecutionTitle(record.event);
    case 'custom':
      return record.name;
    case 'prompt_built':
      return `${record.prompt_chars} prompt chars`;
    default:
      return eventKind(record).replaceAll('_', ' ');
  }
}

function eventSummary(record) {
  switch (eventKind(record)) {
    case 'llm_call_started':
      return summarizeRequest(record.request);
    case 'llm_call_completed':
      return summarizeCompleted(record);
    case 'llm_call_failed':
      return record.error.raw || record.error.code || '';
    case 'tool_call_started':
      return JSON.stringify(record.args);
    case 'tool_call_completed':
      return `${record.success ? 'ok' : 'error'} in ${record.duration_ms} ms\n${JSON.stringify(record.result)}`;
    case 'provider_stream_event':
      return `seq ${record.event.sequence}, ${record.event.elapsed_ms} ms, raw ${record.event.raw_len} chars, sha ${record.event.raw_sha256}`;
    case 'runtime_stream_event':
      return record.event.visible_text || record.event.raw_text || JSON.stringify(record.event);
    case 'protocol_step':
      return `${record.plugin_id}\n${JSON.stringify(record.payload)}`;
    case 'token_usage':
      return usageText(record.usage, record.cumulative);
    case 'lashlang_execution':
      return lashlangExecutionSummary(record.event);
    case 'custom':
      return JSON.stringify(record.payload);
    case 'prompt_built':
      return (record.components || []).map(c => `${c.kind}:${c.id} ${c.chars ?? '?'} chars`).join('\n');
    default:
      return JSON.stringify(record).slice(0, 800);
  }
}

function summarizeRequest(request) {
  const parts = [];
  parts.push(`${request.messages.length} messages`);
  parts.push(`${request.tools.length} tools`);
  if (request.attachments?.length) parts.push(`${request.attachments.length} attachments`);
  const text = request.messages.flatMap(m => m.blocks || []).filter(b => b.kind === 'text').map(b => b.text).join('\n\n');
  return `${parts.join(', ')}\n${text.slice(0, 2000)}`;
}

function summarizeCompleted(record) {
  const usage = record.usage ? usageText(record.usage) : 'usage unavailable';
  return `${usage}\n${record.response.text || ''}`;
}

function usageText(usage, cumulative = null) {
  const total = (usage.input_tokens || 0) + (usage.output_tokens || 0) + (usage.cache_read_input_tokens || 0) + (usage.cache_write_input_tokens || 0);
  let text = `tokens ${total} = in ${usage.input_tokens || 0}, out ${usage.output_tokens || 0}, cache read ${usage.cache_read_input_tokens || 0}, cache write ${usage.cache_write_input_tokens || 0}, reasoning ${usage.reasoning_output_tokens || 0}`;
  if (cumulative) {
    const ctotal = (cumulative.input_tokens || 0) + (cumulative.output_tokens || 0) + (cumulative.cache_read_input_tokens || 0) + (cumulative.cache_write_input_tokens || 0);
    text += `\ncumulative ${ctotal}`;
  }
  return text;
}

function lashlangExecutionTitle(event) {
  if (!event) return 'Lashlang execution';
  switch (event.kind) {
    case 'execution_started':
      return `${event.identity?.entry_name || 'Lashlang'} started`;
    case 'execution_finished':
      return `${event.identity?.entry_name || 'Lashlang'} ${event.status}`;
    case 'node_started':
      return `${event.label} started`;
    case 'node_completed':
      return `${event.label} completed`;
    case 'node_failed':
      return `${event.label} failed`;
    case 'branch_selected':
      return `branch selected: ${event.selected}`;
    case 'child_started':
      return `${event.identity?.entry_name || 'Lashlang'} started child ${event.child?.entry_name || subjectSummary(event.child?.subject) || 'execution'}`;
    default:
      return event.kind || 'Lashlang execution';
  }
}

function lashlangExecutionSummary(event) {
  if (!event) return '';
  const identity = event.identity || {};
  const parts = [];
  if (identity.entry_name) parts.push(`entry ${identity.entry_name}`);
  if (identity.entry_kind) parts.push(`kind ${identity.entry_kind}`);
  if (identity.subject) parts.push(`subject ${subjectSummary(identity.subject)}`);
  if (identity.scope?.session_id) parts.push(`session ${identity.scope.session_id}`);
  if (identity.scope?.turn_id) parts.push(`turn ${identity.scope.turn_id}`);
  if (identity.module_ref) parts.push(`module ${identity.module_ref}`);
  if (event.node_id) parts.push(`node ${event.node_id}`);
  if (event.occurrence) parts.push(`occurrence ${event.occurrence}`);
  if (event.edge_id) parts.push(`edge ${event.edge_id}`);
  if (event.child) parts.push(`child ${subjectSummary(event.child.subject)}`);
  if (event.error) parts.push(`error ${event.error}`);
  if (event.execution_map) {
    parts.push(`${event.execution_map.nodes?.length || 0} nodes`);
    parts.push(`${event.execution_map.edges?.length || 0} edges`);
  }
  return parts.join('\n');
}

function subjectSummary(subject) {
  if (!subject) return '';
  if (subject.type === 'process') return `process ${subject.process_id}`;
  if (subject.type === 'effect') return `effect ${subject.kind || 'effect'}:${subject.effect_id}`;
  return JSON.stringify(subject);
}

function pillRow(record) {
  const pills = [];
  const ctx = record.context || {};
  if (ctx.session_id) pills.push(`session ${ctx.session_id}`);
  if (ctx.turn_id) pills.push(`turn ${ctx.turn_id}`);
  if (ctx.llm_call_id) pills.push(`llm ${ctx.llm_call_id}`);
  if (ctx.effect_id) pills.push(`effect ${ctx.effect_id}`);
  if (record.schema_version) pills.push(`schema ${record.schema_version}`);
  if (!pills.length) return '';
  return `<div class="pill-row">${pills.map(p => `<span class="pill">${escapeHtml(p)}</span>`).join('')}</div>`;
}

function isFailed(record) {
  return eventKind(record).includes('failed') || record.success === false || record.status === 'failed';
}

function shortTime(timestamp) {
  if (!timestamp) return '';
  const match = timestamp.match(/T(\d\d:\d\d:\d\d)/);
  return match ? match[1] : timestamp;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

renderAll();
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use lash_trace::{TraceContext, TraceEvent, TraceRecord};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn loaded_trace(records: Vec<TraceRecord>) -> LoadedTrace {
        LoadedTrace {
            records: records
                .into_iter()
                .map(|record| TraceEntry {
                    raw: serde_json::to_value(&record).expect("trace record JSON"),
                    typed: Some(record),
                })
                .collect(),
        }
    }

    fn temp_trace_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lash-trace-viewer-{name}-{}-{nanos}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn default_output_path_replaces_extension() {
        assert_eq!(
            default_output_path(Path::new("session.trace.jsonl")),
            PathBuf::from("session.trace.html")
        );
    }

    #[test]
    fn render_contains_embedded_trace_data() {
        let trace = loaded_trace(vec![TraceRecord::new(
            TraceContext::default().for_session("s1"),
            TraceEvent::TurnStarted {
                metadata: Default::default(),
            },
        )]);

        let html = render_html("title", Path::new("trace.jsonl"), &trace).expect("render");

        assert!(html.contains("Lash Trace Viewer"));
        assert!(html.contains("trace-data"));
        assert!(html.contains("turn_started"));
    }

    #[test]
    fn render_escapes_script_breakout_sequences() {
        let trace = loaded_trace(vec![TraceRecord::new(
            TraceContext::default(),
            TraceEvent::Custom {
                name: "x".to_string(),
                payload: serde_json::json!({"text": "</script><script>alert(1)</script>"}),
            },
        )]);

        let html = render_html("title", Path::new("trace.jsonl"), &trace).expect("render");

        assert!(!html.contains("</script><script>alert"));
        assert!(html.contains("\\u003c/script\\u003e"));
    }

    #[test]
    fn load_trace_keeps_unknown_valid_json_records() {
        let path = temp_trace_path("unknown");
        std::fs::write(
            &path,
            r#"{"schema_version":999,"type":"future_event","payload":{"text":"kept raw"}}"#,
        )
        .expect("write trace");

        let trace = load_trace(&path).expect("load trace");
        std::fs::remove_file(&path).ok();

        assert_eq!(trace.records.len(), 1);
        assert!(trace.records[0].typed.is_none());
        assert_eq!(trace.records[0].raw["type"], "future_event");

        let html = render_html("title", &path, &trace).expect("render");
        assert!(html.contains("future_event"));
        assert!(html.contains("kept raw"));
    }

    #[test]
    fn load_trace_fails_on_invalid_json() {
        let path = temp_trace_path("invalid");
        std::fs::write(&path, "{\"type\":\"turn_started\"").expect("write trace");

        let err = load_trace(&path).expect_err("invalid JSON should fail");
        std::fs::remove_file(&path).ok();

        assert!(format!("{err:#}").contains("parse trace JSONL line 1"));
    }

    #[test]
    fn load_trace_fails_on_empty_files() {
        let path = temp_trace_path("empty");
        std::fs::write(&path, "\n  \n").expect("write trace");

        let err = load_trace(&path).expect_err("empty trace should fail");
        std::fs::remove_file(&path).ok();

        assert!(format!("{err:#}").contains("trace file contains no records"));
    }
}
