use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use fff_search::git::format_git_status_opt;
use fff_search::grep::{GrepMode, GrepSearchOptions, has_regex_metacharacters, is_import_line};
use fff_search::{
    AiGrepConfig, ContentCacheBudget, FFFMode, FileItem, FilePicker, FilePickerOptions,
    FuzzySearchOptions, GrepMatch, PaginationArgs, QueryParser, SharedFrecency, SharedPicker,
};
use serde_json::json;

use crate::{ToolDefinition, ToolExecutionMode, ToolParam, ToolProvider, ToolResult};

use super::require_str;

const DEFAULT_MAX_RESULTS: usize = 20;
const MAX_CURSORS: usize = 20;
const MAX_LINE_LEN: usize = 180;
const MAX_FFF_FUZZY_QUERY_BYTES: usize = (u16::MAX as usize) / (16 * 50);

/// Search file contents using an indexed fff-search backend.
pub struct Grep {
    base_path: Result<PathBuf, String>,
    backend: OnceLock<Result<Arc<GrepBackend>, String>>,
    cursor_store: Mutex<CursorStore>,
}

impl Grep {
    pub fn new() -> Self {
        match std::env::current_dir() {
            Ok(path) => Self::with_base_path(path),
            Err(err) => {
                Self::with_init_error(format!("failed to resolve current directory: {err}"))
            }
        }
    }

    fn with_init_error(message: String) -> Self {
        Self {
            base_path: Err(message),
            backend: OnceLock::new(),
            cursor_store: Mutex::new(CursorStore::new()),
        }
    }

    fn with_base_path(base_path: PathBuf) -> Self {
        Self {
            base_path: Ok(base_path),
            backend: OnceLock::new(),
            cursor_store: Mutex::new(CursorStore::new()),
        }
    }

    fn ensure_ready(&self) -> Result<Arc<GrepBackend>, ToolResult> {
        let backend = self
            .backend
            .get_or_init(|| self.shared_backend())
            .as_ref()
            .map_err(|err| ToolResult::err_fmt(format_args!("{err}")))?;
        if !backend.picker.wait_for_scan(Duration::from_secs(30)) {
            return Err(ToolResult::err_fmt(format_args!(
                "fff-search initial scan timed out"
            )));
        }
        Ok(Arc::clone(backend))
    }

    fn shared_backend(&self) -> Result<Arc<GrepBackend>, String> {
        let base_path = self.base_path.as_ref().map_err(Clone::clone)?;
        backend_for_base(base_path)
    }

    fn lock_cursors(&self) -> Result<MutexGuard<'_, CursorStore>, ToolResult> {
        self.cursor_store
            .lock()
            .map_err(|_| ToolResult::err_fmt(format_args!("Failed to acquire cursor store lock")))
    }

    fn perform_grep(
        &self,
        backend: &GrepBackend,
        query: &str,
        mode: GrepMode,
        max_results: usize,
        cursor_id: Option<&str>,
    ) -> Result<serde_json::Value, ToolResult> {
        let file_offset = cursor_id
            .and_then(|id| self.cursor_store.lock().ok()?.get(id))
            .unwrap_or(0);

        let (options, auto_expand) = make_grep_options(mode, file_offset);

        let guard = backend.picker.read().map_err(|err| {
            ToolResult::err_fmt(format_args!("Failed to acquire picker lock: {err}"))
        })?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| ToolResult::err_fmt(format_args!("File picker not initialized")))?;

        let parser = QueryParser::new(AiGrepConfig);
        let parsed = parser.parse(query);
        let result = picker.grep(&parsed, &options);

        if result.matches.is_empty() && file_offset == 0 {
            let parts = query.split_whitespace().collect::<Vec<_>>();
            if parts.len() >= 2 {
                let first_word = parts[0];
                let is_valid_constraint = first_word.starts_with('!')
                    || first_word.starts_with('*')
                    || first_word.ends_with('/');

                if !is_valid_constraint {
                    let rest_query = parts[1..].join(" ");
                    let rest_parsed = parser.parse(&rest_query);
                    let rest_text = rest_parsed.grep_text();
                    let retry_mode = if has_regex_metacharacters(&rest_text) {
                        GrepMode::Regex
                    } else {
                        mode
                    };
                    let (retry_options, retry_auto_expand) = make_grep_options(retry_mode, 0);
                    let retry_result = picker.grep(&rest_parsed, &retry_options);

                    if !retry_result.matches.is_empty() && retry_result.matches.len() <= 10 {
                        let mut cursors = self.lock_cursors()?;
                        return Ok(structured_grep_result(
                            StructuredGrepInput {
                                query,
                                query_used: &rest_query,
                                matches: &retry_result.matches,
                                files: &retry_result.files,
                                total_matched: retry_result.matches.len(),
                                files_with_matches: retry_result.files_with_matches,
                                next_file_offset: retry_result.next_file_offset,
                                regex_fallback_error: retry_result.regex_fallback_error.as_deref(),
                                max_results,
                                auto_expand_defs: retry_auto_expand,
                                broadened_from: Some(query),
                                approximate: false,
                                picker,
                            },
                            &mut cursors,
                        ));
                    }
                }
            }

            let fuzzy_query = cleanup_fuzzy_query(query);
            let (fuzzy_options, fuzzy_auto_expand) = make_grep_options(GrepMode::Fuzzy, 0);
            let fuzzy_parsed = parser.parse(&fuzzy_query);
            let fuzzy_result = picker.grep(&fuzzy_parsed, &fuzzy_options);
            if !fuzzy_result.matches.is_empty() {
                let mut cursors = self.lock_cursors()?;
                return Ok(structured_grep_result(
                    StructuredGrepInput {
                        query,
                        query_used: &fuzzy_query,
                        matches: &fuzzy_result.matches,
                        files: &fuzzy_result.files,
                        total_matched: fuzzy_result.matches.len(),
                        files_with_matches: fuzzy_result.files_with_matches,
                        next_file_offset: fuzzy_result.next_file_offset,
                        regex_fallback_error: fuzzy_result.regex_fallback_error.as_deref(),
                        max_results,
                        auto_expand_defs: fuzzy_auto_expand,
                        broadened_from: None,
                        approximate: true,
                        picker,
                    },
                    &mut cursors,
                ));
            }

            if query.contains('/') {
                let file_query = QueryParser::default().parse(query);
                let file_result = picker.fuzzy_search(
                    &file_query,
                    None,
                    FuzzySearchOptions {
                        max_threads: 0,
                        current_file: None,
                        project_path: Some(picker.base_path()),
                        combo_boost_score_multiplier: 100,
                        min_combo_count: 3,
                        pagination: PaginationArgs {
                            offset: 0,
                            limit: 1,
                        },
                    },
                );
                if let (Some(top), Some(score)) =
                    (file_result.items.first(), file_result.scores.first())
                {
                    let query_len = query.len() as i32;
                    if score.base_score > query_len * 10 {
                        return Ok(json!({
                            "query": query,
                            "query_used": query,
                            "matches": [],
                            "files": [],
                            "count": 0,
                            "files_with_matches": 0,
                            "truncated": false,
                            "cursor": null,
                            "suggested_path": top.relative_path(picker),
                            "approximate": false,
                        }));
                    }
                }
            }

            return Ok(empty_grep_result(query));
        }

        if result.matches.is_empty() {
            return Ok(empty_grep_result(query));
        }

        let mut cursors = self.lock_cursors()?;
        Ok(structured_grep_result(
            StructuredGrepInput {
                query,
                query_used: query,
                matches: &result.matches,
                files: &result.files,
                total_matched: result.matches.len(),
                files_with_matches: result.files_with_matches,
                next_file_offset: result.next_file_offset,
                regex_fallback_error: result.regex_fallback_error.as_deref(),
                max_results,
                auto_expand_defs: auto_expand,
                broadened_from: None,
                approximate: false,
                picker,
            },
            &mut cursors,
        ))
    }
}

impl Default for Grep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for Grep {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "grep".into(),
            description: "Search file contents. Search for bare identifiers (e.g. 'InProgressQuote', 'ActorAuth'), NOT code syntax or regex. By default searches the current workspace. Pass `path` to point the search at a specific file or directory anywhere on the filesystem (including outside the workspace). If `query` accidentally starts with an obvious filesystem path followed by search text, grep treats that prefix as `path`. Within a search root, use inline constraints in the query to further narrow (e.g. '*.rs query', 'src/ query'). See server instructions for constraint syntax and core rules.".into(),
            params: vec![
                ToolParam {
                    name: "query".into(),
                    r#type: "str".into(),
                    description: "Search text or regex query with optional constraint prefixes. Pattern is matched within a single line (no cross-line matches). Use a literal token, a short phrase, or a regex — not a multi-clause natural-language query.".into(),
                    default_value: None,
                    required: true,
                },
                ToolParam {
                    name: "path".into(),
                    r#type: "str".into(),
                    description: "Optional file or directory to search within. Accepts absolute paths or paths relative to the workspace root. A directory becomes the search root; a file searches that one file only. When omitted, searches the current workspace.".into(),
                    default_value: None,
                    required: false,
                },
                ToolParam {
                    name: "maxResults".into(),
                    r#type: "int".into(),
                    description: "Max matching lines (default 20).".into(),
                    default_value: Some(json!(20)),
                    required: false,
                },
                ToolParam {
                    name: "cursor".into(),
                    r#type: "str".into(),
                    description:
                        "Cursor from a previous grep result. Only use if previous results were not sufficient."
                            .into(),
                    default_value: None,
                    required: false,
                },
            ],
            returns: "dict".into(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: ToolExecutionMode::Parallel,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let raw_query = match require_str(args, "query") {
            Ok(query) => query,
            Err(err) => return err,
        };
        let max_results = match parse_max_results(args) {
            Ok(max_results) => max_results,
            Err(err) => return err,
        };
        let cursor = args.get("cursor").and_then(|value| value.as_str());
        let path_arg = args
            .get("path")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());

        let default_base = self.base_path.as_ref().cloned().ok();
        let inferred_scope = path_arg
            .is_none()
            .then(|| infer_path_prefix(default_base.as_deref(), raw_query))
            .flatten();
        let path_arg_owned;
        let query_owned;
        let (path_arg, raw_query) = if let Some((path, query)) = inferred_scope {
            path_arg_owned = path;
            query_owned = query;
            (Some(path_arg_owned.as_str()), query_owned.as_str())
        } else {
            (path_arg, raw_query)
        };

        let (backend, query) = match path_arg {
            Some(path) => match resolve_path_scope(default_base.as_deref(), path) {
                Ok(scope) => {
                    let backend = match backend_for_base(&scope.base_path) {
                        Ok(backend) => backend,
                        Err(err) => return ToolResult::err_fmt(format_args!("{err}")),
                    };
                    let query = match scope.file_constraint {
                        Some(filename) => format!("{filename} {raw_query}"),
                        None => raw_query.to_string(),
                    };
                    if !backend.picker.wait_for_scan(Duration::from_secs(30)) {
                        return ToolResult::err_fmt(format_args!(
                            "fff-search initial scan timed out for {}",
                            scope.base_path.display()
                        ));
                    }
                    (backend, query)
                }
                Err(err) => return err,
            },
            None => match self.ensure_ready() {
                Ok(backend) => (backend, raw_query.to_string()),
                Err(err) => return err,
            },
        };

        let grep_text = QueryParser::new(AiGrepConfig).parse(&query).grep_text();
        let mode = if has_regex_metacharacters(&grep_text) {
            GrepMode::Regex
        } else {
            GrepMode::PlainText
        };

        match self.perform_grep(&backend, &query, mode, max_results, cursor) {
            Ok(value) => ToolResult::ok(value),
            Err(err) => err,
        }
    }
}

struct PathScope {
    base_path: PathBuf,
    file_constraint: Option<String>,
}

/// Resolve a user-supplied `path` into a search root and optional
/// single-file filter. A directory becomes the search root directly; a
/// file becomes its parent directory plus a `<filename>` constraint.
/// Relative paths resolve against the workspace root when available
/// and fall back to the current directory otherwise.
fn resolve_path_scope(
    default_base: Option<&Path>,
    requested: &str,
) -> Result<PathScope, ToolResult> {
    let candidate = Path::new(requested);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else if let Some(base) = default_base {
        base.join(candidate)
    } else {
        std::env::current_dir()
            .map_err(|err| {
                ToolResult::err_fmt(format_args!("failed to resolve current directory: {err}"))
            })?
            .join(candidate)
    };
    let canonical = std::fs::canonicalize(&absolute).map_err(|err| {
        ToolResult::err_fmt(format_args!(
            "`path` {requested} does not exist or is not accessible: {err}"
        ))
    })?;
    if canonical.is_dir() {
        Ok(PathScope {
            base_path: canonical,
            file_constraint: None,
        })
    } else {
        let filename = canonical
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .ok_or_else(|| {
                ToolResult::err_fmt(format_args!("`path` {requested} has no filename component"))
            })?;
        let parent = canonical.parent().map(Path::to_path_buf).ok_or_else(|| {
            ToolResult::err_fmt(format_args!("`path` {requested} has no parent directory"))
        })?;
        Ok(PathScope {
            base_path: parent,
            file_constraint: Some(filename),
        })
    }
}

fn infer_path_prefix(default_base: Option<&Path>, query: &str) -> Option<(String, String)> {
    let trimmed = query.trim();
    let (candidate, rest) = split_first_query_token(trimmed)?;
    let candidate = candidate.trim_matches(['"', '\'']);
    if candidate.is_empty() || rest.trim().is_empty() || !looks_like_path(candidate) {
        return None;
    }

    let path = Path::new(candidate);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        default_base?.join(path)
    };
    absolute
        .exists()
        .then(|| (candidate.to_string(), rest.trim().to_string()))
}

fn split_first_query_token(query: &str) -> Option<(&str, &str)> {
    let mut chars = query.char_indices();
    let (_, first) = chars.next()?;
    if first == '"' || first == '\'' {
        for (index, ch) in chars {
            if ch == first {
                let rest = query[index + ch.len_utf8()..].trim_start();
                return Some((&query[..=index], rest));
            }
        }
        return None;
    }

    query
        .char_indices()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, _)| (&query[..index], query[index..].trim_start()))
}

fn looks_like_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.contains('/')
}

/// Look up — or create — a shared fff-search backend rooted at
/// `base_path`. Reuses the process-wide backend cache so repeat
/// searches against the same path avoid the initial scan cost.
fn backend_for_base(base_path: &Path) -> Result<Arc<GrepBackend>, String> {
    let cache_key = std::fs::canonicalize(base_path).unwrap_or_else(|_| base_path.to_path_buf());
    let cache = shared_backend_cache();
    let mut cache = cache
        .lock()
        .map_err(|_| "failed to lock shared grep backend cache".to_string())?;
    if let Some(existing) = cache.get(&cache_key) {
        return existing.clone();
    }
    let backend = initialize_backend_at(base_path).map(Arc::new);
    cache.insert(cache_key, backend.clone());
    backend
}

fn initialize_backend_at(base_path: &Path) -> Result<GrepBackend, String> {
    let picker = SharedPicker::default();
    FilePicker::new_with_shared_state(
        picker.clone(),
        SharedFrecency::default(),
        FilePickerOptions {
            base_path: base_path.to_string_lossy().into_owned(),
            enable_mmap_cache: false,
            enable_content_indexing: false,
            mode: FFFMode::Ai,
            cache_budget: Some(grep_content_cache_budget()),
            watch: false,
        },
    )
    .map_err(|err| format!("failed to initialize indexed grep backend: {err}"))?;
    Ok(GrepBackend { picker })
}

struct GrepBackend {
    picker: SharedPicker,
}

type SharedBackendCache = Mutex<HashMap<PathBuf, Result<Arc<GrepBackend>, String>>>;

fn shared_backend_cache() -> &'static SharedBackendCache {
    static CACHE: OnceLock<SharedBackendCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn grep_content_cache_budget() -> ContentCacheBudget {
    ContentCacheBudget {
        max_files: 0,
        max_bytes: 0,
        max_file_size: 10 * 1024 * 1024,
        cached_count: Default::default(),
        cached_bytes: Default::default(),
    }
}

fn parse_max_results(args: &serde_json::Value) -> Result<usize, ToolResult> {
    match args.get("maxResults") {
        None => Ok(DEFAULT_MAX_RESULTS),
        Some(value) if value.is_null() => Ok(DEFAULT_MAX_RESULTS),
        Some(value) => {
            let parsed = value
                .as_u64()
                .map(|number| number as usize)
                .or_else(|| value.as_f64().map(|number| number as usize))
                .ok_or_else(|| {
                    ToolResult::err_fmt(format_args!("Invalid maxResults: expected number"))
                })?;
            if parsed == 0 {
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid maxResults: must be >= 1"
                )));
            }
            Ok(parsed)
        }
    }
}

fn cleanup_fuzzy_query(input: &str) -> String {
    let mut output = String::with_capacity(input.len().min(MAX_FFF_FUZZY_QUERY_BYTES));
    for ch in input.chars() {
        if !matches!(ch, ':' | '-' | '_') {
            for lower in ch.to_lowercase() {
                let next_len = output.len() + lower.len_utf8();
                if next_len > MAX_FFF_FUZZY_QUERY_BYTES {
                    return output;
                }
                output.push(lower);
            }
        }
    }
    output
}

fn make_grep_options(mode: GrepMode, file_offset: usize) -> (GrepSearchOptions, bool) {
    let max_matches_per_file = 10;
    let before_context = 0;
    let auto_expand_defs = before_context == 0;
    let after_context = if auto_expand_defs { 8 } else { before_context };

    (
        GrepSearchOptions {
            max_file_size: 10 * 1024 * 1024,
            max_matches_per_file,
            smart_case: true,
            file_offset,
            page_limit: 50,
            mode,
            time_budget_ms: 0,
            before_context,
            after_context,
            classify_definitions: true,
            trim_whitespace: false,
            abort_signal: None,
        },
        auto_expand_defs,
    )
}

#[derive(Default)]
struct CursorStore {
    counter: u64,
    cursors: HashMap<String, usize>,
    insertion_order: VecDeque<String>,
}

impl CursorStore {
    fn new() -> Self {
        Self::default()
    }

    fn store(&mut self, file_offset: usize) -> String {
        self.counter = self.counter.wrapping_add(1);
        let id = self.counter.to_string();
        self.cursors.insert(id.clone(), file_offset);
        self.insertion_order.push_back(id.clone());
        while self.cursors.len() > MAX_CURSORS {
            if let Some(oldest) = self.insertion_order.pop_front() {
                self.cursors.remove(&oldest);
            }
        }
        id
    }

    fn get(&self, id: &str) -> Option<usize> {
        self.cursors.get(id).copied()
    }
}

fn truncate_line_for_ai(line: &str, match_ranges: Option<&[(u32, u32)]>, max_len: usize) -> String {
    let trimmed = line.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.len() <= max_len {
        return trimmed.to_string();
    }

    if let Some(ranges) = match_ranges
        && let Some(&(match_start, match_end)) = ranges.first()
    {
        let match_start = match_start as usize;
        let match_end = match_end as usize;
        let match_len = match_end.saturating_sub(match_start);
        let budget = max_len.saturating_sub(match_len);
        let before = budget / 3;
        let after = budget - before;
        let win_start = floor_char_boundary(trimmed, match_start.saturating_sub(before));
        let win_end = ceil_char_boundary(trimmed, (match_end + after).min(trimmed.len()));

        let mut result = trimmed[win_start..win_end].to_string();
        if win_start > 0 {
            result.insert_str(0, "...");
        }
        if win_end < trimmed.len() {
            result.push_str("...");
        }
        return result;
    }

    let end = ceil_char_boundary(trimmed, max_len);
    format!("{}...", &trimmed[..end])
}

fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut idx = index;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut idx = index;
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

struct StructuredGrepInput<'a> {
    query: &'a str,
    query_used: &'a str,
    matches: &'a [GrepMatch],
    files: &'a [&'a FileItem],
    total_matched: usize,
    files_with_matches: usize,
    next_file_offset: usize,
    regex_fallback_error: Option<&'a str>,
    max_results: usize,
    auto_expand_defs: bool,
    broadened_from: Option<&'a str>,
    approximate: bool,
    picker: &'a FilePicker,
}

fn structured_grep_result(
    input: StructuredGrepInput<'_>,
    cursor_store: &mut CursorStore,
) -> serde_json::Value {
    let mut indices = (0..input.matches.len()).collect::<Vec<_>>();
    if input.auto_expand_defs {
        indices.sort_unstable_by_key(|&index| {
            if input.matches[index].is_definition {
                0
            } else if is_import_line(&input.matches[index].line_content) {
                2
            } else {
                1
            }
        });
    }
    indices.truncate(input.max_results);

    let cursor = (input.next_file_offset > 0).then(|| cursor_store.store(input.next_file_offset));
    let mut per_file: HashMap<String, usize> = HashMap::new();
    let mut file_order: Vec<String> = Vec::new();
    let mut suggested_path = None::<String>;
    let matches = indices
        .iter()
        .map(|&index| {
            let matched = &input.matches[index];
            let file = input.files[matched.file_index];
            let path = file.relative_path(input.picker);
            let count = per_file.entry(path.clone()).or_insert_with(|| {
                file_order.push(path.clone());
                0
            });
            *count += 1;
            if suggested_path.is_none() || matched.is_definition {
                suggested_path = Some(path.clone());
            }
            let ranges = matched
                .match_byte_offsets
                .iter()
                .map(|(start, end)| {
                    json!({
                        "start": start,
                        "end": end,
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "path": path,
                "line": matched.line_number,
                "column": matched.col.saturating_add(1),
                "byte_column": matched.col,
                "excerpt": truncate_line_for_ai(
                    &matched.line_content,
                    Some(matched.match_byte_offsets.as_ref()),
                    MAX_LINE_LEN
                ),
                "match": first_match_text(matched),
                "ranges": ranges,
                "is_definition": matched.is_definition,
            })
        })
        .collect::<Vec<_>>();

    let files = file_order
        .into_iter()
        .map(|path| {
            let file = input
                .files
                .iter()
                .find(|file| file.relative_path(input.picker) == path)
                .expect("file_order only contains known files");
            json!({
                "path": path,
                "count": per_file[&path],
                "size_bytes": file.size,
                "is_binary": file.is_binary(),
                "git_status": format_git_status_opt(file.git_status),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "query": input.query,
        "query_used": input.query_used,
        "broadened_from": input.broadened_from,
        "approximate": input.approximate,
        "matches": matches,
        "files": files,
        "count": input.total_matched,
        "shown": indices.len(),
        "files_with_matches": input.files_with_matches,
        "truncated": input.total_matched > indices.len() || input.next_file_offset > 0,
        "cursor": cursor,
        "suggested_path": suggested_path,
        "regex_fallback_error": input.regex_fallback_error,
    })
}

fn empty_grep_result(query: &str) -> serde_json::Value {
    json!({
        "query": query,
        "query_used": query,
        "broadened_from": null,
        "regex_fallback_error": null,
        "matches": [],
        "files": [],
        "count": 0,
        "shown": 0,
        "files_with_matches": 0,
        "truncated": false,
        "cursor": null,
        "suggested_path": null,
        "approximate": false,
    })
}

fn first_match_text(matched: &GrepMatch) -> String {
    let Some((start, end)) = matched.match_byte_offsets.first().copied() else {
        return String::new();
    };
    let start = floor_char_boundary(&matched.line_content, start as usize);
    let end = ceil_char_boundary(&matched.line_content, end as usize);
    matched.line_content[start..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_grep_matches_with_query() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool.execute("grep", &json!({"query": "hello"})).await;
        assert!(result.success);
        assert_eq!(result.result["count"], 2);
        assert_eq!(result.result["matches"][0]["path"], "test.txt");
        assert_eq!(result.result["matches"][0]["excerpt"], "hello world");
        assert_eq!(result.result["matches"][1]["excerpt"], "hello again");
    }

    #[tokio::test]
    async fn test_grep_returns_structured_file_summaries() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "fn thing() {}\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool.execute("grep", &json!({"query": "thing"})).await;
        assert!(result.success);
        assert_eq!(result.result["files"][0]["path"], "alpha.rs");
        assert_eq!(result.result["files"][0]["count"], 1);
        assert_eq!(result.result["suggested_path"], "alpha.rs");
    }

    #[tokio::test]
    async fn test_grep_structured_counts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\nctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool.execute("grep", &json!({"query": "ctx"})).await;
        assert!(result.success);
        assert_eq!(result.result["count"], 2);
        assert_eq!(result.result["files"][0]["count"], 2);
    }

    #[tokio::test]
    async fn test_grep_empty_result_keeps_structured_metadata() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool.execute("grep", &json!({"query": "missing"})).await;
        assert!(result.success);
        assert_eq!(result.result["matches"].as_array().unwrap().len(), 0);
        assert!(result.result["broadened_from"].is_null());
        assert!(result.result["regex_fallback_error"].is_null());
    }

    #[tokio::test]
    async fn test_grep_long_query_does_not_panic_in_fuzzy_fallback() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "short searchable content\n").unwrap();

        let query = "definitely missing ".repeat(20);
        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool.execute("grep", &json!({"query": query})).await;

        assert!(
            result.success,
            "long query should not panic or fail: {:?}",
            result.result
        );
    }

    #[test]
    fn test_cleanup_fuzzy_query_caps_to_fff_score_limit() {
        let query = "Ä".repeat(MAX_FFF_FUZZY_QUERY_BYTES + 10);
        let cleaned = cleanup_fuzzy_query(&query);

        assert!(cleaned.len() <= MAX_FFF_FUZZY_QUERY_BYTES);
        assert!(cleaned.is_char_boundary(cleaned.len()));
    }

    #[tokio::test]
    async fn test_grep_initializes_backend_lazily() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        assert!(tool.backend.get().is_none());

        let result = tool.execute("grep", &json!({"query": "ctx"})).await;
        assert!(result.success);
        assert!(tool.backend.get().is_some());
    }

    #[tokio::test]
    async fn test_grep_path_scopes_search_to_subdirectory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("inner")).unwrap();
        std::fs::write(dir.path().join("outer.txt"), "banana at root\n").unwrap();
        std::fs::write(dir.path().join("inner/inner.txt"), "banana in inner\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool
            .execute("grep", &json!({"query": "banana", "path": "inner"}))
            .await;
        assert!(result.success);
        assert!(
            result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "inner.txt"),
            "expected inner.txt match, got {:?}",
            result.result
        );
        assert!(
            !result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "outer.txt"),
            "path scope should exclude outer.txt, got {:?}",
            result.result
        );
    }

    #[tokio::test]
    async fn test_grep_path_constrains_search_to_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "banana\n").unwrap();
        std::fs::write(dir.path().join("other.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool
            .execute("grep", &json!({"query": "banana", "path": "notes.txt"}))
            .await;
        assert!(result.success);
        assert!(
            result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "notes.txt"),
            "expected notes.txt match, got {:?}",
            result.result
        );
        assert!(
            !result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "other.txt"),
            "file path should exclude other.txt"
        );
    }

    #[tokio::test]
    async fn test_grep_path_can_search_outside_workspace() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = tool
            .execute(
                "grep",
                &json!({
                    "query": "banana",
                    "path": outside.path().to_string_lossy(),
                }),
            )
            .await;
        assert!(
            result.success,
            "expected search outside workspace to succeed, got {:?}",
            result.result
        );
        assert!(
            result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected external.txt match, got {:?}",
            result.result
        );
    }

    #[tokio::test]
    async fn test_grep_infers_obvious_path_prefix_from_query() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = tool
            .execute(
                "grep",
                &json!({"query": format!("{} banana", outside.path().display())}),
            )
            .await;
        assert!(result.success);
        assert!(
            result.result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected inferred path search to find external.txt, got {:?}",
            result.result
        );
    }

    #[tokio::test]
    async fn test_grep_path_missing_returns_clear_error() {
        let workspace = TempDir::new().unwrap();
        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = tool
            .execute(
                "grep",
                &json!({"query": "banana", "path": "/nonexistent/totally/fake"}),
            )
            .await;
        assert!(!result.success);
        let message = result.result.as_str().unwrap_or("");
        assert!(
            message.contains("does not exist"),
            "expected missing-path error, got {message:?}"
        );
    }

    #[tokio::test]
    async fn test_grep_backend_is_shared_process_wide_for_same_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let left = Grep::with_base_path(dir.path().to_path_buf());
        let right = Grep::with_base_path(dir.path().to_path_buf());

        let left_backend = left.ensure_ready().expect("left backend");
        let right_backend = right.ensure_ready().expect("right backend");

        assert!(Arc::ptr_eq(&left_backend, &right_backend));
    }
}
