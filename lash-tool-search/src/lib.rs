use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use fff_search::git::format_git_status_opt;
use fff_search::grep::{GrepMode, GrepSearchOptions, has_regex_metacharacters, is_import_line};
use fff_search::{
    AiGrepConfig, ContentCacheBudget, FFFMode, FileItem, FilePicker, FilePickerOptions,
    FuzzySearchOptions, GrepMatch, PaginationArgs, QueryParser, SharedFrecency, SharedPicker,
};
use serde_json::json;

use lash_core::{
    ToolCall, ToolContract, ToolDefinition, ToolExecutionMode, ToolFailureClass, ToolManifest,
    ToolProvider, ToolResult, ToolRetryPolicy,
};

use lash_tool_support::{object_schema, require_str};

const DEFAULT_MAX_RESULTS: usize = 20;
const MAX_CURSORS: usize = 20;
const MAX_LINE_LEN: usize = 180;
const MAX_FFF_FUZZY_QUERY_BYTES: usize = (u16::MAX as usize) / (16 * 50);
const GREP_WALL_TIMEOUT: Duration = Duration::from_secs(5);
const FFF_SEARCH_BUDGET: Duration = Duration::from_secs(3);
const DIRECT_FILE_MAX_SIZE: u64 = 10 * 1024 * 1024;

/// Search file contents using an indexed fff-search backend.
pub struct Grep {
    base_path: Result<PathBuf, String>,
    backend: OnceLock<Result<Arc<GrepBackend>, String>>,
    cursor_store: Arc<Mutex<CursorStore>>,
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
            cursor_store: Arc::new(Mutex::new(CursorStore::new())),
        }
    }

    fn with_base_path(base_path: PathBuf) -> Self {
        Self {
            base_path: Ok(base_path),
            backend: OnceLock::new(),
            cursor_store: Arc::new(Mutex::new(CursorStore::new())),
        }
    }

    fn ensure_ready_for_query(&self, query: &str) -> Result<Arc<GrepBackend>, ToolResult> {
        let backend = self
            .backend
            .get_or_init(|| self.shared_backend())
            .as_ref()
            .map_err(|err| ToolResult::err_fmt(format_args!("{err}")))?;
        if !backend.picker.wait_for_scan(GREP_WALL_TIMEOUT) {
            return Err(timeout_grep_result(
                query,
                "index_scan",
                GREP_WALL_TIMEOUT,
                "fff-search initial scan timed out",
            ));
        }
        Ok(Arc::clone(backend))
    }

    fn shared_backend(&self) -> Result<Arc<GrepBackend>, String> {
        let base_path = self.base_path.as_ref().map_err(Clone::clone)?;
        backend_for_base(base_path)
    }

    fn lock_cursors(
        cursor_store: &Mutex<CursorStore>,
    ) -> Result<MutexGuard<'_, CursorStore>, ToolResult> {
        cursor_store
            .lock()
            .map_err(|_| ToolResult::err_fmt(format_args!("Failed to acquire cursor store lock")))
    }

    fn perform_grep(
        backend: &GrepBackend,
        cursor_store: &Mutex<CursorStore>,
        query: &str,
        mode: GrepMode,
        max_results: usize,
        cursor_id: Option<&str>,
        control: &GrepRunControl,
    ) -> Result<serde_json::Value, ToolResult> {
        control.check(query)?;
        let file_offset = cursor_id
            .and_then(|id| cursor_store.lock().ok()?.get(id))
            .unwrap_or(0);

        let (options, auto_expand) = make_grep_options(mode, file_offset, control);

        let guard = backend.picker.read().map_err(|err| {
            ToolResult::err_fmt(format_args!("Failed to acquire picker lock: {err}"))
        })?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| ToolResult::err_fmt(format_args!("File picker not initialized")))?;

        let parser = QueryParser::new(AiGrepConfig);
        let parsed = parser.parse(query);
        control.check(query)?;
        let result = picker.grep(&parsed, &options);

        if result.matches.is_empty() && file_offset == 0 {
            control.check(query)?;
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
                    let (retry_options, retry_auto_expand) =
                        make_grep_options(retry_mode, 0, control);
                    control.check(query)?;
                    let retry_result = picker.grep(&rest_parsed, &retry_options);

                    if !retry_result.matches.is_empty() && retry_result.matches.len() <= 10 {
                        let mut cursors = Self::lock_cursors(cursor_store)?;
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
            let (fuzzy_options, fuzzy_auto_expand) = make_grep_options(GrepMode::Fuzzy, 0, control);
            let fuzzy_parsed = parser.parse(&fuzzy_query);
            control.check(query)?;
            let fuzzy_result = picker.grep(&fuzzy_parsed, &fuzzy_options);
            if !fuzzy_result.matches.is_empty() {
                let mut cursors = Self::lock_cursors(cursor_store)?;
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
                control.check(query)?;
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
                            "shown": 0,
                            "files_with_matches": 0,
                            "truncated": false,
                            "cursor": null,
                            "suggested_path": top.relative_path(picker),
                            "approximate": false,
                            "broadened_from": null,
                            "regex_fallback_error": null,
                            "timed_out": false,
                            "cancelled": false,
                            "error": null,
                        }));
                    }
                }
            }

            return Ok(empty_grep_result(query));
        }

        if result.matches.is_empty() {
            return Ok(empty_grep_result(query));
        }

        let mut cursors = Self::lock_cursors(cursor_store)?;
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
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![grep_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "grep").then(|| Arc::new(grep_tool_definition().contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let cancellation_token = call.context.cancellation_token().cloned();
        self.execute_inner(call.args, cancellation_token).await
    }
}

fn grep_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
                "grep",
                "Search file contents. Search for bare identifiers (e.g. 'InProgressQuote', 'ActorAuth'), NOT code syntax or regex. By default searches the current workspace. Pass `path` to point the search at a specific file or directory anywhere on the filesystem (including outside the workspace). If `query` accidentally starts with an obvious filesystem path followed by search text, grep treats that prefix as `path`. Within a search root, use inline constraints in the query as a leading token: `*.rs term` (extension), `src/ term` (path segment), `**/foo/* term` (glob), `!*.test.ts term` (negate). Constraints AND together; one search term per query.",
                object_schema(
                    json!({
                        "query": {
                            "type": "string",
                            "description": "Search text or regex query with optional constraint prefixes. Pattern is matched within a single line (no cross-line matches). Use a literal token, a short phrase, or a regex — not a multi-clause natural-language query."
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional file or directory to search within. Accepts absolute paths or paths relative to the workspace root. A directory becomes the search root; a file searches that one file only. When omitted, searches the current workspace."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": DEFAULT_MAX_RESULTS,
                            "description": "Max matching lines (default 20)."
                        },
                        "cursor": {
                            "type": "string",
                            "description": "Cursor from a previous grep result. Only use if previous results were not sufficient."
                        }
                    }),
                    &["query"],
                ),
                json!({ "type": "object", "additionalProperties": true }),
            )
            .with_examples(vec![
                r#"grep(query="ToolProvider", path="lash/src")"#.into(),
                r#"grep(query="*.rs apply_patch", path=".")"#.into(),
                r#"grep(query="current_query")"#.into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata(
                "filesystem",
                &["search_files", "ripgrep"],
            ))
            .with_execution_mode(ToolExecutionMode::Parallel)
            .with_retry_policy(ToolRetryPolicy::safe(2, 50, 150))
}

impl Grep {
    async fn execute_inner(
        &self,
        args: &serde_json::Value,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
    ) -> ToolResult {
        let raw_query = match require_str(args, "query") {
            Ok(query) => query,
            Err(err) => return err,
        };
        let max_results = match parse_limit(args) {
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
                Ok(PathScope::File(file_path)) => {
                    return direct_file_grep(
                        raw_query,
                        &file_path,
                        default_base.as_deref(),
                        max_results,
                        cancellation_token,
                    )
                    .await;
                }
                Ok(PathScope::Directory(base_path)) => {
                    let backend = match backend_for_base(&base_path) {
                        Ok(backend) => backend,
                        Err(err) => return ToolResult::err_fmt(format_args!("{err}")),
                    };
                    if !backend.picker.wait_for_scan(GREP_WALL_TIMEOUT) {
                        return timeout_grep_result(
                            raw_query,
                            "index_scan",
                            GREP_WALL_TIMEOUT,
                            &format!(
                                "fff-search initial scan timed out for {}",
                                base_path.display()
                            ),
                        );
                    }
                    (backend, raw_query.to_string())
                }
                Err(err) => return err,
            },
            None => match self.ensure_ready_for_query(raw_query) {
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

        bounded_indexed_grep(
            Arc::clone(&backend),
            Arc::clone(&self.cursor_store),
            query,
            mode,
            max_results,
            cursor.map(str::to_string),
            cancellation_token,
        )
        .await
    }
}

enum PathScope {
    Directory(PathBuf),
    File(PathBuf),
}

#[derive(Clone)]
struct GrepRunControl {
    abort_signal: Arc<AtomicBool>,
    deadline: Instant,
    budget: Duration,
}

impl GrepRunControl {
    fn new(abort_signal: Arc<AtomicBool>, budget: Duration) -> Self {
        Self {
            abort_signal,
            deadline: Instant::now() + budget,
            budget,
        }
    }

    fn check(&self, query: &str) -> Result<(), ToolResult> {
        if self.abort_signal.load(Ordering::Relaxed) {
            return Err(cancelled_grep_result(query));
        }
        if Instant::now() >= self.deadline {
            self.abort_signal.store(true, Ordering::Relaxed);
            return Err(timeout_grep_result(
                query,
                "fff_search",
                self.budget,
                "grep search timed out",
            ));
        }
        Ok(())
    }

    fn remaining_budget_ms(&self) -> u64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .max(1) as u64
    }
}

async fn bounded_indexed_grep(
    backend: Arc<GrepBackend>,
    cursor_store: Arc<Mutex<CursorStore>>,
    query: String,
    mode: GrepMode,
    max_results: usize,
    cursor: Option<String>,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
) -> ToolResult {
    let abort_signal = Arc::new(AtomicBool::new(false));
    let cancellation_watcher = cancellation_token.map(|token| {
        let abort_signal = Arc::clone(&abort_signal);
        tokio::spawn(async move {
            token.cancelled().await;
            abort_signal.store(true, Ordering::Relaxed);
        })
    });
    let control = GrepRunControl::new(Arc::clone(&abort_signal), FFF_SEARCH_BUDGET);
    let timeout_query = query.clone();
    let handle = tokio::task::spawn_blocking(move || {
        Grep::perform_grep(
            &backend,
            &cursor_store,
            &query,
            mode,
            max_results,
            cursor.as_deref(),
            &control,
        )
    });

    let result = match tokio::time::timeout(GREP_WALL_TIMEOUT, handle).await {
        Ok(Ok(Ok(value))) => ToolResult::ok(value),
        Ok(Ok(Err(err))) => err,
        Ok(Err(err)) => ToolResult::err(serde_json::json!({
            "query": timeout_query,
            "query_used": timeout_query,
            "matches": [],
            "files": [],
            "count": 0,
            "shown": 0,
            "files_with_matches": 0,
            "truncated": false,
            "cursor": null,
            "suggested_path": null,
            "approximate": false,
            "timed_out": false,
            "cancelled": false,
            "error": {
                "kind": "panic",
                "message": format!("grep worker failed: {err}"),
                "stage": "fff_search",
            },
        })),
        Err(_) => {
            abort_signal.store(true, Ordering::Relaxed);
            timeout_grep_result(
                &timeout_query,
                "fff_search",
                GREP_WALL_TIMEOUT,
                "grep search timed out",
            )
        }
    };
    if let Some(watcher) = cancellation_watcher {
        watcher.abort();
    }
    result
}

async fn direct_file_grep(
    query: &str,
    file_path: &Path,
    default_base: Option<&Path>,
    max_results: usize,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
) -> ToolResult {
    let query = query.to_string();
    let file_path = file_path.to_path_buf();
    let default_base = default_base.map(Path::to_path_buf);
    let abort_signal = Arc::new(AtomicBool::new(false));
    let cancellation_watcher = cancellation_token.map(|token| {
        let abort_signal = Arc::clone(&abort_signal);
        tokio::spawn(async move {
            token.cancelled().await;
            abort_signal.store(true, Ordering::Relaxed);
        })
    });
    let worker_abort = Arc::clone(&abort_signal);
    let timeout_query = query.clone();
    let handle = tokio::task::spawn_blocking(move || {
        direct_file_grep_sync(
            &query,
            &file_path,
            default_base.as_deref(),
            max_results,
            &worker_abort,
        )
    });
    let result = match tokio::time::timeout(GREP_WALL_TIMEOUT, handle).await {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => ToolResult::err(serde_json::json!({
            "query": timeout_query,
            "query_used": timeout_query,
            "matches": [],
            "files": [],
            "count": 0,
            "shown": 0,
            "files_with_matches": 0,
            "truncated": false,
            "cursor": null,
            "suggested_path": null,
            "approximate": false,
            "timed_out": false,
            "cancelled": false,
            "error": {
                "kind": "panic",
                "message": format!("direct grep worker failed: {err}"),
                "stage": "direct_file",
            },
        })),
        Err(_) => {
            abort_signal.store(true, Ordering::Relaxed);
            timeout_grep_result(
                &timeout_query,
                "direct_file",
                GREP_WALL_TIMEOUT,
                "direct file grep timed out",
            )
        }
    };
    if let Some(watcher) = cancellation_watcher {
        watcher.abort();
    }
    result
}

/// Resolve a user-supplied `path` into either an indexed directory search root
/// or a direct single-file scan. Relative paths resolve against the workspace
/// root when available and fall back to the current directory otherwise.
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
        Ok(PathScope::Directory(canonical))
    } else {
        Ok(PathScope::File(canonical))
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
        max_file_size: DIRECT_FILE_MAX_SIZE,
        cached_count: Default::default(),
        cached_bytes: Default::default(),
    }
}

fn direct_file_grep_sync(
    query: &str,
    file_path: &Path,
    default_base: Option<&Path>,
    max_results: usize,
    abort_signal: &AtomicBool,
) -> ToolResult {
    if abort_signal.load(Ordering::Relaxed) {
        return cancelled_grep_result(query);
    }
    let metadata = match std::fs::metadata(file_path) {
        Ok(metadata) => metadata,
        Err(err) => {
            return ToolResult::err(serde_json::json!({
                "query": query,
                "query_used": query,
                "matches": [],
                "files": [],
                "count": 0,
                "shown": 0,
                "files_with_matches": 0,
                "truncated": false,
                "cursor": null,
                "suggested_path": null,
                "approximate": false,
                "timed_out": false,
                "cancelled": false,
                "error": {
                    "kind": "io",
                    "message": format!("failed to stat file: {err}"),
                    "stage": "direct_file",
                },
            }));
        }
    };
    if !metadata.is_file() {
        return ToolResult::err(serde_json::json!({
            "query": query,
            "query_used": query,
            "matches": [],
            "files": [],
            "count": 0,
            "shown": 0,
            "files_with_matches": 0,
            "truncated": false,
            "cursor": null,
            "suggested_path": null,
            "approximate": false,
            "timed_out": false,
            "cancelled": false,
            "error": {
                "kind": "not_a_file",
                "message": "path is not a regular file",
                "stage": "direct_file",
            },
        }));
    }
    if metadata.len() > DIRECT_FILE_MAX_SIZE {
        return ToolResult::err(serde_json::json!({
            "query": query,
            "query_used": query,
            "matches": [],
            "files": [],
            "count": 0,
            "shown": 0,
            "files_with_matches": 0,
            "truncated": false,
            "cursor": null,
            "suggested_path": null,
            "approximate": false,
            "timed_out": false,
            "cancelled": false,
            "error": {
                "kind": "file_too_large",
                "message": format!("file exceeds grep limit of {DIRECT_FILE_MAX_SIZE} bytes"),
                "stage": "direct_file",
                "size_bytes": metadata.len(),
                "max_size_bytes": DIRECT_FILE_MAX_SIZE,
            },
        }));
    }

    let parsed = QueryParser::new(AiGrepConfig).parse(query);
    let grep_text = parsed.grep_text();
    if grep_text.is_empty() {
        return ToolResult::ok(empty_grep_result(query));
    }

    let bytes = match std::fs::read(file_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            return ToolResult::err(serde_json::json!({
                "query": query,
                "query_used": grep_text,
                "matches": [],
                "files": [],
                "count": 0,
                "shown": 0,
                "files_with_matches": 0,
                "truncated": false,
                "cursor": null,
                "suggested_path": null,
                "approximate": false,
                "timed_out": false,
                "cancelled": false,
                "error": {
                    "kind": "io",
                    "message": format!("failed to read file: {err}"),
                    "stage": "direct_file",
                },
            }));
        }
    };
    if abort_signal.load(Ordering::Relaxed) {
        return cancelled_grep_result(query);
    }

    let display_path = display_path_for_direct_file(file_path, default_base);
    let matcher = match DirectMatcher::new(&grep_text) {
        Ok(matcher) => matcher,
        Err(regex_error) => DirectMatcher::literal_with_error(&grep_text, regex_error),
    };

    let text = String::from_utf8_lossy(&bytes);
    let mut matches = Vec::new();
    let mut total_matches = 0usize;
    for (line_index, segment) in text.split_inclusive('\n').enumerate() {
        if abort_signal.load(Ordering::Relaxed) {
            return cancelled_grep_result(query);
        }
        let line = segment.trim_end_matches(['\r', '\n']);
        let ranges = matcher.ranges(line);
        if !ranges.is_empty() {
            total_matches += 1;
            if matches.len() < max_results {
                let first = ranges[0];
                let json_ranges = ranges
                    .iter()
                    .map(|(start, end)| {
                        json!({
                            "start": start,
                            "end": end,
                        })
                    })
                    .collect::<Vec<_>>();
                let match_text =
                    direct_match_text(line, first.0 as usize, first.1 as usize).to_string();
                matches.push(json!({
                    "path": display_path.clone(),
                    "line": (line_index + 1) as u64,
                    "column": first.0.saturating_add(1),
                    "byte_column": first.0,
                    "excerpt": truncate_line_for_ai(line, Some(ranges.as_slice()), MAX_LINE_LEN),
                    "match": match_text,
                    "ranges": json_ranges,
                    "is_definition": looks_like_definition_line(line),
                }));
            }
        }
    }

    let shown = matches.len();
    let files = if total_matches > 0 {
        vec![json!({
            "path": display_path.clone(),
            "count": total_matches,
            "size_bytes": metadata.len(),
            "is_binary": bytes.contains(&0),
            "git_status": null,
        })]
    } else {
        Vec::new()
    };

    ToolResult::ok(json!({
        "query": query,
        "query_used": grep_text,
        "broadened_from": null,
        "regex_fallback_error": matcher.regex_error(),
        "matches": matches,
        "files": files,
        "count": total_matches,
        "shown": shown,
        "files_with_matches": if total_matches > 0 { 1 } else { 0 },
        "truncated": total_matches > shown,
        "cursor": null,
        "suggested_path": if total_matches > 0 { Some(display_path) } else { None },
        "approximate": false,
        "timed_out": false,
        "cancelled": false,
        "error": null,
    }))
}

enum DirectMatcher {
    Literal {
        needle: String,
        case_insensitive: bool,
        regex_error: Option<String>,
    },
    Regex(regex::Regex),
}

impl DirectMatcher {
    fn new(pattern: &str) -> Result<Self, regex::Error> {
        if has_regex_metacharacters(pattern) {
            let case_insensitive = !pattern.chars().any(|ch| ch.is_uppercase());
            let regex = regex::RegexBuilder::new(pattern)
                .case_insensitive(case_insensitive)
                .build()?;
            Ok(Self::Regex(regex))
        } else {
            Ok(Self::Literal {
                needle: pattern.to_string(),
                case_insensitive: !pattern.chars().any(|ch| ch.is_uppercase()),
                regex_error: None,
            })
        }
    }

    fn literal_with_error(pattern: &str, error: regex::Error) -> Self {
        Self::Literal {
            needle: pattern.to_string(),
            case_insensitive: !pattern.chars().any(|ch| ch.is_uppercase()),
            regex_error: Some(error.to_string()),
        }
    }

    fn regex_error(&self) -> Option<&str> {
        match self {
            Self::Literal { regex_error, .. } => regex_error.as_deref(),
            Self::Regex(_) => None,
        }
    }

    fn ranges(&self, line: &str) -> Vec<(u32, u32)> {
        match self {
            Self::Literal {
                needle,
                case_insensitive,
                ..
            } => literal_ranges(line, needle, *case_insensitive),
            Self::Regex(regex) => regex
                .find_iter(line)
                .take(16)
                .map(|matched| (matched.start() as u32, matched.end() as u32))
                .collect(),
        }
    }
}

fn literal_ranges(line: &str, needle: &str, case_insensitive: bool) -> Vec<(u32, u32)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let haystack = if case_insensitive {
        line.to_ascii_lowercase()
    } else {
        line.to_string()
    };
    let needle = if case_insensitive {
        needle.to_ascii_lowercase()
    } else {
        needle.to_string()
    };
    let mut ranges = Vec::new();
    let mut offset = 0usize;
    while let Some(found) = haystack[offset..].find(&needle) {
        let start = offset + found;
        let end = start + needle.len();
        ranges.push((start as u32, end as u32));
        if ranges.len() >= 16 {
            break;
        }
        offset = end.max(start + 1);
    }
    ranges
}

fn display_path_for_direct_file(file_path: &Path, default_base: Option<&Path>) -> String {
    if let Some(base) = default_base
        && let Ok(relative) = file_path.strip_prefix(base)
    {
        return relative.to_string_lossy().to_string();
    }
    file_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| file_path.display().to_string())
}

fn direct_match_text(line: &str, start: usize, end: usize) -> &str {
    let start = floor_char_boundary(line, start);
    let end = ceil_char_boundary(line, end);
    &line[start..end]
}

fn looks_like_definition_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    [
        "fn ",
        "pub fn ",
        "async fn ",
        "def ",
        "class ",
        "struct ",
        "enum ",
        "trait ",
        "impl ",
        "function ",
    ]
    .iter()
    .any(|prefix| trimmed.starts_with(prefix))
}

fn parse_limit(args: &serde_json::Value) -> Result<usize, ToolResult> {
    match args.get("limit") {
        None => Ok(DEFAULT_MAX_RESULTS),
        Some(value) if value.is_null() => Ok(DEFAULT_MAX_RESULTS),
        Some(value) => {
            let parsed = value
                .as_u64()
                .map(|number| number as usize)
                .or_else(|| value.as_f64().map(|number| number as usize))
                .ok_or_else(|| {
                    ToolResult::err_fmt(format_args!("Invalid limit: expected number"))
                })?;
            if parsed == 0 {
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid limit: must be >= 1"
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

fn make_grep_options(
    mode: GrepMode,
    file_offset: usize,
    control: &GrepRunControl,
) -> (GrepSearchOptions, bool) {
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
            time_budget_ms: control.remaining_budget_ms(),
            before_context,
            after_context,
            classify_definitions: true,
            trim_whitespace: false,
            abort_signal: Some(Arc::clone(&control.abort_signal)),
        },
        auto_expand_defs,
    )
}

fn timeout_grep_result(query: &str, stage: &str, budget: Duration, message: &str) -> ToolResult {
    let raw = json!({
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
        "timed_out": true,
        "cancelled": false,
        "error": {
            "kind": "timeout",
            "message": message,
            "stage": stage,
            "budget_ms": budget.as_millis() as u64,
        },
    });
    let mut failure = lash_core::ToolFailure::safe_retry(
        ToolFailureClass::Timeout,
        "grep_timeout",
        message,
        Some(50),
    );
    failure.raw = Some(lash_core::ToolValue::from(raw));
    ToolResult::failure(failure)
}

fn cancelled_grep_result(query: &str) -> ToolResult {
    ToolResult::cancelled_with_raw(
        "grep cancelled",
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
            "timed_out": false,
            "cancelled": true,
            "error": {
                "kind": "cancelled",
                "message": "grep cancelled",
                "stage": "grep",
            },
        }),
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
        "timed_out": false,
        "cancelled": false,
        "error": null,
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
        "timed_out": false,
        "cancelled": false,
        "error": null,
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

    #[test]
    fn grep_uses_limit_argument_in_model_contract() {
        let definition = grep_tool_definition();
        let properties = definition
            .input_schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .expect("object properties");

        assert!(properties.contains_key("limit"));
        assert!(!properties.contains_key("maxResults"));
        assert_eq!(properties["limit"]["default"], serde_json::json!(20));
    }

    #[tokio::test]
    async fn test_grep_matches_with_query() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("test.txt"),
            "hello world\nfoo bar\nhello again\n",
        )
        .unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "hello"})).await;
        assert!(result.output.is_success());
        assert_eq!(result.output.value_for_projection()["count"], 2);
        assert_eq!(
            result.output.value_for_projection()["matches"][0]["path"],
            "test.txt"
        );
        assert_eq!(
            result.output.value_for_projection()["matches"][0]["excerpt"],
            "hello world"
        );
        assert_eq!(
            result.output.value_for_projection()["matches"][1]["excerpt"],
            "hello again"
        );
    }

    #[tokio::test]
    async fn test_grep_returns_structured_file_summaries() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "fn thing() {}\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "thing"})).await;
        assert!(result.output.is_success());
        assert_eq!(
            result.output.value_for_projection()["files"][0]["path"],
            "alpha.rs"
        );
        assert_eq!(result.output.value_for_projection()["files"][0]["count"], 1);
        assert_eq!(
            result.output.value_for_projection()["suggested_path"],
            "alpha.rs"
        );
    }

    #[tokio::test]
    async fn test_grep_structured_counts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\nctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "ctx"})).await;
        assert!(result.output.is_success());
        assert_eq!(result.output.value_for_projection()["count"], 2);
        assert_eq!(result.output.value_for_projection()["files"][0]["count"], 2);
    }

    #[tokio::test]
    async fn test_grep_empty_result_keeps_structured_metadata() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result =
            lash_core::testing::run_tool(&tool, "grep", &json!({"query": "missing"})).await;
        assert!(result.output.is_success());
        assert_eq!(
            result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert!(result.output.value_for_projection()["broadened_from"].is_null());
        assert!(result.output.value_for_projection()["regex_fallback_error"].is_null());
    }

    #[tokio::test]
    async fn test_grep_long_query_does_not_panic_in_fuzzy_fallback() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "short searchable content\n").unwrap();

        let query = "definitely missing ".repeat(20);
        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": query})).await;

        assert!(
            result.output.is_success(),
            "long query should not panic or fail: {:?}",
            result.output.value_for_projection()
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

        let result = lash_core::testing::run_tool(&tool, "grep", &json!({"query": "ctx"})).await;
        assert!(result.output.is_success());
        assert!(tool.backend.get().is_some());
    }

    #[tokio::test]
    async fn test_grep_path_scopes_search_to_subdirectory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("inner")).unwrap();
        std::fs::write(dir.path().join("outer.txt"), "banana at root\n").unwrap();
        std::fs::write(dir.path().join("inner/inner.txt"), "banana in inner\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "inner"}),
        )
        .await;
        assert!(result.output.is_success());
        assert!(
            result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "inner.txt"),
            "expected inner.txt match, got {:?}",
            result.output.value_for_projection()
        );
        assert!(
            !result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "outer.txt"),
            "path scope should exclude outer.txt, got {:?}",
            result.output.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_path_constrains_search_to_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "banana\n").unwrap();
        std::fs::write(dir.path().join("other.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "notes.txt"}),
        )
        .await;
        assert!(result.output.is_success());
        assert!(
            result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "notes.txt"),
            "expected notes.txt match, got {:?}",
            result.output.value_for_projection()
        );
        assert!(
            !result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "other.txt"),
            "file path should exclude other.txt"
        );
        assert!(
            tool.backend.get().is_none(),
            "single-file grep should bypass the indexed backend"
        );
        assert_eq!(result.output.value_for_projection()["timed_out"], false);
        assert_eq!(
            result.output.value_for_projection()["error"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn test_grep_file_path_uses_direct_scan_for_multiword_query() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("bottle.py"),
            "header cookie static_file abort redirect request response\nunrelated\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("other.py"),
            "header cookie static_file abort redirect request response\n",
        )
        .unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({
                "query": "header cookie static_file abort redirect request response",
                "path": "bottle.py",
                "limit": 80,
            }),
        )
        .await;

        assert!(
            result.output.is_success(),
            "direct grep failed: {:?}",
            result.output.value_for_projection()
        );
        assert_eq!(result.output.value_for_projection()["count"], 1);
        assert_eq!(result.output.value_for_projection()["shown"], 1);
        assert_eq!(
            result.output.value_for_projection()["matches"][0]["path"],
            "bottle.py"
        );
        assert_eq!(
            result.output.value_for_projection()["matches"][0]["match"],
            "header cookie static_file abort redirect request response"
        );
        assert!(
            tool.backend.get().is_none(),
            "single-file grep should not initialize fff"
        );
        assert_eq!(result.output.value_for_projection()["timed_out"], false);
        assert_eq!(
            result.output.value_for_projection()["error"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn test_grep_path_can_search_outside_workspace() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({
                "query": "banana",
                "path": outside.path().to_string_lossy(),
            }),
        )
        .await;
        assert!(
            result.output.is_success(),
            "expected search outside workspace to succeed, got {:?}",
            result.output.value_for_projection()
        );
        assert!(
            result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected external.txt match, got {:?}",
            result.output.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_infers_obvious_path_prefix_from_query() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("external.txt"), "banana\n").unwrap();

        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": format!("{} banana", outside.path().display())}),
        )
        .await;
        assert!(result.output.is_success());
        assert!(
            result.output.value_for_projection()["matches"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["path"] == "external.txt"),
            "expected inferred path search to find external.txt, got {:?}",
            result.output.value_for_projection()
        );
    }

    #[tokio::test]
    async fn test_grep_infers_obvious_file_prefix_without_indexing() {
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let file = outside.path().join("external.txt");
        std::fs::write(&file, "banana split\n").unwrap();

        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": format!("{} banana", file.display())}),
        )
        .await;
        assert!(result.output.is_success());
        assert_eq!(
            result.output.value_for_projection()["matches"][0]["path"],
            "external.txt"
        );
        assert!(
            tool.backend.get().is_none(),
            "inferred single-file grep should bypass fff"
        );
    }

    #[test]
    fn test_direct_file_grep_observes_pre_cancelled_abort_signal() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("notes.txt");
        std::fs::write(&file, "banana\n").unwrap();
        let abort = AtomicBool::new(true);

        let result = direct_file_grep_sync("banana", &file, Some(dir.path()), 20, &abort);

        assert!(!result.output.is_success());
        let value = result.output.value_for_projection();
        assert_eq!(value["raw"]["cancelled"], true);
        assert_eq!(value["message"], "grep cancelled");
        assert_eq!(value["source"], "cancellation");
        assert_eq!(value["raw"]["error"]["kind"], "cancelled");
    }

    #[tokio::test]
    async fn test_grep_path_missing_returns_clear_error() {
        let workspace = TempDir::new().unwrap();
        let tool = Grep::with_base_path(workspace.path().to_path_buf());
        let result = lash_core::testing::run_tool(
            &tool,
            "grep",
            &json!({"query": "banana", "path": "/nonexistent/totally/fake"}),
        )
        .await;
        assert!(!result.output.is_success());
        let value = result.output.value_for_projection();
        let message = value["message"].as_str().unwrap_or("");
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

        let left_backend = left.ensure_ready_for_query("ctx").expect("left backend");
        let right_backend = right.ensure_ready_for_query("ctx").expect("right backend");

        assert!(Arc::ptr_eq(&left_backend, &right_backend));
    }
}
