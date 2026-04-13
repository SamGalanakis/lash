use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use fff_search::git::format_git_status_opt;
use fff_search::grep::{GrepMode, GrepSearchOptions, has_regex_metacharacters, is_import_line};
use fff_search::{
    AiGrepConfig, FFFMode, FileItem, FilePicker, FilePickerOptions, FuzzySearchOptions, GrepMatch,
    PaginationArgs, QueryParser, SharedFrecency, SharedPicker,
};
use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::require_str;

const DEFAULT_MAX_RESULTS: usize = 20;
const MAX_CURSORS: usize = 20;
const LARGE_FILE_BYTES: u64 = 20_000;
const MAX_PREVIEW: usize = 120;
const MAX_LINE_LEN: usize = 180;
const MAX_DEF_EXPAND_FIRST: usize = 8;
const MAX_DEF_EXPAND: usize = 5;
const MAX_FIRST_MATCH_EXPAND: usize = 8;

/// Search file contents using an indexed fff-search backend.
pub struct Grep {
    base_path: Result<PathBuf, String>,
    backend: OnceLock<Result<GrepBackend, String>>,
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

    fn ensure_ready(&self) -> Result<&GrepBackend, ToolResult> {
        let backend = self
            .backend
            .get_or_init(|| self.initialize_backend())
            .as_ref()
            .map_err(|err| ToolResult::err_fmt(format_args!("{err}")))?;
        if !backend.picker.wait_for_scan(Duration::from_secs(30)) {
            return Err(ToolResult::err_fmt(format_args!(
                "fff-search initial scan timed out"
            )));
        }
        Ok(backend)
    }

    fn initialize_backend(&self) -> Result<GrepBackend, String> {
        let base_path = self.base_path.as_ref().map_err(Clone::clone)?;
        let picker = SharedPicker::default();
        FilePicker::new_with_shared_state(
            picker.clone(),
            SharedFrecency::default(),
            FilePickerOptions {
                base_path: base_path.to_string_lossy().into_owned(),
                // Keep the indexed backend lightweight until it is actually
                // used. Long-lived sessions should not pay for watcher or mmap
                // state just because the grep tool exists in the catalog.
                warmup_mmap_cache: false,
                mode: FFFMode::Ai,
                cache_budget: None,
                watch: false,
            },
        )
        .map_err(|err| format!("failed to initialize indexed grep backend: {err}"))?;
        Ok(GrepBackend { picker })
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
        output_mode: OutputMode,
    ) -> Result<String, ToolResult> {
        let file_offset = cursor_id
            .and_then(|id| self.cursor_store.lock().ok()?.get(id))
            .unwrap_or(0);

        let (options, auto_expand) = make_grep_options(output_mode, mode, file_offset);
        let show_context = options.before_context > 0;

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
                    let (retry_options, _) = make_grep_options(output_mode, retry_mode, 0);
                    let retry_result = picker.grep(&rest_parsed, &retry_options);

                    if !retry_result.matches.is_empty() && retry_result.matches.len() <= 10 {
                        let mut cursors = self.lock_cursors()?;
                        let text = GrepFormatter {
                            matches: &retry_result.matches,
                            files: &retry_result.files,
                            total_matched: retry_result.matches.len(),
                            next_file_offset: retry_result.next_file_offset,
                            regex_fallback_error: retry_result.regex_fallback_error.as_deref(),
                            output_mode,
                            max_results,
                            show_context,
                            auto_expand_defs: auto_expand,
                        }
                        .format(&mut cursors);
                        return Ok(format!(
                            "0 matches for '{query}'. Auto-broadened to '{rest_query}':\n{text}"
                        ));
                    }
                }
            }

            let fuzzy_query = cleanup_fuzzy_query(query);
            let (fuzzy_options, _) = make_grep_options(output_mode, GrepMode::Fuzzy, 0);
            let fuzzy_parsed = parser.parse(&fuzzy_query);
            let fuzzy_result = picker.grep(&fuzzy_parsed, &fuzzy_options);
            if !fuzzy_result.matches.is_empty() {
                let mut lines = vec![format!(
                    "0 exact matches. {} approximate:",
                    fuzzy_result.matches.len()
                )];
                let mut current_file = "";
                for matched in fuzzy_result.matches.iter().take(3) {
                    let file = fuzzy_result.files[matched.file_index];
                    if file.relative_path.as_str() != current_file {
                        current_file = file.relative_path.as_str();
                        lines.push(current_file.to_string());
                    }
                    lines.push(format!(
                        " {}: {}",
                        matched.line_number, matched.line_content
                    ));
                }
                return Ok(lines.join("\n"));
            }

            if query.contains('/') {
                let file_query = QueryParser::default().parse(query);
                let file_result = FilePicker::fuzzy_search(
                    picker.get_files(),
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
                        return Ok(format!(
                            "0 content matches. But there is a relevant file path: {}",
                            top.relative_path
                        ));
                    }
                }
            }

            return Ok("0 matches.".to_string());
        }

        if result.matches.is_empty() {
            return Ok("0 matches.".to_string());
        }

        let mut cursors = self.lock_cursors()?;
        Ok(GrepFormatter {
            matches: &result.matches,
            files: &result.files,
            total_matched: result.matches.len(),
            next_file_offset: result.next_file_offset,
            regex_fallback_error: result.regex_fallback_error.as_deref(),
            output_mode,
            max_results,
            show_context,
            auto_expand_defs: auto_expand,
        }
        .format(&mut cursors))
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
            description: "Search file contents. Search for bare identifiers (e.g. 'InProgressQuote', 'ActorAuth'), NOT code syntax or regex. Filter files with constraints (e.g. '*.rs query', 'src/ query'). Use filename, directory (ending with /) or glob expressions to prefilter. See server instructions for constraint syntax and core rules.".into(),
            params: vec![
                ToolParam {
                    name: "query".into(),
                    r#type: "str".into(),
                    description: "Search text or regex query with optional constraint prefixes. Matches within single lines only; use one specific term, not multiple words.".into(),
                    default_value: None,
                    required: true,
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
                ToolParam {
                    name: "output_mode".into(),
                    r#type: "str".into(),
                    description:
                        "Output format. Use 'content' (default), 'files_with_matches', 'count', or 'usage'."
                            .into(),
                    default_value: Some(json!("content")),
                    required: false,
                },
            ],
            returns: "str".into(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let query = match require_str(args, "query") {
            Ok(query) => query,
            Err(err) => return err,
        };
        let max_results = match parse_max_results(args) {
            Ok(max_results) => max_results,
            Err(err) => return err,
        };
        let cursor = args.get("cursor").and_then(|value| value.as_str());
        let output_mode = OutputMode::new(args.get("output_mode").and_then(|value| value.as_str()));

        let backend = match self.ensure_ready() {
            Ok(backend) => backend,
            Err(err) => return err,
        };

        let grep_text = QueryParser::new(AiGrepConfig).parse(query).grep_text();
        let mode = if has_regex_metacharacters(&grep_text) {
            GrepMode::Regex
        } else {
            GrepMode::PlainText
        };

        match self.perform_grep(backend, query, mode, max_results, cursor, output_mode) {
            Ok(text) => ToolResult::ok(json!(text)),
            Err(err) => err,
        }
    }
}

struct GrepBackend {
    picker: SharedPicker,
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
    let mut output = String::with_capacity(input.len());
    for ch in input.chars() {
        if !matches!(ch, ':' | '-' | '_') {
            output.extend(ch.to_lowercase());
        }
    }
    output
}

fn make_grep_options(
    output_mode: OutputMode,
    mode: GrepMode,
    file_offset: usize,
) -> (GrepSearchOptions, bool) {
    let is_usage = output_mode == OutputMode::Usage;
    let max_matches_per_file = match output_mode {
        OutputMode::FilesWithMatches => 1,
        _ if is_usage => 8,
        _ => 10,
    };
    let before_context = if is_usage { 1 } else { 0 };
    let auto_expand_defs = !is_usage && before_context == 0;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Content,
    FilesWithMatches,
    Count,
    Usage,
}

impl OutputMode {
    fn new(value: Option<&str>) -> Self {
        match value {
            Some("files_with_matches") => Self::FilesWithMatches,
            Some("count") => Self::Count,
            Some("usage") => Self::Usage,
            _ => Self::Content,
        }
    }
}

fn frecency_word(score: i32) -> Option<&'static str> {
    if score >= 100 {
        Some("hot")
    } else if score >= 50 {
        Some("warm")
    } else if score >= 10 {
        Some("frequent")
    } else {
        None
    }
}

fn file_suffix(file: &FileItem) -> String {
    match (
        frecency_word(file.total_frecency_score),
        format_git_status_opt(file.git_status),
    ) {
        (Some(frecency), Some(git)) => format!(" - {frecency} git:{git}"),
        (Some(frecency), None) => format!(" - {frecency}"),
        (None, Some(git)) => format!(" git:{git}"),
        (None, None) => String::new(),
    }
}

fn size_tag(bytes: u64) -> String {
    if bytes < LARGE_FILE_BYTES {
        String::new()
    } else {
        let kb = (bytes + 512) / 1024;
        format!(" ({}KB - use offset to read relevant section)", kb)
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

struct FileMeta<'a> {
    file: &'a FileItem,
    line_number: u64,
    line_content: String,
    is_definition: bool,
    match_ranges: Vec<(u32, u32)>,
    context_after: Vec<String>,
}

struct GrepFormatter<'a> {
    matches: &'a [GrepMatch],
    files: &'a [&'a FileItem],
    total_matched: usize,
    next_file_offset: usize,
    regex_fallback_error: Option<&'a str>,
    output_mode: OutputMode,
    max_results: usize,
    show_context: bool,
    auto_expand_defs: bool,
}

impl GrepFormatter<'_> {
    fn format(&self, cursor_store: &mut CursorStore) -> String {
        let items = if self.matches.len() > self.max_results {
            &self.matches[..self.max_results]
        } else {
            self.matches
        };

        match self.output_mode {
            OutputMode::FilesWithMatches => {
                return format_files_with_matches(
                    items,
                    self.files,
                    self.next_file_offset,
                    self.auto_expand_defs,
                    cursor_store,
                );
            }
            OutputMode::Count => {
                return format_count(items, self.files, self.next_file_offset, cursor_store);
            }
            OutputMode::Content | OutputMode::Usage => {}
        }

        let mut lines = Vec::new();
        let unique_files = items
            .iter()
            .map(|matched| matched.file_index)
            .collect::<HashSet<_>>()
            .len();
        let max_output_chars = if self.output_mode == OutputMode::Usage || unique_files <= 3 {
            5_000
        } else if unique_files <= 8 {
            3_500
        } else {
            2_500
        };

        if let Some(err) = self.regex_fallback_error {
            lines.push(format!("! regex failed: {err}, using literal match"));
        }

        let file_preview = collect_file_preview(items, self.files);
        let mut content_def_file = "";
        let mut content_first_file = "";
        for meta in &file_preview {
            if content_first_file.is_empty() {
                content_first_file = meta.file.relative_path.as_str();
            }
            if content_def_file.is_empty() && meta.is_definition {
                content_def_file = meta.file.relative_path.as_str();
            }
        }

        let content_suggest = if !content_def_file.is_empty() {
            content_def_file
        } else {
            content_first_file
        };
        if !content_suggest.is_empty() {
            let file_count = file_preview.len();
            if file_count == 1 {
                lines.push(format!("-> Read {content_suggest} (only match)"));
            } else if !content_def_file.is_empty() {
                lines.push(format!("-> Read {content_suggest} [def]"));
            } else if file_count <= 3 {
                lines.push(format!("-> Read {content_suggest} (best match)"));
            }
        }

        if self.total_matched > items.len() {
            lines.push(format!(
                "{}/{} matches shown",
                items.len(),
                self.total_matched
            ));
        }

        let mut expanded_definitions = HashSet::new();
        let mut char_count = 0usize;
        let mut shown_count = 0usize;
        let mut current_file = "";

        let mut sorted_indices = (0..items.len()).collect::<Vec<_>>();
        if self.auto_expand_defs {
            sorted_indices.sort_unstable_by_key(|&index| {
                if items[index].is_definition {
                    0
                } else if is_import_line(&items[index].line_content) {
                    2
                } else {
                    1
                }
            });
        }

        for index in sorted_indices {
            let matched = &items[index];
            let file = self.files[matched.file_index];
            let mut chunk_lines = Vec::new();

            if file.relative_path.as_str() != current_file {
                current_file = file.relative_path.as_str();
                chunk_lines.push(current_file.to_string());
            }

            if self.auto_expand_defs
                && is_import_line(&matched.line_content)
                && !expanded_definitions.is_empty()
            {
                continue;
            }

            if self.show_context && !matched.context_before.is_empty() {
                let start_line = matched
                    .line_number
                    .saturating_sub(matched.context_before.len() as u64);
                for (offset, context) in matched.context_before.iter().enumerate() {
                    chunk_lines.push(format!(
                        " {}-{}",
                        start_line + offset as u64,
                        truncate_line_for_ai(context, None, MAX_LINE_LEN)
                    ));
                }
            }

            chunk_lines.push(format!(
                " {}: {}",
                matched.line_number,
                truncate_line_for_ai(
                    &matched.line_content,
                    Some(matched.match_byte_offsets.as_ref()),
                    MAX_LINE_LEN
                )
            ));

            if self.show_context && !matched.context_after.is_empty() {
                let start_line = matched.line_number + 1;
                for (offset, context) in matched.context_after.iter().enumerate() {
                    chunk_lines.push(format!(
                        " {}-{}",
                        start_line + offset as u64,
                        truncate_line_for_ai(context, None, MAX_LINE_LEN)
                    ));
                }
                chunk_lines.push("--".to_string());
            }

            if self.auto_expand_defs
                && !self.show_context
                && matched.is_definition
                && !matched.context_after.is_empty()
                && !expanded_definitions.contains(file.relative_path.as_str())
            {
                let expand_limit = if expanded_definitions.is_empty() {
                    MAX_DEF_EXPAND_FIRST
                } else {
                    MAX_DEF_EXPAND
                };
                expanded_definitions.insert(file.relative_path.clone());
                let start_line = matched.line_number + 1;
                for (offset, context) in matched.context_after.iter().take(expand_limit).enumerate()
                {
                    if context.trim().is_empty() {
                        break;
                    }
                    chunk_lines.push(format!(
                        "  {}| {}",
                        start_line + offset as u64,
                        truncate_line_for_ai(context, None, MAX_LINE_LEN)
                    ));
                }
            }

            let chunk = chunk_lines.join("\n");
            if char_count + chunk.len() > max_output_chars && shown_count > 0 {
                break;
            }
            char_count += chunk.len();
            lines.push(chunk);
            shown_count += 1;
        }

        if self.next_file_offset > 0 {
            let cursor = cursor_store.store(self.next_file_offset);
            lines.push(format!("\ncursor: {cursor}"));
        }

        lines.join("\n")
    }
}

fn format_files_with_matches(
    items: &[GrepMatch],
    files: &[&FileItem],
    next_file_offset: usize,
    auto_expand_defs: bool,
    cursor_store: &mut CursorStore,
) -> String {
    let file_preview = collect_file_preview(items, files);
    let mut lines = Vec::new();
    let file_count = file_preview.len();

    let mut first_def_file = "";
    let mut first_file = "";
    for meta in &file_preview {
        if first_file.is_empty() {
            first_file = meta.file.relative_path.as_str();
        }
        if first_def_file.is_empty() && meta.is_definition {
            first_def_file = meta.file.relative_path.as_str();
        }
    }
    let suggested_path = if !first_def_file.is_empty() {
        first_def_file
    } else {
        first_file
    };

    if !suggested_path.is_empty() {
        if file_count == 1 {
            lines.push(format!(
                "-> Read {suggested_path} (only match - no need to search further)"
            ));
        } else if !first_def_file.is_empty() && file_count <= 5 {
            lines.push(format!("-> Read {suggested_path} (definition found)"));
        } else if !first_def_file.is_empty() {
            lines.push(format!("-> Read {suggested_path} (definition)"));
        } else if file_count <= 3 {
            lines.push(format!("-> Read {suggested_path} (best match)"));
        } else {
            lines.push(format!("-> Read {suggested_path}"));
        }
    }

    let is_small_set = file_count <= 5;
    let mut expanded_definitions = 0usize;
    for (index, meta) in file_preview.iter().enumerate() {
        let def_tag = if meta.is_definition { " [def]" } else { "" };
        lines.push(format!(
            "{}{}{}{}",
            meta.file.relative_path.as_str(),
            def_tag,
            size_tag(meta.file.size),
            file_suffix(meta.file)
        ));

        if !meta.line_content.is_empty() && (meta.is_definition || index == 0 || is_small_set) {
            let ranges = if meta.match_ranges.is_empty() {
                None
            } else {
                Some(meta.match_ranges.as_slice())
            };
            lines.push(format!(
                "  {}: {}",
                meta.line_number,
                truncate_line_for_ai(&meta.line_content, ranges, MAX_PREVIEW)
            ));

            if auto_expand_defs && !meta.context_after.is_empty() {
                let expand_limit = if meta.is_definition {
                    let limit = if expanded_definitions == 0 {
                        MAX_DEF_EXPAND_FIRST
                    } else {
                        MAX_DEF_EXPAND
                    };
                    expanded_definitions += 1;
                    limit
                } else if is_small_set && index == 0 {
                    MAX_FIRST_MATCH_EXPAND
                } else if is_small_set {
                    MAX_DEF_EXPAND
                } else {
                    0
                };

                if expand_limit > 0 {
                    let start_line = meta.line_number + 1;
                    for (offset, context) in
                        meta.context_after.iter().take(expand_limit).enumerate()
                    {
                        if context.trim().is_empty() {
                            break;
                        }
                        lines.push(format!(
                            "  {}| {}",
                            start_line + offset as u64,
                            truncate_line_for_ai(context, None, MAX_PREVIEW)
                        ));
                    }
                }
            }
        }
    }

    if next_file_offset > 0 {
        let cursor = cursor_store.store(next_file_offset);
        lines.push(format!("\ncursor: {cursor}"));
    }

    lines.join("\n")
}

fn format_count(
    items: &[GrepMatch],
    files: &[&FileItem],
    next_file_offset: usize,
    cursor_store: &mut CursorStore,
) -> String {
    let mut counts = HashMap::new();
    let mut order = Vec::new();
    for matched in items {
        let path = files[matched.file_index].relative_path.as_str();
        let count = counts.entry(path).or_insert_with(|| {
            order.push(path);
            0usize
        });
        *count += 1;
    }

    let mut lines = order
        .into_iter()
        .map(|path| format!("{path}: {}", counts[path]))
        .collect::<Vec<_>>();
    if next_file_offset > 0 {
        let cursor = cursor_store.store(next_file_offset);
        lines.push(format!("\ncursor: {cursor}"));
    }
    lines.join("\n")
}

fn collect_file_preview<'a>(items: &[GrepMatch], files: &[&'a FileItem]) -> Vec<FileMeta<'a>> {
    let mut preview = Vec::new();
    let mut seen = HashSet::new();
    for matched in items {
        let file = files[matched.file_index];
        if seen.insert(file.relative_path.clone()) {
            preview.push(FileMeta {
                file,
                line_number: matched.line_number,
                line_content: matched.line_content.clone(),
                is_definition: matched.is_definition,
                match_ranges: matched.match_byte_offsets.iter().copied().collect(),
                context_after: matched.context_after.clone(),
            });
        }
    }
    preview
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
        let text = result.result.as_str().unwrap_or("");
        assert!(text.contains("test.txt"));
        assert!(text.contains("hello world"));
        assert!(text.contains("hello again"));
    }

    #[tokio::test]
    async fn test_grep_files_with_matches_output_mode() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "fn thing() {}\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool
            .execute(
                "grep",
                &json!({"query": "thing", "output_mode": "files_with_matches"}),
            )
            .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap_or("");
        assert!(text.contains("alpha.rs"));
        assert!(text.contains("Read"));
    }

    #[tokio::test]
    async fn test_grep_count_output_mode() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.rs"), "ctx\nctx\n").unwrap();

        let tool = Grep::with_base_path(dir.path().to_path_buf());
        let result = tool
            .execute("grep", &json!({"query": "ctx", "output_mode": "count"}))
            .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap_or("");
        assert!(text.contains("alpha.rs: 2"));
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
}
