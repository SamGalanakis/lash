use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use lash_core::PromptContribution;
use lash_core::PromptUsage;
use lash_rlm_types::{RlmCreateExtras, RlmTermination};

pub(crate) fn decode_rlm_options(
    options: &lash_core::ProtocolTurnOptions,
) -> Result<RlmCreateExtras, String> {
    if options.is_empty() {
        return Ok(RlmCreateExtras::default());
    }
    options
        .decode()
        .map_err(|err| format!("invalid RLM turn options: {err}"))
}

pub(crate) fn decode_rlm_termination_options(
    options: &lash_core::ProtocolTurnOptions,
) -> Result<RlmTermination, String> {
    decode_rlm_options(options).map(|options| options.termination)
}

/// Render the "Context Budget" line for the volatile turn-tail message.
/// Returns the formatted text (status line + optional escalation tail)
/// or `None` when there's nothing to say. The string is intentionally
/// per-turn dynamic; callers must place it AFTER the cache breakpoint,
/// never inside the cached system prompt.
pub fn format_budget_suffix(
    turn_index: usize,
    usage: Option<&PromptUsage>,
    max_budget_tokens: Option<usize>,
) -> Option<String> {
    let max = max_budget_tokens?;
    let usage = usage?;
    let used = usage.context_budget_tokens;
    if used == 0 {
        return None;
    }
    let pct = used.saturating_mul(100) / max.max(1);
    let mut content =
        format!("Turn: {turn_index} · Tokens: {used} · frame switch threshold: {max} ({pct}%).");
    if pct >= 60 {
        let tail = if used >= max {
            "Past the frame switch threshold. End this block with `control.continue_as(...)` now; do not call `submit` or do more work after it. Pack only what the new frame needs into `task` + `seed`."
        } else if pct >= 90 {
            "Budget tight — finish only the current step, then end the block with `control.continue_as(...)`."
        } else {
            "Look for a clean frame switch point; when you switch, make `control.continue_as(...)` the terminal action in the block."
        };
        content.push('\n');
        content.push_str(tail);
    }
    Some(content)
}

/// Render the "Bound Variables" prompt section from the live execution
/// namespace: the model's own scratch variables and any seeded computed
/// globals, shown the same way (value inline when small, type + size hint when
/// large). Read-only values are excluded by the caller and render in their own
/// section without value previews.
#[derive(Debug, Default)]
pub(crate) struct BoundVariableRenderCache {
    next_ordinal: usize,
    entries: BTreeMap<String, BoundVariableRenderCacheEntry>,
}

impl BoundVariableRenderCache {
    fn order_rows(&mut self, mut rows: Vec<BoundVariableRow>) -> Vec<BoundVariableRow> {
        let live_names = rows
            .iter()
            .map(|row| row.name.clone())
            .collect::<BTreeSet<_>>();
        self.entries.retain(|name, _| live_names.contains(name));

        for row in &mut rows {
            row.fingerprint = row_fingerprint(row);
            if let Some(entry) = self.entries.get(&row.name) {
                row.ordinal = entry.ordinal;
                row.unchanged = entry.fingerprint == row.fingerprint;
            } else {
                row.ordinal = self.next_ordinal;
                self.next_ordinal += 1;
                row.unchanged = false;
            }
        }

        for row in &rows {
            self.entries.insert(
                row.name.clone(),
                BoundVariableRenderCacheEntry {
                    ordinal: row.ordinal,
                    fingerprint: row.fingerprint,
                },
            );
        }

        rows.sort_by(|left, right| {
            right
                .unchanged
                .cmp(&left.unchanged)
                .then_with(|| left.ordinal.cmp(&right.ordinal))
                .then_with(|| left.name.cmp(&right.name))
        });
        rows
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundVariableRenderCacheEntry {
    ordinal: usize,
    fingerprint: u64,
}

#[derive(Clone, Debug)]
struct BoundVariableRow {
    name: String,
    inline: Option<String>,
    shape: Option<JsonShape>,
    size_hint: Option<String>,
    ordinal: usize,
    fingerprint: u64,
    unchanged: bool,
}

pub(crate) fn render_bound_variables(
    cache: &mut BoundVariableRenderCache,
    globals: &serde_json::Map<String, serde_json::Value>,
    history_len: usize,
    inline_char_limit: usize,
) -> PromptContribution {
    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
        "Small values are shown inline so you can read them directly; larger ones show only their type and size — `print` such a variable (or the part you need) to see its contents.".to_string(),
    ];
    // Decide per variable: render its value inline when small, otherwise fall
    // back to a type + size hint. Only register a schema definition for the
    // hinted (large) ones: inline values already show their structure.
    let mut rows = globals
        .iter()
        .map(|(name, value)| build_bound_variable_row(name, value, inline_char_limit))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.name.cmp(&right.name));
    let rows = cache.order_rows(rows);

    let mut registry = SchemaRegistry::default();
    for row in &rows {
        if let Some(shape) = &row.shape {
            registry.register_root(&row.name, shape);
        }
    }

    lines.push(String::new());
    lines.push("Available variables:".to_string());
    lines.push("- `history`: `list<HistoryItem>`, read-only".to_string());
    for row in &rows {
        if let Some(inline) = &row.inline {
            // Value shown explicitly; no type/size hint needed.
            lines.push(format!("- `{}` = {inline}", row.name));
            continue;
        }
        let shape = row
            .shape
            .as_ref()
            .expect("hinted variable has an inferred shape");
        let type_text = render_shape_inline(shape, &registry);
        if let Some(size_hint) = &row.size_hint {
            lines.push(format!("- `{}`: `{type_text}`, {size_hint}", row.name));
        } else {
            lines.push(format!("- `{}`: `{type_text}`", row.name));
        }
    }

    if !registry.definitions.is_empty() {
        lines.push(String::new());
        lines.push("Schema:".to_string());
        lines.push("```text".to_string());
        for (idx, (name, shape)) in registry.definitions.iter().enumerate() {
            if idx > 0 {
                lines.push(String::new());
            }
            lines.extend(render_type_definition(name, shape, &registry));
        }
        lines.push("```".to_string());
    }

    lines.push(String::new());
    lines.push("Runtime notes:".to_string());
    lines.push(format!(
        "- `history` currently has {history_len} {}",
        history_count_unit(history_len)
    ));

    PromptContribution::guidance("Bound Variables", lines.join("\n"))
}

fn build_bound_variable_row(
    name: &str,
    value: &serde_json::Value,
    inline_char_limit: usize,
) -> BoundVariableRow {
    match render_inline_value(value, inline_char_limit) {
        Some(inline) => BoundVariableRow {
            name: name.to_string(),
            inline: Some(inline),
            shape: None,
            size_hint: None,
            ordinal: 0,
            fingerprint: 0,
            unchanged: false,
        },
        None => BoundVariableRow {
            name: name.to_string(),
            inline: None,
            shape: Some(infer_json_shape(value)),
            size_hint: render_value_size_hint(value),
            ordinal: 0,
            fingerprint: 0,
            unchanged: false,
        },
    }
}

fn row_fingerprint(row: &BoundVariableRow) -> u64 {
    let mut hasher = DefaultHasher::new();
    row.name.hash(&mut hasher);
    match (&row.inline, &row.shape) {
        (Some(inline), _) => {
            "inline".hash(&mut hasher);
            inline.hash(&mut hasher);
        }
        (None, Some(shape)) => {
            "hint".hash(&mut hasher);
            shape.hash(&mut hasher);
            row.size_hint.hash(&mut hasher);
        }
        (None, None) => {
            "empty".hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn history_count_unit(count: usize) -> &'static str {
    if count == 1 { "entry" } else { "entries" }
}

/// Render a variable's value inline as compact JSON when it fits within
/// `limit` characters, so the model can read it directly without a `print`.
/// Returns `None` (use a type/size hint instead) for `0` limits or values too
/// large to embed each turn.
fn render_inline_value(value: &serde_json::Value, limit: usize) -> Option<String> {
    if limit == 0 {
        return None;
    }
    let compact = serde_json::to_string(value).ok()?;
    (compact.chars().count() <= limit).then_some(compact)
}

fn render_value_size_hint(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(_) => None,
        serde_json::Value::Number(_) => None,
        serde_json::Value::String(text) => {
            let lines = text.lines().count();
            if lines > 1 {
                Some(format!("len={}, lines={lines}", text.chars().count()))
            } else {
                Some(format!("len={}", text.chars().count()))
            }
        }
        serde_json::Value::Array(values) => Some(format!("len={}", values.len())),
        serde_json::Value::Object(map) => Some(format!("keys={}", map.len())),
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum JsonShape {
    Any,
    Null,
    Bool,
    Int,
    Float,
    Str,
    List(Box<JsonShape>),
    Record(BTreeMap<String, JsonShape>),
    Union(Vec<JsonShape>),
}

#[derive(Default)]
struct SchemaRegistry {
    names_by_key: BTreeMap<String, String>,
    definitions: Vec<(String, JsonShape)>,
    used_names: BTreeSet<String>,
}

impl SchemaRegistry {
    fn register_root(&mut self, root_name: &str, shape: &JsonShape) {
        self.register_shape(shape, &[root_name.to_string()]);
    }

    fn register_shape(&mut self, shape: &JsonShape, hint_segments: &[String]) {
        match shape {
            JsonShape::Record(fields) => {
                let key = canonical_shape_key(shape);
                if self.names_by_key.contains_key(&key) {
                    return;
                }
                for (field, child) in fields {
                    let child_segments = vec![singularize_segment(field)];
                    self.register_nested_shape(child, &child_segments);
                }
                let name = self.allocate_name(type_name_from_segments(hint_segments));
                self.names_by_key.insert(key, name.clone());
                self.definitions.push((name, shape.clone()));
            }
            JsonShape::List(item) => self.register_nested_list_item(item, hint_segments),
            JsonShape::Union(items) => {
                for item in items {
                    self.register_shape(item, hint_segments);
                }
            }
            JsonShape::Any
            | JsonShape::Null
            | JsonShape::Bool
            | JsonShape::Int
            | JsonShape::Float
            | JsonShape::Str => {}
        }
    }

    fn register_nested_shape(&mut self, shape: &JsonShape, hint_segments: &[String]) {
        match shape {
            JsonShape::Record(_) => self.register_shape(shape, hint_segments),
            JsonShape::List(item) => self.register_nested_list_item(item, hint_segments),
            JsonShape::Union(items) => {
                for item in items {
                    self.register_nested_shape(item, hint_segments);
                }
            }
            JsonShape::Any
            | JsonShape::Null
            | JsonShape::Bool
            | JsonShape::Int
            | JsonShape::Float
            | JsonShape::Str => {}
        }
    }

    fn register_nested_list_item(&mut self, item: &JsonShape, hint_segments: &[String]) {
        match item {
            JsonShape::Record(_) => {
                let mut item_segments = hint_segments.to_vec();
                item_segments.push("item".to_string());
                self.register_shape(item, &item_segments);
            }
            JsonShape::List(inner) => self.register_nested_list_item(inner, hint_segments),
            JsonShape::Union(items) => {
                for item in items {
                    self.register_nested_list_item(item, hint_segments);
                }
            }
            JsonShape::Any
            | JsonShape::Null
            | JsonShape::Bool
            | JsonShape::Int
            | JsonShape::Float
            | JsonShape::Str => {}
        }
    }

    fn allocate_name(&mut self, base: String) -> String {
        let base = if base.is_empty() {
            "Value".to_string()
        } else {
            base
        };
        if self.used_names.insert(base.clone()) {
            return base;
        }
        let mut suffix = 2usize;
        loop {
            let candidate = format!("{base}{suffix}");
            if self.used_names.insert(candidate.clone()) {
                return candidate;
            }
            suffix += 1;
        }
    }
}

fn infer_json_shape(value: &serde_json::Value) -> JsonShape {
    match value {
        serde_json::Value::Null => JsonShape::Null,
        serde_json::Value::Bool(_) => JsonShape::Bool,
        serde_json::Value::Number(n) => {
            if n.is_f64() && n.as_f64().is_some_and(|v| v.fract() != 0.0) {
                JsonShape::Float
            } else {
                JsonShape::Int
            }
        }
        serde_json::Value::String(_) => JsonShape::Str,
        serde_json::Value::Array(values) => {
            let item_shape = values
                .iter()
                .map(infer_json_shape)
                .reduce(merge_shapes)
                .unwrap_or(JsonShape::Any);
            JsonShape::List(Box::new(item_shape))
        }
        serde_json::Value::Object(map) => JsonShape::Record(
            map.iter()
                .map(|(key, value)| (key.clone(), infer_json_shape(value)))
                .collect(),
        ),
    }
}

fn merge_shapes(left: JsonShape, right: JsonShape) -> JsonShape {
    use JsonShape as Shape;
    match (left, right) {
        (Shape::Any, _) | (_, Shape::Any) => Shape::Any,
        (left, right) if left == right => left,
        (Shape::Int, Shape::Float) | (Shape::Float, Shape::Int) => Shape::Float,
        (Shape::List(left), Shape::List(right)) => {
            Shape::List(Box::new(merge_shapes(*left, *right)))
        }
        (Shape::Record(left), Shape::Record(right)) if left.keys().eq(right.keys()) => {
            let merged = left
                .into_iter()
                .map(|(key, left_shape)| {
                    let right_shape = right.get(&key).cloned().unwrap_or(JsonShape::Any);
                    (key, merge_shapes(left_shape, right_shape))
                })
                .collect();
            Shape::Record(merged)
        }
        (Shape::Union(left), Shape::Union(right)) => {
            flatten_union(left.into_iter().chain(right).collect())
        }
        (Shape::Union(mut union), other) | (other, Shape::Union(mut union)) => {
            union.push(other);
            flatten_union(union)
        }
        (left, right) => flatten_union(vec![left, right]),
    }
}

fn flatten_union(shapes: Vec<JsonShape>) -> JsonShape {
    let mut flattened = Vec::new();
    for shape in shapes {
        match shape {
            JsonShape::Union(items) => flattened.extend(items),
            other => flattened.push(other),
        }
    }
    let mut by_key = BTreeMap::new();
    for shape in flattened {
        by_key.insert(canonical_shape_key(&shape), shape);
    }
    let deduped = by_key.into_values().collect::<Vec<_>>();
    if deduped.len() == 1 {
        deduped.into_iter().next().unwrap_or(JsonShape::Any)
    } else {
        JsonShape::Union(deduped)
    }
}

fn canonical_shape_key(shape: &JsonShape) -> String {
    match shape {
        JsonShape::Any => "any".to_string(),
        JsonShape::Null => "null".to_string(),
        JsonShape::Bool => "bool".to_string(),
        JsonShape::Int => "int".to_string(),
        JsonShape::Float => "float".to_string(),
        JsonShape::Str => "str".to_string(),
        JsonShape::List(item) => format!("list[{}]", canonical_shape_key(item)),
        JsonShape::Record(fields) => {
            let body = fields
                .iter()
                .map(|(field, shape)| format!("{field}:{}", canonical_shape_key(shape)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        JsonShape::Union(items) => {
            let mut parts = items.iter().map(canonical_shape_key).collect::<Vec<_>>();
            parts.sort();
            format!("union({})", parts.join("|"))
        }
    }
}

fn render_shape_inline(shape: &JsonShape, registry: &SchemaRegistry) -> String {
    match shape {
        JsonShape::Any => "any".to_string(),
        JsonShape::Null => "null".to_string(),
        JsonShape::Bool => "bool".to_string(),
        JsonShape::Int => "int".to_string(),
        JsonShape::Float => "float".to_string(),
        JsonShape::Str => "str".to_string(),
        JsonShape::List(item) => format!("list[{}]", render_shape_inline(item, registry)),
        JsonShape::Record(_) => registry
            .names_by_key
            .get(&canonical_shape_key(shape))
            .cloned()
            .unwrap_or_else(|| "record".to_string()),
        JsonShape::Union(items) => items
            .iter()
            .map(|item| render_shape_inline(item, registry))
            .collect::<Vec<_>>()
            .join(" | "),
    }
}

fn render_type_definition(name: &str, shape: &JsonShape, registry: &SchemaRegistry) -> Vec<String> {
    match shape {
        JsonShape::Record(fields) => {
            let mut lines = vec![format!("type {name} = {{")];
            for (field, shape) in fields {
                lines.push(format!(
                    "  {field}: {},",
                    render_shape_inline(shape, registry)
                ));
            }
            lines.push("}".to_string());
            lines
        }
        _ => vec![format!(
            "type {name} = {}",
            render_shape_inline(shape, registry)
        )],
    }
}

fn type_name_from_segments(segments: &[String]) -> String {
    let joined = segments
        .iter()
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.trim_matches('_'))
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    let mut out = String::new();
    for part in joined
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .collect::<String>()
        })
        .filter(|part| !part.is_empty())
    {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.push(first.to_ascii_uppercase());
            for ch in chars {
                out.push(ch.to_ascii_lowercase());
            }
        }
    }
    out
}

fn singularize_segment(segment: &str) -> String {
    if let Some(prefix) = segment.strip_suffix("ies") {
        return format!("{prefix}y");
    }
    if segment.len() > 1 && segment.ends_with('s') && !segment.ends_with("ss") {
        return segment[..segment.len() - 1].to_string();
    }
    segment.to_string()
}

#[cfg(test)]
mod bound_variable_tests {
    use super::*;
    use serde_json::json;

    fn globals(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().expect("object").clone()
    }

    fn render_with_cache(
        cache: &mut BoundVariableRenderCache,
        value: serde_json::Value,
        history_len: usize,
        inline_char_limit: usize,
    ) -> String {
        render_bound_variables(cache, &globals(value), history_len, inline_char_limit)
            .content
            .to_string()
    }

    #[test]
    fn small_values_render_inline_without_type_or_size() {
        let g = globals(json!({ "inventory": ["lantern", "sword"], "count": 3 }));
        let mut cache = BoundVariableRenderCache::default();
        let rendered = render_bound_variables(&mut cache, &g, 0, 1024);
        let s = &rendered.content;
        assert!(s.contains("- `inventory` = [\"lantern\",\"sword\"]"), "{s}");
        assert!(s.contains("- `count` = 3"), "{s}");
        // Inline values carry no redundant type/size hint.
        assert!(!s.contains("`inventory`:"), "{s}");
        assert!(!s.contains("len="), "{s}");
    }

    #[test]
    fn large_values_fall_back_to_type_and_size_hint() {
        let big: Vec<String> = (0..500).map(|i| format!("item-{i}")).collect();
        let g = globals(json!({ "big": big }));
        let mut cache = BoundVariableRenderCache::default();
        let rendered = render_bound_variables(&mut cache, &g, 0, 64);
        let s = &rendered.content;
        assert!(s.contains("- `big`:"), "{s}");
        assert!(s.contains("len=500"), "{s}");
    }

    #[test]
    fn zero_limit_always_hints() {
        let g = globals(json!({ "x": [1, 2, 3] }));
        let mut cache = BoundVariableRenderCache::default();
        let rendered = render_bound_variables(&mut cache, &g, 0, 0);
        let s = &rendered.content;
        assert!(s.contains("- `x`:"), "{s}");
        assert!(s.contains("len=3"), "{s}");
    }

    #[test]
    fn history_len_renders_in_tail_runtime_notes() {
        let mut cache = BoundVariableRenderCache::default();
        let s = render_with_cache(&mut cache, json!({ "task": "ship" }), 7, 1024);

        assert!(
            s.contains("- `history`: `list<HistoryItem>`, read-only"),
            "{s}"
        );
        assert!(
            !s.contains("- `history`: `list<HistoryItem>`, read-only, 7 entries"),
            "{s}"
        );

        let task_idx = s.find("- `task` = \"ship\"").expect("task row");
        let notes_idx = s.find("Runtime notes:").expect("runtime notes");
        let history_len_idx = s
            .find("- `history` currently has 7 entries")
            .expect("history len note");
        assert!(task_idx < notes_idx, "{s}");
        assert!(notes_idx < history_len_idx, "{s}");
    }

    #[test]
    fn unchanged_rows_stay_before_changed_rows() {
        let mut cache = BoundVariableRenderCache::default();
        let _ = render_with_cache(
            &mut cache,
            json!({ "a": "old", "b": "steady", "c": "steady" }),
            0,
            1024,
        );

        let s = render_with_cache(
            &mut cache,
            json!({ "a": "new", "b": "steady", "c": "steady" }),
            0,
            1024,
        );

        let b_idx = s.find("- `b` = \"steady\"").expect("b row");
        let c_idx = s.find("- `c` = \"steady\"").expect("c row");
        let a_idx = s.find("- `a` = \"new\"").expect("a row");
        assert!(b_idx < c_idx, "{s}");
        assert!(c_idx < a_idx, "{s}");
    }

    #[test]
    fn new_rows_append_after_unchanged_rows() {
        let mut cache = BoundVariableRenderCache::default();
        let _ = render_with_cache(&mut cache, json!({ "b": 1 }), 0, 1024);

        let s = render_with_cache(&mut cache, json!({ "a": 2, "b": 1 }), 0, 1024);

        let b_idx = s.find("- `b` = 1").expect("b row");
        let a_idx = s.find("- `a` = 2").expect("a row");
        assert!(b_idx < a_idx, "{s}");
    }

    #[test]
    fn new_rows_do_not_rename_unchanged_row_schema() {
        let mut cache = BoundVariableRenderCache::default();
        let _ = render_with_cache(&mut cache, json!({ "z": { "id": 1 } }), 0, 0);

        let s = render_with_cache(
            &mut cache,
            json!({ "a": { "id": 2 }, "z": { "id": 1 } }),
            0,
            0,
        );

        let z_idx = s.find("- `z`: `Z`, keys=1").expect("z row");
        let a_idx = s.find("- `a`: `Z`, keys=1").expect("a row");
        assert!(z_idx < a_idx, "{s}");
    }

    #[test]
    fn summarized_shape_or_count_changes_update_row_hash() {
        let mut cache = BoundVariableRenderCache::default();
        let _ = render_with_cache(
            &mut cache,
            json!({ "big": [1, 2, 3], "steady": "same" }),
            0,
            0,
        );

        let s = render_with_cache(
            &mut cache,
            json!({ "big": [1, 2, 3, 4], "steady": "same" }),
            0,
            0,
        );

        let steady_idx = s.find("- `steady`: `str`").expect("steady row");
        let big_idx = s.find("- `big`: `list[int]`, len=4").expect("big row");
        assert!(steady_idx < big_idx, "{s}");
    }

    #[test]
    fn removed_rows_do_not_keep_stale_order_slots() {
        let mut cache = BoundVariableRenderCache::default();
        let _ = render_with_cache(&mut cache, json!({ "a": 1, "b": 2 }), 0, 1024);
        let _ = render_with_cache(&mut cache, json!({ "b": 2 }), 0, 1024);

        let s = render_with_cache(&mut cache, json!({ "a": 1, "b": 2 }), 0, 1024);

        let b_idx = s.find("- `b` = 2").expect("b row");
        let a_idx = s.find("- `a` = 1").expect("a row");
        assert!(b_idx < a_idx, "{s}");
    }
}
