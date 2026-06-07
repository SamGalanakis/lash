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

#[derive(Clone, Debug)]
struct BoundVariableRenderCacheEntry {
    ordinal: usize,
    /// Cheap structural hash of the variable's value. When it matches, the
    /// expensive rebuild (serialize / shape inference / preview) is skipped and
    /// the cached `shape` + `line` are reused.
    value_hash: u64,
    shape: Option<JsonShape>,
    line: String,
}

/// The expensive-to-compute parts of a variable's rendering.
struct BuiltRow {
    inline: Option<String>,
    shape: Option<JsonShape>,
    size_hint: Option<String>,
    preview: Option<String>,
}

/// One variable resolved for this render pass — either reused from the cache
/// (`cached_line` set) or freshly built.
struct WorkRow {
    name: String,
    ordinal: usize,
    unchanged: bool,
    value_hash: u64,
    shape: Option<JsonShape>,
    inline: Option<String>,
    size_hint: Option<String>,
    preview: Option<String>,
    cached_line: Option<String>,
}

pub(crate) fn render_bound_variables(
    cache: &mut BoundVariableRenderCache,
    globals: &serde_json::Map<String, serde_json::Value>,
    history_len: usize,
    inline_char_limit: usize,
) -> PromptContribution {
    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
        "Small values are shown in full; larger ones show their type, size, and a truncated preview (record keys, or the head and tail of a list/string) — `print` such a variable (or the part you need) to see the rest.".to_string(),
    ];

    // Drop cache slots for variables that no longer exist.
    cache.entries.retain(|name, _| globals.contains_key(name));

    // Reuse the cached build for an unchanged variable (detected by a cheap
    // structural value hash); only rebuild changed/new ones. Rebuilding —
    // serialize, shape inference, preview — is the expensive part, so this is
    // the win for globals that are stable across prompt builds.
    let mut next_ordinal = cache.next_ordinal;
    let mut rows: Vec<WorkRow> = Vec::with_capacity(globals.len());
    for (name, value) in globals {
        let hash = value_hash(value);
        match cache.entries.get(name) {
            Some(entry) if entry.value_hash == hash => rows.push(WorkRow {
                name: name.clone(),
                ordinal: entry.ordinal,
                unchanged: true,
                value_hash: hash,
                shape: entry.shape.clone(),
                inline: None,
                size_hint: None,
                preview: None,
                cached_line: Some(entry.line.clone()),
            }),
            existing => {
                let ordinal = existing.map(|entry| entry.ordinal).unwrap_or_else(|| {
                    let ordinal = next_ordinal;
                    next_ordinal += 1;
                    ordinal
                });
                let built = build_bound_variable_row(value, inline_char_limit);
                rows.push(WorkRow {
                    name: name.clone(),
                    ordinal,
                    unchanged: false,
                    value_hash: hash,
                    shape: built.shape,
                    inline: built.inline,
                    size_hint: built.size_hint,
                    preview: built.preview,
                    cached_line: None,
                });
            }
        }
    }
    cache.next_ordinal = next_ordinal;

    // Stable order: unchanged rows first (so their schema names stay fixed and
    // the prompt prefix stays cacheable for the provider), then by first-seen
    // ordinal. This ordering is what makes reusing a cached line safe.
    rows.sort_by(|left, right| {
        right
            .unchanged
            .cmp(&left.unchanged)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
            .then_with(|| left.name.cmp(&right.name))
    });

    // Rebuild the schema registry from every row's shape in stable order so
    // names are deterministic and match those baked into cached lines.
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
        let line = render_row_line(row, &registry);
        if row.cached_line.is_none() {
            cache.entries.insert(
                row.name.clone(),
                BoundVariableRenderCacheEntry {
                    ordinal: row.ordinal,
                    value_hash: row.value_hash,
                    shape: row.shape.clone(),
                    line: line.clone(),
                },
            );
        }
        lines.push(line);
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

fn render_row_line(row: &WorkRow, registry: &SchemaRegistry) -> String {
    if let Some(cached) = &row.cached_line {
        return cached.clone();
    }
    if let Some(inline) = &row.inline {
        // Value shown explicitly; no type/size hint needed.
        return format!("- `{}` = {inline}", row.name);
    }
    let shape = row
        .shape
        .as_ref()
        .expect("hinted variable has an inferred shape");
    let type_text = render_shape_inline(shape, registry);
    let mut line = match &row.size_hint {
        Some(size_hint) => format!("- `{}`: `{type_text}`, {size_hint}", row.name),
        None => format!("- `{}`: `{type_text}`", row.name),
    };
    if let Some(preview) = &row.preview {
        line.push_str(&format!(" ≈ {preview}"));
    }
    line
}

fn build_bound_variable_row(value: &serde_json::Value, inline_char_limit: usize) -> BuiltRow {
    match render_inline_value(value, inline_char_limit) {
        Some(inline) => BuiltRow {
            inline: Some(inline),
            shape: None,
            size_hint: None,
            preview: None,
        },
        None => BuiltRow {
            inline: None,
            shape: Some(infer_json_shape(value)),
            size_hint: render_value_size_hint(value),
            preview: render_degraded_preview(value, inline_char_limit),
        },
    }
}

/// Cheap structural hash of a JSON value for change detection. Walks the value
/// but allocates nothing — unlike serializing it or inferring its shape, which
/// is exactly the work this lets us skip when the value is unchanged.
fn value_hash(value: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_json_value(value, &mut hasher);
    hasher.finish()
}

fn hash_json_value<H: Hasher>(value: &serde_json::Value, hasher: &mut H) {
    match value {
        serde_json::Value::Null => 0u8.hash(hasher),
        serde_json::Value::Bool(flag) => {
            1u8.hash(hasher);
            flag.hash(hasher);
        }
        serde_json::Value::Number(number) => {
            2u8.hash(hasher);
            if let Some(int) = number.as_i64() {
                0u8.hash(hasher);
                int.hash(hasher);
            } else if let Some(uint) = number.as_u64() {
                1u8.hash(hasher);
                uint.hash(hasher);
            } else {
                2u8.hash(hasher);
                number.as_f64().unwrap_or(0.0).to_bits().hash(hasher);
            }
        }
        serde_json::Value::String(text) => {
            3u8.hash(hasher);
            text.hash(hasher);
        }
        serde_json::Value::Array(items) => {
            4u8.hash(hasher);
            items.len().hash(hasher);
            for item in items {
                hash_json_value(item, hasher);
            }
        }
        serde_json::Value::Object(map) => {
            5u8.hash(hasher);
            map.len().hash(hasher);
            for (key, val) in map {
                key.hash(hasher);
                hash_json_value(val, hasher);
            }
        }
    }
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

/// A bounded, shape-aware preview of a value too large to render in full.
///
/// General degradation, top level only — nested values are flattened to clipped
/// JSON, never recursed into, so cost and complexity stay predictable:
/// - record  -> its keys (`{ a, b, …(+N more) }`)
/// - list    -> head and tail elements (`[a, b, …(+N more)…, y, z]`)
/// - string  -> head and tail (`"abc…xyz"`)
///
/// Returns `None` for scalars (which always render in full) and for a `0`
/// budget (preview disabled, type + size hint only).
fn render_degraded_preview(value: &serde_json::Value, budget: usize) -> Option<String> {
    if budget == 0 {
        return None;
    }
    match value {
        serde_json::Value::Object(map) => Some(preview_object_keys(map, budget)),
        serde_json::Value::Array(items) => Some(preview_array(items, budget)),
        serde_json::Value::String(text) => Some(preview_string(text, budget)),
        _ => None,
    }
}

/// `{ key_a, key_b, …(+N more) }` — keys only.
fn preview_object_keys(map: &serde_json::Map<String, serde_json::Value>, budget: usize) -> String {
    let total = map.len();
    let mut shown: Vec<String> = Vec::new();
    let mut used = 0usize;
    for (idx, key) in map.keys().enumerate() {
        let token = clip(key, ELEMENT_CLIP);
        let cost = token.chars().count() + 2; // ", "
        if used + cost > budget && !shown.is_empty() {
            return format!("{{ {}, …(+{} more) }}", shown.join(", "), total - idx);
        }
        used += cost;
        shown.push(token);
    }
    format!("{{ {} }}", shown.join(", "))
}

/// `[a, b, …(+N more)…, y, z]` — middle truncation keeping head and tail.
fn preview_array(items: &[serde_json::Value], budget: usize) -> String {
    let rendered: Vec<String> = items
        .iter()
        .map(|item| clip(&compact(item), ELEMENT_CLIP))
        .collect();
    let joined = format!("[{}]", rendered.join(", "));
    if joined.chars().count() <= budget {
        return joined;
    }
    let half = budget / 2;
    let take_fitting = |iter: &mut dyn Iterator<Item = &String>| -> Vec<String> {
        let mut out = Vec::new();
        let mut used = 0usize;
        for item in iter {
            let cost = item.chars().count() + 2;
            if used + cost > half && !out.is_empty() {
                break;
            }
            used += cost;
            out.push(item.clone());
        }
        out
    };
    let head = take_fitting(&mut rendered.iter());
    let mut tail = take_fitting(&mut rendered.iter().rev());
    tail.reverse();
    let shown = head.len() + tail.len();
    if shown >= rendered.len() {
        return format!("[{}]", rendered.join(", "));
    }
    format!(
        "[{}, …(+{} more)…, {}]",
        head.join(", "),
        rendered.len() - shown,
        tail.join(", ")
    )
}

/// `"head…tail"` — head and tail of a long string.
fn preview_string(text: &str, budget: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= budget {
        return format!("{text:?}");
    }
    let keep = (budget.saturating_sub(3) / 2).max(1);
    let head: String = chars[..keep].iter().collect();
    let tail: String = chars[chars.len() - keep..].iter().collect();
    format!("\"{head}…{tail}\"")
}

/// Per-element/key char cap inside a preview. Keeps a single nested value from
/// dominating the budget (the anti-hairiness rule: clip nested, never recurse).
const ELEMENT_CLIP: usize = 80;

fn compact(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

/// Clip a string to at most `max` chars, marking truncation with `…`.
fn clip(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        return text.to_string();
    }
    let head: String = chars[..max.saturating_sub(1)].iter().collect();
    format!("{head}…")
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
        // Budget 0 disables the preview: type + size hint only, no `≈`.
        assert!(!s.contains('≈'), "{s}");
    }

    #[test]
    fn large_record_degrades_to_keys_preview() {
        let rooms = serde_json::Value::Object(
            (0..30)
                .map(|i| (format!("room_{i:02}"), json!({ "exits": ["n", "s"] })))
                .collect(),
        );
        let g = globals(json!({ "map": rooms }));
        let mut cache = BoundVariableRenderCache::default();
        let s = render_bound_variables(&mut cache, &g, 0, 80)
            .content
            .to_string();
        assert!(s.contains("`map`:"), "{s}"); // type still shown
        assert!(s.contains("keys=30"), "{s}"); // size still shown
        assert!(s.contains("≈ {"), "{s}"); // preview present
        assert!(s.contains("room_00"), "{s}"); // some keys shown
        assert!(s.contains("more)"), "{s}"); // and the rest elided
    }

    #[test]
    fn large_list_degrades_to_head_and_tail() {
        let items: Vec<_> = (0..40).map(|i| json!(format!("note-{i:02}"))).collect();
        let g = globals(json!({ "notes": items }));
        let mut cache = BoundVariableRenderCache::default();
        let s = render_bound_variables(&mut cache, &g, 0, 80)
            .content
            .to_string();
        assert!(s.contains("len=40"), "{s}");
        assert!(s.contains("note-00"), "{s}"); // head retained
        assert!(s.contains("note-39"), "{s}"); // tail retained
        assert!(s.contains("more)…"), "{s}"); // middle elided
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
