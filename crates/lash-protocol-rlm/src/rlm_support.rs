use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use lash_core::PromptContribution;
use lash_core::PromptUsage;
use lash_core::plugin::PromptHookContext;
use lash_rlm_types::RlmTermination;

use crate::projection::{project_rlm_globals_from_events, rlm_history_projection};

pub(crate) fn decode_rlm_termination_options(
    options: &lash_core::ProtocolTurnOptions,
) -> Result<RlmTermination, String> {
    if options.is_empty() {
        return Ok(RlmTermination::default());
    }
    options
        .decode()
        .map_err(|err| format!("invalid RLM turn options: {err}"))
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
            "Past the frame switch threshold. End this block with `control.continue_as(...)` now; do not call `submit` or do more work after it. Pack only what the new frame needs into `task` + `seed`; carry only necessary live process handles."
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

/// Memoizes the rendered "Bound Variables" `PromptContribution`. The
/// cache hits when the defaults `Arc` identity is unchanged between LLM
/// iterations. Saves the JSON-shape inference + the
/// `format!()`/`String::push_str` walk per iteration.
#[derive(Default)]
pub struct BoundVariablesCache {
    inner: Mutex<Option<CachedBoundVariables>>,
}

struct CachedBoundVariables {
    globals: Arc<serde_json::Map<String, serde_json::Value>>,
    history_len: usize,
    rendered: PromptContribution,
}

impl BoundVariablesCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contributions(&self, ctx: &PromptHookContext) -> Vec<PromptContribution> {
        let globals = Arc::new(project_rlm_globals_from_events(ctx.state.active_events()));
        let history_len = rlm_history_projection(&ctx.state.chronological_projection()).len();
        if let Ok(guard) = self.inner.lock()
            && let Some(cached) = guard.as_ref()
            && cached.globals.as_ref() == globals.as_ref()
            && cached.history_len == history_len
        {
            return vec![cached.rendered.clone()];
        }
        let rendered = render_bound_variables(&globals, history_len);
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(CachedBoundVariables {
                globals,
                history_len,
                rendered: rendered.clone(),
            });
        }
        vec![rendered]
    }
}

fn render_bound_variables(
    globals: &Arc<serde_json::Map<String, serde_json::Value>>,
    history_len: usize,
) -> PromptContribution {
    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
        "Type and size hints help you plan before inspecting a value.".to_string(),
    ];
    let mut entries = globals.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));

    let mut registry = SchemaRegistry::default();
    let mut variable_types = Vec::new();
    for (name, value) in entries {
        let shape = infer_json_shape(value);
        registry.register_root(name, &shape);
        variable_types.push((name.as_str(), shape));
    }

    lines.push(String::new());
    lines.push("Available variables:".to_string());
    lines.push(format!(
        "- `history`: `list<HistoryItem>`, Readonly: true, projected binding, {history_len} entries"
    ));
    for (name, shape) in &variable_types {
        let value = globals
            .get(*name)
            .expect("bound variable should still exist while rendering prompt contribution");
        let type_text = render_shape_inline(shape, &registry);
        if let Some(size_hint) = render_value_size_hint(value) {
            lines.push(format!(
                "- `{name}`: `{type_text}`, Readonly: true, projected binding, {size_hint}"
            ));
        } else {
            lines.push(format!(
                "- `{name}`: `{type_text}`, Readonly: true, projected binding"
            ));
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

    PromptContribution::guidance("Bound Variables", lines.join("\n"))
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
