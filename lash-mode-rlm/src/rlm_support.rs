use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use lash::PromptContribution;
use lash::plugin::{ModeSessionContext, PromptHookContext};
use lash::{SessionAppendNode, SessionError, SessionEventRecord};
use lash_rlm_types::RlmModeEvent;

pub async fn restore_execution_state_and_globals(
    ctx: &mut ModeSessionContext<'_>,
    state: &lash::runtime::PersistedSessionState,
) -> Result<(), SessionError> {
    if let Some(snapshot) = state.execution_state_snapshot().map(|bytes| bytes.to_vec()) {
        ctx.restore_execution_state(&snapshot).await?;
    }
    let read_view = state.read_view();
    apply_globals_patch_events(ctx, read_view.active_events()).await
}

pub async fn apply_globals_patch_nodes(
    ctx: &mut ModeSessionContext<'_>,
    nodes: &[SessionAppendNode],
) -> Result<(), SessionError> {
    for node in nodes {
        if let SessionAppendNode::Event {
            event: SessionEventRecord::Mode(event),
        } = node
            && let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = event.rlm_event()
        {
            ctx.apply_mode_globals_patch(&patch).await?;
        }
    }
    Ok(())
}

async fn apply_globals_patch_events<'a>(
    ctx: &mut ModeSessionContext<'_>,
    events: impl IntoIterator<Item = &'a SessionEventRecord>,
) -> Result<(), SessionError> {
    for event in events {
        if let SessionEventRecord::Mode(event) = event
            && let Some(RlmModeEvent::RlmGlobalsPatch(patch)) = event.rlm_event()
        {
            ctx.apply_mode_globals_patch(&patch).await?;
        }
    }
    Ok(())
}

/// Render the "Context Budget" prompt section. Returns an empty vec when
/// no budget is configured or the most recent prompt has no usable token
/// accounting yet.
pub fn budget_prompt_contributions(
    ctx: &PromptHookContext,
    max_budget_tokens: Option<usize>,
) -> Vec<PromptContribution> {
    let Some(max) = max_budget_tokens else {
        return Vec::new();
    };
    let Some(usage) = ctx.state.last_prompt_usage() else {
        return Vec::new();
    };
    let used = usage.context_budget_tokens;
    if used == 0 {
        return Vec::new();
    }
    let pct = used.saturating_mul(100) / max.max(1);
    if pct < 60 {
        return Vec::new();
    }
    let tail = if used >= max {
        "Past the handoff threshold. Do not continue ordinary work — `continue_as` now and pack only what the successor needs into `task` + `seed`."
    } else if pct >= 90 {
        "Budget tight — finish the current step, then `continue_as`."
    } else {
        "Look for a clean handoff point and `continue_as` rather than starting new work."
    };
    let content = format!("Tokens: {used} · handoff threshold: {max} ({pct}%).\n{tail}");
    vec![PromptContribution::execution("Context Budget", content)]
}

pub fn bound_variables_prompt_contributions(ctx: &PromptHookContext) -> Vec<PromptContribution> {
    let globals = ctx.state.shared_rlm_globals();
    if globals.is_empty() {
        return Vec::new();
    }
    vec![render_bound_variables(&globals)]
}

/// Memoizes the rendered "Bound Variables" `PromptContribution`. The
/// cache hits when the projected-globals `Arc` identity is unchanged
/// between LLM iterations — common during a single turn since the
/// `SessionGraphCache` only mints a new globals `Arc` when a
/// `RlmGlobalsPatch` is applied. Saves the JSON-shape inference + the
/// `format!()`/`String::push_str` walk per iteration.
#[derive(Default)]
pub struct BoundVariablesCache {
    inner: Mutex<Option<CachedBoundVariables>>,
}

struct CachedBoundVariables {
    globals: Arc<serde_json::Map<String, serde_json::Value>>,
    rendered: PromptContribution,
}

impl BoundVariablesCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contributions(&self, ctx: &PromptHookContext) -> Vec<PromptContribution> {
        let globals = ctx.state.shared_rlm_globals();
        if globals.is_empty() {
            return Vec::new();
        }
        if let Ok(guard) = self.inner.lock()
            && let Some(cached) = guard.as_ref()
            && Arc::ptr_eq(&cached.globals, &globals)
        {
            return vec![cached.rendered.clone()];
        }
        let rendered = render_bound_variables(&globals);
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(CachedBoundVariables {
                globals,
                rendered: rendered.clone(),
            });
        }
        vec![rendered]
    }
}

fn render_bound_variables(
    globals: &Arc<serde_json::Map<String, serde_json::Value>>,
) -> PromptContribution {
    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
        "The type and size hints below are there to help you choose the right strategy before inspecting a value.".to_string(),
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
    for (name, shape) in &variable_types {
        let value = globals
            .get(*name)
            .expect("bound variable should still exist while rendering prompt contribution");
        let type_text = render_shape_inline(shape, &registry);
        if let Some(size_hint) = render_value_size_hint(value) {
            lines.push(format!("- `{name}`: `{type_text}`, {size_hint}"));
        } else {
            lines.push(format!("- `{name}`: `{type_text}`"));
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
