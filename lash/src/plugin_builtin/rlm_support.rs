#[cfg(feature = "sqlite-store")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::sync::Arc;

use serde_json::json;

use crate::plugin::{PromptHookContext, ToolSurfaceContribution, ToolSurfaceOverride};
#[cfg(feature = "sqlite-store")]
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs};
use crate::tools::run_blocking;
use crate::{
    PromptContext, PromptContribution, ToolDefinition, ToolExecutionContext, ToolParam,
    ToolProvider, ToolResult,
};

#[derive(Clone)]
pub(super) struct SearchToolsProvider;

impl SearchToolsProvider {
    pub fn new() -> Self {
        Self
    }

    #[cfg(feature = "sqlite-store")]
    fn search_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let injected_only = args.get("injected_only").and_then(|v| v.as_bool());

        let mut filtered = Vec::new();
        for tool in &catalog {
            if !tool
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true)
            {
                continue;
            }
            if let Some(injected) = injected_only {
                let inject = tool
                    .get("injected")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if injected != inject {
                    continue;
                }
            }
            filtered.push(tool.clone());
        }

        if browse_all {
            filtered.sort_by(|left, right| {
                let left_name = left
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let right_name = right
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                left_name.cmp(right_name)
            });
            return ToolResult::ok(json!(filtered.into_iter().take(limit).collect::<Vec<_>>()));
        }

        let docs: Vec<SearchDoc> = filtered
            .iter()
            .map(|tool| {
                let examples = tool
                    .get("examples")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let mut fields = HashMap::new();
                fields.insert(
                    "name",
                    tool.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "description",
                    tool.get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert("examples", examples);
                SearchDoc { fields }
            })
            .collect();
        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[("name", 4.0), ("description", 2.0), ("examples", 1.0)],
        );
        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, _)| {
                let mut tool = filtered[idx].clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.insert("score".to_string(), json!(score));
                }
                tool
            })
            .collect();
        ToolResult::ok(json!(items))
    }

    #[cfg(not(feature = "sqlite-store"))]
    fn search_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        let mut filtered = catalog
            .into_iter()
            .filter(|tool| {
                tool.get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
            })
            .filter(|tool| {
                if query.is_empty() {
                    return true;
                }
                let name = tool
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let description = tool
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                name.contains(&query) || description.contains(&query)
            })
            .collect::<Vec<_>>();

        filtered.sort_by(|left, right| {
            let left_name = left
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let right_name = right
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            left_name.cmp(right_name)
        });

        ToolResult::ok(json!(filtered))
    }
}

impl Default for SearchToolsProvider {
    fn default() -> Self {
        Self::new()
    }
}

pub(super) fn tool_discovery_prompt_contributions(
    context: &PromptContext,
) -> Vec<PromptContribution> {
    if context.omitted_tool_count == 0 {
        return Vec::new();
    }

    vec![PromptContribution::guidance(
        "tool_discovery",
        "Tool Discovery",
        "Use `search_tools` to inspect the additional available tools that are omitted from Available Tools for brevity. With no query, it browses the full active tool catalog; use focused queries when you know the kind of tool you need.",
    )]
}

pub(super) fn bound_variables_prompt_contributions(
    ctx: &PromptHookContext,
) -> Vec<PromptContribution> {
    let globals = ctx.state.projected_rlm_globals();
    if globals.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
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
    lines.push("Variables:".to_string());
    for (name, shape) in &variable_types {
        lines.push(format!(
            "- `{name}`: `{}`",
            render_shape_inline(shape, &registry)
        ));
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

    vec![PromptContribution::guidance(
        "bound_variables",
        "Bound Variables",
        lines.join("\n"),
    )]
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

pub(super) fn tool_surface_contribution(
    ctx: &crate::plugin::ToolSurfaceContext,
) -> ToolSurfaceContribution {
    let omitted_tool_count = ctx
        .tools
        .iter()
        .filter(|tool| tool.enabled)
        .filter(|tool| tool.name != "search_tools")
        .filter(|tool| !tool.injected)
        .count();
    let has_search_tools = ctx.tools.iter().any(|tool| tool.name == "search_tools");
    let mut overrides = Vec::new();
    if has_search_tools {
        overrides.push(ToolSurfaceOverride {
            tool_name: "search_tools".to_string(),
            enabled: Some(omitted_tool_count > 0),
            injected: Some(omitted_tool_count > 0),
        });
    }
    let tool_list_notes = if omitted_tool_count > 0 {
        vec![format!(
            "- **Note:** {omitted_tool_count} additional tool(s) are available but omitted from this prompt for brevity."
        )]
    } else {
        Vec::new()
    };
    ToolSurfaceContribution {
        overrides,
        tool_list_notes,
    }
}

#[async_trait::async_trait]
impl ToolProvider for SearchToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search_tools".into(),
            description: "Discover available tools. With a focused `query`, returns ranked matches using hybrid/literal/regex search. With no `query`, returns the full active tool catalog in stable name order.".into(),
            params: vec![
                ToolParam::optional("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("injected_only", "bool"),
            ],
            returns: "list".into(),
            examples: vec![
                "call search_tools { query: \"task planning\" }".into(),
                "call search_tools {}".into(),
            ],
            enabled: true,
            injected: false,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("Unknown tool: {name}"))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "search_tools" => match context.host.tool_catalog(&context.session_id).await {
                Ok(catalog) => {
                    let this = self.clone();
                    let args = args.clone();
                    run_blocking(move || this.search_tools(&args, catalog)).await
                }
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            _ => self.execute(name, args).await,
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::MockSessionManager;
    use crate::{ExecutionMode, ToolDefinition};

    #[tokio::test]
    async fn search_tools_lists_all_without_query() {
        let store = SearchToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default().with_tool_catalog(vec![
                json!({"name":"read_file","description":"Read files","enabled":true,"injected":true,"examples":[]}),
                json!({"name":"apply_patch","description":"Apply patches","enabled":true,"injected":true,"examples":[]}),
            ])),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = store
            .execute_with_context("search_tools", &json!({}), &context)
            .await;

        assert!(result.success);
        let items = result.result.as_array().expect("array");
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].get("name").and_then(|v| v.as_str()),
            Some("apply_patch")
        );
        assert_eq!(
            items[1].get("name").and_then(|v| v.as_str()),
            Some("read_file")
        );
    }

    #[tokio::test]
    async fn search_tools_streaming_execution_preserves_session_context() {
        let store = SearchToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default().with_tool_catalog(vec![
                json!({"name":"ask","description":"Pause and ask the user a targeted question.","enabled":true,"injected":true,"examples":[]}),
            ])),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = store
            .execute_streaming_with_context("search_tools", &json!({"query":"ask"}), &context, None)
            .await;

        assert!(result.success);
        let items = result.result.as_array().expect("array");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].get("name").and_then(|v| v.as_str()), Some("ask"));
    }

    #[test]
    fn prompt_contributions_only_describe_tool_discovery_when_needed() {
        let mut prompt = PromptContext {
            omitted_tool_count: 1,
            ..PromptContext::default()
        };
        let with_omissions = tool_discovery_prompt_contributions(&prompt);
        assert_eq!(with_omissions.len(), 1);
        assert!(with_omissions[0].content.contains("search_tools"));

        prompt.omitted_tool_count = 0;
        assert!(tool_discovery_prompt_contributions(&prompt).is_empty());
    }

    #[test]
    fn prompt_contributions_include_bound_rlm_variables_from_graph_nodes() {
        let mut snapshot = crate::SessionStateEnvelope::default();
        snapshot.policy.execution_mode = crate::ExecutionMode::Rlm;
        snapshot.session_graph.append_plugin(
            crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE,
            serde_json::to_value(crate::RlmGlobalsPatchPluginBody {
                set: serde_json::Map::from_iter([
                    (
                        "input".to_string(),
                        json!({
                            "haystack_sessions": [[{"role":"user","content":"hello"}]],
                            "haystack_dates": ["2023/04/02 (Sun) 01:09"],
                            "question": "When did I graduate?",
                        }),
                    ),
                    ("limit".to_string(), json!(200)),
                ]),
                unset: Vec::new(),
            })
            .expect("patch json"),
        );
        let ctx = crate::PromptHookContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default()),
            prompt: PromptContext::default(),
            state: crate::SessionReadView::new(snapshot),
        };

        let contributions = bound_variables_prompt_contributions(&ctx);
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].title.as_deref(), Some("Bound Variables"));
        assert!(contributions[0].content.contains("Variables:"));
        assert!(contributions[0].content.contains("- `input`: `Input`"));
        assert!(contributions[0].content.contains("- `limit`: `int`"));
        assert!(contributions[0].content.contains("type Input = {"));
        assert!(
            contributions[0]
                .content
                .contains("haystack_sessions: list[list[HaystackSessionItem]]")
        );
        assert!(
            contributions[0]
                .content
                .contains("type HaystackSessionItem = {")
        );
        assert!(contributions[0].content.contains("role: str"));
        assert!(contributions[0].content.contains("content: str"));
    }

    #[test]
    fn rlm_tool_surface_hides_search_tools_when_nothing_is_omitted() {
        let host = crate::PluginHost::new(vec![]);
        let session = host
            .build_session(
                "root",
                ExecutionMode::Rlm,
                crate::ContextApproach::default(),
                None,
            )
            .expect("session");

        let surface = session
            .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Rlm,
                tools: vec![
                    ToolDefinition {
                        name: "search_tools".to_string(),
                        description: "Discover tools".to_string(),
                        params: vec![],
                        returns: "list".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                        input_schema_override: None,
                        output_schema_override: None,
                    },
                    ToolDefinition {
                        name: "read_file".to_string(),
                        description: "Read files".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                        input_schema_override: None,
                        output_schema_override: None,
                    },
                ],
            })
            .expect("tool surface");

        assert!(
            surface
                .tools
                .iter()
                .any(|tool| tool.name == "search_tools" && !tool.enabled && !tool.injected)
        );
    }
}
