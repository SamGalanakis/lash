use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use lash::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolSurfaceContext,
};
use lash::{
    ToolActivation, ToolAvailability, ToolAvailabilityConfig, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolParam, ToolProvider, ToolResult, ToolSurfaceContribution,
    ToolSurfaceOverride,
};
use serde_json::{Value, json};

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 100;
const FUZZY_SCORE_CAP: f64 = 1.25;

#[derive(Clone, Debug)]
struct CatalogTool {
    raw: Value,
    name: String,
    namespace: Option<String>,
    aliases: Vec<String>,
    callable: bool,
    documented: bool,
    discoverable: bool,
    loadable: bool,
}

#[derive(Clone, Debug)]
struct FieldIndex {
    name: &'static str,
    tokens: Vec<String>,
    raw: String,
    weight: f64,
    fuzzy: bool,
}

#[derive(Clone, Debug)]
struct DiscoveryDoc {
    tool: CatalogTool,
    fields: Vec<FieldIndex>,
}

#[derive(Clone, Debug)]
pub struct ToolDiscoveryIndex {
    key: u64,
    docs: Vec<DiscoveryDoc>,
    avg_len: f64,
    doc_freq: HashMap<String, usize>,
}

#[derive(Clone, Default)]
struct IndexCache {
    index: Option<Arc<ToolDiscoveryIndex>>,
}

#[derive(Clone, Default)]
pub struct ToolDiscoveryToolsProvider {
    cache: Arc<RwLock<IndexCache>>,
}

impl ToolDiscoveryToolsProvider {
    pub fn new() -> Self {
        Self::default()
    }

    fn index_for_catalog(&self, catalog: Vec<Value>) -> Arc<ToolDiscoveryIndex> {
        let key = catalog_key(&catalog);
        if let Some(index) = self
            .cache
            .read()
            .expect("tool discovery cache lock poisoned")
            .index
            .as_ref()
            .filter(|index| index.key == key)
            .cloned()
        {
            return index;
        }

        let index = Arc::new(ToolDiscoveryIndex::build(key, catalog));
        self.cache
            .write()
            .expect("tool discovery cache lock poisoned")
            .index = Some(Arc::clone(&index));
        index
    }

    fn search_tools(&self, args: &Value, catalog: Vec<Value>) -> ToolResult {
        let index = self.index_for_catalog(catalog);
        ToolResult::ok(json!(index.search(args)))
    }

    fn requested_names(args: &Value) -> Result<Vec<String>, ToolResult> {
        let Some(raw) = args.get("names") else {
            return Ok(Vec::new());
        };
        match raw {
            Value::Null => Ok(Vec::new()),
            Value::String(name) => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![trimmed.to_string()])
                }
            }
            Value::Array(values) => values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                        .ok_or_else(|| {
                            ToolResult::err_fmt("load_tools.names must contain non-empty strings")
                        })
                })
                .collect(),
            _ => Err(ToolResult::err_fmt(
                "load_tools.names must be a string or list of strings",
            )),
        }
    }

    async fn load_tools(&self, args: &Value, context: &ToolExecutionContext) -> ToolResult {
        let catalog = match context.host.tool_catalog(&context.session_id).await {
            Ok(catalog) => catalog,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let requested_names = match Self::requested_names(args) {
            Ok(names) => names,
            Err(err) => return err,
        };
        if requested_names.is_empty() {
            return ToolResult::err_fmt("load_tools requires non-empty `names`");
        }

        let by_name = catalog
            .into_iter()
            .filter_map(CatalogTool::from_value)
            .map(|tool| (tool.name.clone(), tool))
            .collect::<BTreeMap<_, _>>();

        let mut loaded = Vec::new();
        let mut already_callable = Vec::new();
        let mut already_documented = Vec::new();
        let mut not_loadable = Vec::new();
        let mut unknown = Vec::new();
        let mut to_promote = Vec::new();

        for name in requested_names {
            let Some(tool) = by_name.get(&name) else {
                unknown.push(name);
                continue;
            };
            if tool.callable {
                already_callable.push(name);
            } else if tool.documented {
                already_documented.push(name);
            } else if tool.loadable {
                to_promote.push(name.clone());
                loaded.push(name);
            } else {
                not_loadable.push(name);
            }
        }

        if !to_promote.is_empty()
            && let Err(err) = context
                .host
                .set_tools_availability(
                    &context.session_id,
                    &to_promote,
                    Some(ToolAvailability::Documented),
                )
                .await
        {
            return ToolResult::err_fmt(format_args!("failed to load tools: {err}"));
        }

        ToolResult::ok(json!({
            "loaded": loaded,
            "already_callable": already_callable,
            "already_documented": already_documented,
            "not_loadable": not_loadable,
            "unknown": unknown,
        }))
    }
}

impl ToolDiscoveryIndex {
    fn build(key: u64, catalog: Vec<Value>) -> Self {
        let docs: Vec<DiscoveryDoc> = catalog
            .into_iter()
            .filter_map(CatalogTool::from_value)
            .filter(|tool| tool.discoverable)
            .map(DiscoveryDoc::from_tool)
            .collect();
        let avg_len = if docs.is_empty() {
            1.0
        } else {
            (docs.iter().map(DiscoveryDoc::weighted_len).sum::<f64>() / docs.len() as f64).max(1.0)
        };
        let mut doc_freq = HashMap::new();
        for doc in &docs {
            let mut seen = BTreeSet::new();
            for token in doc.fields.iter().flat_map(|field| field.tokens.iter()) {
                seen.insert(token.clone());
            }
            for token in seen {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }
        Self {
            key,
            docs,
            avg_len,
            doc_freq,
        }
    }

    fn search(&self, args: &Value) -> Vec<Value> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let limit = limit_from_args(args);
        let query_tokens = tokenize(query);

        let mut ranked = Vec::new();
        for (idx, doc) in self.docs.iter().enumerate() {
            if !doc.matches_filters(args) {
                continue;
            }
            let score = bm25_score(&query_tokens, doc, self) + fuzzy_score(&query_tokens, doc);
            let matched_fields = matched_fields(&query_tokens, doc);
            let include = if query.is_empty() {
                true
            } else {
                !matched_fields.is_empty()
            };
            if include {
                ranked.push((idx, score, matched_fields));
            }
        }

        if query.is_empty() {
            ranked.sort_by(|(left, _, _), (right, _, _)| {
                self.docs[*left].tool.name.cmp(&self.docs[*right].tool.name)
            });
        } else {
            ranked.sort_by(|(left_idx, left_score, _), (right_idx, right_score, _)| {
                right_score
                    .partial_cmp(left_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.docs[*left_idx]
                            .tool
                            .name
                            .cmp(&self.docs[*right_idx].tool.name)
                    })
            });
        }

        ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, matched_fields)| self.docs[idx].tool.project(score, matched_fields))
            .collect()
    }
}

impl CatalogTool {
    fn from_value(raw: Value) -> Option<Self> {
        let obj = raw.as_object()?;
        let name = obj.get("name")?.as_str()?.to_string();
        let namespace = obj
            .get("namespace")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let aliases = string_vec(obj.get("aliases"));
        let callable = obj
            .get("callable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let documented = obj
            .get("documented")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let discoverable = obj
            .get("discoverable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let loadable = obj
            .get("loadable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Some(Self {
            raw,
            name,
            namespace,
            aliases,
            callable,
            documented,
            discoverable,
            loadable,
        })
    }

    fn project(&self, score: f64, matched_fields: Vec<String>) -> Value {
        let mut out = serde_json::Map::new();
        out.insert("name".to_string(), json!(self.name));
        if let Some(namespace) = &self.namespace {
            out.insert("namespace".to_string(), json!(namespace));
        }
        out.insert(
            "description".to_string(),
            json!(string_field(&self.raw, "description")),
        );
        out.insert(
            "params".to_string(),
            self.raw.get("params").cloned().unwrap_or_else(|| json!([])),
        );
        out.insert(
            "returns".to_string(),
            json!(string_field(&self.raw, "returns")),
        );
        out.insert(
            "examples".to_string(),
            self.raw
                .get("examples")
                .cloned()
                .unwrap_or_else(|| json!([])),
        );
        let activation_hint = string_field(&self.raw, "activation_hint");
        if !activation_hint.is_empty() {
            out.insert("activation_hint".to_string(), json!(activation_hint));
        }
        out.insert("score".to_string(), json!(score));
        out.insert("matched_fields".to_string(), json!(matched_fields));
        Value::Object(out)
    }
}

impl DiscoveryDoc {
    fn from_tool(tool: CatalogTool) -> Self {
        let mut fields = Vec::new();
        push_field(&mut fields, "name", vec![tool.name.clone()], 6.0, true);
        push_field(
            &mut fields,
            "namespace",
            tool.namespace.iter().cloned().collect(),
            3.0,
            true,
        );
        push_field(&mut fields, "aliases", tool.aliases.clone(), 5.0, true);
        push_field(
            &mut fields,
            "description",
            vec![string_field(&tool.raw, "description")],
            2.0,
            false,
        );
        push_field(
            &mut fields,
            "params",
            vec![json_field(&tool.raw, "params")],
            1.4,
            false,
        );
        push_field(
            &mut fields,
            "returns",
            vec![string_field(&tool.raw, "returns")],
            0.8,
            false,
        );
        push_field(
            &mut fields,
            "examples",
            string_vec(tool.raw.get("examples")),
            1.2,
            false,
        );

        Self { tool, fields }
    }

    fn matches_filters(&self, args: &Value) -> bool {
        let namespaces = namespace_filter(args.get("namespace"));
        if !namespaces.is_empty() {
            return self.tool.namespace.as_deref().is_some_and(|namespace| {
                namespaces.iter().any(|candidate| candidate == namespace)
            });
        }
        true
    }

    fn weighted_len(&self) -> f64 {
        self.fields
            .iter()
            .map(|field| field.tokens.len() as f64 * field.weight)
            .sum::<f64>()
            .max(1.0)
    }
}

fn push_field(
    fields: &mut Vec<FieldIndex>,
    name: &'static str,
    values: Vec<String>,
    weight: f64,
    fuzzy: bool,
) {
    let raw = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let tokens = tokenize(&raw);
    fields.push(FieldIndex {
        name,
        tokens,
        raw,
        weight,
        fuzzy,
    });
}

fn bm25_score(query_tokens: &[String], doc: &DiscoveryDoc, index: &ToolDiscoveryIndex) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    let doc_len = doc.weighted_len();
    let k1 = 1.5;
    let b = 0.75;

    for query in query_tokens {
        let doc_freq = index.doc_freq.get(query).copied().unwrap_or(0) as f64;
        if doc_freq <= 0.0 {
            continue;
        }
        let idf = ((index.docs.len() as f64 - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
        let freq = doc
            .fields
            .iter()
            .map(|field| {
                field.tokens.iter().filter(|token| *token == query).count() as f64 * field.weight
            })
            .sum::<f64>();
        if freq <= 0.0 {
            continue;
        }
        let denom = freq + k1 * (1.0 - b + b * doc_len / index.avg_len);
        score += idf * (freq * (k1 + 1.0)) / denom;
    }
    score
}

fn fuzzy_score(query_tokens: &[String], doc: &DiscoveryDoc) -> f64 {
    let mut score = 0.0;
    for query in query_tokens.iter().filter(|token| token.len() >= 3) {
        let mut best = 0.0;
        for field in doc.fields.iter().filter(|field| field.fuzzy) {
            for token in field.tokens.iter().filter(|token| token.len() >= 3) {
                if token == query || token.contains(query) || query.contains(token) {
                    best = f64::max(best, 0.35 * field.weight);
                    continue;
                }
                let similarity = strsim::jaro_winkler(query, token);
                if similarity >= 0.88 {
                    best = f64::max(best, (similarity - 0.84) * field.weight);
                }
            }
        }
        score += best;
    }
    score.min(FUZZY_SCORE_CAP)
}

fn matched_fields(query_tokens: &[String], doc: &DiscoveryDoc) -> Vec<String> {
    let mut hits = BTreeSet::new();
    for field in &doc.fields {
        if field.raw.is_empty() {
            continue;
        }
        let hit = !query_tokens.is_empty()
            && query_tokens.iter().any(|query| {
                field.tokens.iter().any(|token| {
                    token == query
                        || token.contains(query)
                        || query.contains(token)
                        || (query.len() >= 3
                            && field.fuzzy
                            && strsim::jaro_winkler(query, token) >= 0.88)
                })
            });
        if hit {
            hits.insert(field.name.to_string());
        }
    }
    hits.into_iter().collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn limit_from_args(args: &Value) -> usize {
    args.get("limit")
        .and_then(Value::as_i64)
        .and_then(|value| usize::try_from(value).ok())
        .map(|value| value.clamp(1, MAX_LIMIT))
        .unwrap_or(DEFAULT_LIMIT)
}

fn namespace_filter(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::String(namespace)) => namespace
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn json_field(value: &Value, key: &str) -> String {
    value.get(key).map(Value::to_string).unwrap_or_default()
}

fn string_vec(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        Some(Value::String(value)) => vec![value.trim().to_string()],
        _ => Vec::new(),
    }
}

fn catalog_key(catalog: &[Value]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    catalog.len().hash(&mut hasher);
    for value in catalog {
        value.to_string().hash(&mut hasher);
    }
    hasher.finish()
}

fn search_tools_definition() -> ToolDefinition {
    ToolDefinition {
        name: "search_tools".into(),
        description: "Search catalogued tool names, namespaces, aliases, descriptions, parameters, and examples. Use this when the tool you need is not showcased in the prompt.".into(),
        params: vec![
            ToolParam::typed("query", "str"),
            ToolParam::optional("namespace", "str | list[str]"),
            ToolParam::optional("limit", "int"),
        ],
        returns: "list".into(),
        examples: vec![
            "search_tools(query=\"read files\")".into(),
            "search_tools(query=\"songs\", namespace=\"appworld\")".into(),
        ],
        availability: ToolAvailabilityConfig::documented(),
        activation: ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        discovery: lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["tool_search".to_string()],
        },
        execution_mode: ToolExecutionMode::Parallel,
    }
}

fn load_tools_definition() -> ToolDefinition {
    ToolDefinition {
        name: "load_tools".into(),
        description: "Promote loadable tools into the documented surface. Callable omitted tools do not need loading and can be called directly.".into(),
        params: vec![ToolParam::optional("names", "list")],
        returns: "dict".into(),
        examples: vec!["load_tools(names=[\"search_web\", \"fetch_url\"])".into()],
        availability: ToolAvailabilityConfig::documented(),
        activation: ToolActivation::Always,
        availability_override: None,
        input_schema_override: None,
        output_schema_override: None,
        discovery: lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["activate_tools".to_string()],
        },
        execution_mode: ToolExecutionMode::Serial,
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolDiscoveryToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![search_tools_definition(), load_tools_definition()]
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("Unknown tool: {name}"))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "search_tools" => match context.host.tool_catalog(&context.session_id).await {
                Ok(catalog) => self.search_tools(args, catalog),
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            "load_tools" => self.load_tools(args, context).await,
            _ => self.execute(name, args).await,
        }
    }
}

#[derive(Default)]
pub struct ToolDiscoveryPluginFactory;

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for ToolDiscoveryPluginFactory {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ToolDiscoveryPlugin {
            provider: Arc::new(ToolDiscoveryToolsProvider::new()),
        }))
    }
}

struct ToolDiscoveryPlugin {
    provider: Arc<ToolDiscoveryToolsProvider>,
}

impl SessionPlugin for ToolDiscoveryPlugin {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.surface().contribute(Arc::new(rlm_tool_surface));
        Ok(())
    }
}

fn rlm_tool_surface(ctx: ToolSurfaceContext) -> Result<ToolSurfaceContribution, PluginError> {
    if ctx.mode.plugin_id() != "rlm" {
        return Ok(ToolSurfaceContribution::default());
    }

    let overrides = ctx
        .tools
        .iter()
        .filter_map(|tool| {
            if tool.name == "load_tools" {
                return Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Hidden),
                });
            }
            let availability = tool.effective_availability(&ctx.mode);
            if availability == ToolAvailability::Discoverable {
                Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Callable),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(ToolSurfaceContribution {
        overrides,
        tool_list_notes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::plugin::{
        PluginError, SessionHandle, SessionManager, SessionSnapshot, SessionTurnHandle,
    };
    use lash::{
        AssembledTurn, ExecutionMode, ToolExecutionContext, ToolSurfaceBuildInput, TurnInput,
        build_tool_surface,
    };
    use std::sync::Mutex;

    fn catalog_tool(name: &str, description: &str) -> Value {
        json!({
            "name": name,
            "description": description,
            "params": [],
            "returns": "any",
            "examples": [],
            "aliases": [],
            "availability": "discoverable",
            "callable": false,
            "documented": false,
            "discoverable": true,
            "activation": "loadable",
            "loadable": true,
            "activation_hint": "",
        })
    }

    fn callable_undocumented_tool(name: &str) -> Value {
        let mut tool = catalog_tool(name, "callable omitted tool");
        let obj = tool.as_object_mut().unwrap();
        obj.insert("callable".to_string(), json!(true));
        obj.insert("documented".to_string(), json!(false));
        obj.insert("loadable".to_string(), json!(false));
        tool
    }

    #[derive(Default)]
    struct FakeSessionManager {
        catalog: Vec<Value>,
        promoted: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl SessionManager for FakeSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<Value>, PluginError> {
            Ok(self.catalog.clone())
        }

        async fn set_tools_availability(
            &self,
            _session_id: &str,
            tool_names: &[String],
            _availability: Option<ToolAvailability>,
        ) -> Result<u64, PluginError> {
            self.promoted
                .lock()
                .expect("promoted lock poisoned")
                .extend(tool_names.iter().cloned());
            Ok(2)
        }

        async fn create_session(
            &self,
            _request: lash::plugin::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    fn catalog_tool_with_metadata(
        name: &str,
        description: &str,
        namespace: Option<&str>,
        aliases: Vec<&str>,
    ) -> Value {
        let mut tool = catalog_tool(name, description);
        let obj = tool.as_object_mut().unwrap();
        if let Some(namespace) = namespace {
            obj.insert("namespace".to_string(), json!(namespace));
        }
        obj.insert("aliases".to_string(), json!(aliases));
        tool
    }

    #[test]
    fn exact_name_beats_fuzzy_typo() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool("spotify_search_songs", "Find songs in Spotify"),
                catalog_tool("spotty_notes", "Scratch notes"),
            ],
        );
        let results = index.search(&json!({ "query": "spotify songs" }));
        assert_eq!(results[0]["name"], json!("spotify_search_songs"));
        let typo = index.search(&json!({ "query": "spotfy songs" }));
        assert_eq!(typo[0]["name"], json!("spotify_search_songs"));
    }

    #[test]
    fn default_limit_is_bounded_for_empty_query() {
        let catalog = (0..75)
            .map(|idx| catalog_tool(&format!("tool_{idx:02}"), "tool"))
            .collect();
        let index = ToolDiscoveryIndex::build(1, catalog);
        assert_eq!(index.search(&json!({})).len(), DEFAULT_LIMIT);
    }

    #[test]
    fn limit_is_capped_at_max_limit() {
        let catalog = (0..150)
            .map(|idx| catalog_tool(&format!("tool_{idx:03}"), "tool"))
            .collect();
        let index = ToolDiscoveryIndex::build(1, catalog);
        assert_eq!(index.search(&json!({ "limit": 500 })).len(), MAX_LIMIT);
    }

    #[test]
    fn ranking_finds_core_tool_categories() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "spawn_agent",
                    "Delegate work to subagents",
                    Some("agents"),
                    vec!["subagent"],
                ),
                catalog_tool_with_metadata(
                    "search_web",
                    "Search the web for current sources",
                    Some("web"),
                    vec!["web_search"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_search_songs",
                    "Find songs in Spotify",
                    Some("appworld"),
                    vec!["spotify_search_songs"],
                ),
            ],
        );
        assert_eq!(
            index.search(&json!({ "query": "read files" }))[0]["name"],
            json!("read_file")
        );
        assert_eq!(
            index.search(&json!({ "query": "delegate agent" }))[0]["name"],
            json!("spawn_agent")
        );
        assert_eq!(
            index.search(&json!({ "query": "web search" }))[0]["name"],
            json!("search_web")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify songs", "namespace": "appworld" }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify", "namespace": ["web", "appworld"] }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify", "namespace": "web, appworld" }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
    }

    #[tokio::test]
    async fn search_tools_uses_host_catalog_and_projects_search_metadata() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "search_web",
                    "Search the web",
                    Some("web"),
                    vec!["web_search"],
                ),
            ],
            promoted: Mutex::default(),
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host,
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "search_tools",
                &json!({
                    "query": "cat",
                    "namespace": "filesystem",
                    "limit": 1,
                }),
                &context,
            )
            .await;

        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("search result list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], json!("read_file"));
        assert_eq!(results[0]["namespace"], json!("filesystem"));
        assert_eq!(results[0]["matched_fields"], json!(["aliases"]));
        assert!(results[0]["score"].as_f64().expect("numeric score") > 0.0);
    }

    #[tokio::test]
    async fn load_tools_reports_callable_undocumented_as_already_callable() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                callable_undocumented_tool("mcp__appworld__spotify_search_songs"),
                catalog_tool("fetch_url", "Fetch a URL"),
            ],
            promoted: Mutex::default(),
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host: host.clone(),
            cancellation_token: None,
            async_task_id: None,
        };
        let result = provider
            .execute_with_context(
                "load_tools",
                &json!({
                    "names": ["mcp__appworld__spotify_search_songs", "fetch_url"]
                }),
                &context,
            )
            .await;
        assert!(result.success, "{result:?}");
        assert_eq!(
            result.result["already_callable"],
            json!(["mcp__appworld__spotify_search_songs"])
        );
        assert_eq!(result.result["loaded"], json!(["fetch_url"]));
        assert_eq!(
            *host.promoted.lock().expect("promoted lock poisoned"),
            vec!["fetch_url".to_string()]
        );
    }

    #[test]
    fn rlm_surface_hides_load_tools_and_promotes_discoverable_tools() {
        let tools = vec![
            search_tools_definition(),
            load_tools_definition(),
            ToolDefinition {
                name: "fetch_url".to_string(),
                description: "Fetch URL".to_string(),
                params: Vec::new(),
                returns: "str".to_string(),
                examples: Vec::new(),
                availability: ToolAvailabilityConfig::same(ToolAvailability::Discoverable),
                activation: ToolActivation::Loadable,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                discovery: Default::default(),
                execution_mode: ToolExecutionMode::Parallel,
            },
        ];
        let mode = ExecutionMode::new("rlm");
        let contribution = rlm_tool_surface(ToolSurfaceContext {
            session_id: "session".to_string(),
            mode: mode.clone(),
            tools: tools.clone(),
        })
        .unwrap();
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools,
            mode,
            contributions: vec![contribution],
        });

        assert_eq!(
            surface.tool_availability("load_tools"),
            Some(ToolAvailability::Hidden)
        );
        assert_eq!(
            surface.tool_availability("fetch_url"),
            Some(ToolAvailability::Callable)
        );
    }
}
