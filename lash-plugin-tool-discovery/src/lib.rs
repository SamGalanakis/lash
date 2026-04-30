use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use lash::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolSurfaceContext,
};
use lash::{
    ToolActivation, ToolAvailability, ToolAvailabilityConfig, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolProvider, ToolResult, ToolSurfaceContribution, ToolSurfaceOverride,
};
use serde_json::{Value, json};

const DEFAULT_LIMIT: usize = 10;
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
        let debug = args.get("debug").and_then(Value::as_bool).unwrap_or(false);
        let query_tokens = tokenize(query);

        let mut ranked = Vec::new();
        for (idx, doc) in self.docs.iter().enumerate() {
            if !doc.matches_filters(args) {
                continue;
            }
            let matched_fields = matched_fields(&query_tokens, doc);
            let score = adjusted_score(
                bm25_score(&query_tokens, doc, self) + fuzzy_score(&query_tokens, doc),
                &query_tokens,
                doc,
                &matched_fields,
            );
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
            .map(|(idx, score, _matched_fields)| self.docs[idx].tool.project(score, debug))
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

    fn project(&self, score: f64, debug: bool) -> Value {
        let definition = self.compact_definition();
        let contract = definition.compact_contract();
        let mut out = serde_json::Map::new();
        out.insert("name".to_string(), json!(contract.name));
        out.insert("signature".to_string(), json!(contract.signature));
        out.insert("returns".to_string(), json!(contract.returns));
        if !contract.parameters.is_empty() {
            out.insert("parameters".to_string(), json!(contract.parameters));
        }
        if !contract.return_fields.is_empty() {
            out.insert("return_fields".to_string(), json!(contract.return_fields));
        }
        if !contract.description.is_empty() {
            out.insert("description".to_string(), json!(contract.description));
        }
        if !contract.examples.is_empty() {
            out.insert("examples".to_string(), json!(contract.examples));
        }
        if debug {
            out.insert("score".to_string(), json!(round_score(score)));
        }
        Value::Object(out)
    }

    fn compact_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            string_field(&self.raw, "description"),
            self.raw
                .get("input_schema")
                .cloned()
                .unwrap_or_else(ToolDefinition::default_input_schema),
            self.raw
                .get("output_schema")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .with_examples(string_vec(self.raw.get("examples")))
    }
}

impl DiscoveryDoc {
    fn from_tool(tool: CatalogTool) -> Self {
        let mut fields = Vec::new();
        push_field(&mut fields, "name", vec![tool.name.clone()], 9.0, true);
        push_field(
            &mut fields,
            "namespace",
            tool.namespace.iter().cloned().collect(),
            3.0,
            true,
        );
        push_field(&mut fields, "aliases", tool.aliases.clone(), 8.0, true);
        push_field(
            &mut fields,
            "description",
            vec![string_field(&tool.raw, "description")],
            1.8,
            false,
        );
        push_field(
            &mut fields,
            "params",
            vec![json_field(&tool.raw, "params")],
            0.3,
            false,
        );
        push_field(
            &mut fields,
            "input_fields",
            vec![schema_index_text(tool.raw.get("input_schema"))],
            0.9,
            false,
        );
        push_field(
            &mut fields,
            "output_fields",
            vec![schema_index_text(tool.raw.get("output_schema"))],
            2.4,
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

fn adjusted_score(
    base_score: f64,
    query_tokens: &[String],
    doc: &DiscoveryDoc,
    matched_fields: &[String],
) -> f64 {
    if query_tokens.is_empty() {
        return base_score;
    }

    let name_hits = exact_field_token_hits(query_tokens, doc, "name");
    let alias_hits = exact_field_token_hits(query_tokens, doc, "aliases");
    let description_hits = exact_field_token_hits(query_tokens, doc, "description");
    let output_hits = exact_field_token_hits(query_tokens, doc, "output_fields");
    let primary_hits = name_hits + alias_hits + description_hits;

    let mut score = base_score;
    score += name_hits as f64 * 1.5;
    score += alias_hits as f64 * 1.2;
    score += output_hits.min(query_tokens.len()) as f64 * 0.45;
    score = adjust_for_payment_action_intent(score, query_tokens, doc);
    if name_hits + alias_hits == query_tokens.len() {
        score += 4.0;
    }

    let input_only = primary_hits == 0
        && output_hits == 0
        && matched_fields
            .iter()
            .any(|field| field == "params" || field == "input_fields");
    if input_only {
        score *= 0.35;
    }

    score
}

fn adjust_for_payment_action_intent(
    score: f64,
    query_tokens: &[String],
    doc: &DiscoveryDoc,
) -> f64 {
    if !has_payment_action_intent(query_tokens) || has_query_token(query_tokens, "request") {
        return score;
    }

    let mut adjusted = score;
    if doc_has_any_token(doc, &["transaction"])
        || doc_has_phrase(doc, "send money")
        || doc_has_phrase(doc, "pay user")
    {
        adjusted += 6.0;
    }
    if doc_has_any_token(doc, &["remind", "reminder"]) {
        adjusted *= 0.05;
    } else if doc_has_any_token(doc, &["request"]) {
        adjusted *= 0.8;
    }
    if doc_has_phrase(doc, "venmo balance") || doc_has_phrase(doc, "bank transfer") {
        adjusted *= 0.65;
    }
    adjusted
}

fn has_payment_action_intent(query_tokens: &[String]) -> bool {
    let has_send = has_query_token(query_tokens, "send");
    let has_payment = has_query_token(query_tokens, "payment");
    let has_money = has_query_token(query_tokens, "money");
    let has_make = has_query_token(query_tokens, "make");
    let has_pay = has_query_token(query_tokens, "pay");
    let has_transfer = has_query_token(query_tokens, "transfer");

    (has_send && (has_payment || has_money))
        || (has_make && has_payment)
        || has_pay
        || (has_transfer && has_money)
}

fn has_query_token(query_tokens: &[String], needle: &str) -> bool {
    query_tokens.iter().any(|token| token == needle)
}

fn doc_has_any_token(doc: &DiscoveryDoc, needles: &[&str]) -> bool {
    doc.fields.iter().any(|field| {
        field
            .tokens
            .iter()
            .any(|token| needles.iter().any(|needle| token == needle))
    })
}

fn doc_has_phrase(doc: &DiscoveryDoc, phrase: &str) -> bool {
    let phrase = phrase.to_ascii_lowercase();
    doc.fields
        .iter()
        .any(|field| field.raw.to_ascii_lowercase().contains(&phrase))
}

fn exact_field_token_hits(query_tokens: &[String], doc: &DiscoveryDoc, field_name: &str) -> usize {
    doc.fields
        .iter()
        .filter(|field| field.name == field_name)
        .flat_map(|field| field.tokens.iter())
        .filter(|token| query_tokens.iter().any(|query| query == *token))
        .count()
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

fn schema_index_text(schema: Option<&Value>) -> String {
    let mut parts = Vec::new();
    if let Some(schema) = schema {
        collect_schema_index_text("", schema, &mut parts);
    }
    parts.join("\n")
}

fn collect_schema_index_text(path: &str, schema: &Value, parts: &mut Vec<String>) {
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .or_else(|| schema.get("allOf"))
        .and_then(Value::as_array)
    {
        for subschema in any_of {
            collect_schema_index_text(path, subschema, parts);
        }
    }

    if !path.is_empty() {
        parts.push(path.to_string());
    }
    if let Some(description) = schema
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        parts.push(description.to_string());
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        parts.extend(values.iter().filter_map(enum_index_value));
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property_schema) in properties {
            let field_path = join_schema_path(path, name);
            collect_schema_index_text(&field_path, property_schema, parts);
        }
    }

    if let Some(items) = schema.get("items") {
        let item_path = if path.is_empty() {
            "[]".to_string()
        } else {
            format!("{path}[]")
        };
        collect_schema_index_text(&item_path, items, parts);
    }
}

fn join_schema_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}.{child}")
    }
}

fn enum_index_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn round_score(score: f64) -> f64 {
    (score * 100.0).round() / 100.0
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
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct SearchToolsArgs {
        query: String,
        namespace: Option<NamespaceFilter>,
        #[schemars(range(min = 1, max = 100))]
        limit: Option<usize>,
        debug: Option<bool>,
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    #[serde(untagged)]
    enum NamespaceFilter {
        One(String),
        Many(Vec<String>),
    }

    ToolDefinition::native::<SearchToolsArgs>(
        "search_tools",
        "Search catalogued tool names, namespaces, aliases, descriptions, parameters, and examples. Use this when the tool you need is not showcased in the prompt. Query with concise keywords and intent phrases, not a full sentence: include the app/domain, action, object, and important fields or constraints.",
    )
        .with_examples(vec![
            "search_tools(query=\"spotify song details play_count genre title song_id\", namespace=\"appworld\")".into(),
            "search_tools(query=\"venmo send money private payment_card receiver_email\", namespace=\"appworld\")".into(),
        ])
        .with_availability(ToolAvailabilityConfig::documented())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["tool_search".to_string()],
        })
        .with_execution_mode(ToolExecutionMode::Parallel)
}

fn load_tools_definition() -> ToolDefinition {
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct LoadToolsArgs {
        names: Option<Vec<String>>,
    }

    ToolDefinition::native::<LoadToolsArgs>(
        "load_tools",
        "Promote loadable tools into the documented surface. Callable omitted tools do not need loading and can be called directly.",
    )
        .with_examples(vec![
            "load_tools(names=[\"search_web\", \"fetch_url\"])".into()
        ])
        .with_availability(ToolAvailabilityConfig::documented())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["activate_tools".to_string()],
        })
        .with_execution_mode(ToolExecutionMode::Serial)
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
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": true
            },
            "output_schema": {},
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

    fn ranked_names(results: &[Value]) -> Vec<String> {
        results
            .iter()
            .map(|result| {
                result
                    .get("name")
                    .and_then(Value::as_str)
                    .expect("ranked result name")
                    .to_string()
            })
            .collect()
    }

    fn venmo_payment_tools() -> Vec<Value> {
        let mut create_transaction = catalog_tool_with_metadata(
            "mcp__appworld__venmo_create_transaction",
            "Create a Venmo transaction to send money to another user.",
            Some("appworld"),
            vec!["venmo_send_money", "pay_user"],
        );
        create_transaction.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "receiver_email": {
                        "type": "string",
                        "description": "Email of the person receiving the money."
                    },
                    "payment_card_id": {
                        "type": "integer",
                        "description": "Payment card to fund the transaction."
                    },
                    "private": {
                        "type": "boolean",
                        "description": "Whether the transaction is private."
                    }
                },
                "required": ["access_token", "receiver_email", "payment_card_id"]
            }),
        );

        let mut create_request = catalog_tool_with_metadata(
            "mcp__appworld__venmo_create_payment_request",
            "Create a Venmo payment request asking another user to pay you.",
            Some("appworld"),
            vec!["venmo_request_payment"],
        );
        create_request.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "payer_email": {
                        "type": "string",
                        "description": "Email of the person who should pay the request."
                    },
                    "amount": {"type": "number"}
                },
                "required": ["access_token", "payer_email", "amount"]
            }),
        );

        let mut remind_request = catalog_tool_with_metadata(
            "mcp__appworld__venmo_remind_payment_request",
            "Send a reminder for a pending Venmo payment request.",
            Some("appworld"),
            vec!["venmo_remind_request"],
        );
        remind_request.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "request_id": {"type": "integer"}
                },
                "required": ["access_token", "request_id"]
            }),
        );

        let mut add_balance = catalog_tool_with_metadata(
            "mcp__appworld__venmo_add_to_venmo_balance",
            "Add money from a funding source to your Venmo balance.",
            Some("appworld"),
            vec!["venmo_balance_transfer"],
        );
        add_balance.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "amount": {"type": "number"},
                    "payment_card_id": {"type": "integer"}
                },
                "required": ["access_token", "amount", "payment_card_id"]
            }),
        );

        vec![
            create_request,
            remind_request,
            add_balance,
            create_transaction,
        ]
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

    #[test]
    fn ranking_prefers_name_matches_over_parameter_only_matches() {
        let mut login = catalog_tool_with_metadata(
            "mcp__appworld__spotify_login",
            "Login to your account.",
            Some("appworld"),
            vec!["spotify_login"],
        );
        login.as_object_mut().unwrap().insert(
            "params".to_string(),
            json!([
                {"name": "username", "type": "str", "required": true},
                {"name": "password", "type": "str", "required": true}
            ]),
        );

        let mut logout = catalog_tool_with_metadata(
            "mcp__appworld__spotify_logout",
            "Logout from your account.",
            Some("appworld"),
            vec!["spotify_logout"],
        );
        logout.as_object_mut().unwrap().insert(
            "params".to_string(),
            json!([
                {"name": "access_token", "type": "str", "required": true}
            ]),
        );

        let index = ToolDiscoveryIndex::build(1, vec![logout, login]);
        let results = index.search(&json!({
            "query": "spotify login access token",
            "namespace": "appworld"
        }));

        assert_eq!(results[0]["name"], json!("mcp__appworld__spotify_login"));
    }

    #[test]
    fn ranking_prefers_output_fields_over_input_filter_matches() {
        let mut filter_songs = catalog_tool_with_metadata(
            "mcp__appworld__spotify_filter_songs",
            "Search Spotify songs by filters.",
            Some("appworld"),
            vec!["spotify_filter_songs"],
        );
        filter_songs.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "genre": {
                        "type": "string",
                        "description": "Genre filter."
                    },
                    "play_count": {
                        "type": "integer",
                        "description": "Minimum play count filter."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title filter."
                    }
                },
                "required": ["access_token"]
            }),
        );
        filter_songs.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song_id": {"type": "integer"}
                            },
                            "required": ["song_id"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let mut show_song = catalog_tool_with_metadata(
            "mcp__appworld__spotify_show_song",
            "Get a Spotify song record.",
            Some("appworld"),
            vec!["spotify_show_song", "song_details"],
        );
        show_song.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "song_id": {"type": "integer"}
                },
                "required": ["access_token", "song_id"]
            }),
        );
        show_song.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "description": "Detailed song record.",
                        "properties": {
                            "genre": {
                                "type": "string",
                                "description": "Song genre."
                            },
                            "play_count": {
                                "type": "integer",
                                "description": "Number of times the song was played."
                            },
                            "title": {
                                "type": "string",
                                "description": "Song title."
                            }
                        },
                        "required": ["genre", "play_count", "title"]
                    }
                },
                "required": ["response"]
            }),
        );

        let index = ToolDiscoveryIndex::build(1, vec![filter_songs, show_song]);
        let results = index.search(&json!({
            "query": "play_count genre title",
            "namespace": "appworld"
        }));

        assert_eq!(
            results[0]["name"],
            json!("mcp__appworld__spotify_show_song")
        );
    }

    #[test]
    fn ranking_orders_tools_by_expected_spotify_field_utility() {
        let mut search_songs = catalog_tool_with_metadata(
            "mcp__appworld__spotify_search_songs",
            "Search Spotify songs by filters.",
            Some("appworld"),
            vec!["spotify_search_songs"],
        );
        search_songs.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "genre": {"type": "string"},
                    "play_count": {"type": "integer"},
                    "title": {"type": "string"}
                },
                "required": ["access_token"]
            }),
        );
        search_songs.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song_id": {"type": "integer"}
                            },
                            "required": ["song_id"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let mut show_song = catalog_tool_with_metadata(
            "mcp__appworld__spotify_show_song",
            "Get a Spotify song record.",
            Some("appworld"),
            vec!["spotify_show_song", "song_details"],
        );
        show_song.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "song_id": {"type": "integer"}
                },
                "required": ["access_token", "song_id"]
            }),
        );
        show_song.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "description": "Detailed song record.",
                        "properties": {
                            "genre": {"type": "string"},
                            "play_count": {"type": "integer"},
                            "title": {"type": "string"}
                        },
                        "required": ["genre", "play_count", "title"]
                    }
                },
                "required": ["response"]
            }),
        );

        let mut list_albums = catalog_tool_with_metadata(
            "mcp__appworld__spotify_list_albums",
            "List Spotify albums.",
            Some("appworld"),
            vec!["spotify_albums"],
        );
        list_albums.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "album_id": {"type": "integer"},
                                "title": {"type": "string"}
                            },
                            "required": ["album_id", "title"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let index = ToolDiscoveryIndex::build(1, vec![search_songs, show_song, list_albums]);
        let results = index.search(&json!({
            "query": "spotify song details play_count genre title",
            "namespace": "appworld",
            "limit": 3
        }));

        assert_eq!(
            ranked_names(&results),
            vec![
                "mcp__appworld__spotify_show_song",
                "mcp__appworld__spotify_search_songs",
                "mcp__appworld__spotify_list_albums",
            ]
        );
    }

    #[test]
    fn ranking_orders_venmo_tools_by_send_money_intent() {
        let index = ToolDiscoveryIndex::build(1, venmo_payment_tools());
        let results = index.search(&json!({
            "query": "venmo send money private payment_card receiver_email",
            "namespace": "appworld",
            "limit": 3
        }));

        assert_eq!(
            ranked_names(&results),
            vec![
                "mcp__appworld__venmo_create_transaction",
                "mcp__appworld__venmo_create_payment_request",
                "mcp__appworld__venmo_add_to_venmo_balance",
            ]
        );
    }

    #[test]
    fn ranking_orders_venmo_short_payment_queries_by_action_intent() {
        for query in [
            "venmo send payment",
            "venmo send payment to user",
            "venmo make payment transfer money",
        ] {
            let index = ToolDiscoveryIndex::build(1, venmo_payment_tools());
            let results = index.search(&json!({
                "query": query,
                "namespace": "appworld",
                "limit": 4
            }));

            assert_eq!(
                results[0]["name"],
                json!("mcp__appworld__venmo_create_transaction"),
                "unexpected ranking for query {query:?}"
            );
        }
    }

    #[test]
    fn search_results_include_compact_schema_parameter_restrictions() {
        let mut spotify = catalog_tool("mcp__appworld__spotify_search_songs", "Find songs");
        spotify.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "Access token obtained from spotify app login."
                    },
                    "genre": {
                        "type": ["string", "null"],
                        "description": "Only include songs from this genre.",
                        "default": null
                    },
                    "page_limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 20
                    }
                },
                "required": ["access_token"]
            }),
        );
        spotify.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "description": "Matched songs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "album_id": {"type": ["integer", "null"]},
                                "duration": {"type": "integer"},
                                "genre": {"type": "string"},
                                "like_count": {"type": "integer"},
                                "play_count": {
                                    "type": "integer",
                                    "description": "Number of times the song was played.",
                                    "minimum": 0
                                },
                                "rating": {"type": "number"},
                                "release_date": {"type": "string"},
                                "song_id": {
                                    "type": "integer",
                                    "description": "Stable song identifier."
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Song title."
                                }
                            },
                            "required": [
                                "album_id",
                                "duration",
                                "genre",
                                "like_count",
                                "play_count",
                                "rating",
                                "release_date",
                                "song_id",
                                "title"
                            ]
                        }
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message when search fails."
                    }
                },
                "required": ["response"]
            }),
        );
        let index = ToolDiscoveryIndex::build(1, vec![spotify]);

        let results = index.search(&json!({ "query": "spotify" }));
        assert_eq!(
            results[0]["signature"],
            json!(
                "mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 20)"
            )
        );
        assert_eq!(
            results[0]["returns"],
            json!(
                "record{error?: str, response: list[record{album_id: int | null, duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]}"
            )
        );
        assert_eq!(
            results[0]["parameters"],
            json!([
                {
                    "name": "access_token",
                    "type": "str",
                    "required": true,
                    "description": "Access token obtained from spotify app login.",
                    "signature": "access_token: str"
                },
                {
                    "name": "genre",
                    "type": "str | null",
                    "required": false,
                    "nullable": true,
                    "description": "Only include songs from this genre.",
                    "default": null,
                    "signature": "genre?: str | null = null"
                },
                {
                    "name": "page_limit",
                    "type": "int",
                    "required": false,
                    "description": "Maximum number of results to return.",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 20,
                    "signature": "page_limit?: int >= 1 <= 20 = 20"
                }
            ])
        );
        assert_eq!(
            results[0]["return_fields"],
            json!([
                {
                    "path": "error",
                    "type": "str",
                    "required": false,
                    "description": "Error message when search fails.",
                    "signature": "error?: str"
                },
                {
                    "path": "response",
                    "type": "list[record]",
                    "required": true,
                    "description": "Matched songs.",
                    "items": "record",
                    "signature": "response: list[record]"
                },
                {
                    "path": "response[].album_id",
                    "type": "int | null",
                    "required": true,
                    "nullable": true,
                    "signature": "response[].album_id: int | null"
                },
                {
                    "path": "response[].duration",
                    "type": "int",
                    "required": true,
                    "signature": "response[].duration: int"
                },
                {
                    "path": "response[].genre",
                    "type": "str",
                    "required": true,
                    "signature": "response[].genre: str"
                },
                {
                    "path": "response[].like_count",
                    "type": "int",
                    "required": true,
                    "signature": "response[].like_count: int"
                },
                {
                    "path": "response[].play_count",
                    "type": "int",
                    "required": true,
                    "description": "Number of times the song was played.",
                    "minimum": 0,
                    "signature": "response[].play_count: int >= 0"
                },
                {
                    "path": "response[].rating",
                    "type": "float",
                    "required": true,
                    "signature": "response[].rating: float"
                },
                {
                    "path": "response[].release_date",
                    "type": "str",
                    "required": true,
                    "signature": "response[].release_date: str"
                },
                {
                    "path": "response[].song_id",
                    "type": "int",
                    "required": true,
                    "description": "Stable song identifier.",
                    "signature": "response[].song_id: int"
                },
                {
                    "path": "response[].title",
                    "type": "str",
                    "required": true,
                    "description": "Song title.",
                    "signature": "response[].title: str"
                }
            ])
        );
        assert!(results[0].get("params").is_none());
        assert!(results[0].get("input_schema").is_none());
        assert!(results[0].get("output_schema").is_none());
        assert!(results[0].get("matched_fields").is_none());
        assert!(results[0].get("score").is_none());
    }

    #[tokio::test]
    async fn search_tools_uses_host_catalog_and_projects_compact_contract() {
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
            .execute_streaming_with_context(
                "search_tools",
                &json!({
                    "query": "cat",
                    "namespace": "filesystem",
                    "limit": 1,
                }),
                &context,
                None,
            )
            .await;

        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("search result list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], json!("read_file"));
        assert_eq!(results[0]["signature"], json!("read_file()"));
        assert_eq!(results[0]["returns"], json!("any"));
        assert_eq!(results[0]["description"], json!("Read file contents"));
        assert!(results[0].get("namespace").is_none());
        assert!(results[0].get("matched_fields").is_none());
        assert!(results[0].get("score").is_none());
    }

    #[test]
    fn debug_search_includes_minimal_score() {
        let index = ToolDiscoveryIndex::build(1, vec![catalog_tool("read_file", "Read files")]);

        let results = index.search(&json!({ "query": "read", "debug": true }));

        assert!(results[0]["score"].as_f64().is_some());
        assert!(results[0].get("matched_fields").is_none());
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
            ToolDefinition::new(
                "fetch_url",
                "Fetch URL",
                ToolDefinition::default_input_schema(),
                serde_json::json!({ "type": "string" }),
            )
            .with_availability(ToolAvailabilityConfig::same(ToolAvailability::Discoverable))
            .with_activation(ToolActivation::Loadable)
            .with_execution_mode(ToolExecutionMode::Parallel),
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
