use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
#[cfg(feature = "semantic-tool-search")]
use std::sync::{Mutex, OnceLock};

use lash::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolSurfaceContext,
};
use lash::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
    ToolActivation, ToolAvailability, ToolAvailabilityConfig, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolProvider, ToolResult, ToolSurfaceContribution, ToolSurfaceOverride,
};
use serde_json::{Value, json};

const DEFAULT_LIMIT: usize = 10;
const MAX_LIMIT: usize = 100;
const LLM_CANDIDATE_LIMIT: usize = 100;
const DEFAULT_LLM_RERANK_MODEL: &str = "medium";
const FUZZY_SCORE_CAP: f64 = 1.25;
const SEMANTIC_CANDIDATE_FLOOR: usize = 50;
const RRF_K: f64 = 60.0;

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
    #[cfg(feature = "semantic-tool-search")]
    semantic_text: String,
}

#[derive(Clone, Debug)]
struct RankedCandidate {
    idx: usize,
    lexical_score: f64,
    semantic_score: Option<f64>,
}

#[derive(Debug)]
pub struct ToolDiscoveryIndex {
    key: u64,
    docs: Vec<DiscoveryDoc>,
    avg_len: f64,
    doc_freq: HashMap<String, usize>,
    #[cfg(feature = "semantic-tool-search")]
    semantic_embeddings: OnceLock<Vec<Vec<f32>>>,
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

    async fn search_tools(
        &self,
        args: &Value,
        catalog: Vec<Value>,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let index = self.index_for_catalog(catalog);
        let limit = limit_from_args(args);
        let candidate_args = args_with_limit(args, LLM_CANDIDATE_LIMIT);
        let candidates = index.search(&candidate_args);
        if candidates.is_empty() {
            return ToolResult::ok(json!([]));
        }

        let request = llm_rerank_request(args, &candidates, limit);
        let completion = match context
            .host
            .direct_completion(request, "search_tools")
            .await
        {
            Ok(completion) => completion,
            Err(err) => return ToolResult::err_fmt(format_args!("search_tools failed: {err}")),
        };

        let selected_names = match parse_llm_tool_names(&completion.text) {
            Ok(names) => names,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "search_tools returned invalid JSON: {err}"
                ));
            }
        };

        ToolResult::ok(json!(merge_llm_selection(
            candidates,
            selected_names,
            limit
        )))
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
            #[cfg(feature = "semantic-tool-search")]
            semantic_embeddings: OnceLock::new(),
        }
    }

    fn search(&self, args: &Value) -> Vec<Value> {
        let semantic_scores = self.semantic_scores(args);
        self.search_with_semantic_scores(args, semantic_scores.as_deref())
    }

    fn search_with_semantic_scores(
        &self,
        args: &Value,
        semantic_scores: Option<&[f64]>,
    ) -> Vec<Value> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let limit = limit_from_args(args);
        let debug = args.get("debug").and_then(Value::as_bool).unwrap_or(false);
        let query_tokens = tokenize(query);
        let mut lexical = Vec::new();
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
            if query.is_empty() || !matched_fields.is_empty() {
                lexical.push(RankedCandidate {
                    idx,
                    lexical_score: score,
                    semantic_score: semantic_scores
                        .filter(|scores| scores.len() == self.docs.len())
                        .and_then(|scores| scores.get(idx))
                        .copied(),
                });
            }
        }

        if query.is_empty() {
            lexical.sort_by(|left, right| {
                self.docs[left.idx]
                    .tool
                    .name
                    .cmp(&self.docs[right.idx].tool.name)
            });
        } else {
            lexical.sort_by(|left, right| {
                right
                    .lexical_score
                    .partial_cmp(&left.lexical_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.docs[left.idx]
                            .tool
                            .name
                            .cmp(&self.docs[right.idx].tool.name)
                    })
            });
        }

        let ranked = if query.is_empty() {
            lexical
        } else if let Some(scores) =
            semantic_scores.filter(|scores| scores.len() == self.docs.len())
        {
            let semantic = self.semantic_candidates(args, scores, limit);
            reciprocal_rank_fusion(lexical, semantic)
        } else {
            lexical
        };

        ranked
            .into_iter()
            .take(limit)
            .map(|candidate| {
                let score = candidate
                    .semantic_score
                    .map(|semantic| {
                        round_score(candidate.lexical_score) + round_score(semantic.max(0.0))
                    })
                    .unwrap_or(candidate.lexical_score);
                self.docs[candidate.idx].tool.project(score, debug)
            })
            .collect()
    }

    fn semantic_candidates(
        &self,
        args: &Value,
        semantic_scores: &[f64],
        limit: usize,
    ) -> Vec<RankedCandidate> {
        let candidate_limit = limit
            .saturating_mul(5)
            .max(SEMANTIC_CANDIDATE_FLOOR)
            .min(semantic_scores.len());
        let mut ranked = semantic_scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(idx, score)| {
                score.is_finite() && *score > 0.0 && self.docs[*idx].matches_filters(args)
            })
            .map(|(idx, score)| RankedCandidate {
                idx,
                lexical_score: 0.0,
                semantic_score: Some(score),
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .semantic_score
                .partial_cmp(&left.semantic_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.docs[left.idx]
                        .tool
                        .name
                        .cmp(&self.docs[right.idx].tool.name)
                })
        });
        ranked.truncate(candidate_limit);
        ranked
    }

    fn semantic_scores(&self, args: &Value) -> Option<Vec<f64>> {
        #[cfg(feature = "semantic-tool-search")]
        {
            self.semantic_scores_enabled(args)
        }
        #[cfg(not(feature = "semantic-tool-search"))]
        {
            let _ = args;
            None
        }
    }

    #[cfg(feature = "semantic-tool-search")]
    fn semantic_scores_enabled(&self, args: &Value) -> Option<Vec<f64>> {
        if !semantic_requested(args) {
            return None;
        }
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if query.is_empty() || self.docs.is_empty() {
            return None;
        }
        let model_guard = semantic_model()?;
        let model = model_guard.as_ref()?;
        let doc_embeddings = self.semantic_embeddings.get_or_init(|| {
            let docs = self
                .docs
                .iter()
                .map(|doc| doc.semantic_text.clone())
                .collect::<Vec<_>>();
            model.encode(&docs)
        });
        let query_embedding = model.encode_single(query);
        Some(
            doc_embeddings
                .iter()
                .map(|embedding| cosine_similarity(&query_embedding, embedding) as f64)
                .collect(),
        )
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
        out.insert("signature".to_string(), json!(contract.render_signature()));
        out.insert("returns".to_string(), json!(contract.render_returns()));
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

        #[cfg(feature = "semantic-tool-search")]
        let semantic_text = semantic_index_text(&tool);

        Self {
            tool,
            fields,
            #[cfg(feature = "semantic-tool-search")]
            semantic_text,
        }
    }

    fn matches_filters(&self, args: &Value) -> bool {
        let namespaces = namespace_filter(args.get("namespace"));
        if !namespaces.is_empty()
            && !self
                .tool
                .namespace
                .as_deref()
                .is_some_and(|namespace| namespaces.iter().any(|candidate| candidate == namespace))
        {
            return false;
        }
        !exclude_filter(args.get("exclude")).contains(&self.tool.name)
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
                        || (query.len() >= 3 && token.len() >= 3 && token.contains(query))
                        || (query.len() >= 3 && token.len() >= 3 && query.contains(token))
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

fn reciprocal_rank_fusion(
    lexical: Vec<RankedCandidate>,
    semantic: Vec<RankedCandidate>,
) -> Vec<RankedCandidate> {
    let mut fused: HashMap<usize, (f64, RankedCandidate)> = HashMap::new();
    add_rrf_candidates(&mut fused, lexical, |candidate| candidate.lexical_score);
    add_rrf_candidates(&mut fused, semantic, |candidate| {
        candidate.semantic_score.unwrap_or_default()
    });

    let mut fused = fused.into_values().collect::<Vec<_>>();
    fused.sort_by(|(left_score, left), (right_score, right)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .lexical_score
                    .partial_cmp(&left.lexical_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .semantic_score
                    .partial_cmp(&left.semantic_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.idx.cmp(&right.idx))
    });
    fused.into_iter().map(|(_, candidate)| candidate).collect()
}

fn add_rrf_candidates(
    fused: &mut HashMap<usize, (f64, RankedCandidate)>,
    candidates: Vec<RankedCandidate>,
    score_of: impl Fn(&RankedCandidate) -> f64,
) {
    for (rank, candidate) in candidates.into_iter().enumerate() {
        let rank_score = 1.0 / (RRF_K + rank as f64 + 1.0);
        let raw_score = score_of(&candidate).max(0.0) * 0.000_001;
        let entry = fused
            .entry(candidate.idx)
            .or_insert_with(|| (0.0, candidate.clone()));
        entry.0 += rank_score + raw_score;
        entry.1.lexical_score = entry.1.lexical_score.max(candidate.lexical_score);
        if let Some(score) = candidate.semantic_score {
            let current = entry.1.semantic_score.unwrap_or(f64::NEG_INFINITY);
            if score > current {
                entry.1.semantic_score = Some(score);
            }
        }
    }
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

fn args_with_limit(args: &Value, limit: usize) -> Value {
    let mut args = args.as_object().cloned().unwrap_or_default();
    args.insert("limit".to_string(), json!(limit.clamp(1, MAX_LIMIT)));
    args.insert("debug".to_string(), json!(false));
    Value::Object(args)
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

fn exclude_filter(value: Option<&Value>) -> BTreeSet<String> {
    match value {
        Some(Value::String(name)) => name
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
        _ => BTreeSet::new(),
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

#[cfg(feature = "semantic-tool-search")]
fn semantic_index_text(tool: &CatalogTool) -> String {
    let definition = tool.compact_definition();
    let contract = definition.compact_contract();
    let mut parts = vec![
        contract.name.clone(),
        contract.render_signature(),
        contract.render_returns(),
        contract.description.clone(),
    ];
    parts.extend(contract.examples.clone());
    parts
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
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

#[cfg(feature = "semantic-tool-search")]
fn semantic_requested(args: &Value) -> bool {
    if let Some(value) = args.get("semantic").and_then(Value::as_bool) {
        return value;
    }
    if let Ok(value) = std::env::var("LASH_TOOL_SEARCH_SEMANTIC") {
        return truthy_env_value(&value);
    }
    false
}

#[cfg(feature = "semantic-tool-search")]
fn truthy_env_value(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "semantic-tool-search")]
fn semantic_model()
-> Option<std::sync::MutexGuard<'static, Option<model2vec_rs::model::StaticModel>>> {
    static MODEL: OnceLock<Mutex<Option<model2vec_rs::model::StaticModel>>> = OnceLock::new();
    let lock = MODEL.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().ok()?;
    if guard.is_none() {
        if let Ok(model) = {
            let model_id = std::env::var("LASH_TOOL_SEARCH_SEMANTIC_MODEL")
                .unwrap_or_else(|_| "minishlab/potion-base-2M".to_string());
            model2vec_rs::model::StaticModel::from_pretrained(model_id, None, None, None)
        } {
            *guard = Some(model);
        }
    }
    guard.as_ref()?;
    Some(guard)
}

#[cfg(feature = "semantic-tool-search")]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right.iter()) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
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

fn llm_rerank_request(args: &Value, candidates: &[Value], limit: usize) -> DirectRequest {
    let model = std::env::var("LASH_TOOL_SEARCH_LLM_MODEL")
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_LLM_RERANK_MODEL.to_string());
    let model_variant = std::env::var("LASH_TOOL_SEARCH_LLM_VARIANT")
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty());
    let candidate_names = candidates
        .iter()
        .filter_map(|candidate| candidate.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let schema = DirectJsonSchema {
        name: "tool_search_rerank".to_string(),
        strict: true,
        schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["tool_names"],
            "properties": {
                "tool_names": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": candidate_names,
                    }
                }
            }
        }),
    };
    let prompt = llm_rerank_prompt(args, candidates, limit);
    DirectRequest {
        model,
        model_variant,
        messages: vec![
            DirectMessage {
                role: DirectRole::System,
                parts: vec![DirectPart::Text(
                    "You select API tools for another agent. Return tool names only through the requested JSON schema. Maximize recall for the caller's query while keeping the ranking precise."
                        .to_string(),
                )],
            },
            DirectMessage {
                role: DirectRole::User,
                parts: vec![DirectPart::Text(prompt)],
            },
        ],
        attachments: Vec::new(),
        output: DirectOutputSpec::JsonSchema(schema),
        stream_events: None,
        session_id: None,
    }
}

fn llm_rerank_prompt(args: &Value, candidates: &[Value], limit: usize) -> String {
    let compact_candidates = candidates
        .iter()
        .map(llm_candidate_payload)
        .collect::<Vec<_>>();
    json!({
        "instructions": [
            "Select tools from candidates for the caller's query.",
            "Return only tools that may be needed to complete the task, best to worst.",
            "Prefer read/list/show/search tools for inspection or counting tasks.",
            "Prefer mutation tools only when the query explicitly asks to change state.",
            "For combined constraints, include complementary tools for each constraint.",
            "Do not include tools outside the candidate list."
        ],
        "query": args.get("query").and_then(Value::as_str).unwrap_or_default(),
        "namespace": args.get("namespace").cloned().unwrap_or(Value::Null),
        "exclude": args.get("exclude").cloned().unwrap_or_else(|| json!([])),
        "limit": limit,
        "candidates": compact_candidates,
    })
    .to_string()
}

fn llm_candidate_payload(candidate: &Value) -> Value {
    json!({
        "name": candidate.get("name").cloned().unwrap_or(Value::Null),
        "signature": candidate.get("signature").cloned().unwrap_or(Value::Null),
        "returns": candidate.get("returns").cloned().unwrap_or(Value::Null),
        "description": candidate.get("description").cloned().unwrap_or(Value::Null),
        "examples": candidate.get("examples").cloned().unwrap_or(Value::Null),
    })
}

fn parse_llm_tool_names(text: &str) -> Result<Vec<String>, serde_json::Error> {
    let trimmed = text.trim();
    let value = match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => value,
        Err(err) => {
            let Some(start) = trimmed.find('{') else {
                return Err(err);
            };
            let Some(end) = trimmed.rfind('}') else {
                return Err(err);
            };
            serde_json::from_str::<Value>(&trimmed[start..=end])?
        }
    };
    Ok(value
        .get("tool_names")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect())
}

fn merge_llm_selection(
    candidates: Vec<Value>,
    selected_names: Vec<String>,
    limit: usize,
) -> Vec<Value> {
    let mut by_name = BTreeMap::new();
    let mut deterministic_names = Vec::new();
    for candidate in candidates {
        let Some(name) = candidate.get("name").and_then(Value::as_str) else {
            continue;
        };
        if !by_name.contains_key(name) {
            deterministic_names.push(name.to_string());
        }
        by_name.insert(name.to_string(), candidate);
    }

    let mut seen = BTreeSet::new();
    let mut ranked = Vec::new();
    for name in selected_names.into_iter().chain(deterministic_names) {
        if ranked.len() >= limit {
            break;
        }
        if !seen.insert(name.clone()) {
            continue;
        }
        if let Some(candidate) = by_name.get(&name) {
            ranked.push(candidate.clone());
        }
    }
    ranked
}

fn search_tools_definition() -> ToolDefinition {
    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct SearchToolsArgs {
        #[schemars(
            description = "Concise tool search query. Prefer keywords and short intent phrases with the app/domain, action, object, qualifiers, and important fields; for multi-constraint tasks include every constraint, such as \"spotify liked songs library\"."
        )]
        query: String,
        #[schemars(description = "Optional namespace filter, such as \"appworld\".")]
        namespace: Option<NamespaceFilter>,
        #[schemars(range(min = 1, max = 100))]
        #[schemars(description = "Maximum number of results to return. Defaults to 10.")]
        limit: Option<usize>,
        #[schemars(description = "Exact tool name or names to exclude from results.")]
        exclude: Option<NameFilter>,
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    #[serde(untagged)]
    enum NamespaceFilter {
        One(String),
        Many(Vec<String>),
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    #[serde(untagged)]
    enum NameFilter {
        One(String),
        Many(Vec<String>),
    }

    ToolDefinition::native::<SearchToolsArgs>(
        "search_tools",
        "Search catalogued tool names, namespaces, aliases, descriptions, signatures, return fields, and examples. Use this when the tool you need is not showcased in the prompt. Query with concise keywords and short intent phrases: include the app/domain, action, object, qualifiers, and important fields or constraints. For initial exploration, print only result names and signatures; inspect descriptions and examples only when you need to choose between close matches or learn call idioms.",
    )
        .with_examples(vec![
            "search_tools(query=\"spotify liked songs library\", namespace=\"appworld\")".into(),
            "search_tools(query=\"spotify song details play_count genre title song_id\", namespace=\"appworld\")".into(),
            "search_tools(query=\"venmo send money private payment_card receiver_email\", namespace=\"appworld\")".into(),
        ])
        .with_availability(ToolAvailabilityConfig::documented())
        .with_activation(ToolActivation::Always)
        .with_discovery(lash::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["tool_search".to_string()],
        })
        .with_execution_mode(ToolExecutionMode::Serial)
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
                Ok(catalog) => self.search_tools(args, catalog, context).await,
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
        AssembledTurn, DirectCompletion, ExecutionMode, TokenUsage, ToolExecutionContext,
        ToolSurfaceBuildInput, TurnInput, build_tool_surface,
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
        direct_response: Mutex<Option<String>>,
        direct_requests: Mutex<Vec<lash::DirectRequest>>,
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

        async fn direct_completion(
            &self,
            request: lash::DirectRequest,
            _usage_source: &str,
        ) -> Result<DirectCompletion, PluginError> {
            self.direct_requests
                .lock()
                .expect("direct requests lock poisoned")
                .push(request);
            let text = self
                .direct_response
                .lock()
                .expect("direct response lock poisoned")
                .clone()
                .unwrap_or_else(|| "{\"tool_names\":[]}".to_string());
            Ok(DirectCompletion {
                text,
                usage: TokenUsage::default(),
            })
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
    fn short_query_words_do_not_create_substring_matches() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_album",
                    "Show details for a Spotify album.",
                    Some("appworld"),
                    vec!["spotify_album_details"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_song_library",
                    "Show songs saved in your collection.",
                    Some("appworld"),
                    vec!["spotify_song_library"],
                ),
            ],
        );

        let results = index.search(&json!({
            "query": "tracks in collection",
            "namespace": "appworld",
            "limit": 5
        }));

        assert_eq!(
            ranked_names(&results),
            vec!["mcp__appworld__spotify_show_song_library"]
        );
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
    fn semantic_fusion_adds_recall_candidates_without_replacing_exact_matches() {
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
                    "mcp__appworld__music_collection",
                    "Show saved tracks.",
                    Some("appworld"),
                    vec!["music_collection"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_liked_songs",
                    "Show songs that you have favorited.",
                    Some("appworld"),
                    vec!["spotify_favorite_songs"],
                ),
            ],
        );

        let lexical_only = index.search(&json!({
            "query": "spotify liked songs library",
            "namespace": "appworld",
            "limit": 2
        }));
        assert_eq!(
            ranked_names(&lexical_only),
            vec!["mcp__appworld__spotify_show_liked_songs"]
        );

        let semantic = index.search_with_semantic_scores(
            &json!({
                "query": "spotify liked songs library",
                "namespace": "appworld",
                "limit": 2
            }),
            Some(&[0.0, 0.92, 0.86]),
        );
        assert_eq!(
            ranked_names(&semantic),
            vec![
                "mcp__appworld__spotify_show_liked_songs",
                "mcp__appworld__music_collection",
            ]
        );

        let exact = index.search_with_semantic_scores(
            &json!({ "query": "read file", "limit": 2 }),
            Some(&[0.45, 0.9, 0.8]),
        );
        assert_eq!(exact[0]["name"], json!("read_file"));
    }

    #[test]
    fn semantic_fusion_promotes_complementary_tools_for_mixed_qualifier_queries() {
        let catalog = vec![
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_liked_songs",
                "Show songs that you have liked.",
                Some("appworld"),
                vec!["spotify_liked_songs"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_song_library",
                "Show songs in your saved Spotify library.",
                Some("appworld"),
                vec!["spotify_song_library"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_liked_playlists",
                "Show Spotify playlists that you have liked.",
                Some("appworld"),
                vec!["spotify_liked_playlists"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_playlist_library",
                "Show playlists in your Spotify library.",
                Some("appworld"),
                vec!["spotify_playlist_library"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_playlist",
                "Show songs from a Spotify playlist.",
                Some("appworld"),
                vec!["spotify_playlist_songs"],
            ),
        ];
        let index = ToolDiscoveryIndex::build(1, catalog);

        let library_and_liked = index.search_with_semantic_scores(
            &json!({
                "query": "spotify liked songs library",
                "namespace": "appworld",
                "limit": 3
            }),
            Some(&[0.92, 0.91, 0.76, 0.74, 0.68]),
        );
        let names = ranked_names(&library_and_liked);
        assert_eq!(names.len(), 3);
        assert!(names[..2].contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names[..2].contains(&"mcp__appworld__spotify_show_song_library".to_string()));

        let playlist_and_liked = index.search_with_semantic_scores(
            &json!({
                "query": "spotify playlist songs liked",
                "namespace": "appworld",
                "limit": 4
            }),
            Some(&[0.88, 0.62, 0.91, 0.84, 0.87]),
        );
        let names = ranked_names(&playlist_and_liked);
        assert!(names.contains(&"mcp__appworld__spotify_show_playlist".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_playlists".to_string()));
    }

    #[cfg(feature = "semantic-tool-search")]
    #[test]
    #[ignore = "downloads and loads an external embedding model"]
    fn semantic_model_smoke_retrieves_paraphrased_tool_matches() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_liked_songs",
                    "Show songs that you have liked.",
                    Some("appworld"),
                    vec!["spotify_liked_songs"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_song_library",
                    "Show songs saved in your Spotify library.",
                    Some("appworld"),
                    vec!["spotify_song_library"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_album",
                    "Show details for a Spotify album.",
                    Some("appworld"),
                    vec!["spotify_album_details"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__venmo_create_transaction",
                    "Send money to another Venmo user.",
                    Some("appworld"),
                    vec!["venmo_send_money"],
                ),
            ],
        );

        let lexical = index.search(&json!({
            "query": "favorite tracks in my saved collection",
            "namespace": "appworld",
            "limit": 3
        }));
        let semantic = index.search(&json!({
            "query": "favorite tracks in my saved collection",
            "namespace": "appworld",
            "semantic": true,
            "limit": 3
        }));

        eprintln!("lexical={:?}", ranked_names(&lexical));
        eprintln!("semantic={:?}", ranked_names(&semantic));

        let names = ranked_names(&semantic);
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_song_library".to_string()));
        assert!(!names.contains(&"mcp__appworld__venmo_create_transaction".to_string()));
    }

    #[test]
    fn reciprocal_rank_fusion_keeps_cross_list_hits_ahead_of_single_list_noise() {
        let fused = reciprocal_rank_fusion(
            vec![
                RankedCandidate {
                    idx: 0,
                    lexical_score: 10.0,
                    semantic_score: None,
                },
                RankedCandidate {
                    idx: 1,
                    lexical_score: 8.0,
                    semantic_score: None,
                },
                RankedCandidate {
                    idx: 2,
                    lexical_score: 6.0,
                    semantic_score: None,
                },
            ],
            vec![
                RankedCandidate {
                    idx: 3,
                    lexical_score: 0.0,
                    semantic_score: Some(0.99),
                },
                RankedCandidate {
                    idx: 1,
                    lexical_score: 0.0,
                    semantic_score: Some(0.88),
                },
                RankedCandidate {
                    idx: 4,
                    lexical_score: 0.0,
                    semantic_score: Some(0.87),
                },
            ],
        );

        let names = fused
            .iter()
            .map(|candidate| candidate.idx)
            .collect::<Vec<_>>();
        assert_eq!(names[..3], [1, 0, 3]);
    }

    #[test]
    fn search_results_include_compact_schema_parameter_restrictions() {
        let mut spotify = catalog_tool("mcp__appworld__spotify_search_songs", "Find songs");
        spotify.as_object_mut().unwrap().insert(
            "examples".to_string(),
            json!(["search songs by genre", "search songs by play count"]),
        );
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
                "mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 20)\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 20` — Maximum number of results to return."
            )
        );
        assert_eq!(
            results[0]["returns"],
            json!(
                "record{error?: str, response: list[record{album_id: int | null, duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]}\nReturn fields:\n- `error?: str` — Error message when search fails.\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str`\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title."
            )
        );
        assert!(results[0].get("params").is_none());
        assert!(results[0].get("parameters").is_none());
        assert!(results[0].get("return_fields").is_none());
        assert_eq!(
            results[0]["examples"],
            json!(["search songs by genre", "search songs by play count"])
        );
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
            ..Default::default()
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

    #[test]
    fn exclude_filter_removes_exact_tool_names() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool("read_file", "Read files"),
                catalog_tool("search_web", "Search the web"),
            ],
        );

        let results = index.search(&json!({
            "query": "",
            "exclude": ["read_file"],
        }));

        assert_eq!(ranked_names(&results), vec!["search_web"]);
    }

    #[test]
    fn llm_rerank_request_uses_structured_name_enum_schema() {
        let candidates = vec![
            json!({"name": "read_file", "signature": "read_file()", "returns": "str", "description": "Read file"}),
            json!({"name": "search_web", "signature": "search_web(query: str)", "returns": "record", "description": "Search web"}),
        ];

        let request = llm_rerank_request(&json!({"query": "find docs"}), &candidates, 2);

        assert_eq!(
            request.model,
            std::env::var("LASH_TOOL_SEARCH_LLM_MODEL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_LLM_RERANK_MODEL.to_string())
        );
        let DirectOutputSpec::JsonSchema(schema) = request.output else {
            panic!("expected json schema output");
        };
        assert_eq!(schema.name, "tool_search_rerank");
        assert!(
            schema.schema["properties"]["tool_names"]
                .get("uniqueItems")
                .is_none()
        );
        assert!(
            schema.schema["properties"]["tool_names"]
                .get("maxItems")
                .is_none()
        );
        assert_eq!(
            schema.schema["properties"]["tool_names"]["items"]["enum"],
            json!(["read_file", "search_web"])
        );
    }

    #[test]
    fn merge_llm_selection_dedupes_and_fills_from_deterministic_order() {
        let candidates = vec![
            json!({"name": "a"}),
            json!({"name": "b"}),
            json!({"name": "c"}),
        ];

        let merged = merge_llm_selection(
            candidates,
            vec!["b".to_string(), "b".to_string(), "missing".to_string()],
            3,
        );

        assert_eq!(ranked_names(&merged), vec!["b", "a", "c"]);
    }

    #[tokio::test]
    async fn search_tools_reranks_candidates_with_direct_completion() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                catalog_tool_with_metadata("read_file", "Read file contents", None, vec!["cat"]),
                catalog_tool_with_metadata("search_web", "Search the web", None, vec!["web"]),
            ],
            promoted: Mutex::default(),
            direct_response: Mutex::new(Some(
                "{\"tool_names\":[\"search_web\",\"search_web\",\"unknown\"]}".to_string(),
            )),
            ..Default::default()
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
                "search_tools",
                &json!({
                    "query": "",
                    "exclude": ["read_file"],
                    "limit": 2,
                }),
                &context,
            )
            .await;

        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("search result list");
        assert_eq!(ranked_names(results), vec!["search_web"]);
        let requests = host
            .direct_requests
            .lock()
            .expect("direct requests lock poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "medium");
        assert!(
            requests[0]
                .messages
                .iter()
                .flat_map(|message| message.parts.iter())
                .any(|part| matches!(
                    part,
                    DirectPart::Text(text)
                        if text.contains("\"exclude\":[\"read_file\"]")
                            && !text.contains("\"name\":\"read_file\"")
                ))
        );
        let DirectOutputSpec::JsonSchema(schema) = &requests[0].output else {
            panic!("expected json schema output");
        };
        assert_eq!(
            schema.schema["properties"]["tool_names"]["items"]["enum"],
            json!(["search_web"])
        );
    }

    #[tokio::test]
    async fn load_tools_reports_callable_undocumented_as_already_callable() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                callable_undocumented_tool("mcp__appworld__spotify_search_songs"),
                catalog_tool("fetch_url", "Fetch a URL"),
            ],
            promoted: Mutex::default(),
            ..Default::default()
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
