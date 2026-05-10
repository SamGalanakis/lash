use std::collections::{BTreeSet, HashMap};
#[cfg(feature = "semantic-tool-search")]
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

use crate::catalog::CatalogTool;
use crate::common::{
    FUZZY_SCORE_CAP, RRF_K, SEMANTIC_CANDIDATE_FLOOR, exclude_filter, json_field, limit_from_args,
    namespace_filter, round_score, string_field, string_vec, tokenize,
};
use crate::schema_index::schema_index_text;
#[cfg(feature = "semantic-tool-search")]
use crate::schema_index::semantic_index_text;

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
pub(crate) struct RankedCandidate {
    pub(crate) idx: usize,
    pub(crate) lexical_score: f64,
    pub(crate) semantic_score: Option<f64>,
}

#[derive(Debug)]
pub struct ToolDiscoveryIndex {
    pub(crate) key: u64,
    docs: Vec<DiscoveryDoc>,
    avg_len: f64,
    doc_freq: HashMap<String, usize>,
    #[cfg(feature = "semantic-tool-search")]
    semantic_embeddings: OnceLock<Vec<Vec<f32>>>,
}

impl ToolDiscoveryIndex {
    pub(crate) fn build(key: u64, catalog: Vec<Value>) -> Self {
        let docs: Vec<DiscoveryDoc> = catalog
            .into_iter()
            .filter_map(CatalogTool::from_value)
            .filter(|tool| tool.searchable)
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

    pub(crate) fn search(&self, args: &Value) -> Vec<Value> {
        let semantic_scores = self.semantic_scores(args);
        self.search_with_semantic_scores(args, semantic_scores.as_deref())
    }

    pub(crate) fn search_with_semantic_scores(
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
            vec![
                schema_index_text(tool.raw.get("output_schema")),
                tool.compact_definition()
                    .compact_contract()
                    .render_returns(),
            ],
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

pub(crate) fn reciprocal_rank_fusion(
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
    if guard.is_none()
        && let Ok(model) = {
            let model_id = std::env::var("LASH_TOOL_SEARCH_SEMANTIC_MODEL")
                .unwrap_or_else(|_| "minishlab/potion-base-2M".to_string());
            model2vec_rs::model::StaticModel::from_pretrained(model_id, None, None, None)
        }
    {
        *guard = Some(model);
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
