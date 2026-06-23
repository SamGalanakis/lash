use std::collections::{BTreeSet, HashMap};
#[cfg(feature = "semantic-tool-search")]
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

use super::catalog::CatalogTool;
#[cfg(feature = "lashlang")]
use super::common::module_filter;
use super::common::{
    FUZZY_SCORE_CAP, RRF_K, SEMANTIC_CANDIDATE_FLOOR, exclude_filter, limit_from_args, round_score,
    tokenize,
};
#[cfg(feature = "semantic-tool-search")]
use super::schema_index::semantic_index_text;

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

#[derive(Clone, Copy, Debug, Default)]
struct MatchedFields {
    any: bool,
    params_or_input_fields: bool,
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
    pub(crate) fn build(key: u64, catalog: &[Value]) -> Self {
        let docs: Vec<DiscoveryDoc> = catalog
            .iter()
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
            if query.is_empty() || matched_fields.any {
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
        push_field(&mut fields, "id", vec![tool.id.clone()], 4.0, true);
        push_field(&mut fields, "name", vec![tool.name.clone()], 9.0, true);
        #[cfg(feature = "lashlang")]
        push_field(&mut fields, "call", vec![tool.call.clone()], 9.0, true);
        #[cfg(feature = "lashlang")]
        push_field(
            &mut fields,
            "module",
            vec![tool.module_path.join(".")],
            3.0,
            true,
        );
        #[cfg(feature = "lashlang")]
        push_field(&mut fields, "aliases", tool.aliases.clone(), 8.0, true);
        push_field(
            &mut fields,
            "description",
            vec![tool.contract.description.clone()],
            1.8,
            false,
        );
        push_field(
            &mut fields,
            "params",
            vec![tool.contract.signature.clone()],
            0.3,
            false,
        );
        push_field(
            &mut fields,
            "input_fields",
            vec![compact_values_index_text(&tool.contract.parameters)],
            0.9,
            false,
        );
        push_field(
            &mut fields,
            "output_fields",
            vec![
                compact_values_index_text(&tool.contract.return_fields),
                tool.contract.render_returns(),
            ],
            2.4,
            false,
        );
        push_field(
            &mut fields,
            "examples",
            tool.contract.examples.clone(),
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
        #[cfg(feature = "lashlang")]
        {
            let modules = module_filter(args.get("module"));
            if !modules.is_empty()
                && !self
                    .tool
                    .module_path
                    .iter()
                    .any(|segment| modules.iter().any(|candidate| candidate == segment))
                && !modules
                    .iter()
                    .any(|candidate| candidate == &self.tool.module_path.join("."))
            {
                return false;
            }
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

fn compact_values_index_text(values: &[Value]) -> String {
    values
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n")
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
    matched_fields: &MatchedFields,
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
    if name_hits + alias_hits == query_tokens.len() {
        score += 4.0;
    }

    let input_only = primary_hits == 0 && output_hits == 0 && matched_fields.params_or_input_fields;
    if input_only {
        score *= 0.35;
    }

    score
}

fn exact_field_token_hits(query_tokens: &[String], doc: &DiscoveryDoc, field_name: &str) -> usize {
    doc.fields
        .iter()
        .filter(|field| field.name == field_name)
        .flat_map(|field| field.tokens.iter())
        .filter(|token| query_tokens.iter().any(|query| query == *token))
        .count()
}

fn matched_fields(query_tokens: &[String], doc: &DiscoveryDoc) -> MatchedFields {
    let mut hits = MatchedFields::default();
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
            hits.any = true;
            hits.params_or_input_fields |= matches!(field.name, "params" | "input_fields");
        }
    }
    hits
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{ToolContract, ToolDefinition};
    use lash_tool_support::{LashlangToolBinding, ToolDefinitionLashlangExt};
    use serde_json::{Value, json};

    fn catalog_tool(name: &str, description: &str) -> Value {
        catalog_tool_with_metadata(name, description, Some("tools"), Vec::new())
    }

    fn catalog_tool_with_metadata(
        name: &str,
        description: &str,
        module: Option<&str>,
        aliases: Vec<&str>,
    ) -> Value {
        let tool = ToolDefinition::raw(
            format!("tool:test/{name}"),
            name,
            description,
            ToolContract::default_input_schema(),
            json!({}),
        )
        .with_lashlang_binding(
            LashlangToolBinding::new(
                [module.unwrap_or(match name {
                    "read_file" => "files",
                    "search_web" => "web",
                    _ => "tools",
                })],
                match name {
                    "read_file" => "read",
                    "search_web" => "search",
                    _ => name,
                },
            )
            .with_aliases(aliases),
        );
        catalog_tool_from_definition(tool)
    }

    fn catalog_tool_from_definition(tool: ToolDefinition) -> Value {
        // Mirror the flat `project_tool_catalog` shape: id/name/description/
        // bindings/activation/contract. The Lashlang call-path is derived from
        // the `lashlang.tool` binding by `CatalogTool::from_value`.
        let manifest = tool.manifest();
        json!({
            "id": manifest.id,
            "name": manifest.name,
            "description": manifest.description,
            "bindings": manifest.bindings,
            "activation": manifest.activation,
            "contract": manifest.compact_contract.clone().expect("compact contract"),
        })
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

    #[test]
    fn exact_name_beats_fuzzy_typo() {
        let index = ToolDiscoveryIndex::build(
            1,
            &[
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
    fn ranking_prefers_output_fields_over_input_filter_matches() {
        let filter_songs = ToolDefinition::raw(
            "mcp:appworld/mcp__appworld__spotify_filter_songs",
            "mcp__appworld__spotify_filter_songs",
            "Search Spotify songs by filters.",
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
        )
        .with_lashlang_binding(
            LashlangToolBinding::new(["appworld"], "spotify_filter_songs")
                .with_aliases(["spotify_filter_songs"]),
        );
        let show_song = ToolDefinition::raw(
            "mcp:appworld/mcp__appworld__spotify_show_song",
            "mcp__appworld__spotify_show_song",
            "Get a Spotify song record.",
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "song_id": {"type": "integer"}
                },
                "required": ["access_token", "song_id"]
            }),
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
        )
        .with_lashlang_binding(
            LashlangToolBinding::new(["appworld"], "spotify_show_song")
                .with_aliases(["spotify_show_song", "song_details"]),
        );

        let index = ToolDiscoveryIndex::build(
            1,
            &[
                catalog_tool_from_definition(filter_songs),
                catalog_tool_from_definition(show_song),
            ],
        );
        let results = index.search(&json!({
            "query": "play_count genre title",
            "module": "appworld"
        }));

        assert_eq!(
            results[0]["name"],
            json!("mcp__appworld__spotify_show_song")
        );
    }

    #[test]
    fn search_results_include_compact_schema_parameter_restrictions() {
        let spotify = ToolDefinition::raw(
            "mcp:appworld/mcp__appworld__spotify_search_songs",
            "mcp__appworld__spotify_search_songs",
            "Find songs",
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
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "description": "Matched songs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "play_count": {
                                    "type": "integer",
                                    "description": "Number of times the song was played.",
                                    "minimum": 0
                                },
                                "song_id": {
                                    "type": "integer",
                                    "description": "Stable song identifier."
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Song title."
                                }
                            },
                            "required": ["play_count", "song_id", "title"]
                        }
                    }
                },
                "required": ["response"]
            }),
        )
        .with_examples(vec![
            "search songs by genre".to_string(),
            "search songs by play count".to_string(),
        ])
        .with_lashlang_binding(
            LashlangToolBinding::new(["appworld"], "spotify_search_songs")
                .with_aliases(["spotify_search_songs"]),
        );
        let index = ToolDiscoveryIndex::build(1, &[catalog_tool_from_definition(spotify)]);

        let results = index.search(&json!({ "query": "spotify" }));
        let signature = results[0]["signature"].as_str().expect("signature");
        assert!(signature.contains("page_limit?: int >= 1 <= 20 = 20"));
        assert!(signature.contains("response[].play_count: int >= 0"));
        assert_eq!(
            results[0]["examples"],
            json!(["search songs by genre", "search songs by play count"])
        );
        assert!(results[0].get("input_schema").is_none());
        assert!(results[0].get("output_schema").is_none());
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
    fn ranked_names_extracts_result_names() {
        let index = ToolDiscoveryIndex::build(
            1,
            &[
                catalog_tool_with_metadata("read_file", "Read file contents", None, vec!["cat"]),
                catalog_tool_with_metadata("search_web", "Search the web", None, vec!["web"]),
            ],
        );

        assert_eq!(
            ranked_names(&index.search(&json!({ "query": "read file" }))),
            vec!["read_file"]
        );
    }
}
