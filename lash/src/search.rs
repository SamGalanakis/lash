use std::collections::{HashMap, HashSet};

use regex::Regex;

const DEFAULT_LIMIT: usize = 10;
const MAX_LIMIT: usize = 100;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchMode {
    Hybrid,
    Literal,
    Regex,
}

impl SearchMode {
    pub(crate) fn parse(value: Option<&str>) -> Self {
        match value
            .unwrap_or("hybrid")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "literal" => Self::Literal,
            "regex" => Self::Regex,
            _ => Self::Hybrid,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SearchDoc {
    pub(crate) fields: HashMap<&'static str, String>,
}

pub(crate) fn limit_from_args(args: &serde_json::Value) -> usize {
    args.get("limit")
        .and_then(|v| v.as_i64())
        .and_then(|n| usize::try_from(n).ok())
        .map(|n| n.clamp(1, MAX_LIMIT))
        .unwrap_or(DEFAULT_LIMIT)
}

fn compile_regex(pattern: Option<&str>) -> Option<Regex> {
    let raw = pattern.unwrap_or_default();
    if raw.is_empty() {
        return None;
    }
    match Regex::new(&format!("(?i){raw}")) {
        Ok(re) => Some(re),
        Err(_) => {
            let escaped = regex::escape(raw);
            Regex::new(&format!("(?i){escaped}")).ok()
        }
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn bm25_scores(
    query_tokens: &[String],
    docs: &[SearchDoc],
    field_weights: &[(&'static str, f64)],
) -> Vec<f64> {
    let n_docs = docs.len();
    if n_docs == 0 {
        return Vec::new();
    }

    let mut doc_tfs: Vec<HashMap<String, f64>> = Vec::with_capacity(n_docs);
    let mut doc_lens: Vec<f64> = Vec::with_capacity(n_docs);
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for doc in docs {
        let mut tf: HashMap<String, f64> = HashMap::new();
        let mut dlen = 0.0_f64;
        for (field, weight) in field_weights {
            if *weight <= 0.0 {
                continue;
            }
            let text = doc
                .fields
                .get(field)
                .map(String::as_str)
                .unwrap_or_default();
            let tokens = tokenize(text);
            if tokens.is_empty() {
                continue;
            }
            let mut counts: HashMap<String, usize> = HashMap::new();
            for tok in &tokens {
                *counts.entry(tok.clone()).or_insert(0) += 1;
            }
            for (tok, count) in counts {
                *tf.entry(tok).or_insert(0.0) += (count as f64) * *weight;
            }
            dlen += (tokens.len() as f64) * *weight;
        }
        for tok in tf.keys() {
            *doc_freq.entry(tok.clone()).or_insert(0) += 1;
        }
        doc_tfs.push(tf);
        doc_lens.push(dlen);
    }

    let avgdl = {
        let sum: f64 = doc_lens.iter().sum();
        let avg = sum / (n_docs as f64);
        if avg <= 0.0 { 1.0 } else { avg }
    };

    let mut qtf: HashMap<String, usize> = HashMap::new();
    for tok in query_tokens {
        *qtf.entry(tok.clone()).or_insert(0) += 1;
    }

    let k1 = 1.5_f64;
    let b = 0.75_f64;
    let mut scores = vec![0.0_f64; n_docs];
    for (i, tf) in doc_tfs.iter().enumerate() {
        let dl = doc_lens[i];
        let norm = 1.0 - b + b * (dl / avgdl);
        for (tok, qcount) in &qtf {
            let freq = *tf.get(tok).unwrap_or(&0.0);
            if freq <= 0.0 {
                continue;
            }
            let df = *doc_freq.get(tok).unwrap_or(&0) as f64;
            let idf = ((n_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();
            let denom = freq + k1 * norm;
            if denom <= 0.0 {
                continue;
            }
            let term = idf * ((freq * (k1 + 1.0)) / denom);
            scores[i] += term * (1.0 + (*qcount as f64).ln());
        }
    }

    scores
}

fn field_hits(
    fields: &HashMap<&'static str, String>,
    query: &str,
    mode: SearchMode,
    regex_filter: Option<&Regex>,
) -> Vec<String> {
    let query_lower = query.to_ascii_lowercase();
    let query_tokens: HashSet<String> = tokenize(query).into_iter().collect();
    let mut hits = Vec::new();
    for (field, value) in fields {
        if value.is_empty() {
            continue;
        }
        let value_lower = value.to_ascii_lowercase();
        let mut hit = match mode {
            SearchMode::Regex => regex_filter.is_some_and(|re| re.is_match(value)),
            SearchMode::Literal => !query.is_empty() && value_lower.contains(&query_lower),
            SearchMode::Hybrid => {
                if !query_tokens.is_empty() {
                    let tokens: HashSet<String> = tokenize(value).into_iter().collect();
                    query_tokens.iter().any(|t| tokens.contains(t))
                } else if query.is_empty() {
                    true
                } else {
                    value_lower.contains(&query_lower)
                }
            }
        };
        if hit && regex_filter.is_some() && mode != SearchMode::Regex {
            hit = regex_filter.is_some_and(|re| re.is_match(value));
        }
        if hit {
            hits.push((*field).to_string());
        }
    }
    hits
}

pub(crate) fn rank_docs(
    docs: &[SearchDoc],
    query: &str,
    mode: SearchMode,
    regex: Option<&str>,
    field_weights: &[(&'static str, f64)],
) -> Vec<(usize, f64, Vec<String>)> {
    let query_tokens = tokenize(query);
    let query_lower = query.to_ascii_lowercase();
    let mut scores = vec![0.0_f64; docs.len()];
    if mode == SearchMode::Hybrid && !query_tokens.is_empty() {
        scores = bm25_scores(&query_tokens, docs, field_weights);
    }
    let regex_filter = match mode {
        SearchMode::Regex => compile_regex(regex.or(Some(query))),
        _ => compile_regex(regex),
    };

    let mut indices: Vec<usize> = (0..docs.len()).collect();
    if mode == SearchMode::Hybrid {
        indices.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
    }

    let mut ranked = Vec::new();
    for idx in indices {
        let haystack = docs[idx]
            .fields
            .values()
            .filter(|value| !value.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let haystack_lower = haystack.to_ascii_lowercase();

        let mut include = match mode {
            SearchMode::Regex => regex_filter
                .as_ref()
                .is_some_and(|re| re.is_match(&haystack)),
            SearchMode::Literal => !query.is_empty() && haystack_lower.contains(&query_lower),
            SearchMode::Hybrid => {
                if query.is_empty() {
                    true
                } else if !query_tokens.is_empty() {
                    scores[idx] > 0.0 || haystack_lower.contains(&query_lower)
                } else {
                    haystack_lower.contains(&query_lower)
                }
            }
        };
        if include && regex_filter.is_some() && mode != SearchMode::Regex {
            include = regex_filter
                .as_ref()
                .is_some_and(|re| re.is_match(&haystack));
        }
        if !include {
            continue;
        }

        let hits = field_hits(&docs[idx].fields, query, mode, regex_filter.as_ref());
        if hits.is_empty() && !(query.is_empty() && mode != SearchMode::Regex) {
            continue;
        }
        ranked.push((idx, scores[idx], hits));
    }
    ranked
}

pub(crate) fn truncate_preview(text: &str, limit: usize) -> String {
    let compact = text.trim().replace('\n', " ");
    if compact.len() <= limit {
        compact
    } else {
        format!("{}...", &compact[..limit.saturating_sub(3)])
    }
}
