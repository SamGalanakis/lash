use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

use serde_json::{Value, json};

pub(crate) const DEFAULT_LIMIT: usize = 10;
pub(crate) const MAX_LIMIT: usize = 100;
pub(crate) const LLM_CANDIDATE_LIMIT: usize = 100;
pub(crate) const FUZZY_SCORE_CAP: f64 = 1.25;
pub(crate) const SEMANTIC_CANDIDATE_FLOOR: usize = 50;
pub(crate) const RRF_K: f64 = 60.0;

pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

pub(crate) fn limit_from_args(args: &Value) -> usize {
    args.get("limit")
        .and_then(Value::as_i64)
        .and_then(|value| usize::try_from(value).ok())
        .map(|value| value.clamp(1, MAX_LIMIT))
        .unwrap_or(DEFAULT_LIMIT)
}

pub(crate) fn args_with_limit(args: &Value, limit: usize) -> Value {
    let mut args = args.as_object().cloned().unwrap_or_default();
    args.insert("limit".to_string(), json!(limit.clamp(1, MAX_LIMIT)));
    args.insert("debug".to_string(), json!(false));
    Value::Object(args)
}

pub(crate) fn module_filter(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::String(module)) => module
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

pub(crate) fn exclude_filter(value: Option<&Value>) -> BTreeSet<String> {
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

pub(crate) fn round_score(score: f64) -> f64 {
    (score * 100.0).round() / 100.0
}

pub(crate) fn string_vec(value: Option<&Value>) -> Vec<String> {
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

pub(crate) fn catalog_key(catalog: &[Value]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    catalog.len().hash(&mut hasher);
    for value in catalog {
        hash_json_value(value, &mut hasher);
    }
    hasher.finish()
}

fn hash_json_value(value: &Value, state: &mut impl Hasher) {
    match value {
        Value::Null => 0_u8.hash(state),
        Value::Bool(value) => {
            1_u8.hash(state);
            value.hash(state);
        }
        Value::Number(value) => {
            2_u8.hash(state);
            value.to_string().hash(state);
        }
        Value::String(value) => {
            3_u8.hash(state);
            value.hash(state);
        }
        Value::Array(values) => {
            4_u8.hash(state);
            values.len().hash(state);
            for value in values {
                hash_json_value(value, state);
            }
        }
        Value::Object(values) => {
            5_u8.hash(state);
            values.len().hash(state);
            for (key, value) in values {
                key.hash(state);
                hash_json_value(value, state);
            }
        }
    }
}
