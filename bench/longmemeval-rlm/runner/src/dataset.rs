use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LongMemEvalTurn {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LongMemEvalQuestion {
    pub question_id: String,
    pub question: String,
    #[serde(default)]
    pub answer: Option<serde_json::Value>,
    #[serde(default)]
    pub question_type: Option<String>,
    #[serde(default)]
    pub question_date: Option<String>,
    #[serde(default)]
    pub haystack_dates: Vec<String>,
    #[serde(default)]
    pub haystack_session_ids: Vec<String>,
    #[serde(default)]
    pub haystack_sessions: Vec<Vec<LongMemEvalTurn>>,
}

pub fn load_questions(path: &Path) -> anyhow::Result<Vec<LongMemEvalQuestion>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read dataset {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parse dataset {}", path.display()))
}
