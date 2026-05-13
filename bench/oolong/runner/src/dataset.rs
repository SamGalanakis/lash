use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OolongSuite {
    Synth,
    SynthWithLabels,
    Real,
}

impl OolongSuite {
    pub fn label(self) -> &'static str {
        match self {
            Self::Synth => "synth",
            Self::SynthWithLabels => "synth_with_labels",
            Self::Real => "real",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OolongQuestion {
    pub question_id: String,
    pub suite: OolongSuite,
    pub split: String,
    pub dataset: Option<String>,
    pub config: Option<String>,
    pub context_len: Option<u64>,
    pub context_window_id: Option<Value>,
    pub task_group: Option<String>,
    pub task: Option<String>,
    pub answer_type: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_bool")]
    pub input_subset: Option<bool>,
    pub prompt: String,
    pub context: String,
    pub question: String,
    pub answer: Value,
    #[serde(default)]
    pub source: Value,
}

fn deserialize_optional_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(value)),
        Some(Value::String(value)) => match value.as_str() {
            "true" | "True" | "TRUE" => Ok(Some(true)),
            "false" | "False" | "FALSE" => Ok(Some(false)),
            other => Err(serde::de::Error::custom(format!(
                "invalid boolean string `{other}`"
            ))),
        },
        Some(other) => Err(serde::de::Error::custom(format!(
            "invalid boolean value `{other}`"
        ))),
    }
}

pub fn default_dataset_path(state_root: &Path, suite: OolongSuite) -> std::path::PathBuf {
    state_root
        .join("data")
        .join(format!("oolong_{}.jsonl", suite.label()))
}

pub fn load_questions(path: &Path) -> anyhow::Result<Vec<OolongQuestion>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: OolongQuestion = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), idx + 1))?;
        out.push(row);
    }
    if out.is_empty() {
        bail!("no OOLONG questions found in {}", path.display());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_accepts_string_encoded_input_subset() {
        let row = r#"{
            "question_id": "1",
            "suite": "synth",
            "split": "validation",
            "dataset": "spam",
            "config": null,
            "context_len": 1024,
            "context_window_id": null,
            "task_group": "counting",
            "task": null,
            "answer_type": null,
            "input_subset": "False",
            "prompt": "p",
            "context": "c",
            "question": "q",
            "answer": ["spam"],
            "source": {}
        }"#;

        let question: OolongQuestion = serde_json::from_str(row).expect("question");

        assert_eq!(question.input_subset, Some(false));
    }

    #[test]
    fn question_accepts_real_suite_uuid_context_window_id() {
        let row = r#"{
            "question_id": "1",
            "suite": "real",
            "split": "test",
            "dataset": null,
            "config": "dnd",
            "context_len": null,
            "context_window_id": "29f1fec6-ebf3-378f-201d-8a20db19eecd",
            "task_group": null,
            "task": null,
            "answer_type": null,
            "input_subset": null,
            "prompt": "p",
            "context": "c",
            "question": "q",
            "answer": 114,
            "source": {}
        }"#;

        let question: OolongQuestion = serde_json::from_str(row).expect("question");

        assert_eq!(
            question.context_window_id,
            Some(Value::String(
                "29f1fec6-ebf3-378f-201d-8a20db19eecd".to_string()
            ))
        );
    }
}
