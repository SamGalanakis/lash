use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

pub const DOMAINS: &[&str] = &["logic", "cs", "chemistry", "chess", "math"];
pub const DIFFICULTIES: &[&str] = &["easy", "medium", "hard"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LongCoTQuestion {
    pub question_id: String,
    pub domain: String,
    pub difficulty: String,
    pub prompt: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<serde_json::Value>,
}

pub fn select_difficulties(difficulty: Option<&str>) -> Vec<&'static str> {
    match difficulty {
        None => DIFFICULTIES.to_vec(),
        Some("longcot") => vec!["medium", "hard"],
        Some("longcot-mini") => vec!["easy"],
        Some(d) if DIFFICULTIES.contains(&d) => {
            vec![DIFFICULTIES.iter().copied().find(|x| *x == d).unwrap()]
        }
        Some(other) => panic!("unsupported difficulty: {other}"),
    }
}

pub fn load_questions(
    data_dir: &Path,
    domains: &[String],
    difficulty: Option<&str>,
) -> anyhow::Result<Vec<LongCoTQuestion>> {
    let difficulties = select_difficulties(difficulty);
    let selected_domains: Vec<&str> = if domains.is_empty() {
        DOMAINS.to_vec()
    } else {
        domains.iter().map(String::as_str).collect()
    };

    let mut out = Vec::new();
    for dom in &selected_domains {
        for diff in &difficulties {
            let path: PathBuf = data_dir.join(dom).join(format!("{diff}.json"));
            if !path.exists() {
                continue;
            }
            let raw =
                fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
            let value: serde_json::Value =
                serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
            let arr = value
                .get("questions")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            for row in arr {
                let qid = row
                    .get("question_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let prompt = row
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let (Some(qid), Some(prompt)) = (qid, prompt) else {
                    continue;
                };
                out.push(LongCoTQuestion {
                    question_id: qid,
                    domain: (*dom).to_string(),
                    difficulty: (*diff).to_string(),
                    prompt,
                    template: row
                        .get("template")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    answer: row.get("answer").cloned(),
                    problem: row.get("problem").cloned(),
                });
            }
        }
    }
    if out.is_empty() {
        bail!(
            "no questions found under {} for domains={:?} difficulty={:?}",
            data_dir.display(),
            selected_domains,
            difficulty
        );
    }
    Ok(out)
}
