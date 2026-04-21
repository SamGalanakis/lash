use std::fs;
use std::path::Path;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

/// Subset of SWE-bench's `SWEbenchInstance` TypedDict that we care about.
/// Upstream dataset rows carry more (hints_text, created_at, …), but those
/// aren't load-bearing for predictions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweBenchInstance {
    pub repo: String,
    pub instance_id: String,
    pub base_commit: String,
    #[serde(default)]
    pub patch: String,
    #[serde(default)]
    pub test_patch: String,
    pub problem_statement: String,
    #[serde(default)]
    pub version: String,
    #[serde(default, rename = "FAIL_TO_PASS")]
    pub fail_to_pass: serde_json::Value,
    #[serde(default, rename = "PASS_TO_PASS")]
    pub pass_to_pass: serde_json::Value,
    #[serde(default)]
    pub environment_setup_commit: Option<String>,
}

pub fn load_instances(path: &Path) -> anyhow::Result<Vec<SweBenchInstance>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "jsonl" => {
            for (idx, line) in raw.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let row: SweBenchInstance = serde_json::from_str(trimmed)
                    .with_context(|| format!("parse {} line {}", path.display(), idx + 1))?;
                out.push(row);
            }
        }
        "json" => {
            let value: serde_json::Value =
                serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
            let arr = value
                .as_array()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{} is not a JSON array", path.display()))?;
            for (idx, row) in arr.into_iter().enumerate() {
                let parsed: SweBenchInstance = serde_json::from_value(row)
                    .with_context(|| format!("parse {} row {}", path.display(), idx))?;
                out.push(parsed);
            }
        }
        other => bail!(
            "unsupported dataset extension `.{other}` for {} — expected .jsonl or .json",
            path.display()
        ),
    }
    if out.is_empty() {
        bail!("no instances loaded from {}", path.display());
    }
    Ok(out)
}
