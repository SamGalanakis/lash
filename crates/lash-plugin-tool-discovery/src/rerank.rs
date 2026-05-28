use std::collections::{BTreeMap, BTreeSet};

use lash_core::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
};
use serde_json::{Value, json};

use crate::common::DEFAULT_LLM_RERANK_MODEL;

pub(crate) fn llm_rerank_request(
    args: &Value,
    candidates: &[Value],
    limit: usize,
) -> DirectRequest {
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
        generation: lash_core::GenerationOptions::default(),
        session_id: None,
        caused_by: None,
                    replay: None,
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
        "module": args.get("module").cloned().unwrap_or(Value::Null),
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
        "description": candidate.get("description").cloned().unwrap_or(Value::Null),
        "examples": candidate.get("examples").cloned().unwrap_or(Value::Null),
    })
}

pub(crate) fn parse_llm_tool_names(text: &str) -> Result<Vec<String>, serde_json::Error> {
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

pub(crate) fn merge_llm_selection(
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{DirectOutputSpec, DirectPart};

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
    fn llm_rerank_request_uses_structured_name_enum_schema() {
        let candidates = vec![
            json!({"name": "read_file", "signature": "await files.read({ path: str })? -> str", "description": "Read file"}),
            json!({"name": "search_web", "signature": "await web.search({ query: str })? -> record", "description": "Search web"}),
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

    #[test]
    fn llm_rerank_prompt_excludes_filtered_candidates() {
        let candidates = vec![
            json!({"name": "search_web", "signature": "await web.search({ query: str })? -> record"}),
        ];
        let request = llm_rerank_request(
            &json!({"query": "", "exclude": ["read_file"]}),
            &candidates,
            1,
        );

        assert!(
            request
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
    }
}
