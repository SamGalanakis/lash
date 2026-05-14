use lash_core::llm::types::{LlmOutputPart, LlmResponse, LlmStreamEvent, LlmUsage};
use lash_core::testing::TestProvider;
use lash_core::{
    ToolAvailabilityConfig, ToolContract, ToolDefinition, ToolDiscoveryMetadata, ToolExecutionMode,
    ToolManifest, ToolOutputContract, ToolProvider, ToolResult,
};

use super::scenarios::RuntimePerfScenario;

const OPENAI_COMPAT_STREAM_CHUNK_COUNT: usize = 256;
const OPENAI_COMPAT_STREAM_CHUNK_BYTES: usize = 96;

pub(crate) struct BenchmarkStreamProfile {
    pub(crate) full_text: String,
    pub(crate) deltas: Vec<String>,
}

pub(crate) fn benchmark_provider(scenario: RuntimePerfScenario) -> TestProvider {
    TestProvider::builder()
        .kind("benchmark")
        .default_model("mock-model")
        .requires_streaming(true)
        .complete(move |req| async move {
            let profile = benchmark_stream_profile(scenario);
            let usage = LlmUsage {
                input_tokens: 1_024,
                output_tokens: 64,
                cached_input_tokens: 512,
                reasoning_tokens: 48,
            };
            if let Some(tx) = req.stream_events.as_ref() {
                for delta in &profile.deltas {
                    tx.send(LlmStreamEvent::Delta(delta.clone()));
                }
                tx.send(LlmStreamEvent::Usage(usage.clone()));
            }
            Ok(LlmResponse {
                full_text: profile.full_text.clone(),
                deltas: profile.deltas.clone(),
                parts: vec![LlmOutputPart::Text {
                    text: profile.full_text,
                    response_meta: None,
                }],
                usage,
                provider_usage: None,
                request_body: None,
                http_summary: None,
            })
        })
        .build()
}

pub(crate) struct BenchmarkEchoTool;
pub(crate) struct BenchmarkLargeToolSurface;

#[async_trait::async_trait]
impl ToolProvider for BenchmarkEchoTool {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![benchmark_echo_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
        (name == "benchmark_echo")
            .then(|| std::sync::Arc::new(benchmark_echo_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> ToolResult {
        if call.name != "benchmark_echo" {
            return ToolResult::err_fmt(format_args!("Unknown benchmark tool: {}", call.name));
        }
        tokio::task::yield_now().await;
        ToolResult::ok(serde_json::json!({
            "value": call.args.get("value").cloned().unwrap_or(serde_json::Value::Null),
            "ordinal": call.args.get("ordinal").cloned().unwrap_or(serde_json::Value::Null),
        }))
    }
}

fn benchmark_echo_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "benchmark_echo",
        "Return the input payload with a tiny async yield for runtime profiling.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": ["string", "number", "boolean", "object", "array", "null"] },
                "ordinal": { "type": "integer" }
            },
            "additionalProperties": true
        }),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
}

#[async_trait::async_trait]
impl ToolProvider for BenchmarkLargeToolSurface {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
        self.tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| std::sync::Arc::new(tool.contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> ToolResult {
        if !GMAIL_LIKE_TOOL_NAMES.contains(&call.name) {
            return ToolResult::err_fmt(format_args!("Unknown benchmark tool: {}", call.name));
        }
        tokio::task::yield_now().await;
        ToolResult::ok(serde_json::json!({
            "tool": call.name,
            "ok": true,
            "echo": call.args,
        }))
    }
}

impl BenchmarkLargeToolSurface {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        GMAIL_LIKE_TOOL_NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| gmail_like_tool_definition(index, name))
            .collect()
    }
}

fn gmail_like_tool_definition(index: usize, name: &str) -> ToolDefinition {
    let mut definition = ToolDefinition::raw(
        name,
        gmail_like_tool_description(index, name),
        gmail_like_input_schema(name),
        gmail_like_output_schema(name),
    )
    .with_availability(ToolAvailabilityConfig::callable())
    .with_examples(vec![
        format!(
            r#"call {name} {{ user_id: "me", message_id: "msg_123", payload: {{ label_ids: ["INBOX", "IMPORTANT"] }} }}"#
        ),
        format!(
            r#"call {name} {{ user_id: "me", query: "from:alerts@example.com newer_than:7d", limit: 25 }}"#
        ),
    ])
    .with_discovery(ToolDiscoveryMetadata {
        namespace: Some("gmail".to_string()),
        aliases: vec![
            name.trim_start_matches("GMAIL_")
                .to_ascii_lowercase()
                .replace('_', " "),
        ],
    })
    .with_execution_mode(ToolExecutionMode::Parallel);

    if index.is_multiple_of(7) {
        definition.output_contract = ToolOutputContract::from_input_schema(
            "projection",
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "status": { "type": "string" }
                },
                "additionalProperties": true
            })),
        );
    }

    definition
}

fn gmail_like_tool_description(index: usize, name: &str) -> String {
    let verb = name
        .trim_start_matches("GMAIL_")
        .to_ascii_lowercase()
        .replace('_', " ");
    format!(
        "Synthetic Gmail toolkit operation #{index}: {verb}. Mirrors a real provider action with OAuth-scoped Gmail semantics, mailbox resource identifiers, optional label/filter/thread payloads, and structured result projection metadata. The description is intentionally long enough to exercise prompt-side tool documentation, compact catalog rendering, and RLM callable surface construction for provider-sized toolkits."
    )
}

fn gmail_like_input_schema(name: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "$defs": {
            "email_address": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Display name for the mailbox participant." },
                    "email": { "type": "string", "format": "email", "description": "RFC 5322 email address." }
                },
                "required": ["email"],
                "additionalProperties": false
            },
            "message_part": {
                "type": "object",
                "properties": {
                    "mime_type": { "type": "string" },
                    "filename": { "type": "string" },
                    "headers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "value": { "type": "string" }
                            },
                            "required": ["name", "value"],
                            "additionalProperties": false
                        }
                    },
                    "body": {
                        "type": "object",
                        "properties": {
                            "size": { "type": "integer" },
                            "data": { "type": "string" },
                            "attachment_id": { "type": "string" }
                        },
                        "additionalProperties": false
                    },
                    "parts": {
                        "type": "array",
                        "items": { "$ref": "#/$defs/message_part" }
                    }
                },
                "additionalProperties": false
            },
            "label_mutation": {
                "type": "object",
                "properties": {
                    "add_label_ids": { "type": "array", "items": { "type": "string" }, "maxItems": 50 },
                    "remove_label_ids": { "type": "array", "items": { "type": "string" }, "maxItems": 50 }
                },
                "additionalProperties": false
            }
        },
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Gmail user id. Use `me` for the authenticated user.",
                "default": "me"
            },
            "message_id": {
                "type": "string",
                "description": "Gmail message, thread, draft, label, filter, or settings resource id."
            },
            "thread_id": { "type": "string" },
            "query": {
                "type": "string",
                "description": "Search query, label expression, email address, or filter criteria."
            },
            "projection": {
                "description": "Optional output type witness for tools whose response should be compacted to selected fields.",
                "anyOf": [
                    { "type": "string", "enum": ["summary", "full", "ids_only"] },
                    {
                        "type": "object",
                        "properties": {
                            "fields": { "type": "array", "items": { "type": "string" } },
                            "include_headers": { "type": "boolean" },
                            "include_body": { "type": "boolean" }
                        },
                        "additionalProperties": false
                    }
                ]
            },
            "payload": {
                "description": "Operation-specific Gmail request payload.",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "raw": { "type": "string", "description": "Base64url encoded RFC 2822 message." },
                            "subject": { "type": "string" },
                            "body": {
                                "type": "object",
                                "properties": {
                                    "plain": { "type": "string" },
                                    "html": { "type": "string" }
                                },
                                "additionalProperties": false
                            },
                            "to": { "type": "array", "items": { "$ref": "#/$defs/email_address" } },
                            "cc": { "type": "array", "items": { "$ref": "#/$defs/email_address" } },
                            "bcc": { "type": "array", "items": { "$ref": "#/$defs/email_address" } },
                            "attachments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "filename": { "type": "string" },
                                        "mime_type": { "type": "string" },
                                        "content_base64": { "type": "string" },
                                        "document_id": { "type": "string" }
                                    },
                                    "additionalProperties": false
                                }
                            }
                        },
                        "additionalProperties": false
                    },
                    { "$ref": "#/$defs/label_mutation" },
                    {
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "object",
                                "properties": {
                                    "from": { "type": "string" },
                                    "to": { "type": "string" },
                                    "subject": { "type": "string" },
                                    "has_attachment": { "type": "boolean" },
                                    "query": { "type": "string" }
                                },
                                "additionalProperties": false
                            },
                            "action": { "$ref": "#/$defs/label_mutation" }
                        },
                        "additionalProperties": false
                    },
                    { "$ref": "#/$defs/message_part" }
                ]
            },
            "page_token": { "type": "string" },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum number of records to inspect or mutate."
            },
            "include_spam_trash": {
                "type": "boolean",
                "description": "Whether to include spam and trash folders when the Gmail API supports it."
            },
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "visibility": { "type": "string", "enum": ["labelShow", "labelHide", "labelShowIfUnread"] },
                        "color": {
                            "type": "object",
                            "properties": {
                                "text_color": { "type": "string" },
                                "background_color": { "type": "string" }
                            },
                            "additionalProperties": false
                        }
                    },
                    "additionalProperties": false
                }
            },
            "operation": {
                "type": "string",
                "const": name
            }
        },
        "required": ["user_id"],
        "additionalProperties": false
    })
}

fn gmail_like_output_schema(_name: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": { "type": "string" },
            "status": { "type": "string", "enum": ["ok", "queued", "partial", "not_modified"] },
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "thread_id": { "type": "string" },
                        "label_ids": { "type": "array", "items": { "type": "string" } },
                        "snippet": { "type": "string" },
                        "payload": {
                            "type": "object",
                            "properties": {
                                "mime_type": { "type": "string" },
                                "headers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": { "type": "string" },
                                            "value": { "type": "string" }
                                        },
                                        "additionalProperties": false
                                    }
                                },
                                "body": { "type": "object", "additionalProperties": true },
                                "parts": {
                                    "type": "array",
                                    "items": { "type": "object", "additionalProperties": true }
                                }
                            },
                            "additionalProperties": false
                        }
                    },
                    "additionalProperties": false
                }
            },
            "thread": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "history_id": { "type": "string" },
                    "messages": { "type": "array", "items": { "type": "object", "additionalProperties": true } }
                },
                "additionalProperties": false
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "next_page_token": { "type": "string" },
                    "result_size_estimate": { "type": "integer" },
                    "rate_limit": {
                        "type": "object",
                        "properties": {
                            "remaining": { "type": "integer" },
                            "reset_at": { "type": "string", "format": "date-time" }
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": true
            }
        },
        "additionalProperties": true
    })
}

const GMAIL_LIKE_TOOL_NAMES: &[&str] = &[
    "GMAIL_ADD_LABEL_TO_EMAIL",
    "GMAIL_BATCH_DELETE_MESSAGES",
    "GMAIL_BATCH_MODIFY_MESSAGES",
    "GMAIL_CREATE_EMAIL_DRAFT",
    "GMAIL_CREATE_FILTER",
    "GMAIL_CREATE_LABEL",
    "GMAIL_CREATE_PROMPT_POST",
    "GMAIL_DELETE_DRAFT",
    "GMAIL_DELETE_FILTER",
    "GMAIL_DELETE_LABEL",
    "GMAIL_DELETE_MESSAGE",
    "GMAIL_DELETE_THREAD",
    "GMAIL_FETCH_EMAILS",
    "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
    "GMAIL_FETCH_MESSAGE_BY_THREAD_ID",
    "GMAIL_FORWARD_MESSAGE",
    "GMAIL_GET_ATTACHMENT",
    "GMAIL_GET_AUTO_FORWARDING",
    "GMAIL_GET_CONTACTS",
    "GMAIL_GET_DRAFT",
    "GMAIL_GET_FILTER",
    "GMAIL_GET_LABEL",
    "GMAIL_GET_LANGUAGE_SETTINGS",
    "GMAIL_GET_PEOPLE",
    "GMAIL_GET_PROFILE",
    "GMAIL_GET_VACATION_SETTINGS",
    "GMAIL_IMPORT_MESSAGE",
    "GMAIL_INSERT_MESSAGE",
    "GMAIL_LIST_CSE_IDENTITIES",
    "GMAIL_LIST_CSE_KEYPAIRS",
    "GMAIL_LIST_DRAFTS",
    "GMAIL_LIST_FILTERS",
    "GMAIL_LIST_FORWARDING_ADDRESSES",
    "GMAIL_LIST_HISTORY",
    "GMAIL_LIST_LABELS",
    "GMAIL_LIST_MESSAGES",
    "GMAIL_LIST_SEND_AS",
    "GMAIL_LIST_SMIME_INFO",
    "GMAIL_LIST_THREADS",
    "GMAIL_MODIFY_THREAD_LABELS",
    "GMAIL_MOVE_THREAD_TO_TRASH",
    "GMAIL_MOVE_TO_TRASH",
    "GMAIL_PATCH_LABEL",
    "GMAIL_PATCH_SEND_AS",
    "GMAIL_REMOVE_LABEL",
    "GMAIL_REPLY_TO_THREAD",
    "GMAIL_SEARCH_PEOPLE",
    "GMAIL_SEND_DRAFT",
    "GMAIL_SEND_EMAIL",
    "GMAIL_SETTINGS_GET_IMAP",
    "GMAIL_SETTINGS_GET_POP",
    "GMAIL_SETTINGS_SEND_AS_GET",
    "GMAIL_STOP_WATCH",
    "GMAIL_UNTRASH_MESSAGE",
    "GMAIL_UNTRASH_THREAD",
    "GMAIL_UPDATE_DRAFT",
    "GMAIL_UPDATE_IMAP_SETTINGS",
    "GMAIL_UPDATE_LABEL",
    "GMAIL_UPDATE_LANGUAGE_SETTINGS",
    "GMAIL_UPDATE_POP_SETTINGS",
    "GMAIL_UPDATE_SEND_AS",
    "GMAIL_UPDATE_USER_ATTRIBUTES_VALUES",
    "GMAIL_UPDATE_VACATION_SETTINGS",
];

pub(crate) fn benchmark_stream_profile(scenario: RuntimePerfScenario) -> BenchmarkStreamProfile {
    match scenario {
        RuntimePerfScenario::OpenAiCompatStream => {
            let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
            let mut deltas = Vec::with_capacity(OPENAI_COMPAT_STREAM_CHUNK_COUNT + 1);
            for index in 0..OPENAI_COMPAT_STREAM_CHUNK_COUNT {
                let prefix = format!("chunk-{index:03}: ");
                let fill_len = OPENAI_COMPAT_STREAM_CHUNK_BYTES.saturating_sub(prefix.len() + 1);
                let body: String = alphabet
                    .chars()
                    .cycle()
                    .skip(index % alphabet.len())
                    .take(fill_len)
                    .collect();
                deltas.push(format!("{prefix}{body}\n"));
            }
            deltas.push("runtime perf benchmark ok".to_string());
            BenchmarkStreamProfile {
                full_text: deltas.concat(),
                deltas,
            }
        }
        RuntimePerfScenario::Rlm
        | RuntimePerfScenario::RlmGlobals
        | RuntimePerfScenario::RlmLargeToolSurface
        | RuntimePerfScenario::EmbedRlm => {
            let text = "```lashlang\nsubmit \"runtime perf benchmark ok\"\n```".to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
        RuntimePerfScenario::RlmToolCalls => {
            let text = r#"```lashlang
fanout = parallel {
  a: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 1 }
  b: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 2 }
  c: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 3 }
  d: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 4 }
}
first = fanout.a?
submit first.value
```"#
                .to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
        _ => {
            let text = "runtime perf benchmark ok".to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{ToolAvailability, ToolSurfaceBuildInput, build_tool_surface};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn large_tool_surface_fixture_matches_gmail_sized_callable_catalog() {
        let defs = BenchmarkLargeToolSurface.tool_definitions();
        assert_eq!(defs.len(), 63);
        assert!(defs.iter().all(|def| {
            def.availability.standard == ToolAvailability::Callable
                && def.discovery.namespace.as_deref() == Some("gmail")
                && !def.input_schema["properties"]
                    .as_object()
                    .expect("object schema")
                    .is_empty()
        }));
        assert!(
            defs.iter().any(|def| !def.output_contract.is_static()),
            "fixture should cover dynamic output contracts"
        );
        let first = defs.first().expect("fixture tool");
        assert!(
            first.input_schema["$defs"]["message_part"]["properties"]["parts"]["items"]["$ref"]
                .as_str()
                == Some("#/$defs/message_part"),
            "fixture should include recursive nested schema refs"
        );
        assert!(
            first.input_schema["properties"]["payload"]["oneOf"]
                .as_array()
                .is_some_and(|variants| variants.len() >= 4),
            "fixture should include provider-style payload unions"
        );
        assert!(
            first.input_schema["properties"]["projection"]["anyOf"]
                .as_array()
                .is_some_and(|variants| variants.len() >= 2),
            "fixture should include output projection unions"
        );
    }

    #[test]
    fn rlm_large_tool_surface_does_not_resolve_nested_schema_contracts_without_tool_calls() {
        let provider = BenchmarkLargeToolSurface;
        let definitions = provider.tool_definitions();
        let manifests = definitions
            .iter()
            .map(|definition| definition.manifest())
            .collect::<Vec<_>>();
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let resolver_count = Arc::clone(&contract_resolutions);

        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools: manifests,
            mode: lash_core::ExecutionMode::new("rlm"),
            resolve_contract: Some(Arc::new(move |name| {
                resolver_count.fetch_add(1, Ordering::SeqCst);
                definitions
                    .iter()
                    .find(|definition| definition.name == name)
                    .map(|definition| Arc::new(definition.contract()))
            })),
            contributions: Vec::new(),
        });

        assert_eq!(surface.callable_tools().len(), GMAIL_LIKE_TOOL_NAMES.len());
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 0);
        assert_eq!(surface.prompt_tool_docs(), "");
    }
}
