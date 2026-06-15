use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use lash_core::llm::types::{
    LlmContentBlock, LlmOutputPart, LlmOutputSpec, LlmRequest, LlmResponse, LlmStreamEvent,
    LlmUsage,
};
use lash_core::testing::TestProvider;
use lash_core::{
    LashlangToolBinding, Resolution, RuntimeEffectController, ToolAvailabilityConfig, ToolContract,
    ToolDefinition, ToolManifest, ToolOutputContract, ToolProvider, ToolResult, ToolScheduling,
};

use super::scenarios::RuntimePerfScenario;

const OPENAI_COMPAT_STREAM_CHUNK_COUNT: usize = 256;
const OPENAI_COMPAT_STREAM_CHUNK_BYTES: usize = 96;

pub(crate) struct BenchmarkStreamProfile {
    pub(crate) full_text: String,
    pub(crate) deltas: Vec<String>,
    pub(crate) parts: Vec<LlmOutputPart>,
}

pub(crate) fn benchmark_provider(scenario: RuntimePerfScenario) -> TestProvider {
    TestProvider::builder()
        .kind("benchmark")
        .serialize_config(move || {
            serde_json::json!({
                "scenario": scenario.name(),
            })
        })
        .requires_streaming(true)
        .complete(move |req| async move {
            let profile = benchmark_stream_profile_for_request(scenario, &req);
            let usage = LlmUsage {
                input_tokens: 1_024,
                output_tokens: 64,
                cached_input_tokens: 512,
                reasoning_tokens: 48,
            };
            if let Some(tx) = req.stream_events.as_ref() {
                if profile.deltas.is_empty() {
                    for part in &profile.parts {
                        tx.send(LlmStreamEvent::Part(part.clone()));
                    }
                } else {
                    for delta in &profile.deltas {
                        tx.send(LlmStreamEvent::Delta(delta.clone()));
                    }
                }
                tx.send(LlmStreamEvent::Usage(usage.clone()));
            }
            let parts = if profile.parts.is_empty() {
                vec![LlmOutputPart::Text {
                    text: profile.full_text.clone(),
                    response_meta: None,
                }]
            } else {
                profile.parts
            };
            Ok(LlmResponse {
                full_text: profile.full_text.clone(),
                parts,
                usage,
                terminal_reason: lash_core::LlmTerminalReason::Stop,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body: None,
                http_summary: None,
            })
        })
        .build()
}

#[derive(Default)]
pub(crate) struct BenchmarkEchoTool;

#[derive(Clone)]
pub(crate) struct BenchmarkLargeToolCatalog {
    cache: Arc<BenchmarkLargeToolCatalogCache>,
}

struct BenchmarkLargeToolCatalogCache {
    manifests: Vec<ToolManifest>,
    contracts: HashMap<String, Arc<ToolContract>>,
}

impl Default for BenchmarkLargeToolCatalog {
    fn default() -> Self {
        Self {
            cache: Arc::clone(large_tool_catalog_cache()),
        }
    }
}

fn large_tool_catalog_cache() -> &'static Arc<BenchmarkLargeToolCatalogCache> {
    static CACHE: OnceLock<Arc<BenchmarkLargeToolCatalogCache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let definitions = BenchmarkLargeToolCatalog::build_tool_definitions();
        let manifests = definitions
            .iter()
            .map(ToolDefinition::manifest)
            .collect::<Vec<_>>();
        let contracts = definitions
            .iter()
            .map(|definition| {
                (
                    definition.name().to_string(),
                    Arc::new(definition.contract()) as Arc<ToolContract>,
                )
            })
            .collect::<HashMap<_, _>>();
        Arc::new(BenchmarkLargeToolCatalogCache {
            manifests,
            contracts,
        })
    })
}

#[async_trait::async_trait]
impl ToolProvider for BenchmarkEchoTool {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![
            benchmark_echo_tool_definition().manifest(),
            benchmark_slow_tool_definition().manifest(),
            benchmark_async_tool_definition().manifest(),
        ]
    }

    fn resolve_contract(&self, name: &str) -> Option<std::sync::Arc<ToolContract>> {
        match name {
            "benchmark_echo" => Some(std::sync::Arc::new(
                benchmark_echo_tool_definition().contract(),
            )),
            "benchmark_slow" => Some(std::sync::Arc::new(
                benchmark_slow_tool_definition().contract(),
            )),
            "benchmark_async" => Some(std::sync::Arc::new(
                benchmark_async_tool_definition().contract(),
            )),
            _ => None,
        }
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> ToolResult {
        match call.name {
            "benchmark_echo" => execute_benchmark_echo(call).await,
            "benchmark_slow" => execute_benchmark_slow(call).await,
            "benchmark_async" => execute_benchmark_async(call).await,
            _ => ToolResult::err_fmt(format_args!("Unknown benchmark tool: {}", call.name)),
        }
    }
}

async fn execute_benchmark_echo(call: lash_core::ToolCall<'_>) -> ToolResult {
    tokio::task::yield_now().await;
    ToolResult::ok(serde_json::json!({
        "value": call.args.get("value").cloned().unwrap_or(serde_json::Value::Null),
        "ordinal": call.args.get("ordinal").cloned().unwrap_or(serde_json::Value::Null),
    }))
}

async fn execute_benchmark_slow(call: lash_core::ToolCall<'_>) -> ToolResult {
    let delay_ms = call
        .args
        .get("delay_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(25);
    if let Some(cancel) = call.context.cancellation_token().cloned() {
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {}
            _ = cancel.cancelled() => return ToolResult::cancelled("benchmark_slow cancelled"),
        }
    } else {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    ToolResult::ok(serde_json::json!({
        "value": call.args.get("value").cloned().unwrap_or(serde_json::Value::Null),
        "delay_ms": delay_ms,
    }))
}

async fn execute_benchmark_async(call: lash_core::ToolCall<'_>) -> ToolResult {
    let key = match call.context.completion_key().await {
        Ok(key) => key,
        Err(err) => return ToolResult::err_fmt(err),
    };
    let value = call
        .args
        .get("value")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        let _ = lash_core::InlineRuntimeEffectController
            .resolve_await_event(
                &key,
                Resolution::Ok(serde_json::json!({
                    "value": value,
                    "mode": "pending_completion"
                })),
            )
            .await;
    });
    ToolResult::pending(lash_core::PendingCompletion::new())
}

fn benchmark_echo_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:benchmark_echo",
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
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {},
                "ordinal": {
                    "anyOf": [
                        { "type": "integer" },
                        { "type": "null" }
                    ]
                }
            },
            "required": ["value", "ordinal"],
            "additionalProperties": false
        }),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

fn benchmark_slow_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:benchmark_slow",
        "benchmark_slow",
        "Sleep briefly before returning; used to profile process handle cancellation and await paths.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": ["string", "number", "boolean", "object", "array", "null"] },
                "delay_ms": { "type": "integer", "minimum": 0, "maximum": 1000 }
            },
            "additionalProperties": false
        }),
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {},
                "delay_ms": { "type": "integer", "minimum": 0 }
            },
            "required": ["value", "delay_ms"],
            "additionalProperties": false
        }),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

fn benchmark_async_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:benchmark_async",
        "benchmark_async",
        "Return asynchronously through the host AwaitEvent completion path.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": ["string", "number", "boolean", "object", "array", "null"] }
            },
            "required": ["value"],
            "additionalProperties": false
        }),
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {},
                "mode": { "type": "string" }
            },
            "required": ["value", "mode"],
            "additionalProperties": false
        }),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

#[async_trait::async_trait]
impl ToolProvider for BenchmarkLargeToolCatalog {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.cache.manifests.clone()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.cache.contracts.get(name).cloned()
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

impl BenchmarkLargeToolCatalog {
    fn build_tool_definitions() -> Vec<ToolDefinition> {
        GMAIL_LIKE_TOOL_NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| gmail_like_tool_definition(index, name))
            .collect()
    }
}

fn gmail_like_tool_definition(index: usize, name: &str) -> ToolDefinition {
    let mut definition = ToolDefinition::raw(
        format!("tool:{name}"),
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
    .with_lashlang_binding(
        LashlangToolBinding::new(
            ["gmail"],
            name.trim_start_matches("GMAIL_").to_ascii_lowercase(),
        )
        .with_aliases([name
            .trim_start_matches("GMAIL_")
            .to_ascii_lowercase()
            .replace('_', " ")]),
    )
    .with_scheduling(ToolScheduling::Parallel);

    if index.is_multiple_of(7) {
        definition.contract.output_contract = ToolOutputContract::from_input_schema(
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
    benchmark_stream_profile_for_request(scenario, &empty_request())
}

fn benchmark_stream_profile_for_request(
    scenario: RuntimePerfScenario,
    request: &LlmRequest,
) -> BenchmarkStreamProfile {
    if matches!(
        scenario,
        RuntimePerfScenario::ObservationalMemoryMaintenance
    ) && request
        .session_id
        .as_deref()
        .is_some_and(|id| id.ends_with("-om-observer"))
    {
        return text_profile(
            "<observations>Runtime perf observer captured persisted benchmark messages.</observations>\n\
             <current-task>Measure post-persist observational memory maintenance.</current-task>\n\
             <suggested-response>Report the runtime perf benchmark marker.</suggested-response>",
        );
    }

    if request.output_spec.is_some()
        || request
            .session_id
            .as_deref()
            .is_some_and(|id| id.ends_with("-llm-query"))
    {
        if request.output_spec.as_ref().is_some_and(|spec| {
            matches!(spec, LlmOutputSpec::JsonSchema(schema) if schema.name == "tool_search_rerank")
        }) {
            return text_profile(
                serde_json::json!({
                    "tool_names": [
                        "GMAIL_SEND_EMAIL",
                        "GMAIL_CREATE_EMAIL_DRAFT",
                        "GMAIL_LIST_MESSAGES",
                        "exec_command"
                    ]
                })
                .to_string(),
            );
        }
        return text_profile(
            serde_json::json!({
                "kind": "value",
                "value": "runtime perf benchmark ok",
                "error": null,
            })
            .to_string(),
        );
    }

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
                parts: Vec::new(),
            }
        }
        RuntimePerfScenario::StandardToolCalls => {
            if request_has_tool_result(request) {
                text_profile("runtime perf benchmark ok")
            } else {
                tool_call_profile(
                    "standard-batch-call",
                    "batch",
                    serde_json::json!({
                        "tool_calls": [
                            {
                                "tool": "benchmark_echo",
                                "parameters": {
                                    "value": "runtime perf benchmark ok",
                                    "ordinal": 1,
                                }
                            },
                            {
                                "tool": "benchmark_echo",
                                "parameters": {
                                    "value": "runtime perf benchmark ok",
                                    "ordinal": 2,
                                }
                            },
                            {
                                "tool": "benchmark_echo",
                                "parameters": {
                                    "value": "runtime perf benchmark ok",
                                    "ordinal": 3,
                                }
                            }
                        ]
                    }),
                )
            }
        }
        RuntimePerfScenario::StandardAsyncToolCompletion => {
            if request_has_tool_result(request) {
                text_profile("runtime perf benchmark ok")
            } else {
                tool_call_profile(
                    "standard-async-completion-call",
                    "benchmark_async",
                    serde_json::json!({
                        "value": "runtime perf benchmark ok"
                    }),
                )
            }
        }
        RuntimePerfScenario::StandardShellOutput => {
            if request_has_tool_result(request) {
                text_profile("runtime perf benchmark ok")
            } else {
                tool_call_profile(
                    "standard-shell-output-call",
                    "exec_command",
                    serde_json::json!({
                        "cmd": "for i in $(seq 1 160); do printf 'runtime-perf-shell-line-%03d abcdefghijklmnopqrstuvwxyz0123456789\\n' \"$i\"; done",
                        "timeout_ms": 5000,
                        "max_output_tokens": 4096
                    }),
                )
            }
        }
        RuntimePerfScenario::ToolDiscoverySearch => {
            if request_has_tool_result(request) {
                text_profile("runtime perf benchmark ok")
            } else {
                tool_call_profile(
                    "standard-tool-discovery-call",
                    "search_tools",
                    serde_json::json!({
                        "query": "gmail send email draft label message search",
                        "module": "gmail",
                        "limit": 8
                    }),
                )
            }
        }
        RuntimePerfScenario::Rlm
        | RuntimePerfScenario::RlmLargeToolCatalog
        | RuntimePerfScenario::EmbedRlm
        | RuntimePerfScenario::TraceJsonlExtended => {
            let text = "```lashlang\nsubmit \"runtime perf benchmark ok\"\n```".to_string();
            text_profile(text)
        }
        RuntimePerfScenario::RlmGlobals => {
            // Mix of small (inline) and large (degraded preview) globals so the
            // benchmark exercises both branches of `render_bound_variables`:
            // small values render in full, while `big_map`/`big_notes`/`big_text`
            // exceed the inline budget and go through the keys / head-tail
            // truncation path that runs on every prompt build.
            let text = r#"```lashlang
big_map = {}
for i in range(24) {
  big_map[format("room_{}", i)] = { exits: ["north", "south", "east"], items: [format("item_{}", i)] }
}
big_notes = []
for i in range(45) {
  big_notes = push(big_notes, format("note {}: long observation about world state, plan, and next steps", i))
}
big_text = "Loud Room: "
for i in range(40) {
  big_text = format("{}echo step {} dampens the acoustics; ", big_text, i)
}
live_record = {
  status: "ready",
  turn: input.turn,
  goal: input.goal,
  nested: {
    path: input.path,
    labels: ["runtime", "rlm", "globals"],
    counters: { first: 1, second: 2, third: 3 }
  }
}
live_list = [
  { name: "alpha", count: 1 },
  { name: "beta", count: 2 },
  { name: "gamma", count: 3 }
]
live_message = "runtime perf benchmark ok"
host_snapshot = { benchmark: benchmark, input: input, chat: chat }
submit live_message
```"#
                .to_string();
            text_profile(text)
        }
        RuntimePerfScenario::RlmToolCalls => {
            let text = r#"```lashlang
first = await tools.benchmark_echo({ value: "runtime perf benchmark ok", ordinal: 1 })?
second = await tools.benchmark_echo({ value: "runtime perf benchmark ok", ordinal: 2 })?
third = await tools.benchmark_echo({ value: "runtime perf benchmark ok", ordinal: 3 })?
fourth = await tools.benchmark_echo({ value: "runtime perf benchmark ok", ordinal: 4 })?
submit first.value
```"#
                .to_string();
            text_profile(text)
        }
        RuntimePerfScenario::RlmProcessHandles => {
            let text = r#"```lashlang
process benchmark_echo_process(tool: Tools, value: str, ordinal: int) {
  result = await tool.benchmark_echo({ value: value, ordinal: ordinal })?
  finish result
}

process benchmark_slow_process(tool: Tools, value: str, delay_ms: int) {
  result = await tool.benchmark_slow({ value: value, delay_ms: delay_ms })?
  finish result
}

first = start benchmark_echo_process(tool: tools, value: "runtime perf benchmark ok", ordinal: 1)
second = start benchmark_echo_process(tool: tools, value: "runtime perf benchmark ok", ordinal: 2)
slow = start benchmark_slow_process(tool: tools, value: "cancelled", delay_ms: 50)
live = await processes.list({})?
cancel slow
first_result = (await first)?
second_result = (await second)?
submit first_result.value
```"#
                .to_string();
            text_profile(text)
        }
        RuntimePerfScenario::RlmLlmQuery => {
            let text = r#"```lashlang
result = await llm.query({
  task: "Return the exact benchmark marker.",
  inputs: { marker: "runtime perf benchmark ok" }
})?
submit result
```"#
                .to_string();
            text_profile(text)
        }
        _ => text_profile("runtime perf benchmark ok"),
    }
}

fn text_profile(text: impl Into<String>) -> BenchmarkStreamProfile {
    let text = text.into();
    BenchmarkStreamProfile {
        full_text: text.clone(),
        deltas: vec![text],
        parts: Vec::new(),
    }
}

fn tool_call_profile(
    call_id: impl Into<String>,
    tool_name: impl Into<String>,
    args: serde_json::Value,
) -> BenchmarkStreamProfile {
    BenchmarkStreamProfile {
        full_text: String::new(),
        deltas: Vec::new(),
        parts: vec![LlmOutputPart::ToolCall {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            input_json: args.to_string(),
            replay: None,
        }],
    }
}

fn request_has_tool_result(request: &LlmRequest) -> bool {
    request.messages.iter().any(|message| {
        message
            .blocks
            .iter()
            .any(|block| matches!(block, LlmContentBlock::ToolResult { .. }))
    })
}

fn empty_request() -> LlmRequest {
    LlmRequest {
        model: "mock-model".to_string(),
        messages: Vec::new(),
        attachments: Vec::new(),
        tools: std::sync::Arc::new(Vec::new()),
        tool_choice: Default::default(),
        generation: Default::default(),
        model_variant: None,
        session_id: None,
        output_spec: None,
        stream_events: None,
        provider_trace: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{ToolAvailability, ToolCatalogBuildInput, build_tool_catalog};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn large_tool_catalog_fixture_matches_gmail_sized_callable_catalog() {
        let defs = BenchmarkLargeToolCatalog::build_tool_definitions();
        assert_eq!(defs.len(), 63);
        assert!(defs.iter().all(|def| {
            def.manifest.availability.base == ToolAvailability::Callable
                && def.manifest.lashlang_binding.module_path == vec!["gmail".to_string()]
                && !def.contract.input_schema["properties"]
                    .as_object()
                    .expect("object schema")
                    .is_empty()
        }));
        assert!(
            defs.iter()
                .any(|def| !def.contract.output_contract.is_static()),
            "fixture should cover dynamic output contracts"
        );
        let first = defs.first().expect("fixture tool");
        assert!(
            first.contract.input_schema["$defs"]["message_part"]["properties"]["parts"]["items"]
                ["$ref"]
                .as_str()
                == Some("#/$defs/message_part"),
            "fixture should include recursive nested schema refs"
        );
        assert!(
            first.contract.input_schema["properties"]["payload"]["oneOf"]
                .as_array()
                .is_some_and(|variants| variants.len() >= 4),
            "fixture should include provider-style payload unions"
        );
        assert!(
            first.contract.input_schema["properties"]["projection"]["anyOf"]
                .as_array()
                .is_some_and(|variants| variants.len() >= 2),
            "fixture should include output projection unions"
        );
    }

    #[test]
    fn rlm_large_tool_catalog_does_not_resolve_nested_schema_contracts_without_tool_calls() {
        let definitions = BenchmarkLargeToolCatalog::build_tool_definitions();
        let manifests = definitions
            .iter()
            .map(|definition| definition.manifest())
            .collect::<Vec<_>>();
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let resolver_count = Arc::clone(&contract_resolutions);

        let surface = build_tool_catalog(ToolCatalogBuildInput {
            tools: manifests,
            resolve_contract: Some(Arc::new(move |name| {
                resolver_count.fetch_add(1, Ordering::SeqCst);
                definitions
                    .iter()
                    .find(|definition| definition.name() == name)
                    .map(|definition| Arc::new(definition.contract()))
            })),
            contributions: Vec::new(),
        });

        assert_eq!(surface.callable_tools().len(), GMAIL_LIKE_TOOL_NAMES.len());
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 0);
        assert_eq!(surface.prompt_tool_docs(), "");
    }
}
