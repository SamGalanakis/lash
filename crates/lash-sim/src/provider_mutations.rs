use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use lash_core::llm::types::{
    LlmEventSender, LlmMessage, LlmRequest, LlmRole, LlmToolChoice, LlmToolSpec,
};
use lash_core::provider::{DefaultProviderFailureClassifier, Provider, ProviderFailureClassifier};
use lash_llm_transport::LlmHttpTransport;
use lash_provider_anthropic::AnthropicProvider;
use lash_provider_google::GoogleOAuthProvider;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};
use serde_json::{Value, json};

use crate::provider::{ProviderWireScript, ScriptedLlmHttpTransport};
use crate::scheduler::BoundaryEvent;
use crate::trace::value_digest;

const OPENAI_COMPAT_TOOL_CALL: &str = include_str!(
    "../provider-scripts/canonical/openai-compatible.chat-tool-call-split-stream.json"
);
const OPENAI_COMPAT_RATE_LIMIT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-rate-limit-429.json");
const OPENAI_RESPONSES_TEXT: &str =
    include_str!("../provider-scripts/canonical/openai.responses-text-stream.json");
const ANTHROPIC_MESSAGES_TEXT: &str =
    include_str!("../provider-scripts/canonical/anthropic.messages-text-stream.json");
const GOOGLE_STREAM_GENERATE_TEXT: &str =
    include_str!("../provider-scripts/canonical/google.stream-generate-content-text-stream.json");
const GOOGLE_GENERATE_RATE_LIMIT: &str =
    include_str!("../provider-scripts/canonical/google.generate-content-rate-limit-429.json");

#[derive(Debug)]
pub struct ProviderMutationExecutionError {
    message: String,
}

impl ProviderMutationExecutionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ProviderMutationExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "provider mutation execution failed: {}", self.message)
    }
}

impl std::error::Error for ProviderMutationExecutionError {}

impl From<serde_json::Error> for ProviderMutationExecutionError {
    fn from(value: serde_json::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<lash_core::LlmTransportError> for ProviderMutationExecutionError {
    fn from(value: lash_core::LlmTransportError) -> Self {
        Self::new(value.to_string())
    }
}

#[derive(Default)]
pub struct ProviderMutationMatrixCache {
    by_mutation: BTreeMap<String, Value>,
}

impl ProviderMutationMatrixCache {
    pub async fn augment_observation(
        &mut self,
        event: &BoundaryEvent,
        mut observed: Value,
    ) -> Result<Value, ProviderMutationExecutionError> {
        let mutation = event
            .payload
            .get("mutation")
            .and_then(Value::as_str)
            .unwrap_or("unknown_mutation")
            .to_string();
        let matrix = if let Some(matrix) = self.by_mutation.get(&mutation) {
            matrix.clone()
        } else {
            let matrix = execute_provider_mutation_matrix(&mutation).await?;
            self.by_mutation.insert(mutation.clone(), matrix.clone());
            matrix
        };
        observed
            .as_object_mut()
            .ok_or_else(|| {
                ProviderMutationExecutionError::new(
                    "provider mutation observation was not an object",
                )
            })?
            .insert("provider_parser_matrix".to_string(), matrix);
        Ok(observed)
    }
}

pub async fn execute_provider_mutation_matrix(
    mutation: &str,
) -> Result<Value, ProviderMutationExecutionError> {
    let specs = mutation_specs(mutation)?;
    let mut proofs = Vec::with_capacity(specs.len());
    for spec in specs {
        proofs.push(run_mutation_script(spec).await?);
    }
    let provider_kinds = proofs
        .iter()
        .filter_map(|proof| proof.get("provider_kind").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let matrix = json!({
        "schema": "lash.sim.provider-mutation-parser-matrix.v1",
        "mutation": mutation,
        "real_provider_parser_execution": true,
        "provider_kinds": provider_kinds,
        "proof_count": proofs.len(),
        "proofs": proofs,
    });
    Ok(json!({
        "matrix": matrix,
        "digest": value_digest(&matrix),
    }))
}

struct MutationScriptSpec {
    proof_name: &'static str,
    provider_kind: &'static str,
    script_content: String,
    request_kind: MutationRequestKind,
    expected_status: Option<u16>,
}

enum MutationRequestKind {
    OpenAiCompatible { stream: bool },
    OpenAiResponses,
    Anthropic,
    Google { stream: bool },
}

fn mutation_specs(
    mutation: &str,
) -> Result<Vec<MutationScriptSpec>, ProviderMutationExecutionError> {
    match mutation {
        "malformed_sse_chunk" => malformed_parser_specs("malformed-sse", "malformed_sse_chunk"),
        "dropped_terminal_event" => {
            malformed_parser_specs("dropped-terminal-event", "dropped_terminal_event")
        }
        "duplicate_tool_call_delta" => {
            malformed_parser_specs("duplicate-tool-call-delta", "duplicate_tool_call_delta")
        }
        "wrong_provider_schema" => {
            malformed_parser_specs("wrong-provider-schema", "wrong_provider_schema")
        }
        "rate_limit_error_envelope" => Ok(vec![
            MutationScriptSpec {
                proof_name: "mutation.rate-limit.openai-compatible",
                provider_kind: "openai-compatible",
                script_content: OPENAI_COMPAT_RATE_LIMIT.to_string(),
                request_kind: MutationRequestKind::OpenAiCompatible { stream: false },
                expected_status: Some(429),
            },
            MutationScriptSpec {
                proof_name: "mutation.rate-limit.openai",
                provider_kind: "openai",
                script_content: rate_limit_script(
                    OPENAI_RESPONSES_TEXT,
                    "openai.responses-rate-limit-429",
                    json!({
                        "error": {
                            "message": "Rate limit reached for responses",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded"
                        }
                    }),
                )?,
                request_kind: MutationRequestKind::OpenAiResponses,
                expected_status: Some(429),
            },
            MutationScriptSpec {
                proof_name: "mutation.rate-limit.anthropic",
                provider_kind: "anthropic",
                script_content: rate_limit_script(
                    ANTHROPIC_MESSAGES_TEXT,
                    "anthropic.messages-rate-limit-429",
                    json!({
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "rate limit exceeded"
                        }
                    }),
                )?,
                request_kind: MutationRequestKind::Anthropic,
                expected_status: Some(429),
            },
            MutationScriptSpec {
                proof_name: "mutation.rate-limit.google",
                provider_kind: "google_oauth",
                script_content: GOOGLE_GENERATE_RATE_LIMIT.to_string(),
                request_kind: MutationRequestKind::Google { stream: false },
                expected_status: Some(429),
            },
        ]),
        other => Err(ProviderMutationExecutionError::new(format!(
            "unknown provider mutation `{other}`"
        ))),
    }
}

fn malformed_parser_specs(
    script_stem: &'static str,
    mutation: &'static str,
) -> Result<Vec<MutationScriptSpec>, ProviderMutationExecutionError> {
    Ok(vec![
        MutationScriptSpec {
            proof_name: "mutation.parser-error.openai-compatible",
            provider_kind: "openai-compatible",
            script_content: malformed_sse_script(
                OPENAI_COMPAT_TOOL_CALL,
                &format!("openai-compatible.chat-{script_stem}"),
                mutation,
            )?,
            request_kind: MutationRequestKind::OpenAiCompatible { stream: true },
            expected_status: None,
        },
        MutationScriptSpec {
            proof_name: "mutation.parser-error.openai",
            provider_kind: "openai",
            script_content: malformed_sse_script(
                OPENAI_RESPONSES_TEXT,
                &format!("openai.responses-{script_stem}"),
                mutation,
            )?,
            request_kind: MutationRequestKind::OpenAiResponses,
            expected_status: None,
        },
        MutationScriptSpec {
            proof_name: "mutation.parser-error.anthropic",
            provider_kind: "anthropic",
            script_content: malformed_sse_script(
                ANTHROPIC_MESSAGES_TEXT,
                &format!("anthropic.messages-{script_stem}"),
                mutation,
            )?,
            request_kind: MutationRequestKind::Anthropic,
            expected_status: None,
        },
        MutationScriptSpec {
            proof_name: "mutation.parser-error.google",
            provider_kind: "google_oauth",
            script_content: malformed_sse_script(
                GOOGLE_STREAM_GENERATE_TEXT,
                &format!("google.stream-generate-content-{script_stem}"),
                mutation,
            )?,
            request_kind: MutationRequestKind::Google { stream: true },
            expected_status: None,
        },
    ])
}

fn malformed_sse_script(
    script: &str,
    name: &str,
    mutation: &str,
) -> Result<String, ProviderMutationExecutionError> {
    let mut value: Value = serde_json::from_str(script)?;
    value["name"] = Value::String(name.to_string());
    let timeline = value
        .get_mut("timeline")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProviderMutationExecutionError::new("script timeline was not an array"))?;
    let Some(event) = timeline
        .iter_mut()
        .find(|event| event.get("event").and_then(Value::as_str) == Some("sse"))
    else {
        return Err(ProviderMutationExecutionError::new(
            "script had no SSE event to mutate",
        ));
    };
    event["data"] = Value::String("{ malformed provider event".to_string());
    value["expected_provider"] = json!({
        "mutation": mutation,
        "expected": "provider parser error",
    });
    Ok(serde_json::to_string(&value)?)
}

fn rate_limit_script(
    script: &str,
    name: &str,
    body: Value,
) -> Result<String, ProviderMutationExecutionError> {
    let mut value: Value = serde_json::from_str(script)?;
    value["name"] = Value::String(name.to_string());
    value["timeline"] = json!([
        {
            "at": 10,
            "event": "http_error",
            "status": 429,
            "headers": [
                { "name": "content-type", "value": "application/json" },
                { "name": "retry-after", "value": "3" },
                { "name": "x-request-id", "value": format!("req-{name}") }
            ],
            "body": body.to_string()
        }
    ]);
    value["expected_provider"] = json!({
        "mutation": "rate_limit_error_envelope",
        "status": 429,
    });
    Ok(serde_json::to_string(&value)?)
}

async fn run_mutation_script(
    spec: MutationScriptSpec,
) -> Result<Value, ProviderMutationExecutionError> {
    let parsed = ProviderWireScript::from_json_str(&spec.script_content)?;
    let script_name = parsed.name.clone();
    let transport = Arc::new(ScriptedLlmHttpTransport::new(parsed));
    let result = match spec.request_kind {
        MutationRequestKind::OpenAiCompatible { stream } => {
            let mut provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
                .with_transport(provider_transport(&transport));
            provider.complete(openai_compatible_request(stream)).await
        }
        MutationRequestKind::OpenAiResponses => {
            let mut provider =
                OpenAiProvider::new("test-key").with_transport(provider_transport(&transport));
            provider.complete(openai_responses_request()).await
        }
        MutationRequestKind::Anthropic => {
            let mut provider = AnthropicProvider::new("test-key")
                .with_base_url(Some("https://anthropic.test".to_string()))
                .with_transport(provider_transport(&transport));
            provider.complete(anthropic_messages_request()).await
        }
        MutationRequestKind::Google { stream } => {
            let mut provider = GoogleOAuthProvider::new("access-token", "refresh-token", 0)
                .with_project_id(Some("project-1".to_string()))
                .with_transport(provider_transport(&transport));
            provider.complete(google_request(stream)).await
        }
    };
    let err = result.map_or_else(
        |err| Ok(err),
        |_| {
            Err(ProviderMutationExecutionError::new(format!(
                "mutated provider script `{script_name}` unexpectedly succeeded"
            )))
        },
    )?;
    if let Some(expected_status) = spec.expected_status
        && err.status != Some(expected_status)
    {
        return Err(ProviderMutationExecutionError::new(format!(
            "mutated provider script `{script_name}` returned status {:?}, expected {expected_status}",
            err.status
        )));
    }
    let classified = DefaultProviderFailureClassifier.classify(err.clone());
    let exchanges = transport.exchanges()?;
    Ok(json!({
        "proof_name": spec.proof_name,
        "provider_kind": spec.provider_kind,
        "script_name": script_name,
        "exchange_count": exchanges.len(),
        "endpoint": exchanges.first().map(|exchange| exchange.request.path.clone()),
        "response_events": exchanges.first().map(|exchange| exchange.response.event_names.clone()).unwrap_or_default(),
        "status": err.status,
        "kind": format!("{:?}", err.kind),
        "retryable": err.retryable,
        "terminal_reason": err.terminal_reason.code(),
        "classification": {
            "kind": format!("{:?}", classified.kind),
            "retryable": classified.retryable,
            "status": classified.status,
            "terminal_reason": classified.terminal_reason.code(),
        },
    }))
}

fn provider_transport(transport: &Arc<ScriptedLlmHttpTransport>) -> Arc<dyn LlmHttpTransport> {
    transport.clone()
}

fn openai_compatible_request(stream: bool) -> LlmRequest {
    LlmRequest {
        model: "openai/gpt-5.4".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "lookup x")],
        attachments: Vec::new(),
        tools: Arc::new(vec![LlmToolSpec {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "q": { "type": "string" }
                }
            }),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
        }]),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: stream.then(|| LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

fn openai_responses_request() -> LlmRequest {
    LlmRequest {
        model: "gpt-5.4".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: Some(LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

fn anthropic_messages_request() -> LlmRequest {
    LlmRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: Some(LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

fn google_request(stream: bool) -> LlmRequest {
    LlmRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: stream.then(|| LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{BoundaryEvent, BoundaryKind};

    #[tokio::test]
    async fn provider_mutation_matrix_runs_real_provider_parsers() {
        let mut cache = ProviderMutationMatrixCache::default();
        let event = BoundaryEvent::new(
            "session-001:provider-mutation:001",
            "session-001",
            BoundaryKind::ProviderMutation,
            1,
            "provider.script-mutation.rejected",
            json!({
                "mutation": "malformed_sse_chunk",
            }),
        );
        let observed = cache
            .augment_observation(
                &event,
                json!({
                    "session": "session-001",
                    "provider_mutation": true,
                    "mutation": "malformed_sse_chunk",
                    "rejected": true,
                    "first_rejection": true,
                }),
            )
            .await
            .expect("provider mutation matrix");

        let matrix = &observed["provider_parser_matrix"]["matrix"];
        assert_eq!(matrix["real_provider_parser_execution"], true);
        assert_eq!(matrix["proof_count"], 4);
        let providers = matrix["provider_kinds"]
            .as_array()
            .expect("providers")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert_eq!(
            providers,
            vec!["anthropic", "google_oauth", "openai", "openai-compatible"]
        );
        assert!(
            matrix["proofs"]
                .as_array()
                .expect("proofs")
                .iter()
                .all(|proof| proof["exchange_count"].as_u64() == Some(1))
        );
    }
}
