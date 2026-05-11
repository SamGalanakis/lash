use lash_core::llm::types::{LlmOutputPart, LlmResponse, LlmStreamEvent, LlmUsage};
use lash_core::testing::TestProvider;
use lash_core::{ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};

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

#[async_trait::async_trait]
impl ToolProvider for BenchmarkEchoTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
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
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
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
