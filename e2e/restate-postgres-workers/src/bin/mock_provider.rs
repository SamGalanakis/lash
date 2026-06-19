use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use lash_postgres_store::PostgresStorage;
use serde_json::{Value, json};
use sqlx::PgPool;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lash_restate_postgres_workers_e2e::{
    EXPECTED_ASYNC_TEXT, EXPECTED_DURABLE_INPUT_TEXT, EXPECTED_FINAL_TEXT,
    EXPECTED_TOOL_BATCH_TEXT, EXPECTED_WAKE_TEXT, ensure_e2e_schema, env, record_provider_call,
    required_env,
};

#[derive(Clone)]
struct AppState {
    calls: Arc<AtomicU64>,
    pool: PgPool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let database_url = required_env("DATABASE_URL")?;
    let storage = PostgresStorage::connect(&database_url).await?;
    ensure_e2e_schema(storage.pool()).await?;
    let port = env("MOCK_PROVIDER_PORT", "18001");
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/responses", post(responses))
        .with_state(AppState {
            calls: Arc::new(AtomicU64::new(0)),
            pool: storage.pool().clone(),
        });
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_completion(State(state): State<AppState>, Json(request): Json<Value>) -> Json<Value> {
    let n = state.calls.fetch_add(1, Ordering::SeqCst) + 1;
    let request_id = format!("chatcmpl-e2e-{n}");
    let latest_user = latest_user_text(&request);
    let full_text = request.to_string();
    let workflow_id = extract_latest_marker(&latest_user, "workflow_id=")
        .or_else(|| extract_latest_marker(&full_text, "workflow_id="))
        .unwrap_or_else(|| "unknown".to_string());
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let scenario = latest_scenario_marker(&latest_user)
        .or_else(|| latest_scenario_marker(&full_text))
        .unwrap_or(MockScenario::KitchenSink);
    let (scenario, content) = match scenario {
        MockScenario::TriggerSetup => ("trigger_setup", trigger_setup_script()),
        MockScenario::QueuedWake => ("queued_wake", queued_wake_script()),
        MockScenario::SignalSuspend => ("signal_suspend", signal_suspend_script(&workflow_id)),
        MockScenario::AsyncCompletion => {
            ("async_completion", async_completion_script(&workflow_id))
        }
        MockScenario::DurableInputRequest => (
            "durable_input_request",
            durable_input_request_script(&workflow_id),
        ),
        MockScenario::ToolBatch => (
            "tool_batch",
            tool_batch_script(&workflow_id, full_text.contains("fail_once=true")),
        ),
        MockScenario::KitchenSink => (
            "kitchen_sink",
            kitchen_sink_script(&workflow_id, full_text.contains("fail_once=true")),
        ),
    };
    let response = json!({
        "id": request_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": content,
            }
        }],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 31,
            "total_tokens": 48
        }
    });
    if let Err(err) = record_provider_call(
        &state.pool,
        response["id"].as_str().unwrap_or("chatcmpl-e2e"),
        scenario,
        &workflow_id,
        model,
        &request,
        &response,
    )
    .await
    {
        tracing::error!(workflow_id, scenario, error = %err, "failed to record provider call");
    }
    Json(response)
}

async fn responses(State(state): State<AppState>, Json(request): Json<Value>) -> Json<Value> {
    let n = state.calls.fetch_add(1, Ordering::SeqCst) + 1;
    let request_id = format!("resp-e2e-{n}");
    let response = json!({
        "id": request_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": request.get("model").and_then(Value::as_str).unwrap_or("e2e-mock"),
        "output": [{
            "id": format!("msg-e2e-{n}"),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "ok"
            }]
        }],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2
        }
    });
    Json(response)
}

fn latest_user_text(request: &Value) -> String {
    request
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .rev()
                .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .map(message_content_text)
        .unwrap_or_else(|| request.to_string())
}

fn message_content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.map(Value::to_string).unwrap_or_default(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MockScenario {
    KitchenSink,
    QueuedWake,
    TriggerSetup,
    SignalSuspend,
    AsyncCompletion,
    DurableInputRequest,
    ToolBatch,
}

fn latest_scenario_marker(text: &str) -> Option<MockScenario> {
    [
        ("tool_batch=true", MockScenario::ToolBatch),
        (
            "durable_input_request=true",
            MockScenario::DurableInputRequest,
        ),
        ("async_completion=true", MockScenario::AsyncCompletion),
        ("trigger_setup=true", MockScenario::TriggerSetup),
        ("signal_suspend=true", MockScenario::SignalSuspend),
        ("Background process wake", MockScenario::QueuedWake),
        ("fail_once=", MockScenario::KitchenSink),
    ]
    .into_iter()
    .filter_map(|(marker, scenario)| text.rfind(marker).map(|idx| (idx, scenario)))
    .max_by_key(|(idx, _)| *idx)
    .map(|(_, scenario)| scenario)
}

fn extract_latest_marker(text: &str, marker: &str) -> Option<String> {
    let start = text.rfind(marker)? + marker.len();
    let rest = &text[start..];
    let value: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ':' | '.'))
        .collect();
    (!value.is_empty()).then_some(value)
}

fn kitchen_sink_script(workflow_id: &str, fail_once: bool) -> String {
    let crash = if fail_once {
        format!(
            r#"
crash = await tools.crash_once({{ workflow_id: "{workflow_id}" }})?
"#
        )
    } else {
        String::new()
    };
    format!(
        r#"
Execute this program.

```lashlang
process child(tools: Tools, value: str) {{
  lookup = await tools.app_lookup({{ key: value }})?
  finish {{ child: value, lookup: lookup.value }}
}}

process parent(tools: Tools, workflow_id: str) {{
  parent_lookup = await tools.app_lookup({{ key: "parent" }})?
  nested_handle = start child(tools: tools, value: "nested")
  nested_result = (await nested_handle)?
  parallel = await {{
    left: start child(tools: tools, value: "left"),
    right: start child(tools: tools, value: "right")
  }}
  left_result = parallel.left?
  right_result = parallel.right?
  sleep for "1ms"
  finish {{
    parent_lookup: parent_lookup.value,
    nested: nested_result.lookup,
    parallel: {{
      left: left_result.lookup,
      right: right_result.lookup
    }},
    slept: true,
    wake: "deferred"
  }}
}}

process waker(workflow_id: str) {{
  sleep for "1500ms"
  wake {{
    kind: "parent_wake",
    workflow_id: workflow_id,
    text: "deploy complete"
  }}
  finish {{ wake: "sent" }}
}}

foreground = await tools.app_lookup({{ key: "foreground" }})?
attachment = await tools.make_attachment({{
  workflow_id: "{workflow_id}",
  name: "kitchen-sink.png"
}})?
{crash}
parent_handle = start parent(tools: tools, workflow_id: "{workflow_id}")
process_result = (await parent_handle)?
waker_handle = start waker(workflow_id: "{workflow_id}")
sleep for "0ms"
submit {{
  workflow_id: "{workflow_id}",
  foreground: foreground.value,
  attachment_id: attachment.id,
  attachment_mime: attachment.mime,
  wake_process: waker_handle,
  process: process_result,
  final: "{EXPECTED_FINAL_TEXT}"
}}
```
"#
    )
}

fn trigger_setup_script() -> String {
    r#"
Register this trigger.

```lashlang
process on_button(event: ui.button.Pressed) {
  finish { triggered: event.button, message: event.message }
}

handle = await triggers.register({
  source: ui.button.pressed({}),
  target: on_button,
  inputs: { event: trigger.event },
  name: "button watcher"
})?
submit { registered: true, handle: handle }
```
"#
    .to_string()
}

fn signal_suspend_script(workflow_id: &str) -> String {
    format!(
        r#"
Start the signal suspension process but do not await it.

```lashlang
process waiter(workflow_id: str) signals {{ first: any, second: any }} {{
  first = wait_signal("first")
  second = wait_signal("second")
  finish {{
    workflow_id: workflow_id,
    first: first,
    second: second
  }}
}}

handle = start waiter(workflow_id: "{workflow_id}")
submit {{
  workflow_id: "{workflow_id}",
  process_id: handle.id,
  final: "signal-suspend-started"
}}
```
"#
    )
}

fn queued_wake_script() -> String {
    format!(
        r#"
Consume the queued wake.

```lashlang
submit {{ wake_consumed: true, final: "{EXPECTED_WAKE_TEXT}" }}
```
"#
    )
}

fn async_completion_script(workflow_id: &str) -> String {
    format!(
        r#"
Exercise the async host tool completion path.

```lashlang
process async_child(tools: Tools, workflow_id: str) {{
  lookup = await tools.async_lookup({{ workflow_id: workflow_id, key: "detached" }})?
  finish lookup
}}

handle = start async_child(tools: tools, workflow_id: "{workflow_id}")
result = (await handle)?
submit {{
  workflow_id: "{workflow_id}",
  async: result,
  final: "{EXPECTED_ASYNC_TEXT}"
}}
```
"#
    )
}

fn durable_input_request_script(workflow_id: &str) -> String {
    format!(
        r#"
Exercise a durable in-process tool that opens an input request and resumes through a custom await key.

```lashlang
process durable_child(tools: Tools, workflow_id: str) {{
  input = await tools.durable_input_request({{
    workflow_id: workflow_id,
    question: "approve durable input?"
  }})?
  finish input
}}

handle = start durable_child(tools: tools, workflow_id: "{workflow_id}")
result = (await handle)?
submit {{
  workflow_id: "{workflow_id}",
  durable: result,
  final: "{EXPECTED_DURABLE_INPUT_TEXT}"
}}
```
"#
    )
}

fn tool_batch_script(workflow_id: &str, fail_once: bool) -> String {
    let crash = if fail_once {
        format!(
            r#"
crash = await tools.crash_once({{ workflow_id: "{workflow_id}" }})?
"#
        )
    } else {
        String::new()
    };
    format!(
        r#"
Exercise direct aggregate resource batching.

```lashlang
batch = await {{
  slow: tools.batch_side_effect({{
    workflow_id: "{workflow_id}",
    key: "slow",
    delay_ms: 75
  }})?,
  fast: tools.batch_side_effect({{
    workflow_id: "{workflow_id}",
    key: "fast",
    delay_ms: 5
  }})?,
  literal: "kept"
}}
{crash}
submit {{
  workflow_id: "{workflow_id}",
  batch: batch,
  final: "{EXPECTED_TOOL_BATCH_TEXT}"
}}
```
"#
    )
}
