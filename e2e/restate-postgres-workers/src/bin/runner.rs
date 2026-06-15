use anyhow::{Context, Result};
use lash::durability::EffectHost;
use lash::triggers::{TriggerOccurrenceRequest, empty_trigger_source_key};
use lash_core::{
    AwaitEventKey, ExecutionScope, InlineRuntimeEffectController, RuntimePersistence,
    ScopedEffectController,
};
use lash_postgres_store::PostgresStorage;
use lash_restate::{RestateEffectHost, RestateProcessDeployment};
use lash_restate_postgres_workers_e2e::{
    ATTACHMENT_MIME, BUTTON_SOURCE_TYPE, DEFAULT_SESSION_ID, EXPECTED_ASYNC_TEXT,
    EXPECTED_DURABLE_INPUT_TEXT, EXPECTED_FINAL_TEXT, ProcessSignalRequest, TURN_WORKFLOW_NAME,
    TurnRequest, TurnResponse, TurnScenario, build_e2e_core, ensure_e2e_schema, env,
    expected_attachment_bytes, process_registry_from_storage, reset_e2e_rows, s3_store_from_env,
};
use reqwest::StatusCode;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let database_url = lash_restate_postgres_workers_e2e::required_env("DATABASE_URL")?;
    let storage = wait_for_postgres(&database_url).await?;
    ensure_e2e_schema(storage.pool()).await?;
    reset_e2e_rows(storage.pool()).await?;

    let trace_dir = std::env::var("LASH_E2E_TRACE_DIR").ok().map(PathBuf::from);
    if let Some(dir) = &trace_dir {
        reset_trace_dir(dir)?;
    }

    let attachment_store = s3_store_from_env()?;
    wait_for_minio(&attachment_store).await?;
    let mock_provider_base_url = env("MOCK_PROVIDER_BASE_URL", "http://mock-provider:18001");
    wait_for_mock_provider(&mock_provider_base_url).await?;

    let admin_url = env("RESTATE_ADMIN_URL", "http://restate:9070");
    let deployment_url = env("WORKER_DEPLOYMENT_URL", "http://worker-proxy:18100");
    register_restate_deployment(&admin_url, &deployment_url).await?;
    let ingress_url = env("RESTATE_INGRESS_URL", "http://restate:8080");

    let main_request = TurnRequest {
        workflow_id: "e2e-main".to_string(),
        fail_once: false,
        scenario: TurnScenario::KitchenSink,
        signal: None,
    };
    submit_workflow(&ingress_url, &main_request).await?;
    let main_response = wait_for_terminal_result(storage.pool(), &main_request.workflow_id).await?;
    assert_kitchen_sink_response(&main_response, true)?;
    wait_for_queued_work(
        &storage,
        &mock_provider_base_url,
        trace_dir.clone(),
        &ingress_url,
    )
    .await?;
    let main_wake_request = TurnRequest {
        workflow_id: "e2e-main-wake".to_string(),
        fail_once: false,
        scenario: TurnScenario::DrainQueued,
        signal: None,
    };
    submit_workflow(&ingress_url, &main_wake_request).await?;
    let main_wake_response =
        wait_for_terminal_result(storage.pool(), &main_wake_request.workflow_id).await?;
    assert_queued_wake_response(&main_wake_response)?;

    let trigger_request = TurnRequest {
        workflow_id: "e2e-trigger-setup".to_string(),
        fail_once: false,
        scenario: TurnScenario::TriggerSetup,
        signal: None,
    };
    submit_workflow(&ingress_url, &trigger_request).await?;
    let trigger_setup_response =
        wait_for_terminal_result(storage.pool(), &trigger_request.workflow_id).await?;
    assert_trigger_setup_response(&trigger_setup_response)?;
    let trigger_process_id = emit_button_event(
        &storage,
        &mock_provider_base_url,
        trace_dir.clone(),
        &ingress_url,
    )
    .await?;
    wait_for_process_terminal(storage.pool(), &trigger_process_id).await?;

    let signal_setup_request = TurnRequest {
        workflow_id: "e2e-signal-suspend-setup".to_string(),
        fail_once: false,
        scenario: TurnScenario::SignalSuspend,
        signal: None,
    };
    submit_workflow(&ingress_url, &signal_setup_request).await?;
    let signal_setup_response =
        wait_for_terminal_result(storage.pool(), &signal_setup_request.workflow_id).await?;
    let signal_process_id = assert_signal_suspend_setup_response(&signal_setup_response)?;
    wait_for_process_signal_wait(storage.pool(), &signal_process_id, "first", 1).await?;

    let failover_request = TurnRequest {
        workflow_id: "e2e-failover".to_string(),
        fail_once: true,
        scenario: TurnScenario::KitchenSink,
        signal: None,
    };
    submit_workflow(&ingress_url, &failover_request).await?;
    let failover_response =
        wait_for_terminal_result(storage.pool(), &failover_request.workflow_id).await?;
    assert_kitchen_sink_response(&failover_response, true)?;
    wait_for_process_signal_wait(storage.pool(), &signal_process_id, "first", 1).await?;
    wait_for_queued_work(
        &storage,
        &mock_provider_base_url,
        trace_dir.clone(),
        &ingress_url,
    )
    .await?;
    let failover_wake_request = TurnRequest {
        workflow_id: "e2e-failover-wake".to_string(),
        fail_once: false,
        scenario: TurnScenario::DrainQueued,
        signal: None,
    };
    submit_workflow(&ingress_url, &failover_wake_request).await?;
    let failover_wake_response =
        wait_for_terminal_result(storage.pool(), &failover_wake_request.workflow_id).await?;
    assert_queued_wake_response(&failover_wake_response)?;

    submit_signal_workflow(
        &ingress_url,
        storage.pool(),
        "e2e-signal-first",
        &signal_process_id,
        "first",
        "first-1",
        json!({ "phase": "first" }),
    )
    .await?;
    wait_for_process_signal_wait(storage.pool(), &signal_process_id, "second", 1).await?;
    submit_signal_workflow(
        &ingress_url,
        storage.pool(),
        "e2e-signal-second",
        &signal_process_id,
        "second",
        "second-1",
        json!({ "phase": "second" }),
    )
    .await?;
    wait_for_process_terminal(storage.pool(), &signal_process_id).await?;
    assert_signal_process_output(storage.pool(), &signal_process_id).await?;

    let async_request = TurnRequest {
        workflow_id: "e2e-async-completion".to_string(),
        fail_once: false,
        scenario: TurnScenario::AsyncCompletion,
        signal: None,
    };
    submit_workflow(&ingress_url, &async_request).await?;
    let async_response =
        wait_for_terminal_result(storage.pool(), &async_request.workflow_id).await?;
    assert_async_completion_response(&async_response)?;

    let durable_input_request = TurnRequest {
        workflow_id: "e2e-durable-input".to_string(),
        fail_once: false,
        scenario: TurnScenario::DurableInputRequest,
        signal: None,
    };
    submit_workflow(&ingress_url, &durable_input_request).await?;
    let durable_key =
        wait_for_durable_input_key(storage.pool(), &durable_input_request.workflow_id).await?;
    let durable_resolve = RestateEffectHost::with_ingress_url(ingress_url.clone())
        .resolve_await_event(
            &durable_key,
            lash_core::Resolution::Ok(json!({ "answer": "durable-approved" })),
        )
        .await
        .context("resolve durable input await key")?;
    anyhow::ensure!(
        matches!(durable_resolve, lash_core::ResolveOutcome::Accepted),
        "durable input resolve was not accepted: {durable_resolve:?}"
    );
    let durable_response =
        wait_for_terminal_result(storage.pool(), &durable_input_request.workflow_id).await?;
    assert_durable_input_response(&durable_response)?;

    let responses = wait_for_terminal_results(storage.pool(), 10).await?;

    assert_processes_terminal(storage.pool()).await?;
    assert_no_duplicate_runtime_rows(storage.pool()).await?;
    assert_worker_distribution(storage.pool()).await?;
    assert_failover(storage.pool()).await?;
    assert_provider_calls(storage.pool()).await?;
    assert_tool_and_turn_telemetry(storage.pool()).await?;
    assert_durable_input_steps(storage.pool()).await?;
    assert_trigger_delivery(storage.pool(), &trigger_process_id).await?;
    assert_attachments_round_trip(storage.pool(), &attachment_store, &responses).await?;
    assert_reopened_session_agrees(
        &storage,
        &mock_provider_base_url,
        trace_dir.clone(),
        &ingress_url,
        &responses,
    )
    .await?;
    if let Some(dir) = &trace_dir {
        assert_traces(dir).await?;
    }

    println!(
        "restate-postgres-workers e2e passed: {} workflows, trigger process {}, signal process {}, traces {}",
        responses.len(),
        trigger_process_id,
        signal_process_id,
        trace_dir
            .as_ref()
            .map(|dir| dir.display().to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );
    Ok(())
}

fn reset_trace_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create trace dir `{}`", dir.display()))?;
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("read trace dir `{}`", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::remove_file(entry.path())
                .with_context(|| format!("remove stale trace `{}`", entry.path().display()))?;
        }
    }
    Ok(())
}

async fn wait_for_postgres(database_url: &str) -> Result<PostgresStorage> {
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut last_error = None;
    while Instant::now() < deadline {
        match PostgresStorage::connect(database_url).await {
            Ok(storage) => return Ok(storage),
            Err(err) => {
                last_error = Some(err.to_string());
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    anyhow::bail!(
        "Postgres did not become ready: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
}

async fn wait_for_minio(store: &impl lash::persistence::AttachmentStore) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(90);
    let meta = lash_core::AttachmentCreateMeta::new(
        lash_core::MediaType::Image(lash_core::ImageMediaType::Png),
        Some(1),
        Some(1),
        Some("runner-health.png".to_string()),
    );
    let mut last_error = None;
    while Instant::now() < deadline {
        match store
            .put(b"runner-minio-health".to_vec(), meta.clone())
            .await
        {
            Ok(reference) => match store.get(&reference.id).await {
                Ok(stored) if stored.bytes == b"runner-minio-health" => return Ok(()),
                Ok(_) => last_error = Some("MinIO health attachment bytes changed".to_string()),
                Err(err) => last_error = Some(err.to_string()),
            },
            Err(err) => last_error = Some(err.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!(
        "MinIO did not become ready: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
}

async fn register_restate_deployment(admin_url: &str, deployment_url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .context("build Restate admin client")?;
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut last_error = None;
    while Instant::now() < deadline {
        match client
            .post(format!("{}/deployments", admin_url.trim_end_matches('/')))
            .json(&json!({
                "uri": deployment_url,
                "force": true,
                "breaking": true,
            }))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(format!("{status}: {body}"));
            }
            Err(err) => last_error = Some(err.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!(
        "Restate deployment registration failed: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
}

async fn wait_for_mock_provider(base_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut last_error = None;
    while Instant::now() < deadline {
        match client
            .get(format!("{}/health", base_url.trim_end_matches('/')))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(format!("{status}: {body}"));
            }
            Err(err) => last_error = Some(err.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!(
        "mock provider did not become ready: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
}

async fn submit_workflow(ingress_url: &str, request: &TurnRequest) -> Result<()> {
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .context("build Restate ingress client")?;
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut last_error = None;
    while Instant::now() < deadline {
        match client
            .post(workflow_send_url(ingress_url, &request.workflow_id))
            .json(request)
            .send()
            .await
        {
            Ok(response)
                if response.status().is_success() || response.status() == StatusCode::CONFLICT =>
            {
                return Ok(());
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(format!("{status}: {body}"));
            }
            Err(err) => last_error = Some(err.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!(
        "workflow `{}` submit failed: {}",
        request.workflow_id,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )
}

async fn submit_signal_workflow(
    ingress_url: &str,
    pool: &sqlx::PgPool,
    workflow_id: &str,
    process_id: &str,
    signal_name: &str,
    signal_id: &str,
    payload: serde_json::Value,
) -> Result<TurnResponse> {
    let request = TurnRequest {
        workflow_id: workflow_id.to_string(),
        fail_once: false,
        scenario: TurnScenario::SignalProcess,
        signal: Some(ProcessSignalRequest {
            process_id: process_id.to_string(),
            signal_name: signal_name.to_string(),
            signal_id: signal_id.to_string(),
            payload,
        }),
    };
    submit_workflow(ingress_url, &request).await?;
    let response = wait_for_terminal_result(pool, workflow_id).await?;
    anyhow::ensure!(
        response
            .submitted_value
            .get("signalled")
            .and_then(Value::as_bool)
            == Some(true),
        "signal workflow `{workflow_id}` did not submit signalled=true: {}",
        response.submitted_value
    );
    Ok(response)
}

fn workflow_send_url(ingress_url: &str, workflow_id: &str) -> String {
    format!(
        "{}/{}/{}/run/send",
        ingress_url.trim_end_matches('/'),
        TURN_WORKFLOW_NAME,
        workflow_id
    )
}

async fn wait_for_terminal_result(pool: &sqlx::PgPool, workflow_id: &str) -> Result<TurnResponse> {
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        if let Some(response) = load_terminal_result(pool, workflow_id).await? {
            return Ok(response);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("timed out waiting for `{workflow_id}`")
}

async fn wait_for_terminal_results(
    pool: &sqlx::PgPool,
    expected: usize,
) -> Result<Vec<TurnResponse>> {
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        let rows = sqlx::query_as::<_, TerminalResultRow>(
            "SELECT workflow_id, process_id, worker_id, attachment_id, final_text, submitted_json,
                    queued_turn_ran, streamed_event_count, replay_cursor
             FROM lash_e2e_terminal_results
             ORDER BY workflow_id",
        )
        .fetch_all(pool)
        .await
        .context("load terminal results")?;
        if rows.len() >= expected {
            return rows
                .into_iter()
                .map(response_from_row)
                .collect::<Result<Vec<_>>>();
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("timed out waiting for {expected} completed workflows")
}

async fn load_terminal_result(
    pool: &sqlx::PgPool,
    workflow_id: &str,
) -> Result<Option<TurnResponse>> {
    sqlx::query_as::<_, TerminalResultRow>(
        "SELECT workflow_id, process_id, worker_id, attachment_id, final_text, submitted_json,
                queued_turn_ran, streamed_event_count, replay_cursor
         FROM lash_e2e_terminal_results
         WHERE workflow_id = $1",
    )
    .bind(workflow_id)
    .fetch_optional(pool)
    .await
    .context("load terminal result")?
    .map(response_from_row)
    .transpose()
}

type TerminalResultRow = (
    String,
    String,
    String,
    String,
    String,
    String,
    bool,
    i64,
    Option<String>,
);

fn response_from_row(
    (
        workflow_id,
        process_id,
        worker_id,
        attachment_id,
        final_text,
        submitted_json,
        queued_turn_ran,
        streamed_event_count,
        replay_cursor,
    ): TerminalResultRow,
) -> Result<TurnResponse> {
    Ok(TurnResponse {
        workflow_id,
        worker_id,
        process_id,
        process_ids: Vec::new(),
        attachment_id,
        final_text,
        submitted_value: serde_json::from_str(&submitted_json)
            .with_context(|| format!("decode submitted JSON `{submitted_json}`"))?,
        streamed_event_count: streamed_event_count as usize,
        replay_cursor,
        queued_turn_ran,
    })
}

fn assert_kitchen_sink_response(response: &TurnResponse, expect_attachment: bool) -> Result<()> {
    anyhow::ensure!(
        response.final_text == EXPECTED_FINAL_TEXT,
        "workflow `{}` final text mismatch: {}",
        response.workflow_id,
        response.final_text
    );
    let submitted = &response.submitted_value;
    anyhow::ensure!(
        submitted.get("foreground").and_then(Value::as_str) == Some("lookup:foreground"),
        "foreground lookup missing from `{}`: {submitted}",
        response.workflow_id
    );
    anyhow::ensure!(
        submitted
            .pointer("/process/parent_lookup")
            .and_then(Value::as_str)
            == Some("lookup:parent"),
        "parent lookup missing from `{}`: {submitted}",
        response.workflow_id
    );
    anyhow::ensure!(
        submitted.pointer("/process/nested").and_then(Value::as_str) == Some("lookup:nested"),
        "nested process output missing from `{}`: {submitted}",
        response.workflow_id
    );
    anyhow::ensure!(
        submitted
            .pointer("/process/parallel/left")
            .and_then(Value::as_str)
            == Some("lookup:left")
            && submitted
                .pointer("/process/parallel/right")
                .and_then(Value::as_str)
                == Some("lookup:right"),
        "parallel process output missing from `{}`: {submitted}",
        response.workflow_id
    );
    anyhow::ensure!(
        submitted.pointer("/process/wake").and_then(Value::as_str) == Some("deferred"),
        "deferred wake marker missing from `{}`: {submitted}",
        response.workflow_id
    );
    if expect_attachment {
        anyhow::ensure!(
            !response.attachment_id.is_empty(),
            "workflow `{}` did not return an attachment id",
            response.workflow_id
        );
        anyhow::ensure!(
            submitted.get("attachment_mime").and_then(Value::as_str) == Some(ATTACHMENT_MIME),
            "workflow `{}` returned wrong attachment mime: {submitted}",
            response.workflow_id
        );
    }
    Ok(())
}

fn assert_queued_wake_response(response: &TurnResponse) -> Result<()> {
    anyhow::ensure!(
        response.final_text == lash_restate_postgres_workers_e2e::EXPECTED_WAKE_TEXT,
        "workflow `{}` wake final mismatch: {}",
        response.workflow_id,
        response.final_text
    );
    anyhow::ensure!(
        response
            .submitted_value
            .get("wake_consumed")
            .and_then(Value::as_bool)
            == Some(true),
        "queued wake workflow did not submit wake_consumed=true: {}",
        response.submitted_value
    );
    anyhow::ensure!(
        response.queued_turn_ran,
        "workflow `{}` did not run queued work",
        response.workflow_id
    );
    Ok(())
}

fn assert_trigger_setup_response(response: &TurnResponse) -> Result<()> {
    anyhow::ensure!(
        response
            .submitted_value
            .get("registered")
            .and_then(Value::as_bool)
            == Some(true),
        "trigger setup did not submit registered=true: {}",
        response.submitted_value
    );
    Ok(())
}

fn assert_signal_suspend_setup_response(response: &TurnResponse) -> Result<String> {
    anyhow::ensure!(
        response
            .submitted_value
            .get("final")
            .and_then(Value::as_str)
            == Some("signal-suspend-started"),
        "signal setup did not submit signal-suspend-started: {}",
        response.submitted_value
    );
    let process_id = response
        .submitted_value
        .get("process_id")
        .and_then(Value::as_str)
        .context("signal setup submitted no process_id")?;
    Ok(process_id.to_string())
}

fn assert_async_completion_response(response: &TurnResponse) -> Result<()> {
    anyhow::ensure!(
        response.final_text == EXPECTED_ASYNC_TEXT,
        "workflow `{}` async final mismatch: {}",
        response.workflow_id,
        response.final_text
    );
    let async_value = response
        .submitted_value
        .get("async")
        .context("async completion response missing async result")?;
    anyhow::ensure!(
        async_value.get("async").and_then(Value::as_bool) == Some(true),
        "async completion result did not mark async=true: {}",
        response.submitted_value
    );
    anyhow::ensure!(
        async_value.get("value").and_then(Value::as_str) == Some("async:detached"),
        "async completion value mismatch: {}",
        response.submitted_value
    );
    Ok(())
}

fn assert_durable_input_response(response: &TurnResponse) -> Result<()> {
    anyhow::ensure!(
        response.final_text == EXPECTED_DURABLE_INPUT_TEXT,
        "workflow `{}` durable input final mismatch: {}",
        response.workflow_id,
        response.final_text
    );
    let durable = response
        .submitted_value
        .get("durable")
        .context("durable input response missing durable result")?;
    anyhow::ensure!(
        durable.get("answer").and_then(Value::as_str) == Some("durable-approved"),
        "durable input answer mismatch: {}",
        response.submitted_value
    );
    anyhow::ensure!(
        durable
            .get("request_id")
            .and_then(Value::as_str)
            .is_some_and(|request_id| request_id.ends_with(":request-1")),
        "durable input request id mismatch: {}",
        response.submitted_value
    );
    Ok(())
}

async fn wait_for_durable_input_key(
    pool: &sqlx::PgPool,
    workflow_id: &str,
) -> Result<AwaitEventKey> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT result_json
             FROM lash_e2e_tool_events
             WHERE workflow_id = $1 AND tool_name = 'durable_input_request.opened'
             ORDER BY event_id DESC
             LIMIT 1",
        )
        .bind(workflow_id)
        .fetch_optional(pool)
        .await
        .with_context(|| format!("load durable input key for `{workflow_id}`"))?;
        if let Some(result_json) = row {
            let value: Value = serde_json::from_str(&result_json)
                .with_context(|| format!("decode durable input key row for `{workflow_id}`"))?;
            let key_value = value
                .get("await_key")
                .cloned()
                .context("durable input key row missing await_key")?;
            let key: AwaitEventKey =
                serde_json::from_value(key_value).context("decode durable input AwaitEventKey")?;
            return Ok(key);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    anyhow::bail!("timed out waiting for durable input key for `{workflow_id}`")
}

async fn wait_for_queued_work(
    storage: &PostgresStorage,
    mock_provider_base_url: &str,
    trace_dir: Option<PathBuf>,
    ingress_url: &str,
) -> Result<()> {
    let registry = process_registry_from_storage(storage);
    let deployment = RestateProcessDeployment::new(ingress_url.to_string(), registry);
    let process_work_driver = deployment.process_work_driver();
    let core = build_e2e_core(lash_restate_postgres_workers_e2e::E2eCoreConfig {
        worker_id: "runner-queue-watch".to_string(),
        storage: storage.clone(),
        attachment_store: Arc::new(s3_store_from_env()?)
            as Arc<dyn lash::persistence::AttachmentStore>,
        process_work_driver,
        restate_ingress_url: ingress_url.to_string(),
        mock_provider_base_url: mock_provider_base_url.to_string(),
        trace_dir,
        fail_once: false,
    })?;
    let session = core.session(DEFAULT_SESSION_ID).rlm().open().await?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let queued = session.queued_work().await?;
        if !queued.is_empty() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    anyhow::bail!("timed out waiting for queued process wake")
}

async fn wait_for_process_signal_wait(
    pool: &sqlx::PgPool,
    process_id: &str,
    signal_name: &str,
    ordinal: u64,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        let row: Option<(String, String)> =
            sqlx::query_as("SELECT status, record_json FROM lash_processes WHERE process_id = $1")
                .bind(process_id)
                .fetch_optional(pool)
                .await
                .with_context(|| format!("load process `{process_id}` wait state"))?;
        if let Some((status, record_json)) = row {
            anyhow::ensure!(
                status == "running",
                "process `{process_id}` reached status `{status}` before signal `{signal_name}` wait"
            );
            let record: Value = serde_json::from_str(&record_json)
                .with_context(|| format!("decode process `{process_id}` record"))?;
            let wait = record.get("wait").cloned().unwrap_or(Value::Null);
            let kind = wait.get("kind").cloned().unwrap_or(Value::Null);
            if kind.get("kind").and_then(Value::as_str) == Some("signal")
                && kind.get("name").and_then(Value::as_str) == Some(signal_name)
                && kind.get("ordinal").and_then(Value::as_u64) == Some(ordinal)
            {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("timed out waiting for process `{process_id}` signal `{signal_name}` wait")
}

async fn emit_button_event(
    storage: &PostgresStorage,
    mock_provider_base_url: &str,
    trace_dir: Option<PathBuf>,
    ingress_url: &str,
) -> Result<String> {
    let registry = process_registry_from_storage(storage);
    let deployment = RestateProcessDeployment::new(ingress_url.to_string(), registry);
    let process_work_driver = deployment.process_work_driver();
    let _poke = deployment.spawn();
    let core = build_e2e_core(lash_restate_postgres_workers_e2e::E2eCoreConfig {
        worker_id: "runner".to_string(),
        storage: storage.clone(),
        attachment_store: Arc::new(s3_store_from_env()?)
            as Arc<dyn lash::persistence::AttachmentStore>,
        process_work_driver,
        restate_ingress_url: ingress_url.to_string(),
        mock_provider_base_url: mock_provider_base_url.to_string(),
        trace_dir,
        fail_once: false,
    })?;
    let source_key = empty_trigger_source_key(BUTTON_SOURCE_TYPE)?;
    let scoped = ScopedEffectController::shared(
        Arc::new(InlineRuntimeEffectController),
        ExecutionScope::runtime_operation("e2e-button-trigger"),
    )?;
    let report = core
        .triggers()
        .emit(
            TriggerOccurrenceRequest::new(
                BUTTON_SOURCE_TYPE,
                source_key,
                json!({
                    "button": "Red",
                    "message": "pressed from runner",
                    "pressed_at": "2026-06-08T12:00:00Z"
                }),
                "e2e-button-red-1",
            )
            .with_source(json!({"runner": true})),
            scoped,
        )
        .await?;
    report
        .started_process_ids
        .first()
        .cloned()
        .context("trigger occurrence did not start a process")
}

async fn assert_signal_process_output(pool: &sqlx::PgPool, process_id: &str) -> Result<()> {
    let event_json: String = sqlx::query_scalar(
        "SELECT event_json
         FROM lash_process_events
         WHERE process_id = $1 AND event_type = 'process.completed'",
    )
    .bind(process_id)
    .fetch_one(pool)
    .await
    .with_context(|| format!("load completed event for signal process `{process_id}`"))?;
    let event: Value = serde_json::from_str(&event_json)
        .with_context(|| format!("decode completed event for signal process `{process_id}`"))?;
    let await_output = event
        .pointer("/payload/await_output")
        .with_context(|| format!("completed event missing await output: {event}"))?;
    anyhow::ensure!(
        await_output.get("type").and_then(Value::as_str) == Some("success"),
        "signal process `{process_id}` completed with non-success output: {await_output}"
    );
    let value = await_output
        .get("value")
        .with_context(|| format!("completed event missing success value: {event}"))?;
    anyhow::ensure!(
        value.pointer("/first/phase").and_then(Value::as_str) == Some("first")
            && value.pointer("/second/phase").and_then(Value::as_str) == Some("second"),
        "signal process `{process_id}` completed with unexpected value: {value}"
    );

    let events: Vec<String> = sqlx::query_scalar(
        "SELECT event_type
         FROM lash_process_events
         WHERE process_id = $1
         ORDER BY sequence",
    )
    .bind(process_id)
    .fetch_all(pool)
    .await
    .with_context(|| format!("load signal process `{process_id}` events"))?;
    let first_signal = events
        .iter()
        .position(|event| event == "signal.first")
        .context("signal.first event missing")?;
    let second_signal = events
        .iter()
        .position(|event| event == "signal.second")
        .context("signal.second event missing")?;
    let completed = events
        .iter()
        .position(|event| event == "process.completed")
        .context("process.completed event missing")?;
    anyhow::ensure!(
        first_signal < second_signal && second_signal < completed,
        "signal process events out of order: {events:?}"
    );
    Ok(())
}

async fn wait_for_process_terminal(pool: &sqlx::PgPool, process_id: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        let status: Option<String> =
            sqlx::query_scalar("SELECT status FROM lash_processes WHERE process_id = $1")
                .bind(process_id)
                .fetch_optional(pool)
                .await
                .with_context(|| format!("load process `{process_id}` status"))?;
        if matches!(
            status.as_deref(),
            Some("completed" | "failed" | "cancelled")
        ) {
            anyhow::ensure!(
                status.as_deref() == Some("completed"),
                "trigger process `{process_id}` ended with {status:?}"
            );
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("timed out waiting for process `{process_id}`")
}

async fn assert_processes_terminal(pool: &sqlx::PgPool) -> Result<()> {
    let rows = sqlx::query_as::<_, (String, String, String)>(
        "SELECT process_id, status, record_json
         FROM lash_processes
         ORDER BY created_at_ms, process_id",
    )
    .fetch_all(pool)
    .await
    .context("load process rows")?;
    anyhow::ensure!(
        rows.len() >= 11,
        "expected at least 11 process rows for kitchen sink + failover + trigger + signal + async completion, got {}",
        rows.len()
    );
    let terminal = rows
        .iter()
        .filter(|(_, status, _)| matches!(status.as_str(), "completed" | "failed" | "cancelled"))
        .count();
    anyhow::ensure!(
        terminal == rows.len(),
        "expected all process rows terminal, got {terminal}/{}: {rows:?}",
        rows.len()
    );
    let record_text = rows
        .iter()
        .map(|(_, _, record)| record.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for needle in [
        "async_child",
        "async:detached",
        "parent",
        "child",
        "on_button",
        "lookup:left",
        "lookup:right",
    ] {
        anyhow::ensure!(
            record_text.contains(needle),
            "process records did not contain `{needle}`"
        );
    }

    let inconsistent_terminal_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM (
            SELECT p.process_id
            FROM lash_processes p
            LEFT JOIN lash_process_events e
              ON e.process_id = p.process_id
             AND e.event_type IN ('process.completed', 'process.failed', 'process.cancelled')
            GROUP BY p.process_id
            HAVING COUNT(e.process_id) <> 1
        ) inconsistent",
    )
    .fetch_one(pool)
    .await
    .context("count process rows without exactly one terminal event")?;
    anyhow::ensure!(
        inconsistent_terminal_events == 0,
        "expected every process to have exactly one terminal event"
    );
    Ok(())
}

async fn assert_no_duplicate_runtime_rows(pool: &sqlx::PgPool) -> Result<()> {
    let queued_work_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM lash_queued_work_batches WHERE session_id = $1")
            .bind(DEFAULT_SESSION_ID)
            .fetch_one(pool)
            .await
            .context("count queued work rows")?;
    anyhow::ensure!(
        queued_work_count == 0,
        "expected no leftover queued work rows after wake consumption, got {queued_work_count}"
    );
    let duplicate_turn_commits: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM (
            SELECT session_id, turn_id
            FROM lash_runtime_turn_commits
            WHERE session_id = $1
            GROUP BY session_id, turn_id
            HAVING COUNT(*) > 1
        ) duplicates",
    )
    .bind(DEFAULT_SESSION_ID)
    .fetch_one(pool)
    .await
    .context("count duplicate runtime turn commits")?;
    anyhow::ensure!(
        duplicate_turn_commits == 0,
        "duplicate runtime turn commits were recorded"
    );
    let artifacts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lash_lashlang_artifacts")
        .fetch_one(pool)
        .await
        .context("count Lashlang artifacts")?;
    anyhow::ensure!(artifacts > 0, "expected Lashlang artifact rows");
    Ok(())
}

async fn assert_worker_distribution(pool: &sqlx::PgPool) -> Result<()> {
    let workers: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT worker_id FROM lash_e2e_worker_events ORDER BY worker_id",
    )
    .fetch_all(pool)
    .await
    .context("list worker ids")?;
    let workers = workers.into_iter().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        workers.contains("worker-a") && workers.contains("worker-b"),
        "expected both worker-a and worker-b to handle work, got {workers:?}"
    );
    Ok(())
}

async fn assert_failover(pool: &sqlx::PgPool) -> Result<()> {
    let exit_worker: String = sqlx::query_scalar(
        "SELECT worker_id
         FROM lash_e2e_failover_markers
         WHERE workflow_id = 'e2e-failover'",
    )
    .fetch_one(pool)
    .await
    .context("load failover exit marker")?;
    let completed_by: Vec<String> = sqlx::query_scalar(
        "SELECT worker_id
         FROM lash_e2e_worker_events
         WHERE workflow_id = 'e2e-failover'
           AND event_type = 'turn_completed'
         ORDER BY worker_id",
    )
    .fetch_all(pool)
    .await
    .context("load failover completion worker")?;
    anyhow::ensure!(
        completed_by.iter().any(|worker| worker != &exit_worker),
        "expected a peer to complete failover after {exit_worker} exited, got {completed_by:?}"
    );
    let final_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)
         FROM lash_e2e_terminal_results
         WHERE workflow_id = 'e2e-failover'",
    )
    .fetch_one(pool)
    .await
    .context("count failover final rows")?;
    anyhow::ensure!(
        final_rows == 1,
        "failover workflow recorded {final_rows} final rows"
    );
    Ok(())
}

async fn assert_provider_calls(pool: &sqlx::PgPool) -> Result<()> {
    let bad_model: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_e2e_provider_calls WHERE model <> 'e2e-mock'",
    )
    .fetch_one(pool)
    .await
    .context("count provider calls with wrong model")?;
    anyhow::ensure!(bad_model == 0, "provider saw {bad_model} wrong-model calls");
    let failover_calls: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_e2e_provider_calls
         WHERE workflow_id = 'e2e-failover' AND scenario = 'kitchen_sink'",
    )
    .fetch_one(pool)
    .await
    .context("count failover provider calls")?;
    anyhow::ensure!(
        failover_calls == 1,
        "expected one durable failover provider completion, got {failover_calls}"
    );
    let scenarios: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT scenario FROM lash_e2e_provider_calls ORDER BY scenario",
    )
    .fetch_all(pool)
    .await
    .context("list provider scenarios")?;
    for expected in [
        "async_completion",
        "durable_input_request",
        "kitchen_sink",
        "queued_wake",
        "trigger_setup",
        "signal_suspend",
    ] {
        anyhow::ensure!(
            scenarios.iter().any(|scenario| scenario == expected),
            "provider scenario `{expected}` missing from {scenarios:?}"
        );
    }
    Ok(())
}

async fn assert_tool_and_turn_telemetry(pool: &sqlx::PgPool) -> Result<()> {
    for tool in [
        "app_lookup",
        "async_lookup",
        "make_attachment",
        "crash_once",
        "durable_input_request",
    ] {
        let count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM lash_e2e_tool_events WHERE tool_name = $1")
                .bind(tool)
                .fetch_one(pool)
                .await
                .with_context(|| format!("count tool events for `{tool}`"))?;
        anyhow::ensure!(count > 0, "missing tool telemetry for `{tool}`");
    }
    let async_resolutions: Vec<String> = sqlx::query_scalar(
        "SELECT result_json
         FROM lash_e2e_tool_events
         WHERE tool_name = 'async_lookup.resolve'",
    )
    .fetch_all(pool)
    .await
    .context("load async lookup resolution telemetry")?;
    let accepted = async_resolutions
        .iter()
        .filter_map(|row| serde_json::from_str::<Value>(row).ok())
        .any(|row| {
            row.pointer("/outcome/status").and_then(Value::as_str) == Some("accepted")
                && row.pointer("/result/value").and_then(Value::as_str) == Some("async:detached")
        });
    anyhow::ensure!(
        accepted,
        "async lookup did not record an accepted external resolution: {async_resolutions:?}"
    );
    let turn_events: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lash_e2e_turn_events")
        .fetch_one(pool)
        .await
        .context("count streamed turn events")?;
    anyhow::ensure!(turn_events > 0, "no streamed turn activities were recorded");
    let cursor_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_e2e_turn_events
         WHERE stream_name = 'main' AND cursor IS NOT NULL",
    )
    .fetch_one(pool)
    .await
    .context("count cursor-bearing turn events")?;
    anyhow::ensure!(
        cursor_events > 0,
        "no streamed turn event recorded a replay cursor"
    );
    let live_replay_checks: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_e2e_worker_events WHERE event_type = 'live_replay_checked'",
    )
    .fetch_one(pool)
    .await
    .context("count live replay checks")?;
    anyhow::ensure!(
        live_replay_checks >= 5,
        "expected live replay checks for five workflow turns, got {live_replay_checks}"
    );
    Ok(())
}

async fn assert_durable_input_steps(pool: &sqlx::PgPool) -> Result<()> {
    let rows: Vec<(String, i64)> = sqlx::query_as(
        "SELECT step_id, count
         FROM lash_e2e_durable_step_counts
         WHERE workflow_id = 'e2e-durable-input'
         ORDER BY step_id",
    )
    .fetch_all(pool)
    .await
    .context("load durable input step counts")?;
    let counts = rows
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    for step_id in ["complete", "create"] {
        let count = counts.get(step_id).copied().unwrap_or_default();
        anyhow::ensure!(
            count == 1,
            "durable input step `{step_id}` ran {count} times; counts={counts:?}"
        );
    }
    Ok(())
}

async fn assert_trigger_delivery(pool: &sqlx::PgPool, trigger_process_id: &str) -> Result<()> {
    let trigger_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM lash_trigger_subscriptions
         WHERE source_type = $1 AND enabled = true",
    )
    .bind(BUTTON_SOURCE_TYPE)
    .fetch_one(pool)
    .await
    .context("count trigger subscriptions")?;
    anyhow::ensure!(
        trigger_count == 1,
        "expected one enabled trigger, got {trigger_count}"
    );
    let delivery_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM lash_trigger_deliveries WHERE process_id = $1")
            .bind(trigger_process_id)
            .fetch_one(pool)
            .await
            .context("count trigger occurrence deliveries")?;
    anyhow::ensure!(
        delivery_count == 1,
        "expected one trigger delivery for `{trigger_process_id}`, got {delivery_count}"
    );
    Ok(())
}

async fn assert_attachments_round_trip(
    pool: &sqlx::PgPool,
    store: &impl lash::persistence::AttachmentStore,
    responses: &[TurnResponse],
) -> Result<()> {
    for response in responses
        .iter()
        .filter(|response| !response.attachment_id.is_empty())
    {
        let id = lash_core::AttachmentId::new(response.attachment_id.clone());
        let stored = store
            .get(&id)
            .await
            .with_context(|| format!("read worker attachment `{id}` from MinIO"))?;
        anyhow::ensure!(
            stored.bytes == expected_attachment_bytes(&response.workflow_id),
            "worker attachment `{id}` bytes did not match expected content"
        );
        anyhow::ensure!(
            stored.meta.media_type.canonical_mime() == ATTACHMENT_MIME,
            "worker attachment `{id}` had wrong mime"
        );
        anyhow::ensure!(
            stored.meta.label.as_deref() == Some("kitchen-sink.png"),
            "worker attachment `{id}` had wrong label {:?}",
            stored.meta.label
        );
        let manifest: Option<(String, Option<i64>)> = sqlx::query_as(
            "SELECT session_id, committed_at_ms
             FROM lash_attachment_manifest
             WHERE attachment_id = $1",
        )
        .bind(response.attachment_id.as_str())
        .fetch_optional(pool)
        .await
        .with_context(|| format!("load attachment manifest for `{}`", response.attachment_id))?;
        let (session_id, committed_at_ms) = manifest.with_context(|| {
            format!(
                "missing attachment manifest row for `{}`",
                response.attachment_id
            )
        })?;
        anyhow::ensure!(
            session_id == DEFAULT_SESSION_ID,
            "attachment manifest session mismatch for `{}`: {session_id}",
            response.attachment_id
        );
        anyhow::ensure!(
            committed_at_ms.is_some(),
            "attachment manifest row for `{}` was not committed",
            response.attachment_id
        );
    }
    Ok(())
}

async fn assert_reopened_session_agrees(
    storage: &PostgresStorage,
    mock_provider_base_url: &str,
    trace_dir: Option<PathBuf>,
    ingress_url: &str,
    responses: &[TurnResponse],
) -> Result<()> {
    let registry = process_registry_from_storage(storage);
    let deployment = RestateProcessDeployment::new(ingress_url.to_string(), registry);
    let process_work_driver = deployment.process_work_driver();
    let _poke = deployment.spawn();
    let core = build_e2e_core(lash_restate_postgres_workers_e2e::E2eCoreConfig {
        worker_id: "runner-reopen".to_string(),
        storage: storage.clone(),
        attachment_store: Arc::new(s3_store_from_env()?)
            as Arc<dyn lash::persistence::AttachmentStore>,
        process_work_driver,
        restate_ingress_url: ingress_url.to_string(),
        mock_provider_base_url: mock_provider_base_url.to_string(),
        trace_dir,
        fail_once: false,
    })?;
    let session = core.session(DEFAULT_SESSION_ID).rlm().open().await?;
    let read = storage
        .session_store(DEFAULT_SESSION_ID)
        .load_session(lash::persistence::SessionReadScope::FullGraph)
        .await
        .context("load persisted runtime session")?
        .context("runtime session was not persisted")?;
    anyhow::ensure!(
        read.session_id == DEFAULT_SESSION_ID,
        "expected session `{}`, got `{}`",
        DEFAULT_SESSION_ID,
        read.session_id
    );
    let queued = session.queued_work().await?;
    anyhow::ensure!(
        queued.is_empty(),
        "reopened session had queued work: {queued:?}"
    );
    let submitted_finals = responses
        .iter()
        .filter_map(|response| {
            response
                .submitted_value
                .get("final")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        submitted_finals.contains(EXPECTED_FINAL_TEXT),
        "reopened assertion did not see submitted final `{EXPECTED_FINAL_TEXT}` in {submitted_finals:?}"
    );
    Ok(())
}

async fn assert_traces(trace_dir: &Path) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        let files = std::fs::read_dir(trace_dir)
            .with_context(|| format!("read trace dir `{}`", trace_dir.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
            .collect::<Vec<_>>();
        if !files.is_empty() {
            let mut combined = String::new();
            for file in files {
                combined.push_str(&std::fs::read_to_string(&file).unwrap_or_default());
                combined.push('\n');
            }
            for needle in [
                "app_lookup",
                "async_lookup",
                "make_attachment",
                "crash_once",
                "parent",
                "child",
                "parent_wake",
                "on_button",
            ] {
                anyhow::ensure!(
                    combined.contains(needle),
                    "trace JSONL did not contain `{needle}`"
                );
            }
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("no trace JSONL files appeared in `{}`", trace_dir.display())
}
