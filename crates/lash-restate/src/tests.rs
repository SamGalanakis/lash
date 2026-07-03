//! Tests for the Restate adapter (extracted from lib.rs).

use super::*;
use bytes::{BufMut, Bytes, BytesMut};
use http_body_util::{BodyExt, Empty, Full};
use lash_core::{ProcessInput, ProcessRegistration};
use lash_lashlang_runtime::{LashlangToolBinding, ToolDefinitionLashlangExt};
use restate_sdk::prelude::Endpoint;
use restate_sdk::service::Discoverable;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[test]
fn restate_effect_name_uses_lash_replay_key() {
    let invocation = RuntimeInvocation::effect(
        lash_core::runtime::RuntimeScope::for_turn("session", "turn", 1, 2),
        "effect",
        RuntimeEffectKind::ToolAttempt,
        "session:turn:1:2:tool_attempt:effect",
    );

    assert_eq!(
        restate_effect_name(&invocation),
        "lash:session:turn:1:2:tool_attempt:effect"
    );
}

#[tokio::test]
async fn restate_effect_host_satisfies_scope_factory_conformance() {
    lash_core::testing::conformance::effect_host(|| Arc::new(RestateEffectHost::new())).await;
}

#[tokio::test]
async fn restate_handler_controller_satisfies_concurrent_replay_conformance() {
    let context = Arc::new(ReplayableRecordingContext::default());
    let controller = RestateRuntimeEffectController::new(Arc::clone(&context));
    let replay_context = Arc::clone(&context);

    lash_core::testing::conformance::effect_controller_concurrent_replay_deterministic(
        &controller,
        move || replay_context.start_replay(),
    )
    .await;

    let durable_context = Arc::new(ReplayableRecordingContext::default());
    let durable_controller = RestateRuntimeEffectController::new(Arc::clone(&durable_context));
    let durable_replay_context = Arc::clone(&durable_context);
    lash_core::testing::conformance::effect_controller_durable_steps_replay(
        &durable_controller,
        move || durable_replay_context.start_replay(),
    )
    .await;

    let tool_context = Arc::new(ReplayableRecordingContext::default());
    let tool_controller = RestateRuntimeEffectController::new(Arc::clone(&tool_context));
    let tool_replay_context = Arc::clone(&tool_context);
    lash_core::testing::conformance::effect_controller_tool_attempt_fanout_replay_deterministic(
        &tool_controller,
        move || tool_replay_context.start_replay(),
    )
    .await;

    let runs = context.runs();
    assert_eq!(runs.len(), 4);
    assert!(runs.iter().any(|name| name.ends_with(":effect-slow")));
    assert!(runs.iter().any(|name| name.ends_with(":effect-fast")));

    let tool_runs = tool_context.runs();
    assert_eq!(tool_runs.len(), 4);
    assert!(
        tool_runs
            .iter()
            .any(|name| name.ends_with(":tool-attempt-slow"))
    );
    assert!(
        tool_runs
            .iter()
            .any(|name| name.ends_with(":tool-attempt-fast"))
    );
}

#[test]
fn restate_handler_controller_disallows_concurrent_effect_calls() {
    let controller = RestateRuntimeEffectController::new(Arc::new(RecordingContext::default()));

    assert!(
        !controller.supports_concurrent_effects(),
        "Restate handler context calls such as ctx.run must be awaited before the next effect call"
    );
}

#[test]
fn recorded_runtime_effect_hash_mismatch_fails_explicitly() {
    let recorded = RecordedRuntimeEffect {
        envelope_hash: "old".to_string(),
        outcome: Ok(RuntimeEffectOutcome::Sleep),
    };

    let err = validate_recorded_effect_hash(recorded, "new").expect_err("hash mismatch");

    assert_eq!(err.code, "restate_effect_hash_mismatch");
}

#[test]
fn recorded_runtime_effect_hash_match_returns_replayed_outcome() {
    let recorded = RecordedRuntimeEffect {
        envelope_hash: "same".to_string(),
        outcome: Ok(RuntimeEffectOutcome::Sleep),
    };

    let outcome = validate_recorded_effect_hash(recorded, "same")
        .expect("hash match")
        .expect("replayed outcome");

    assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
}

fn llm_spec() -> lash_core::LlmRequestSpec {
    lash_core::LlmRequestSpec {
        model: "model".to_string(),
        messages: Vec::new(),
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: Default::default(),
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        scope: lash_core::LlmRequestScope::new(
            "session".to_string(),
            "session:frame:test".to_string(),
            "session:request:test".to_string(),
        ),
        output_spec: None,
    }
}

fn prepared_tool_call() -> lash_core::PreparedToolCall {
    lash_core::PreparedToolCall::from_parts(
        "call-1",
        "tool:tool",
        "tool",
        serde_json::json!({}),
        None,
        serde_json::Value::Null,
    )
}

fn prepared_tool_call_with(call_id: &str, tool_name: &str) -> lash_core::PreparedToolCall {
    lash_core::PreparedToolCall::from_parts(
        call_id,
        format!("tool:{tool_name}"),
        tool_name,
        serde_json::json!({ "call": call_id }),
        None,
        serde_json::Value::Null,
    )
}

fn completed_tool_record(call_id: &str, tool_name: &str) -> lash_core::ToolCallRecord {
    lash_core::ToolCallRecord {
        call_id: Some(call_id.to_string()),
        tool: tool_name.to_string(),
        args: serde_json::json!({ "call": call_id }),
        output: lash_core::ToolCallOutput::success(serde_json::json!({ "call": call_id })),
        duration_ms: 1,
    }
}

fn external_registration(id: &str) -> ProcessRegistration {
    ProcessRegistration::new(
        id,
        ProcessInput::External {
            metadata: serde_json::Value::Null,
        },
        lash_core::ProcessProvenance::host(),
    )
}

fn sync_await<T, F>(future: F) -> T
where
    T: Send + 'static,
    F: std::future::Future<Output = T> + Send + 'static,
{
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    })
    .join()
    .expect("runtime thread")
}

fn process_registry() -> Arc<dyn ProcessRegistry> {
    Arc::new(sync_await(async {
        lash_sqlite_store::SqliteProcessRegistry::memory()
            .await
            .expect("sqlite registry")
    }))
}

fn lashlang_process_input(input: lash_lashlang_runtime::LashlangProcessInput) -> ProcessInput {
    input
        .into_process_input()
        .expect("serialize lashlang process input")
}

#[derive(Default)]
struct DurableMemoryAttachmentStore {
    inner: lash_core::InMemoryAttachmentStore,
}

#[async_trait::async_trait]
impl lash_core::AttachmentStore for DurableMemoryAttachmentStore {
    fn persistence(&self) -> lash_core::AttachmentStorePersistence {
        lash_core::AttachmentStorePersistence::Durable
    }

    async fn put(
        &self,
        bytes: Vec<u8>,
        meta: lash_core::AttachmentCreateMeta,
    ) -> Result<lash_core::AttachmentRef, lash_core::AttachmentStoreError> {
        self.inner.put(bytes, meta).await
    }

    async fn get(
        &self,
        id: &lash_core::AttachmentId,
    ) -> Result<lash_core::StoredAttachment, lash_core::AttachmentStoreError> {
        self.inner.get(id).await
    }

    async fn delete(
        &self,
        id: &lash_core::AttachmentId,
    ) -> Result<(), lash_core::AttachmentStoreError> {
        self.inner.delete(id).await
    }
}

#[derive(Default)]
struct DurableMemoryProcessEnvStore {
    inner: lash_core::InMemoryProcessExecutionEnvStore,
}

#[async_trait::async_trait]
impl lash_core::ProcessExecutionEnvStore for DurableMemoryProcessEnvStore {
    fn durability_tier(&self) -> lash_core::DurabilityTier {
        lash_core::DurabilityTier::Durable
    }

    async fn put_process_execution_env(
        &self,
        env_ref: &lash_core::ProcessExecutionEnvRef,
        bytes: &[u8],
    ) -> Result<(), lash_core::PluginError> {
        self.inner.put_process_execution_env(env_ref, bytes).await
    }

    async fn get_process_execution_env(
        &self,
        env_ref: &lash_core::ProcessExecutionEnvRef,
    ) -> Result<Option<Vec<u8>>, lash_core::PluginError> {
        self.inner.get_process_execution_env(env_ref).await
    }
}

static RECOVERY_PROCESS_ENV_STORE: LazyLock<Arc<DurableMemoryProcessEnvStore>> =
    LazyLock::new(|| Arc::new(DurableMemoryProcessEnvStore::default()));

struct CommitRetryStore {
    inner: Arc<dyn lash_core::RuntimePersistence>,
}

lash_core::impl_noop_attachment_manifest!(CommitRetryStore);

// Pass-through wrapper over the shared in-memory recovery store; every
// segment delegates to `inner`.
#[async_trait::async_trait]
impl lash_core::SessionCommitStore for CommitRetryStore {
    fn durability_tier(&self) -> lash_core::DurabilityTier {
        lash_core::DurabilityTier::Durable
    }

    async fn load_session(
        &self,
        _scope: lash_core::SessionReadScope,
    ) -> Result<Option<lash_core::store::PersistedSessionRead>, lash_core::StoreError> {
        Ok(None)
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<lash_core::SessionNodeRecord>, lash_core::StoreError> {
        self.inner.load_node(node_id).await
    }

    async fn commit_runtime_state(
        &self,
        commit: lash_core::store::RuntimeCommit,
    ) -> Result<lash_core::store::RuntimeCommitResult, lash_core::StoreError> {
        self.inner.commit_runtime_state(commit).await
    }

    async fn save_session_meta(
        &self,
        meta: lash_core::SessionMeta,
    ) -> Result<(), lash_core::StoreError> {
        self.inner.save_session_meta(meta).await
    }

    async fn load_session_meta(
        &self,
    ) -> Result<Option<lash_core::SessionMeta>, lash_core::StoreError> {
        self.inner.load_session_meta().await
    }
}

#[async_trait::async_trait]
impl lash_core::SessionExecutionLeaseStore for CommitRetryStore {
    async fn try_claim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &lash_core::LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<lash_core::SessionExecutionLeaseClaimOutcome, lash_core::StoreError> {
        self.inner
            .try_claim_session_execution_lease(session_id, owner, lease_ttl_ms)
            .await
    }

    async fn reclaim_session_execution_lease(
        &self,
        session_id: &str,
        owner: &lash_core::LeaseOwnerIdentity,
        observed_holder: &lash_core::SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<lash_core::SessionExecutionLeaseClaimOutcome, lash_core::StoreError> {
        self.inner
            .reclaim_session_execution_lease(session_id, owner, observed_holder, lease_ttl_ms)
            .await
    }

    async fn renew_session_execution_lease(
        &self,
        fence: &lash_core::SessionExecutionLeaseFence,
        lease_ttl_ms: u64,
    ) -> Result<lash_core::SessionExecutionLease, lash_core::StoreError> {
        self.inner
            .renew_session_execution_lease(fence, lease_ttl_ms)
            .await
    }

    async fn release_session_execution_lease(
        &self,
        completion: &lash_core::SessionExecutionLeaseCompletion,
    ) -> Result<(), lash_core::StoreError> {
        self.inner.release_session_execution_lease(completion).await
    }
}

#[async_trait::async_trait]
impl lash_core::QueuedWorkStore for CommitRetryStore {
    async fn enqueue_queued_work(
        &self,
        batch: lash_core::runtime::QueuedWorkBatchDraft,
    ) -> Result<lash_core::runtime::QueuedWorkBatch, lash_core::StoreError> {
        self.inner.enqueue_queued_work(batch).await
    }

    async fn claim_leading_ready_session_command(
        &self,
        session_id: &str,
        session_execution_lease: &lash_core::SessionExecutionLeaseFence,
        owner: &lash_core::LeaseOwnerIdentity,
        lease_ttl_ms: u64,
    ) -> Result<Option<lash_core::runtime::QueuedWorkClaim>, lash_core::StoreError> {
        self.inner
            .claim_leading_ready_session_command(
                session_id,
                session_execution_lease,
                owner,
                lease_ttl_ms,
            )
            .await
    }

    async fn claim_ready_queued_work(
        &self,
        session_id: &str,
        session_execution_lease: &lash_core::SessionExecutionLeaseFence,
        owner: &lash_core::LeaseOwnerIdentity,
        boundary: lash_core::runtime::QueuedWorkClaimBoundary,
        lease_ttl_ms: u64,
        max_batches: usize,
    ) -> Result<Option<lash_core::runtime::QueuedWorkClaim>, lash_core::StoreError> {
        self.inner
            .claim_ready_queued_work(
                session_id,
                session_execution_lease,
                owner,
                boundary,
                lease_ttl_ms,
                max_batches,
            )
            .await
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &lash_core::runtime::QueuedWorkClaim,
        lease_ttl_ms: u64,
    ) -> Result<lash_core::runtime::QueuedWorkClaim, lash_core::StoreError> {
        self.inner
            .renew_queued_work_claim(claim, lease_ttl_ms)
            .await
    }

    async fn abandon_queued_work_claim(
        &self,
        claim: &lash_core::runtime::QueuedWorkClaim,
    ) -> Result<(), lash_core::StoreError> {
        self.inner.abandon_queued_work_claim(claim).await
    }

    async fn cancel_queued_work_batch(
        &self,
        session_id: &str,
        batch_id: &str,
    ) -> Result<Option<lash_core::runtime::QueuedWorkBatch>, lash_core::StoreError> {
        self.inner
            .cancel_queued_work_batch(session_id, batch_id)
            .await
    }

    async fn list_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<lash_core::runtime::QueuedWorkBatch>, lash_core::StoreError> {
        self.inner.list_queued_work(session_id).await
    }

    async fn list_pending_queued_work(
        &self,
        session_id: &str,
    ) -> Result<Vec<lash_core::runtime::QueuedWorkBatch>, lash_core::StoreError> {
        self.inner.list_pending_queued_work(session_id).await
    }
}

#[async_trait::async_trait]
impl lash_core::TurnInputStore for CommitRetryStore {
    async fn enqueue_pending_turn_input(
        &self,
        input: lash_core::PendingTurnInputDraft,
    ) -> Result<lash_core::PendingTurnInput, lash_core::StoreError> {
        self.inner.enqueue_pending_turn_input(input).await
    }

    async fn list_pending_turn_inputs(
        &self,
        session_id: &str,
    ) -> Result<Vec<lash_core::PendingTurnInput>, lash_core::StoreError> {
        self.inner.list_pending_turn_inputs(session_id).await
    }

    async fn cancel_pending_turn_inputs(
        &self,
        session_id: &str,
        targets: &[lash_core::PendingTurnInputCancelTarget],
    ) -> Result<Vec<lash_core::PendingTurnInputCancelResult>, lash_core::StoreError> {
        self.inner
            .cancel_pending_turn_inputs(session_id, targets)
            .await
    }

    async fn cancel_pending_turn_input_suffix(
        &self,
        session_id: &str,
        anchor: &lash_core::PendingTurnInputCancelTarget,
    ) -> Result<lash_core::PendingTurnInputSuffixCancelOutcome, lash_core::StoreError> {
        self.inner
            .cancel_pending_turn_input_suffix(session_id, anchor)
            .await
    }

    async fn claim_active_turn_inputs(
        &self,
        session_id: &str,
        session_execution_lease: &lash_core::SessionExecutionLeaseFence,
        owner: &lash_core::LeaseOwnerIdentity,
        turn_id: &str,
        checkpoint: lash_core::CheckpointKind,
        lease_ttl_ms: u64,
        max_inputs: usize,
    ) -> Result<Option<lash_core::runtime::TurnInputClaim>, lash_core::StoreError> {
        self.inner
            .claim_active_turn_inputs(
                session_id,
                session_execution_lease,
                owner,
                turn_id,
                checkpoint,
                lease_ttl_ms,
                max_inputs,
            )
            .await
    }

    async fn claim_next_turn_inputs(
        &self,
        session_id: &str,
        session_execution_lease: &lash_core::SessionExecutionLeaseFence,
        owner: &lash_core::LeaseOwnerIdentity,
        lease_ttl_ms: u64,
        max_inputs: usize,
    ) -> Result<Option<lash_core::runtime::TurnInputClaim>, lash_core::StoreError> {
        self.inner
            .claim_next_turn_inputs(
                session_id,
                session_execution_lease,
                owner,
                lease_ttl_ms,
                max_inputs,
            )
            .await
    }

    async fn abandon_turn_input_claim(
        &self,
        claim: &lash_core::runtime::TurnInputClaim,
    ) -> Result<(), lash_core::StoreError> {
        self.inner.abandon_turn_input_claim(claim).await
    }
}

#[async_trait::async_trait]
impl lash_core::StoreMaintenance for CommitRetryStore {
    async fn tombstone_nodes(&self, ids: &[String]) -> Result<(), lash_core::StoreError> {
        self.inner.tombstone_nodes(ids).await
    }

    async fn vacuum(&self) -> Result<lash_core::VacuumReport, lash_core::StoreError> {
        self.inner.vacuum().await
    }

    async fn gc_unreachable(&self) -> Result<lash_core::GcReport, lash_core::StoreError> {
        self.inner.gc_unreachable().await
    }
}

const RESTATE_INVOCATION_CONTENT_TYPE: &str = "application/vnd.restate.invocation.v6";

fn encode_restate_message(message_type: u16, payload: Vec<u8>) -> Bytes {
    let mut encoded = BytesMut::with_capacity(8 + payload.len());
    let header = ((message_type as u64) << 48) | payload.len() as u64;
    encoded.put_u64(header);
    encoded.extend_from_slice(&payload);
    encoded.freeze()
}

fn put_varint(buf: &mut BytesMut, mut value: u64) {
    while value >= 0x80 {
        buf.put_u8(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    buf.put_u8(value as u8);
}

fn put_field_key(buf: &mut BytesMut, field_number: u32, wire_type: u8) {
    put_varint(buf, ((field_number as u64) << 3) | wire_type as u64);
}

fn put_varint_field(buf: &mut BytesMut, field_number: u32, value: u64) {
    put_field_key(buf, field_number, 0);
    put_varint(buf, value);
}

fn put_len_field(buf: &mut BytesMut, field_number: u32, value: &[u8]) {
    put_field_key(buf, field_number, 2);
    put_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

fn encode_start_message(workflow_key: &str) -> Bytes {
    let mut payload = BytesMut::new();
    put_len_field(&mut payload, 1, workflow_key.as_bytes());
    put_len_field(&mut payload, 2, workflow_key.as_bytes());
    put_varint_field(&mut payload, 3, 1);
    put_len_field(&mut payload, 6, workflow_key.as_bytes());
    encode_restate_message(0x0000, payload.to_vec())
}

fn encode_input_command(payload: &[u8]) -> Bytes {
    let mut value = BytesMut::new();
    put_len_field(&mut value, 1, payload);

    let mut command = BytesMut::new();
    put_len_field(&mut command, 14, &value);
    encode_restate_message(0x0400, command.to_vec())
}

fn encode_invocation_body<T: serde::Serialize>(
    workflow_key: &str,
    input: &T,
) -> Result<Bytes, TerminalError> {
    let input = serde_json::to_vec(input).map_err(TerminalError::from_error)?;
    let start = encode_start_message(workflow_key);
    let input = encode_input_command(&input);
    let mut body = BytesMut::with_capacity(start.len() + input.len());
    body.extend_from_slice(&start);
    body.extend_from_slice(&input);
    Ok(body.freeze())
}

async fn invoke_process_workflow_endpoint<T: serde::Serialize>(
    endpoint: &Endpoint,
    handler: &str,
    workflow_key: &str,
    input: &T,
) -> Result<Bytes, TerminalError> {
    let response = endpoint.handle(
        http::Request::builder()
            .uri(format!("/invoke/LashProcessWorkflow/{handler}"))
            .header(http::header::CONTENT_TYPE, RESTATE_INVOCATION_CONTENT_TYPE)
            .body(Full::new(encode_invocation_body(workflow_key, input)?))
            .expect("workflow invocation request"),
    );
    let status = response.status();
    if !status.is_success() {
        return Err(TerminalError::new_with_code(
            status.as_u16(),
            format!("workflow endpoint invocation returned status {status}"),
        ));
    }
    response
        .into_body()
        .collect()
        .await
        .map(|body| body.to_bytes())
        .map_err(|err| TerminalError::new(format!("workflow endpoint body failed: {err}")))
}

#[test]
fn restate_command_execution_plan_is_explicit_for_every_command() {
    let cases = vec![
        (
            RuntimeEffectCommand::Sleep { duration_ms: 1 },
            RestateEffectExecution::Timer,
        ),
        (
            RuntimeEffectCommand::process(ProcessCommand::List {
                session_scope: lash_core::SessionScope::new("session"),
                mode: lash_core::ProcessListMode::Live,
            }),
            RestateEffectExecution::DirectProcess,
        ),
        (
            RuntimeEffectCommand::AwaitEvent {
                key: restate_await_event_key(
                    &ExecutionScope::turn("session", "turn"),
                    AwaitEventWaitIdentity::Custom {
                        key: "event".to_string(),
                    },
                )
                .expect("await-event key"),
            },
            RestateEffectExecution::AwaitEvent,
        ),
        (
            RuntimeEffectCommand::LlmCall {
                request: Box::new(llm_spec()),
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::Direct {
                request: Box::new(llm_spec()),
                usage_source: "test".to_string(),
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::ToolAttempt {
                call: prepared_tool_call(),
                execution_grant: None,
                attempt: 1,
                max_attempts: 1,
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::ToolBatch {
                batch: lash_core::PreparedToolBatch::new("batch", vec![prepared_tool_call()]),
            },
            RestateEffectExecution::DirectLocal,
        ),
        (
            RuntimeEffectCommand::ExecCode {
                language: "code".to_string(),
                code: "1 + 1".to_string(),
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::Checkpoint {
                checkpoint: lash_core::CheckpointKind::AfterWork,
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::SyncExecutionEnvironment {
                update_machine_config: true,
            },
            RestateEffectExecution::JournaledRun,
        ),
        (
            RuntimeEffectCommand::DurableStep {
                step_id: "step".to_string(),
                input: serde_json::json!({ "x": 1 }),
            },
            RestateEffectExecution::JournaledRun,
        ),
    ];

    for (command, expected) in cases {
        assert_eq!(restate_effect_execution(&command), expected);
    }
}

#[derive(Default)]
struct RecordingContext {
    endpoint: Option<Endpoint>,
    sleeps: Mutex<Vec<u64>>,
    runs: Mutex<Vec<String>>,
    started: Mutex<Vec<ProcessRegistration>>,
    started_execution_contexts: Mutex<Vec<ProcessExecutionContext>>,
    process_command_log: Mutex<Vec<String>>,
    cancelled: Mutex<Vec<(String, Option<String>)>>,
    resolved_events: Mutex<Vec<RestateProcessEventResolveRequest>>,
    awaited_events: Mutex<HashMap<String, Resolution>>,
}

impl RecordingContext {
    fn with_endpoint(endpoint: Endpoint) -> Self {
        Self {
            endpoint: Some(endpoint),
            ..Default::default()
        }
    }

    fn resolve_process_terminal(&self, process_id: &str, output: &ProcessAwaitOutput) {
        let key = restate_process_terminal_await_key(process_id).expect("terminal await key");
        let resolution =
            restate_process_terminal_resolution(output).expect("terminal await resolution");
        self.awaited_events
            .lock()
            .expect("awaited events lock")
            .insert(key.promise_key(), resolution);
    }
}

impl<'ctx> RestateControllerContext<'ctx> for Arc<RecordingContext> {
    fn sleep_send<'run>(
        &'run self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        self.sleeps
            .lock()
            .expect("sleeps lock")
            .push(duration.as_millis() as u64);
        Box::pin(async { Ok(()) })
    }

    fn run_json_send<'run, T, Fut>(
        &'run self,
        _effect_name: String,
        _retry_policy: Option<RunRetryPolicy>,
        future: Fut,
    ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
        T: Serialize + DeserializeOwned + Send + 'static,
        Fut: Future<Output = T> + Send + 'run,
    {
        self.runs.lock().expect("runs lock").push(_effect_name);
        Box::pin(async move { Ok(Json(future.await)) })
    }

    fn start_process_workflow<'run>(
        &'run self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        let process_id = registration.id.clone();
        let endpoint = self.endpoint.clone();
        self.process_command_log
            .lock()
            .expect("process command log lock")
            .push(format!("send:{process_id}"));
        self.started
            .lock()
            .expect("started lock")
            .push(registration.clone());
        self.started_execution_contexts
            .lock()
            .expect("started execution contexts lock")
            .push(execution_context.clone());
        Box::pin(async move {
            if let Some(endpoint) = endpoint {
                invoke_process_workflow_endpoint(
                    &endpoint,
                    "run",
                    &process_id,
                    &RestateProcessWorkflowInput {
                        registration,
                        execution_context,
                    },
                )
                .await?;
            }
            Ok(format!("invocation-{process_id}"))
        })
    }

    fn request_process_workflow_cancel<'run>(
        &'run self,
        request: RestateProcessCancelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        let endpoint = self.endpoint.clone();
        let process_id = request.process_id.clone();
        self.cancelled
            .lock()
            .expect("cancelled lock")
            .push((request.process_id.clone(), request.reason.clone()));
        Box::pin(async move {
            if let Some(endpoint) = endpoint {
                invoke_process_workflow_endpoint(&endpoint, "cancel", &process_id, &request)
                    .await?;
            }
            Ok(())
        })
    }

    fn await_event<'run>(
        &'run self,
        key: String,
    ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        let resolution = self
            .awaited_events
            .lock()
            .expect("awaited events lock")
            .get(&key)
            .cloned();
        Box::pin(async move {
            resolution
                .ok_or_else(|| TerminalError::new(format!("event await is unresolved: {key}")))
        })
    }

    fn await_process_terminal<'run>(
        &'run self,
        process_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<ProcessAwaitOutput, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        self.process_command_log
            .lock()
            .expect("process command log lock")
            .push(format!("call:{process_id}"));
        let result = restate_process_terminal_await_key(&process_id)
            .map_err(TerminalError::from_error)
            .and_then(|key| {
                self.awaited_events
                    .lock()
                    .expect("awaited events lock")
                    .get(&key.promise_key())
                    .cloned()
                    .ok_or_else(|| {
                        TerminalError::new(format!(
                            "process terminal await is unresolved: {process_id}"
                        ))
                    })
            })
            .and_then(|resolution| {
                restate_process_terminal_output(&process_id, resolution)
                    .map_err(TerminalError::from_error)
            });
        Box::pin(async move { result })
    }

    fn resolve_event<'run>(
        &'run self,
        request: RestateProcessEventResolveRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        self.resolved_events
            .lock()
            .expect("resolved events lock")
            .push(request.clone());
        self.awaited_events
            .lock()
            .expect("awaited events lock")
            .insert(request.key, request.resolution);
        Box::pin(async { Ok(()) })
    }
}

#[derive(Default)]
struct ReplayableRecordingContext {
    sleeps: Mutex<Vec<u64>>,
    runs: Mutex<Vec<String>>,
    records: Mutex<HashMap<String, Vec<u8>>>,
    replaying: AtomicBool,
}

impl ReplayableRecordingContext {
    fn start_replay(&self) {
        self.replaying.store(true, Ordering::SeqCst);
    }

    fn runs(&self) -> Vec<String> {
        self.runs.lock().expect("runs lock").clone()
    }
}

#[derive(Default)]
struct PositionalReplayContext {
    sleeps: Mutex<Vec<u64>>,
    runs: Mutex<Vec<String>>,
    records: Mutex<Vec<(String, Vec<u8>)>>,
    replaying: AtomicBool,
    replay_cursor: AtomicUsize,
}

impl PositionalReplayContext {
    fn start_replay(&self) {
        self.replaying.store(true, Ordering::SeqCst);
        self.replay_cursor.store(0, Ordering::SeqCst);
    }

    fn runs(&self) -> Vec<String> {
        self.runs.lock().expect("runs lock").clone()
    }

    fn record_count(&self) -> usize {
        self.records.lock().expect("records lock").len()
    }
}

impl<'ctx> RestateControllerContext<'ctx> for Arc<PositionalReplayContext> {
    fn sleep_send<'run>(
        &'run self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        self.sleeps
            .lock()
            .expect("sleeps lock")
            .push(duration.as_millis() as u64);
        Box::pin(async { Ok(()) })
    }

    fn run_json_send<'run, T, Fut>(
        &'run self,
        effect_name: String,
        _retry_policy: Option<RunRetryPolicy>,
        future: Fut,
    ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
        T: Serialize + DeserializeOwned + Send + 'static,
        Fut: Future<Output = T> + Send + 'run,
    {
        self.runs
            .lock()
            .expect("runs lock")
            .push(effect_name.clone());
        if self.replaying.load(Ordering::SeqCst) {
            let position = self.replay_cursor.fetch_add(1, Ordering::SeqCst);
            let recorded = self
                .records
                .lock()
                .expect("records lock")
                .get(position)
                .cloned();
            return Box::pin(async move {
                let (recorded_effect_name, bytes) = recorded.ok_or_else(|| {
                    TerminalError::new(format!("missing recorded effect at position {position}"))
                })?;
                if recorded_effect_name != effect_name {
                    return Err(TerminalError::new(format!(
                        "recorded effect at position {position} was `{recorded_effect_name}`, got `{effect_name}`"
                    )));
                }
                serde_json::from_slice(&bytes)
                    .map(Json)
                    .map_err(TerminalError::from_error)
            });
        }

        let context = Arc::clone(self);
        Box::pin(async move {
            let value = future.await;
            let bytes = serde_json::to_vec(&value).map_err(TerminalError::from_error)?;
            context
                .records
                .lock()
                .expect("records lock")
                .push((effect_name, bytes));
            Ok(Json(value))
        })
    }

    fn start_process_workflow<'run>(
        &'run self,
        _registration: ProcessRegistration,
        _execution_context: ProcessExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process workflow start is unsupported")) })
    }

    fn request_process_workflow_cancel<'run>(
        &'run self,
        _request: RestateProcessCancelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process workflow cancel is unsupported")) })
    }

    fn await_event<'run>(
        &'run self,
        _key: String,
    ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("event await is unsupported")) })
    }

    fn await_process_terminal<'run>(
        &'run self,
        _process_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<ProcessAwaitOutput, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process terminal await is unsupported")) })
    }

    fn resolve_event<'run>(
        &'run self,
        _request: RestateProcessEventResolveRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("event resolve is unsupported")) })
    }
}

impl<'ctx> RestateControllerContext<'ctx> for Arc<ReplayableRecordingContext> {
    fn sleep_send<'run>(
        &'run self,
        duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        self.sleeps
            .lock()
            .expect("sleeps lock")
            .push(duration.as_millis() as u64);
        Box::pin(async { Ok(()) })
    }

    fn run_json_send<'run, T, Fut>(
        &'run self,
        effect_name: String,
        _retry_policy: Option<RunRetryPolicy>,
        future: Fut,
    ) -> Pin<Box<dyn Future<Output = Result<Json<T>, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
        T: Serialize + DeserializeOwned + Send + 'static,
        Fut: Future<Output = T> + Send + 'run,
    {
        self.runs
            .lock()
            .expect("runs lock")
            .push(effect_name.clone());
        let replaying = self.replaying.load(Ordering::SeqCst);
        if replaying {
            let recorded = self
                .records
                .lock()
                .expect("records lock")
                .get(&effect_name)
                .cloned();
            return Box::pin(async move {
                let bytes = recorded.ok_or_else(|| {
                    TerminalError::new(format!("missing recorded effect `{effect_name}`"))
                })?;
                serde_json::from_slice(&bytes)
                    .map(Json)
                    .map_err(TerminalError::from_error)
            });
        }

        let context = Arc::clone(self);
        Box::pin(async move {
            let value = future.await;
            let bytes = serde_json::to_vec(&value).map_err(TerminalError::from_error)?;
            context
                .records
                .lock()
                .expect("records lock")
                .insert(effect_name, bytes);
            Ok(Json(value))
        })
    }

    fn start_process_workflow<'run>(
        &'run self,
        _registration: ProcessRegistration,
        _execution_context: ProcessExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process workflow start is unsupported")) })
    }

    fn request_process_workflow_cancel<'run>(
        &'run self,
        _request: RestateProcessCancelRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process workflow cancel is unsupported")) })
    }

    fn await_event<'run>(
        &'run self,
        _key: String,
    ) -> Pin<Box<dyn Future<Output = Result<Resolution, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("event await is unsupported")) })
    }

    fn await_process_terminal<'run>(
        &'run self,
        _process_id: String,
    ) -> Pin<Box<dyn Future<Output = Result<ProcessAwaitOutput, TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("process terminal await is unsupported")) })
    }

    fn resolve_event<'run>(
        &'run self,
        _request: RestateProcessEventResolveRequest,
    ) -> Pin<Box<dyn Future<Output = Result<(), TerminalError>> + Send + 'run>>
    where
        'ctx: 'run,
    {
        Box::pin(async { Err(TerminalError::new("event resolve is unsupported")) })
    }
}

fn runtime_invocation(kind: RuntimeEffectKind, effect_id: &str) -> RuntimeInvocation {
    RuntimeInvocation::effect(
        lash_core::runtime::RuntimeScope::for_turn("session", "turn", 1, 0),
        effect_id,
        kind,
        format!("session:turn:1:0:{}:{effect_id}", kind.as_str()),
    )
}

#[tokio::test]
async fn restate_controller_executes_non_sleep_effect_inside_run() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let err = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::ExecCode, "exec"),
                RuntimeEffectCommand::ExecCode {
                    language: "code".to_string(),
                    code: "1 + 1".to_string(),
                },
            ),
            RuntimeEffectLocalExecutor::unavailable(),
        )
        .await
        .expect_err("unavailable local executor should be returned from ctx.run");

    assert_eq!(err.code, "runtime_effect_local_executor_unavailable");
    assert_eq!(
        context.runs.lock().expect("runs lock").as_slice(),
        &["lash:session:turn:1:0:exec_code:exec".to_string()]
    );
    assert!(context.sleeps.lock().expect("sleeps lock").is_empty());
}

#[tokio::test]
async fn restate_positional_replay_records_tool_attempt_as_one_command() {
    let context = Arc::new(PositionalReplayContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let call = prepared_tool_call_with("call-fast", "fast_tool");
    let envelope = RuntimeEffectEnvelope::new(
        runtime_invocation(RuntimeEffectKind::ToolAttempt, "tool-attempt"),
        RuntimeEffectCommand::ToolAttempt {
            call,
            execution_grant: None,
            attempt: 1,
            max_attempts: 1,
        },
    );
    let local_runs = Arc::new(AtomicUsize::new(0));

    let first = host
        .execute_effect(
            envelope.clone(),
            RuntimeEffectLocalExecutor::testing({
                let local_runs = Arc::clone(&local_runs);
                |_envelope| async move {
                    local_runs.fetch_add(1, Ordering::SeqCst);
                    Ok(RuntimeEffectOutcome::ToolAttempt {
                        launch: lash_core::ToolAttemptLaunch::Done {
                            record: completed_tool_record("call-fast", "fast_tool"),
                        },
                        triggers: Vec::new(),
                    })
                }
            }),
        )
        .await
        .expect("first attempt run");

    let RuntimeEffectOutcome::ToolAttempt { launch, .. } = first else {
        panic!("expected tool attempt outcome");
    };
    assert!(matches!(
        &launch,
        lash_core::ToolAttemptLaunch::Done { record } if record.call_id.as_deref() == Some("call-fast")
    ));
    assert_eq!(context.record_count(), 1);
    assert_eq!(context.runs().len(), 1);
    assert_eq!(local_runs.load(Ordering::SeqCst), 1);

    context.start_replay();
    let replayed = host
        .execute_effect(
            envelope,
            RuntimeEffectLocalExecutor::testing(|_| async {
                panic!("positional replay should not rerun the ToolAttempt executor")
            }),
        )
        .await
        .expect("replayed attempt run");

    let RuntimeEffectOutcome::ToolAttempt { launch, .. } = replayed else {
        panic!("expected replayed tool attempt outcome");
    };
    assert!(matches!(
        &launch,
        lash_core::ToolAttemptLaunch::Done { record } if record.call_id.as_deref() == Some("call-fast")
    ));
    assert_eq!(context.record_count(), 1);
    assert_eq!(context.runs().len(), 2);
    assert_eq!(local_runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn restate_controller_routes_sleep_only_through_timer() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Sleep, "sleep"),
                RuntimeEffectCommand::Sleep { duration_ms: 42 },
            ),
            RuntimeEffectLocalExecutor::unavailable(),
        )
        .await
        .expect("sleep");

    assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
    assert_eq!(
        context.sleeps.lock().expect("sleeps lock").as_slice(),
        &[42]
    );
    assert!(context.runs.lock().expect("runs lock").is_empty());
}

#[tokio::test]
async fn restate_handler_resolves_process_await_events_through_workflow_handler() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let key = AwaitEventResolver::await_event_key(
        &host,
        &ExecutionScope::process("process-await"),
        AwaitEventWaitIdentity::tool_completion("tool-call"),
    )
    .await
    .expect("await event key");
    let resolution = Resolution::Ok(serde_json::json!({ "done": true }));

    let outcome = AwaitEventResolver::resolve_await_event(&host, &key, resolution.clone())
        .await
        .expect("resolve await event");

    assert_eq!(outcome, ResolveOutcome::Accepted);
    let resolved = context
        .resolved_events
        .lock()
        .expect("resolved events lock");
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].process_id, "process-await");
    assert_eq!(resolved[0].key, key.promise_key());
    assert_eq!(resolved[0].resolution, resolution);
}

#[tokio::test]
async fn restate_handler_reports_non_process_await_event_resolution_unknown() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let key = AwaitEventResolver::await_event_key(
        &host,
        &ExecutionScope::turn("session", "turn"),
        AwaitEventWaitIdentity::tool_completion("tool-call"),
    )
    .await
    .expect("await event key");

    let outcome = AwaitEventResolver::resolve_await_event(
        &host,
        &key,
        Resolution::Ok(serde_json::json!({ "done": true })),
    )
    .await
    .expect("resolve await event");

    assert_eq!(outcome, ResolveOutcome::UnknownOrRevoked);
    assert!(
        context
            .resolved_events
            .lock()
            .expect("resolved events lock")
            .is_empty()
    );
}

#[tokio::test]
async fn restate_handler_cancels_individual_wait_via_documented_workaround() {
    // The per-wait workaround the session-wide lever points at: resolving a
    // known key with `Resolution::Cancelled` is accepted and durably recorded
    // by the process workflow handler under the exact promise key.
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let key = AwaitEventResolver::await_event_key(
        &host,
        &ExecutionScope::process("process-cancel-wait"),
        AwaitEventWaitIdentity::tool_completion("tool-call-cancel"),
    )
    .await
    .expect("await event key");

    let outcome = AwaitEventResolver::resolve_await_event(&host, &key, Resolution::Cancelled)
        .await
        .expect("cancel individual wait");

    assert_eq!(outcome, ResolveOutcome::Accepted);
    let resolved = context
        .resolved_events
        .lock()
        .expect("resolved events lock");
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].process_id, "process-cancel-wait");
    assert_eq!(resolved[0].key, key.promise_key());
    assert_eq!(resolved[0].resolution, Resolution::Cancelled);
}

#[tokio::test]
async fn restate_handler_controller_rejects_cancel_by_session_with_documented_error() {
    // Restate cannot enumerate a session's outstanding promises, so the
    // session-wide lever fails loudly with a specific error naming the
    // per-wait workaround instead of faking success.
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());

    let err = AwaitEventResolver::cancel_await_events_for_session(&host, "restate-session")
        .await
        .expect_err("Restate must reject session-wide durable-wait cancellation");

    assert_eq!(
        err.code.as_str(),
        "restate_await_event_cancel_by_session_unsupported"
    );
    assert!(
        err.message.contains("resolve_await_event"),
        "error must name the per-wait workaround: {}",
        err.message
    );
    // Refusing must not resolve anything.
    assert!(
        context
            .resolved_events
            .lock()
            .expect("resolved events lock")
            .is_empty()
    );
}

#[tokio::test]
async fn restate_effect_host_rejects_cancel_by_session_with_documented_error() {
    // The deployment-level host delegates to its controller and surfaces the
    // same documented refusal, so `LashSession::revoke_durable_waits` reports a
    // specific limitation on Restate deployments.
    let host = RestateEffectHost::new();

    let err = AwaitEventResolver::cancel_await_events_for_session(&host, "restate-session")
        .await
        .expect_err("Restate effect host must reject session-wide cancellation");

    assert_eq!(
        err.code.as_str(),
        "restate_await_event_cancel_by_session_unsupported"
    );
    assert!(err.message.contains("resolve_await_event"));
}

fn replay_test_policy(session_id: &str) -> lash_core::SessionPolicy {
    let mut policy = lash_core::testing::mock_session_policy();
    policy.session_id = Some(session_id.to_string());
    policy
}

fn replay_test_state(
    session_id: &str,
    policy: &lash_core::SessionPolicy,
) -> lash_core::RuntimeSessionState {
    lash_core::RuntimeSessionState {
        session_id: session_id.to_string(),
        policy: policy.clone(),
        ..lash_core::RuntimeSessionState::default()
    }
}

fn replay_test_input(turn_id: &str) -> lash_core::TurnInput {
    let mut input = lash_core::TurnInput::text("finish once");
    input.trace_turn_id = Some(turn_id.to_string());
    input
}

async fn replay_test_runtime(
    session_id: &str,
    policy: lash_core::SessionPolicy,
    initial_state: lash_core::RuntimeSessionState,
    host: lash_core::RuntimeHostConfig,
    store: Arc<dyn lash_core::RuntimePersistence>,
) -> lash_core::LashRuntime {
    lash_core::LashRuntime::builder()
        .with_session_id(session_id)
        .with_policy(policy)
        .with_initial_state(initial_state)
        .with_runtime_host(host)
        .with_plugin_factories(lash_core::testing::test_standard_protocol_factories())
        .with_store(store)
        .build()
        .await
        .expect("build replay test runtime")
}

async fn run_restate_replay_turn(
    runtime: &mut lash_core::LashRuntime,
    context: Arc<ReplayableRecordingContext>,
    session_id: &str,
    turn_id: &str,
) -> lash_core::AssembledTurn {
    let controller = RestateRuntimeEffectController::new(context);
    let scoped_effect_controller = controller
        .scoped_effect_controller(ExecutionScope::turn(session_id, turn_id))
        .expect("scoped restate controller");
    runtime
        .stream_turn(
            replay_test_input(turn_id),
            lash_core::TurnOptions::new(
                tokio_util::sync::CancellationToken::new(),
                scoped_effect_controller,
            ),
        )
        .await
        .expect("run replay test turn")
}

#[tokio::test]
async fn restate_handler_replay_retries_final_lash_commit_idempotently() {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_id = "restate-final-commit-replay";
    let turn_id = "restate-turn-1";
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let provider = lash_core::testing::TestProvider::builder()
        .kind("stub")
        .complete({
            let provider_calls = Arc::clone(&provider_calls);
            move |_request| {
                let provider_calls = Arc::clone(&provider_calls);
                async move {
                    let call_index = provider_calls.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(
                        call_index, 0,
                        "Restate replay should return the recorded LLM effect"
                    );
                    Ok(lash_core::LlmResponse {
                        full_text: "committed once".to_string(),
                        parts: vec![lash_core::LlmOutputPart::Text {
                            text: "committed once".to_string(),
                            response_meta: None,
                        }],
                        ..lash_core::LlmResponse::default()
                    })
                }
            }
        })
        .build()
        .into_handle();
    let mut host = lash_core::RuntimeHostConfig::in_memory();
    host.providers.provider_resolver = Arc::new(lash_core::SingleProviderResolver::new(provider));
    host.durability.attachment_store = Arc::new(DurableMemoryAttachmentStore::default());
    host.durability.process_env_store = Arc::new(DurableMemoryProcessEnvStore::default());
    let store = Arc::new(
        lash_sqlite_store::Store::open(&dir.path().join("session.db"))
            .await
            .expect("open session store"),
    );
    let runtime_store: Arc<dyn lash_core::RuntimePersistence> = store.clone();
    let policy = replay_test_policy(session_id);
    let initial_state = replay_test_state(session_id, &policy);
    let context = Arc::new(ReplayableRecordingContext::default());

    let mut first = replay_test_runtime(
        session_id,
        policy.clone(),
        initial_state.clone(),
        host.clone(),
        Arc::clone(&runtime_store),
    )
    .await;
    let first_turn =
        run_restate_replay_turn(&mut first, Arc::clone(&context), session_id, turn_id).await;
    assert!(matches!(
        first_turn.outcome,
        lash_core::TurnOutcome::Finished(_)
    ));
    let first_runs = context.runs();
    assert!(!first_runs.is_empty());

    context.start_replay();
    let retry_store: Arc<dyn lash_core::RuntimePersistence> = Arc::new(CommitRetryStore {
        inner: Arc::clone(&runtime_store),
    });
    let mut replay =
        replay_test_runtime(session_id, policy, initial_state, host, retry_store).await;
    let replay_turn =
        run_restate_replay_turn(&mut replay, Arc::clone(&context), session_id, turn_id).await;
    assert!(matches!(
        replay_turn.outcome,
        lash_core::TurnOutcome::Finished(_)
    ));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);

    let conn = rusqlite::Connection::open(dir.path().join("session.db"))
        .expect("open raw session sqlite store");
    let rows: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM runtime_turn_commits WHERE session_id = ?1 AND turn_id = ?2",
            rusqlite::params![session_id, turn_id],
            |row| row.get(0),
        )
        .expect("count turn commit stamps");
    assert_eq!(rows, 1);
}

#[tokio::test]
async fn restate_controller_schedules_process_workflow_without_running_executor() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let registration = external_registration("task-1");
    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "background-start"),
                RuntimeEffectCommand::process(ProcessCommand::Start {
                    registration,
                    grant: Some(lash_core::ProcessStartGrant {
                        session_scope: lash_core::SessionScope::new("session"),
                        descriptor: lash_core::ProcessHandleDescriptor::new(
                            Some("tool"),
                            Some("task"),
                        ),
                    }),
                    execution_context: Box::new(ProcessExecutionContext::default()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("start");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Start { record },
    } = outcome
    else {
        panic!("wrong outcome");
    };

    assert_eq!(
        record
            .external_ref
            .as_ref()
            .map(|external| external.id.as_str()),
        Some("LashProcessWorkflow/task-1")
    );
    assert_eq!(
        registry
            .get_process("task-1")
            .await
            .expect("get")
            .external_ref
            .as_ref()
            .map(|external| external.id.as_str()),
        Some("LashProcessWorkflow/task-1")
    );
    let session_scope = lash_core::SessionScope::new("session");
    assert_eq!(
        registry
            .list_handle_grants(&session_scope)
            .await
            .expect("grants")
            .into_iter()
            .next()
            .and_then(|(_, record)| record.external_ref)
            .map(|external| (
                external.backend,
                external
                    .metadata
                    .and_then(|metadata| metadata.get("invocation_id").cloned())
            )),
        Some((
            "restate".to_string(),
            Some(serde_json::json!("invocation-task-1"))
        ))
    );
    assert_eq!(
        context
            .started
            .lock()
            .expect("started lock")
            .iter()
            .map(|registration| registration.id.as_str())
            .collect::<Vec<_>>(),
        vec!["task-1"]
    );
    assert!(
        context.runs.lock().expect("runs lock").is_empty(),
        "process workflow scheduling must not call Restate context from inside ctx.run"
    );
}

#[tokio::test]
async fn restate_controller_replays_process_start_await_command_sequence() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let process_id = "task-start-await-replay";

    let start = || {
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::Process, "process-start-replay"),
            RuntimeEffectCommand::process(ProcessCommand::Start {
                registration: external_registration(process_id),
                grant: None,
                execution_context: Box::new(ProcessExecutionContext::default()),
            }),
        )
    };
    let await_terminal = || {
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::Process, "process-await-replay"),
            RuntimeEffectCommand::process(ProcessCommand::Await {
                process_id: process_id.to_string(),
            }),
        )
    };
    let terminal = ProcessAwaitOutput::Success {
        value: serde_json::json!({ "done": true }),
        control: None,
    };

    host.execute_effect(
        start(),
        RuntimeEffectLocalExecutor::processes(registry.clone(), None),
    )
    .await
    .expect("first start");
    registry
        .complete_process(process_id, terminal.clone())
        .await
        .expect("complete child process");
    context.resolve_process_terminal(process_id, &terminal);
    host.execute_effect(
        await_terminal(),
        RuntimeEffectLocalExecutor::processes(registry.clone(), None),
    )
    .await
    .expect("first await");

    // Simulates Restate replay of the same parent handler after a later
    // suspension resumes. The already persisted registry record has an
    // external_ref at this point, but the handler must still issue the same
    // Restate send before the await call so the journal command sequence stays
    // send -> call -> ... on every replay.
    host.execute_effect(
        start(),
        RuntimeEffectLocalExecutor::processes(registry.clone(), None),
    )
    .await
    .expect("replay start");
    host.execute_effect(
        await_terminal(),
        RuntimeEffectLocalExecutor::processes(registry.clone(), None),
    )
    .await
    .expect("replay await");

    assert_eq!(
        context
            .process_command_log
            .lock()
            .expect("process command log lock")
            .as_slice(),
        &[
            format!("send:{process_id}"),
            format!("call:{process_id}"),
            format!("send:{process_id}"),
            format!("call:{process_id}"),
        ],
        "child process start/await must replay the same Restate command sequence"
    );
}

#[tokio::test]
async fn restate_controller_start_emits_send_when_external_ref_already_exists() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let process_id = "task-start-existing-ref";
    let registration = external_registration(process_id);
    registry
        .register_process(registration.clone())
        .await
        .expect("register process");
    registry
        .set_external_ref(
            process_id,
            ProcessExternalRef {
                backend: "restate".to_string(),
                id: format!("LashProcessWorkflow/{process_id}"),
                metadata: Some(serde_json::json!({
                    "invocation_id": format!("invocation-{process_id}")
                })),
            },
        )
        .await
        .expect("pre-set external ref");

    host.execute_effect(
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::Process, "process-start-existing-ref"),
            RuntimeEffectCommand::process(ProcessCommand::Start {
                registration,
                grant: None,
                execution_context: Box::new(ProcessExecutionContext::default()),
            }),
        ),
        RuntimeEffectLocalExecutor::processes(registry, None),
    )
    .await
    .expect("start with existing external ref");

    assert_eq!(
        context
            .process_command_log
            .lock()
            .expect("process command log lock")
            .as_slice(),
        &[format!("send:{process_id}")],
        "pre-existing external_ref must not suppress the journaled Restate send"
    );
}

async fn run_parent_shaped_start_await_suspend_flow(
    host: &RestateRuntimeEffectController<'_, Arc<RecordingContext>>,
    registry: Arc<dyn ProcessRegistry>,
    process_id: &str,
    suspend_key: AwaitEventKey,
) {
    host.execute_effect(
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::Process, "parent-flow-start-child"),
            RuntimeEffectCommand::process(ProcessCommand::Start {
                registration: external_registration(process_id),
                grant: None,
                execution_context: Box::new(ProcessExecutionContext::default()),
            }),
        ),
        RuntimeEffectLocalExecutor::processes(registry.clone(), None),
    )
    .await
    .expect("parent flow start child");

    host.execute_effect(
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::Process, "parent-flow-await-child"),
            RuntimeEffectCommand::process(ProcessCommand::Await {
                process_id: process_id.to_string(),
            }),
        ),
        RuntimeEffectLocalExecutor::processes(registry, None),
    )
    .await
    .expect("parent flow await child");

    host.execute_effect(
        RuntimeEffectEnvelope::new(
            runtime_invocation(RuntimeEffectKind::AwaitEvent, "parent-flow-suspend"),
            RuntimeEffectCommand::AwaitEvent { key: suspend_key },
        ),
        RuntimeEffectLocalExecutor::await_event(tokio_util::sync::CancellationToken::new(), None),
    )
    .await
    .expect("parent flow await resume event");
}

#[tokio::test]
async fn restate_controller_replays_parent_shaped_start_await_suspend_flow() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let process_id = "task-parent-flow-replay";
    let terminal = ProcessAwaitOutput::Success {
        value: serde_json::json!({ "done": true }),
        control: None,
    };
    let suspend_key = restate_await_event_key(
        &ExecutionScope::process(process_id),
        AwaitEventWaitIdentity::Custom {
            key: "parent-resume-input".to_string(),
        },
    )
    .expect("parent suspend key");
    context.resolve_process_terminal(process_id, &terminal);
    context
        .awaited_events
        .lock()
        .expect("awaited events lock")
        .insert(
            suspend_key.promise_key(),
            Resolution::Ok(serde_json::json!({ "answer": "resume" })),
        );

    run_parent_shaped_start_await_suspend_flow(
        &host,
        registry.clone(),
        process_id,
        suspend_key.clone(),
    )
    .await;
    run_parent_shaped_start_await_suspend_flow(&host, registry, process_id, suspend_key).await;

    assert_eq!(
        context
            .process_command_log
            .lock()
            .expect("process command log lock")
            .as_slice(),
        &[
            format!("send:{process_id}"),
            format!("call:{process_id}"),
            format!("send:{process_id}"),
            format!("call:{process_id}"),
        ],
        "a parent-shaped replay after suspension must preserve child start/await command order"
    );
}

#[tokio::test]
async fn restate_controller_schedules_lashlang_process_with_serializable_input() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let module = lashlang::parse("process scan(root: str) { finish root }")
        .expect("lashlang process module");
    let catalog = lashlang::LashlangHostCatalog::new();
    let linked_module = lashlang::LinkedModule::link(
        module.clone(),
        lashlang::LashlangHostEnvironment::new(catalog, lashlang::LashlangAbilities::all()),
    )
    .expect("link lashlang module");
    let process_ref = linked_module
        .artifact
        .process_ref("scan")
        .expect("scan process ref")
        .clone();
    let mut args = serde_json::Map::new();
    args.insert("root".to_string(), serde_json::json!("."));
    let registration = ProcessRegistration::new(
        "process-1",
        lashlang_process_input(lash_lashlang_runtime::LashlangProcessInput {
            module_ref: linked_module.module_ref.clone(),
            process_ref: process_ref.clone(),
            host_requirements_ref: linked_module.host_requirements_ref.clone(),
            process_name: "scan".to_string(),
            args: args.clone(),
        }),
        lash_core::ProcessProvenance::session(lash_core::SessionScope::new("session")),
    )
    .with_extra_event_types(lash_lashlang_runtime::lashlang_process_event_types())
    .with_execution_env_ref(Some(lash_core::ProcessExecutionEnvRef::new(
        "process-env:test:process-1",
    )))
    .with_wake_target(Some(lash_core::SessionScope::new("session")));

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "lashlang-process-start"),
                RuntimeEffectCommand::process(ProcessCommand::Start {
                    registration,
                    grant: None,
                    execution_context: Box::new(ProcessExecutionContext::default()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("start");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Start { record },
    } = outcome
    else {
        panic!("wrong outcome");
    };

    assert_eq!(
        record
            .external_ref
            .as_ref()
            .map(|external| external.backend.as_str()),
        Some("restate")
    );
    assert_eq!(
        registry
            .get_process("process-1")
            .await
            .expect("registered process")
            .external_ref
            .as_ref()
            .map(|external| external.backend.as_str()),
        Some("restate")
    );
    let started = context.started.lock().expect("started lock").clone();
    assert_eq!(started.len(), 1);
    let ProcessInput::Engine { kind, payload } = started[0].input.as_ref() else {
        panic!("expected engine process input");
    };
    assert_eq!(kind, lash_lashlang_runtime::LASHLANG_ENGINE_KIND);
    let sent = lash_lashlang_runtime::LashlangProcessInput::from_payload(payload.clone())
        .expect("typed lashlang payload");
    assert_eq!(sent.module_ref, linked_module.module_ref);
    assert_eq!(sent.process_ref, process_ref);
    assert_eq!(
        sent.host_requirements_ref,
        linked_module.host_requirements_ref
    );
    assert_eq!(sent.process_name, "scan");
    assert_eq!(sent.args, args);
    assert_eq!(
        context
            .started
            .lock()
            .expect("started lock")
            .iter()
            .map(|registration| {
                registration
                    .wake_target
                    .as_ref()
                    .map(|scope| scope.session_id.as_str())
            })
            .collect::<Vec<_>>(),
        vec![Some("session")]
    );
}

#[tokio::test]
async fn restate_controller_lists_and_transfers_grants_through_process_effects() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let s1 = lash_core::SessionScope::new("s1");
    let s2 = lash_core::SessionScope::new("s2");
    registry
        .register_process(external_registration("task-list"))
        .await
        .expect("register");
    registry
        .grant_handle(
            &s1,
            "task-list",
            lash_core::ProcessHandleDescriptor::new(Some("tool"), Some("task")),
        )
        .await
        .expect("grant");

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "process-list-s1"),
                RuntimeEffectCommand::process(ProcessCommand::List {
                    session_scope: s1.clone(),
                    mode: lash_core::ProcessListMode::Live,
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("list");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::List { entries },
    } = outcome
    else {
        panic!("wrong list outcome");
    };
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0.process_id, "task-list");

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "process-transfer"),
                RuntimeEffectCommand::process(ProcessCommand::Transfer {
                    from_scope: s1.clone(),
                    to_scope: s2.clone(),
                    process_ids: vec!["task-list".to_string()],
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("transfer");
    assert!(matches!(
        outcome,
        RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::Transfer
        }
    ));

    let entries = registry.list_handle_grants(&s2).await.expect("s2 grants");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0.process_id, "task-list");
    assert!(
        registry
            .list_handle_grants(&s1)
            .await
            .expect("s1")
            .is_empty()
    );
    assert!(context.started.lock().expect("started lock").is_empty());
}

#[tokio::test]
async fn restate_controller_awaits_and_signals_through_process_effects() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    registry
        .register_process(external_registration("task-await-signal"))
        .await
        .expect("register");
    registry
        .register_process(
            external_registration("task-signal")
                .with_extra_event_types(lash_lashlang_runtime::lashlang_process_event_types())
                .with_extra_event_types([lash_core::ProcessEventType {
                    name: "signal.notify".to_string(),
                    payload_schema: lash_core::LashSchema::any(),
                    semantics: lash_core::ProcessEventSemanticsSpec::default(),
                }]),
        )
        .await
        .expect("register signal target");
    let awaited_output = ProcessAwaitOutput::Success {
        value: serde_json::json!({ "done": true }),
        control: None,
    };
    registry
        .complete_process("task-await-signal", awaited_output.clone())
        .await
        .expect("complete");
    context.resolve_process_terminal("task-await-signal", &awaited_output);

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "process-await"),
                RuntimeEffectCommand::process(ProcessCommand::Await {
                    process_id: "task-await-signal".to_string(),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("await");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Await { output },
    } = outcome
    else {
        panic!("wrong await outcome");
    };
    assert_eq!(
        output,
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "done": true }),
            control: None,
        }
    );

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "process-signal"),
                RuntimeEffectCommand::process(ProcessCommand::Signal {
                    process_id: "task-signal".to_string(),
                    signal_name: "notify".to_string(),
                    signal_id: "notify".to_string(),
                    request: lash_core::ProcessEventAppendRequest::new(
                        "signal.notify",
                        serde_json::json!({ "signal": "notify" }),
                    )
                    .with_replay_key("signal:notify"),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("signal");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Signal { event },
    } = outcome
    else {
        panic!("wrong signal outcome");
    };
    assert_eq!(event.event_type, "signal.notify");
    assert!(context.started.lock().expect("started lock").is_empty());

    // Append-before-resolve discipline: the durable event is the record, the
    // promise resolution is only the wake-up, keyed by the Nth occurrence of
    // this signal name so repeated signals map onto one-shot engine promises.
    {
        let resolved = context.resolved_events.lock().expect("resolved lock");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].process_id, "task-signal");
        assert_eq!(
            resolved[0].key,
            lash_core::process_signal_wait_key("task-signal", "notify", 1)
        );
        assert_eq!(
            resolved[0].resolution,
            Resolution::Ok(serde_json::json!({ "signal": "notify" }))
        );
    }

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "process-signal-2"),
                RuntimeEffectCommand::process(ProcessCommand::Signal {
                    process_id: "task-signal".to_string(),
                    signal_name: "notify".to_string(),
                    signal_id: "notify-2".to_string(),
                    request: lash_core::ProcessEventAppendRequest::new(
                        "signal.notify",
                        serde_json::json!({ "signal": "notify-2" }),
                    )
                    .with_replay_key("signal:notify-2"),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("second signal");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Signal { .. },
    } = outcome
    else {
        panic!("wrong second signal outcome");
    };
    let resolved = context.resolved_events.lock().expect("resolved lock");
    assert_eq!(resolved.len(), 2);
    assert_eq!(
        resolved[1].key,
        lash_core::process_signal_wait_key("task-signal", "notify", 2),
        "second signal must resolve the ordinal-2 wait key"
    );
}

#[tokio::test]
async fn restate_controller_cancel_requests_call_workflow_cancel() {
    let context = Arc::new(RecordingContext::default());
    let host = RestateRuntimeEffectController::new(context.clone());
    let registry = process_registry();
    let registration = external_registration("task-cancel");
    registry
        .register_process(registration)
        .await
        .expect("register");

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "background-cancel"),
                RuntimeEffectCommand::process(ProcessCommand::Cancel {
                    process_id: "task-cancel".to_string(),
                    reason: Some("user requested".to_string()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry, None),
        )
        .await
        .expect("cancel");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Cancel { record },
    } = outcome
    else {
        panic!("wrong outcome");
    };

    assert!(!record.is_terminal());
    assert_eq!(
        context.cancelled.lock().expect("cancelled lock").as_slice(),
        &[(
            "task-cancel".to_string(),
            Some("user requested".to_string())
        )]
    );
}

#[derive(Debug, PartialEq, Eq)]
struct RecordedProcessRun {
    process_id: String,
    wake_target_session_id: Option<String>,
    tool_effect_id: Option<String>,
    execution_scope_id: String,
    controller_tier: lash_core::DurabilityTier,
}

#[derive(Default)]
struct RecordingRunner {
    ran: Mutex<Vec<RecordedProcessRun>>,
    cancelled: Mutex<Vec<RestateProcessCancelRequest>>,
}

#[async_trait::async_trait]
impl RestateProcessRunner for RecordingRunner {
    async fn run_process(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
        scoped_effect_controller: lash_core::ScopedEffectController<'_>,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.ran
            .lock()
            .expect("runner ran lock")
            .push(RecordedProcessRun {
                process_id: registration.id.clone(),
                wake_target_session_id: registration
                    .wake_target
                    .as_ref()
                    .map(|scope| scope.session_id.clone()),
                tool_effect_id: execution_context
                    .causal_invocation
                    .and_then(|invocation| invocation.effect_id().map(str::to_string)),
                execution_scope_id: scoped_effect_controller.scope_id().to_string(),
                controller_tier: scoped_effect_controller.controller().durability_tier(),
            });
        Ok(ProcessAwaitOutput::Success {
            value: serde_json::json!({"ok": true}),
            control: None,
        })
    }

    async fn request_process_cancel(
        &self,
        request: RestateProcessCancelRequest,
    ) -> Result<(), PluginError> {
        self.cancelled
            .lock()
            .expect("runner cancelled lock")
            .push(request);
        Ok(())
    }
}

#[tokio::test]
async fn process_workflow_endpoint_smoke_schedules_runs_and_cancels_process() {
    let runner = Arc::new(RecordingRunner::default());
    let registry = process_registry();
    let endpoint = Endpoint::builder()
        .bind(LashProcessWorkflowImpl::new(runner.clone(), registry.clone()).serve())
        .build();
    let context = Arc::new(RecordingContext::with_endpoint(endpoint));
    let host = RestateRuntimeEffectController::new(context.clone());
    let registration = external_registration("task-smoke")
        .with_wake_target(Some(lash_core::SessionScope::new("wake-smoke")));
    let execution_context = ProcessExecutionContext::default().with_causal_invocation(Some(
        runtime_invocation(RuntimeEffectKind::ToolAttempt, "tool-smoke"),
    ));

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "background-smoke-start"),
                RuntimeEffectCommand::process(ProcessCommand::Start {
                    registration,
                    grant: Some(lash_core::ProcessStartGrant {
                        session_scope: lash_core::SessionScope::new("session"),
                        descriptor: lash_core::ProcessHandleDescriptor::new(
                            Some("tool"),
                            Some("task-smoke"),
                        ),
                    }),
                    execution_context: Box::new(execution_context),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry.clone(), None),
        )
        .await
        .expect("start through endpoint smoke");
    let RuntimeEffectOutcome::Process {
        result: ProcessEffectOutcome::Start { record },
    } = outcome
    else {
        panic!("wrong start outcome");
    };

    let external_ref = record.external_ref.as_ref().expect("external ref");
    assert_eq!(external_ref.backend, "restate");
    assert_eq!(external_ref.id, "LashProcessWorkflow/task-smoke");
    assert_eq!(
        external_ref
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("invocation_id")),
        Some(&serde_json::json!("invocation-task-smoke"))
    );

    let session_scope = lash_core::SessionScope::new("session");
    let grants = registry
        .list_handle_grants(&session_scope)
        .await
        .expect("session grants");
    assert_eq!(grants.len(), 1);
    assert_eq!(grants[0].0.process_id, "task-smoke");
    let granted_external_ref = grants[0].1.external_ref.as_ref().expect("grant ref");
    assert_eq!(granted_external_ref.backend, "restate");
    assert_eq!(granted_external_ref.id, "LashProcessWorkflow/task-smoke");

    assert_eq!(
        context
            .started
            .lock()
            .expect("started lock")
            .iter()
            .map(|registration| registration.id.as_str())
            .collect::<Vec<_>>(),
        vec!["task-smoke"]
    );
    assert_eq!(
        runner.ran.lock().expect("runner ran lock").as_slice(),
        &[RecordedProcessRun {
            process_id: "task-smoke".to_string(),
            wake_target_session_id: Some("wake-smoke".to_string()),
            tool_effect_id: Some("tool-smoke".to_string()),
            execution_scope_id: "task-smoke".to_string(),
            controller_tier: lash_core::DurabilityTier::Durable,
        }]
    );

    let outcome = host
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "background-smoke-cancel"),
                RuntimeEffectCommand::process(ProcessCommand::Cancel {
                    process_id: "task-smoke".to_string(),
                    reason: Some("stop-smoke".to_string()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(registry, None),
        )
        .await
        .expect("cancel through endpoint smoke");
    assert!(matches!(
        outcome,
        RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::Cancel { .. }
        }
    ));
    assert_eq!(
        context.cancelled.lock().expect("cancelled lock").as_slice(),
        &[("task-smoke".to_string(), Some("stop-smoke".to_string()))]
    );
    assert_eq!(
        runner
            .cancelled
            .lock()
            .expect("runner cancelled lock")
            .as_slice(),
        &[RestateProcessCancelRequest {
            process_id: "task-smoke".to_string(),
            reason: Some("stop-smoke".to_string()),
        }]
    );
}

struct RecoveryProcessTool;

impl RecoveryProcessTool {
    fn definition() -> lash_core::ToolDefinition {
        lash_core::ToolDefinition::raw(
            "tool:recovery_echo",
            "recovery_echo",
            "Echo a line and emit a durable process wake.",
            serde_json::json!({
                "type": "object",
                "properties": { "line": { "type": "string" } },
                "required": ["line"],
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new(["tools"], "recovery_echo"))
    }
}

#[async_trait::async_trait]
impl lash_core::ToolProvider for RecoveryProcessTool {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![Self::definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "recovery_echo").then(|| Arc::new(Self::definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        let line = call
            .args
            .get("line")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let event = lash_core::ProcessEventAppendRequest::new(
            "process.wake",
            serde_json::json!({ "message": line, "wake_input": line }),
        )
        .with_replay_key(format!("process.wake:{line}"));
        if let Err(err) = call.context.process_events().emit_request(event).await {
            return lash_core::ToolResult::err_fmt(err);
        }
        lash_core::ToolResult::ok(serde_json::json!({ "echo": line }))
    }
}

struct SnapshotRecoveryTool;

impl SnapshotRecoveryTool {
    fn definition() -> lash_core::ToolDefinition {
        lash_core::ToolDefinition::raw(
            "tool:snapshot_echo",
            "snapshot_echo",
            "Echo a line from a snapshot-backed process tool.",
            serde_json::json!({
                "type": "object",
                "properties": { "line": { "type": "string" } },
                "required": ["line"],
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new(["tools"], "snapshot_echo"))
    }
}

#[async_trait::async_trait]
impl lash_core::ToolProvider for SnapshotRecoveryTool {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![Self::definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "snapshot_echo").then(|| Arc::new(Self::definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        let line = call
            .args
            .get("line")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        lash_core::ToolResult::ok(serde_json::json!({ "echo": format!("snapshot:{line}") }))
    }
}

#[derive(serde::Deserialize)]
struct SnapshotRecoveryToolOptions {
    snapshot_ref: String,
}

fn snapshot_recovery_tool_options(snapshot_ref: &str) -> lash_core::PluginOptions {
    lash_core::PluginOptions::typed(
        "snapshot-recovery-tool",
        serde_json::json!({ "snapshot_ref": snapshot_ref }),
    )
    .expect("snapshot recovery plugin options")
}

fn snapshot_recovery_tool_factory() -> Arc<dyn lash_core::PluginFactory> {
    Arc::new(lash_core::plugin::PluginSpecFactory::new(
        "snapshot-recovery-tool",
        Arc::new(|ctx| {
            let snapshot_available = ctx
                .plugin_options
                .decode::<SnapshotRecoveryToolOptions>("snapshot-recovery-tool")
                .map_err(|err| {
                    lash_core::PluginError::Registration(format!(
                        "invalid snapshot recovery tool options: {err}"
                    ))
                })?
                .is_some_and(|options| options.snapshot_ref == "tool-authority:sha256:ok");
            let spec = if snapshot_available {
                lash_core::PluginSpec::new().with_tool_provider(Arc::new(SnapshotRecoveryTool))
            } else {
                lash_core::PluginSpec::new()
            };
            Ok(spec)
        }),
    ))
}

fn recovery_worker(
    registry: Arc<dyn ProcessRegistry>,
    store_factory: Arc<dyn lash_core::SessionStoreFactory>,
) -> DurableProcessWorker {
    recovery_worker_with_plugins(registry, store_factory, Vec::new())
}

fn recovery_worker_with_plugins(
    registry: Arc<dyn ProcessRegistry>,
    store_factory: Arc<dyn lash_core::SessionStoreFactory>,
    extra_plugins: Vec<Arc<dyn lash_core::PluginFactory>>,
) -> DurableProcessWorker {
    let tools: Arc<dyn lash_core::ToolProvider> = Arc::new(RecoveryProcessTool);
    let mut plugins = vec![
        Arc::new(lash_protocol_standard::StandardProtocolPluginFactory::new())
            as Arc<dyn lash_core::PluginFactory>,
        Arc::new(lash_core::plugin::StaticPluginFactory::new(
            "recovery-tool",
            lash_core::PluginSpec::new().with_tool_provider(tools),
        )),
    ];
    plugins.extend(extra_plugins);
    let plugin_host = lash_core::PluginHost::new(plugins);
    let process_env_store: Arc<dyn lash_core::ProcessExecutionEnvStore> =
        RECOVERY_PROCESS_ENV_STORE.clone();
    let runtime_host = lash_core::RuntimeHostConfig::in_memory()
        .with_process_env_store(process_env_store)
        .with_process_engine(Arc::new(
            lash_lashlang_runtime::LashlangProcessEngine::in_memory(
                lash_lashlang_runtime::LashlangSurface::default(),
            ),
        ));
    DurableProcessWorker::new(
        lash_core::DurableProcessWorkerConfig::new(
            Arc::new(plugin_host),
            runtime_host,
            store_factory,
            registry,
        )
        .with_session_policy(recovery_session_policy()),
    )
}

fn recovery_session_policy() -> lash_core::SessionPolicy {
    lash_core::SessionPolicy {
        model: lash_core::ModelSpec::from_token_limits("mock-model", None, 200_000, None)
            .expect("model spec"),
        ..lash_core::SessionPolicy::default()
    }
}

async fn persist_recovery_env_ref() -> lash_core::ProcessExecutionEnvRef {
    let spec = lash_core::ProcessExecutionEnvSpec::new(
        lash_core::PluginOptions::empty(),
        recovery_session_policy(),
    );
    lash_core::runtime::persist_process_execution_env(RECOVERY_PROCESS_ENV_STORE.as_ref(), &spec)
        .await
        .expect("persist recovery process execution env")
}

async fn persist_snapshot_recovery_env_ref(
    snapshot_ref: &str,
) -> lash_core::ProcessExecutionEnvRef {
    let spec = lash_core::ProcessExecutionEnvSpec::new(
        snapshot_recovery_tool_options(snapshot_ref),
        recovery_session_policy(),
    );
    lash_core::runtime::persist_process_execution_env(RECOVERY_PROCESS_ENV_STORE.as_ref(), &spec)
        .await
        .expect("persist snapshot recovery process execution env")
}

fn process_wake_event_type() -> lash_core::ProcessEventType {
    lash_core::ProcessEventType {
        name: "process.wake".to_string(),
        payload_schema: lash_core::LashSchema::any(),
        semantics: lash_core::ProcessEventSemanticsSpec {
            wake: Some(lash_core::ProcessWakeSpec {
                when: Some(lash_core::ProcessValueSelector::Present(
                    "/wake_input".to_string(),
                )),
                input: lash_core::ProcessValueSelector::Pointer("/wake_input".to_string()),
                dedupe_key: lash_core::ProcessWakeDedupeKey::Selector(
                    lash_core::ProcessValueSelector::Pointer("/message".to_string()),
                ),
            }),
            ..lash_core::ProcessEventSemanticsSpec::default()
        },
    }
}

async fn snapshot_lashlang_registration(
    process_id: &str,
    env_ref: lash_core::ProcessExecutionEnvRef,
) -> ProcessRegistration {
    let module = lashlang::parse(
        r#"
        process main() {
          called = await tools.snapshot_echo({ line: "restored" })?
          finish called.echo
        }
        "#,
    )
    .expect("snapshot lashlang module");
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources.add_module_operation(
        ["tools"],
        "Tools",
        "snapshot_echo",
        "tool:snapshot_echo",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    let linked_module = lashlang::LinkedModule::link(
        module,
        lashlang::LashlangHostEnvironment::new(
            resources,
            lashlang::LashlangAbilities::default()
                .with_processes()
                .with_sleep()
                .with_process_signals(),
        ),
    )
    .expect("link snapshot lashlang module");
    lashlang::LashlangArtifactStore::put_module_artifact(
        lashlang::global_in_memory_lashlang_artifact_store().as_ref(),
        &linked_module.artifact,
    )
    .await
    .expect("store snapshot lashlang module artifact");
    let process_ref = linked_module
        .artifact
        .process_ref("main")
        .expect("main process ref")
        .clone();
    ProcessRegistration::new(
        process_id,
        lashlang_process_input(lash_lashlang_runtime::LashlangProcessInput {
            module_ref: linked_module.module_ref,
            process_ref,
            host_requirements_ref: linked_module.host_requirements_ref,
            process_name: "main".to_string(),
            args: serde_json::Map::new(),
        }),
        lash_core::ProcessProvenance::host(),
    )
    .with_extra_event_types(lash_lashlang_runtime::lashlang_process_event_types())
    .with_execution_env_ref(Some(env_ref))
}

#[tokio::test]
async fn sqlite_process_recovery_reopens_registry_worker_grants_wakes_and_cancel() {
    let temp = tempfile::tempdir().expect("tempdir");
    let process_db = temp.path().join("processes.db");
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        temp.path().join("sessions"),
    )) as Arc<dyn lash_core::SessionStoreFactory>;
    let registry_a = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("open registry"),
    ) as Arc<dyn ProcessRegistry>;
    let worker_a = recovery_worker(Arc::clone(&registry_a), Arc::clone(&store_factory));
    let _root_store = store_factory
        .create_store(&lash_core::SessionStoreCreateRequest {
            session_id: "root".to_string(),
            relation: lash_core::SessionRelation::default(),
            policy: recovery_session_policy(),
        })
        .await
        .expect("create root session store before wake delivery");
    let endpoint_a = Endpoint::builder()
        .bind(
            LashProcessWorkflowImpl::new(
                Arc::new(RestateCoreProcessRunner::new(worker_a)),
                Arc::clone(&registry_a),
            )
            .serve(),
        )
        .build();
    let context_a = Arc::new(RecordingContext::with_endpoint(endpoint_a));
    let host_a = RestateRuntimeEffectController::new(context_a);
    let creator_scope = lash_core::SessionScope::new("root");
    let scope_id = creator_scope.id();
    let env_ref = persist_recovery_env_ref().await;
    let registration = ProcessRegistration::new(
        "recover-tool",
        ProcessInput::ToolCall {
            call: lash_core::PreparedToolCall::from_parts(
                "recover-call",
                "tool:recovery_echo",
                "recovery_echo",
                serde_json::json!({ "line": "wake-after-rebuild" }),
                None,
                serde_json::Value::Null,
            ),
        },
        lash_core::ProcessProvenance::session(creator_scope.clone()),
    )
    .with_extra_event_types([process_wake_event_type()])
    .with_execution_env_ref(Some(env_ref))
    .with_wake_target(Some(creator_scope.clone()));

    host_a
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "recovery-start"),
                RuntimeEffectCommand::process(ProcessCommand::Start {
                    registration,
                    grant: Some(lash_core::ProcessStartGrant {
                        session_scope: creator_scope.clone(),
                        descriptor: lash_core::ProcessHandleDescriptor::new(
                            Some("tool"),
                            Some("recover-tool"),
                        ),
                    }),
                    execution_context: Box::new(ProcessExecutionContext::default()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(Arc::clone(&registry_a), None),
        )
        .await
        .expect("schedule and run process through Restate endpoint");
    drop(host_a);
    drop(registry_a);

    let registry_b = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("reopen registry"),
    ) as Arc<dyn ProcessRegistry>;
    let grants = registry_b
        .list_handle_grants(&creator_scope)
        .await
        .expect("list reopened grants");
    assert_eq!(grants.len(), 1);
    assert_eq!(grants[0].0.process_id, "recover-tool");
    assert_eq!(
        lash_core::ProcessAwaiter::polling(Arc::clone(&registry_b))
            .await_terminal("recover-tool")
            .await
            .expect("await recovered terminal process"),
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "echo": "wake-after-rebuild" }),
            control: None,
        }
    );
    let queue_store = store_factory
        .create_store(&lash_core::SessionStoreCreateRequest {
            session_id: "root".to_string(),
            relation: lash_core::SessionRelation::default(),
            policy: lash_core::SessionPolicy {
                model: lash_core::ModelSpec::from_token_limits("mock-model", None, 200_000, None)
                    .expect("model spec"),
                ..lash_core::SessionPolicy::default()
            },
        })
        .await
        .expect("open root session store");
    let queued = queue_store
        .list_queued_work("root")
        .await
        .expect("list queued wakes");
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].items.len(), 1);
    let lash_core::runtime::QueuedWorkPayload::ProcessWake { wake } = &queued[0].items[0].payload
    else {
        panic!("expected process wake queue payload");
    };
    assert_eq!(wake.input, "wake-after-rebuild");
    assert_eq!(wake.target_scope_id, scope_id);

    let worker_b = recovery_worker(Arc::clone(&registry_b), store_factory);
    let endpoint_b = Endpoint::builder()
        .bind(
            LashProcessWorkflowImpl::new(
                Arc::new(RestateCoreProcessRunner::new(worker_b)),
                Arc::clone(&registry_b),
            )
            .serve(),
        )
        .build();
    let context_b = Arc::new(RecordingContext::with_endpoint(endpoint_b));
    let host_b = RestateRuntimeEffectController::new(context_b);
    host_b
        .execute_effect(
            RuntimeEffectEnvelope::new(
                runtime_invocation(RuntimeEffectKind::Process, "recovery-cancel"),
                RuntimeEffectCommand::process(ProcessCommand::Cancel {
                    process_id: "recover-tool".to_string(),
                    reason: Some("post-rebuild cancel probe".to_string()),
                }),
            ),
            RuntimeEffectLocalExecutor::processes(Arc::clone(&registry_b), None),
        )
        .await
        .expect("cancel through reopened process workflow");
    assert!(
        registry_b
            .events_after("recover-tool", 0)
            .await
            .expect("events after cancel")
            .iter()
            .any(|event| event.event_type == "process.cancel_requested")
    );
}

#[tokio::test]
async fn sqlite_process_recovery_rebuilds_snapshot_plugin_options_after_worker_reopen() {
    let temp = tempfile::tempdir().expect("tempdir");
    let process_db = temp.path().join("processes.db");
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        temp.path().join("sessions"),
    )) as Arc<dyn lash_core::SessionStoreFactory>;
    let registry_a = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("open registry"),
    ) as Arc<dyn ProcessRegistry>;
    let env_ref = persist_snapshot_recovery_env_ref("tool-authority:sha256:ok").await;
    registry_a
        .register_process(snapshot_lashlang_registration("snapshot-ok", env_ref).await)
        .await
        .expect("register snapshot-backed process");
    drop(registry_a);

    let registry_b = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("reopen registry"),
    ) as Arc<dyn ProcessRegistry>;
    let worker_b = recovery_worker_with_plugins(
        Arc::clone(&registry_b),
        store_factory,
        vec![snapshot_recovery_tool_factory()],
    );
    worker_b
        .drive_pending_processes()
        .await
        .expect("recover snapshot-backed process");

    assert_eq!(
        lash_core::ProcessAwaiter::polling(Arc::clone(&registry_b))
            .await_terminal("snapshot-ok")
            .await
            .expect("await recovered snapshot-backed process"),
        ProcessAwaitOutput::Success {
            value: serde_json::json!("snapshot:restored"),
            control: None,
        }
    );
}

#[tokio::test]
async fn sqlite_process_recovery_terminalizes_revoked_snapshot_plugin_options() {
    let temp = tempfile::tempdir().expect("tempdir");
    let process_db = temp.path().join("processes.db");
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        temp.path().join("sessions"),
    )) as Arc<dyn lash_core::SessionStoreFactory>;
    let registry_a = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("open registry"),
    ) as Arc<dyn ProcessRegistry>;
    let env_ref = persist_snapshot_recovery_env_ref("tool-authority:sha256:revoked").await;
    registry_a
        .register_process(snapshot_lashlang_registration("snapshot-revoked", env_ref).await)
        .await
        .expect("register revoked snapshot-backed process");
    drop(registry_a);

    let registry_b = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("reopen registry"),
    ) as Arc<dyn ProcessRegistry>;
    let worker_b = recovery_worker_with_plugins(
        Arc::clone(&registry_b),
        store_factory,
        vec![snapshot_recovery_tool_factory()],
    );
    worker_b
        .drive_pending_processes()
        .await
        .expect("recover revoked snapshot-backed process");

    let await_output = lash_core::ProcessAwaiter::polling(Arc::clone(&registry_b))
        .await_terminal("snapshot-revoked")
        .await
        .expect("await terminal revoked snapshot-backed process");
    let ProcessAwaitOutput::Failure { code, message, .. } = await_output else {
        panic!("expected revoked snapshot process failure, got {await_output:#?}");
    };
    assert_eq!(code, "process_host_environment_incompatible");
    assert!(
        message.contains("module `tools` does not expose operation `snapshot_echo`"),
        "{message}"
    );
}

/// Build a durable registration for a trigger-started Lashlang engine process.
///
/// A trigger-started process carries the trigger route's engine payload and
/// provenance whose `caused_by` is the
/// trigger occurrence that fired it — distinct from a turn-started process, whose
/// provenance traces to a live turn/tool call. The module artifact is stored
/// in the process-global in-memory artifact store, mirroring how a trigger
/// route's linked module is published before the process runs; that store
/// survives the registry/worker reopen within a single test process.
async fn trigger_lashlang_registration(process_id: &str, resource: &str) -> ProcessRegistration {
    let module =
        lashlang::parse("process notify(resource: str) { finish { triggered: resource } }")
            .expect("lashlang trigger module");
    let linked_module = lashlang::LinkedModule::link(
        module,
        lashlang::LashlangHostEnvironment::new(
            lashlang::LashlangHostCatalog::new(),
            lashlang::LashlangAbilities::all(),
        ),
    )
    .expect("link lashlang trigger module");
    lashlang::LashlangArtifactStore::put_module_artifact(
        lashlang::global_in_memory_lashlang_artifact_store().as_ref(),
        &linked_module.artifact,
    )
    .await
    .expect("store lashlang trigger module artifact");
    let process_ref = linked_module
        .artifact
        .process_ref("notify")
        .expect("notify process ref")
        .clone();
    let mut args = serde_json::Map::new();
    args.insert("resource".to_string(), serde_json::json!(resource));
    let env_ref = persist_recovery_env_ref().await;
    ProcessRegistration::new(
        process_id,
        lashlang_process_input(lash_lashlang_runtime::LashlangProcessInput {
            module_ref: linked_module.module_ref,
            process_ref,
            host_requirements_ref: linked_module.host_requirements_ref,
            process_name: "notify".to_string(),
            args,
        }),
        lash_core::ProcessProvenance::session(lash_core::SessionScope::new("root")).with_caused_by(
            Some(lash_core::CausalRef::SessionNode {
                session_id: "root".to_string(),
                node_id: "trigger:resource.updated".to_string(),
            }),
        ),
    )
    .with_extra_event_types(lash_lashlang_runtime::lashlang_process_event_types())
    .with_execution_env_ref(Some(env_ref))
}

fn assert_lashlang_engine_record(
    record: &lash_core::ProcessRecord,
    expected_process_name: &str,
    expected_args: serde_json::Map<String, serde_json::Value>,
) {
    let ProcessInput::Engine { kind, payload } = record.input.as_ref() else {
        panic!(
            "persisted Lashlang process must use generic engine input, got {:?}",
            record.input
        );
    };
    assert_eq!(
        kind,
        lash_lashlang_runtime::LASHLANG_ENGINE_KIND,
        "persisted row must dispatch through the registered Lashlang process engine"
    );
    let decoded = lash_lashlang_runtime::LashlangProcessInput::from_payload(payload.clone())
        .expect("persisted Lashlang engine payload must decode after registry reopen");
    assert_eq!(decoded.process_name, expected_process_name);
    assert_eq!(decoded.args, expected_args);
}

/// Phase-B recovery: a TRIGGER-started process whose worker died mid-flight is
/// left non-terminal in the durable registry; a subsequent worker reopening
/// that registry must drive it to completion via the recovery sweep — the same
/// durable re-execution guarantee a turn-started process has (invariant 3).
///
/// Mirrors `sqlite_process_recovery_reopens_registry_worker_grants_wakes_and_cancel`
/// but the process is started by a trigger occurrence (a `lashlang` engine row
/// with trigger provenance), not by a live turn's tool call.
#[tokio::test]
async fn sqlite_trigger_started_process_recovered_after_worker_registry_reopen() {
    let temp = tempfile::tempdir().expect("tempdir");
    let process_db = temp.path().join("processes.db");
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        temp.path().join("sessions"),
    )) as Arc<dyn lash_core::SessionStoreFactory>;

    // A worker started the trigger process and crashed before it could run:
    // the durable row exists and is non-terminal. We register it directly to
    // model exactly that mid-flight crash state.
    let registry_a = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("open registry"),
    ) as Arc<dyn ProcessRegistry>;
    registry_a
        .register_process(trigger_lashlang_registration("trigger-notify", "issue-42").await)
        .await
        .expect("register trigger-started process");
    let persisted_before_rebuild = registry_a
        .get_process("trigger-notify")
        .await
        .expect("persisted trigger-started process before recovery");
    assert!(
        !persisted_before_rebuild.is_terminal(),
        "freshly trigger-started process must be non-terminal before recovery"
    );
    drop(registry_a);

    // Reopen the registry and stand up a fresh worker over it: the crash
    // recovery counterpart. The recovery sweep submits the non-terminal process
    // by workflow key; Restate coalesces duplicates and the workflow writes the
    // terminal outcome.
    let registry_b = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&process_db)
            .await
            .expect("reopen registry"),
    ) as Arc<dyn ProcessRegistry>;
    let reopened_record = registry_b
        .get_process("trigger-notify")
        .await
        .expect("trigger-started process survives registry reopen");
    assert_lashlang_engine_record(
        &reopened_record,
        "notify",
        serde_json::Map::from_iter([("resource".to_string(), serde_json::json!("issue-42"))]),
    );
    assert_eq!(
        registry_b
            .list_non_terminal()
            .await
            .expect("list non-terminal after reopen")
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        vec!["trigger-notify"],
        "the trigger-started process must be on the recovery worklist after reopen"
    );

    let worker_b = recovery_worker(Arc::clone(&registry_b), Arc::clone(&store_factory));
    worker_b
        .drive_pending_processes()
        .await
        .expect("recover non-terminal trigger-started process");

    assert_eq!(
        lash_core::ProcessAwaiter::polling(Arc::clone(&registry_b))
            .await_terminal("trigger-notify")
            .await
            .expect("await recovered trigger-started process"),
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "triggered": "issue-42" }),
            control: None,
        },
        "the trigger-started process must run to its terminal value on recovery"
    );
    assert!(
        registry_b
            .list_non_terminal()
            .await
            .expect("list non-terminal after recovery")
            .is_empty(),
        "recovery must drive the trigger-started process to terminal"
    );

    // Idempotent by process_id: re-running the sweep over an already-terminal
    // process is a no-op and never double-executes it.
    worker_b
        .drive_pending_processes()
        .await
        .expect("second recovery sweep is idempotent");
    assert_eq!(
        lash_core::ProcessAwaiter::polling(Arc::clone(&registry_b))
            .await_terminal("trigger-notify")
            .await
            .expect("await after idempotent re-sweep"),
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "triggered": "issue-42" }),
            control: None,
        }
    );
}

fn discover_service<S: Discoverable>(_: &S) -> restate_sdk::discovery::Service {
    S::discover()
}

#[tokio::test]
async fn process_workflow_binds_to_restate_endpoint_and_discovers_handlers() {
    let runner = Arc::new(RecordingRunner::default());
    let registry = process_registry();
    let service = LashProcessWorkflowImpl::new(runner, registry).serve();
    let discovery = discover_service(&service);
    let endpoint = Endpoint::builder().bind(service).build();

    assert_eq!(discovery.name.to_string(), "LashProcessWorkflow");
    assert_eq!(
        discovery.ty.to_string(),
        restate_sdk::discovery::ServiceType::Workflow.to_string()
    );
    assert_eq!(discovery.handlers.len(), 4);

    let run = discovery
        .handlers
        .iter()
        .find(|handler| handler.name.to_string() == "run")
        .expect("run handler discovery");
    let cancel = discovery
        .handlers
        .iter()
        .find(|handler| handler.name.to_string() == "cancel")
        .expect("cancel handler discovery");
    let resolve_event = discovery
        .handlers
        .iter()
        .find(|handler| handler.name.to_string() == "resolve_event")
        .expect("resolve_event handler discovery");
    let await_terminal = discovery
        .handlers
        .iter()
        .find(|handler| handler.name.to_string() == "await_terminal")
        .expect("await_terminal handler discovery");

    assert_eq!(
        run.ty.as_ref().map(ToString::to_string).as_deref(),
        Some("WORKFLOW")
    );
    assert_eq!(
        cancel.ty.as_ref().map(ToString::to_string).as_deref(),
        Some("SHARED")
    );
    assert_eq!(
        resolve_event
            .ty
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("SHARED")
    );
    assert_eq!(
        await_terminal
            .ty
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("SHARED")
    );

    let response = endpoint.handle(
        http::Request::builder()
            .uri("/discover")
            .header("accept", "application/vnd.restate.endpointmanifest.v3+json")
            .body(Empty::<bytes::Bytes>::new())
            .expect("discover request"),
    );
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/vnd.restate.endpointmanifest.v3+json")
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("discover response body")
        .to_bytes();
    let manifest: serde_json::Value =
        serde_json::from_slice(&body).expect("discover response json");
    let workflow = manifest["services"]
        .as_array()
        .expect("services array")
        .iter()
        .find(|service| service["name"] == "LashProcessWorkflow")
        .expect("workflow service");
    let handlers = workflow["handlers"].as_array().expect("handlers array");
    assert!(
        handlers
            .iter()
            .any(|handler| handler["name"] == "run" && handler["ty"] == "WORKFLOW")
    );
    assert!(
        handlers
            .iter()
            .any(|handler| handler["name"] == "cancel" && handler["ty"] == "SHARED")
    );
    assert!(
        handlers
            .iter()
            .any(|handler| handler["name"] == "resolve_event" && handler["ty"] == "SHARED")
    );
    assert!(
        handlers
            .iter()
            .any(|handler| handler["name"] == "await_terminal" && handler["ty"] == "SHARED")
    );
}

#[tokio::test]
async fn process_deployment_driver_and_workflow_share_registry() {
    let registry = process_registry();
    let deployment = RestateProcessDeployment::new("http://127.0.0.1:8080", Arc::clone(&registry));
    let driver = deployment.process_work_driver();
    let driver_registry = driver.process_registry();

    assert!(Arc::ptr_eq(&driver.process_registry(), &driver_registry));

    let worker = DurableProcessWorker::new(
        lash_core::DurableProcessWorkerConfig::new(
            Arc::new(lash_core::PluginHost::empty()),
            lash_core::RuntimeHostConfig::in_memory(),
            Arc::new(lash_core::InMemorySessionStoreFactory::new()),
            Arc::clone(&driver_registry),
        )
        .with_change_hub(driver.change_hub()),
    );
    let service = deployment.workflow(worker).serve();
    let discovery = discover_service(&service);
    let endpoint = Endpoint::builder().bind(service).build();

    assert_eq!(discovery.name.to_string(), "LashProcessWorkflow");
    assert!(discovery.handlers.iter().any(|handler| {
        handler.name.to_string() == "run"
            && handler.ty.as_ref().map(ToString::to_string).as_deref() == Some("WORKFLOW")
    }));
    assert!(discovery.handlers.iter().any(|handler| {
        handler.name.to_string() == "cancel"
            && handler.ty.as_ref().map(ToString::to_string).as_deref() == Some("SHARED")
    }));
    assert!(discovery.handlers.iter().any(|handler| {
        handler.name.to_string() == "resolve_event"
            && handler.ty.as_ref().map(ToString::to_string).as_deref() == Some("SHARED")
    }));
    assert!(discovery.handlers.iter().any(|handler| {
        handler.name.to_string() == "await_terminal"
            && handler.ty.as_ref().map(ToString::to_string).as_deref() == Some("SHARED")
    }));

    let response = endpoint.handle(
        http::Request::builder()
            .uri("/discover")
            .header("accept", "application/vnd.restate.endpointmanifest.v3+json")
            .body(Empty::<bytes::Bytes>::new())
            .expect("discover request"),
    );
    assert_eq!(response.status(), http::StatusCode::OK);
}

#[tokio::test]
async fn process_workflow_impl_runs_and_cancels_through_runner() {
    let runner = Arc::new(RecordingRunner::default());
    let registry = process_registry();
    let workflow = LashProcessWorkflowImpl::new(runner.clone(), registry.clone());
    let registration = external_registration("task-workflow")
        .with_wake_target(Some(lash_core::SessionScope::new("wake-session")));
    registry
        .register_process(registration.clone())
        .await
        .expect("register workflow process");
    let execution_context = ProcessExecutionContext::default().with_causal_invocation(Some(
        runtime_invocation(RuntimeEffectKind::ToolAttempt, "tool-effect"),
    ));

    let output = workflow
        .run_registration(
            registration,
            execution_context,
            lash_core::ScopedEffectController::shared(
                Arc::new(lash_core::InlineRuntimeEffectController),
                lash_core::ExecutionScope::process("task-workflow"),
            )
            .expect("inline process scope"),
        )
        .await
        .expect("workflow run");
    workflow
        .cancel_registration(RestateProcessCancelRequest {
            process_id: "task-workflow".to_string(),
            reason: Some("stop".to_string()),
        })
        .await
        .expect("workflow cancel");

    assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
    assert_eq!(
        runner.ran.lock().expect("runner ran lock").as_slice(),
        &[RecordedProcessRun {
            process_id: "task-workflow".to_string(),
            wake_target_session_id: Some("wake-session".to_string()),
            tool_effect_id: Some("tool-effect".to_string()),
            execution_scope_id: "task-workflow".to_string(),
            controller_tier: lash_core::DurabilityTier::Inline,
        }]
    );
    assert_eq!(
        runner
            .cancelled
            .lock()
            .expect("runner cancelled lock")
            .as_slice(),
        &[RestateProcessCancelRequest {
            process_id: "task-workflow".to_string(),
            reason: Some("stop".to_string()),
        }]
    );
}

#[tokio::test]
async fn ingress_runner_submits_non_terminal_process_by_workflow_key() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // A non-terminal process is the durable worklist row the ingress runner
    // must submit.
    let registry = process_registry();
    registry
        .register_process(external_registration("task-1"))
        .await
        .expect("register");

    // Minimal mock ingress: capture two submissions, then reply 202 Accepted
    // so the reqwest submit succeeds. The second submit exercises the
    // registry's exact-repeat external_ref path for a still-running process.
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured_server = captured.clone();
    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.expect("read request");
            captured_server
                .lock()
                .expect("captured lock")
                .push(String::from_utf8_lossy(&buf[..n]).into_owned());
            socket
                .write_all(
                    b"HTTP/1.1 202 Accepted\r\ncontent-type: application/json\r\ncontent-length: 49\r\n\r\n{\"invocationId\":\"inv_task_1\",\"status\":\"Accepted\"}",
                )
                .await
                .expect("write response");
            socket.flush().await.expect("flush");
        }
    });

    let runner = RestateProcessIngressRunner::new(format!("http://{addr}"), registry.clone());
    runner.claim_and_run_pending().await.expect("drive pending");
    runner
        .claim_and_run_pending()
        .await
        .expect("drive pending again");
    server.await.expect("mock ingress server task");

    let requests = captured.lock().expect("captured lock").clone();
    assert_eq!(
        requests.len(),
        2,
        "the non-terminal process must be submitted on both scans"
    );
    let request = &requests[0];
    assert!(
        request.starts_with("POST /LashProcessWorkflow/task-1/run/send "),
        "submits the keyed workflow run: {request}"
    );
    assert!(
        !request.contains("idempotency-key:"),
        "workflow sends must not carry an idempotency header; Restate coalesces by workflow key: {request}"
    );
    assert!(
        requests[1].starts_with("POST /LashProcessWorkflow/task-1/run/send "),
        "repeat scan submits the same keyed workflow run: {}",
        requests[1]
    );

    // The durable backend reference is recorded so the process is observably
    // owned by Restate.
    let record = registry.get_process("task-1").await.expect("get process");
    assert_eq!(
        record.external_ref.as_ref().map(|e| e.backend.as_str()),
        Some("restate"),
        "the durable external_ref must be recorded after a successful submit"
    );
    assert_eq!(
        record
            .external_ref
            .as_ref()
            .and_then(|external| external.metadata.as_ref())
            .and_then(|metadata| metadata.get("invocation_id")),
        Some(&serde_json::json!("inv_task_1"))
    );
}

struct MockHttpResponse {
    status: &'static str,
    body: &'static str,
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::new();
    let mut scratch = [0u8; 1024];
    loop {
        let n = socket.read(&mut scratch).await.expect("read request");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&scratch[..n]);
        let Some(header_end) = buf.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if buf.len() >= header_end + 4 + content_length {
            break;
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

async fn spawn_restate_http_capture(
    responses: Vec<MockHttpResponse>,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured_server = Arc::clone(&captured);
    let server = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_http_request(&mut socket).await;
            captured_server.lock().expect("captured lock").push(request);
            let body = response.body.as_bytes();
            let header = format!(
                "HTTP/1.1 {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.status,
                body.len()
            );
            socket
                .write_all(header.as_bytes())
                .await
                .expect("write response headers");
            socket.write_all(body).await.expect("write response body");
            socket.flush().await.expect("flush");
        }
    });
    (format!("http://{addr}"), captured, server)
}

#[tokio::test]
async fn restate_ingress_client_parses_send_invocation_id() {
    let (base_url, captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "202 Accepted",
        body: r#"{"invocationId":"inv_123","status":"Accepted"}"#,
    }])
    .await;
    let client = RestateIngressClient::new(base_url);

    let invocation_id = client
        .send_workflow_json(
            "WorkbenchTurnWorkflow",
            "turn-1",
            "run",
            &serde_json::json!({ "turn_id": "turn-1" }),
        )
        .await
        .expect("send workflow");
    server.await.expect("capture server");

    assert_eq!(invocation_id.as_str(), "inv_123");
    let requests = captured.lock().expect("captured lock");
    assert!(
        requests[0].starts_with("POST /WorkbenchTurnWorkflow/turn-1/run/send "),
        "unexpected request: {}",
        requests[0]
    );
    assert!(requests[0].contains(r#""turn_id":"turn-1""#));
}

#[tokio::test]
async fn restate_ingress_client_accepts_previously_accepted_send() {
    let (base_url, _captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "202 Accepted",
        body: r#"{"invocationId":"inv_duplicate","status":"PreviouslyAccepted"}"#,
    }])
    .await;
    let client = RestateIngressClient::new(base_url);

    let invocation_id = client
        .send_workflow_json(
            "LashProcessWorkflow",
            "process-1",
            "run",
            &serde_json::json!({ "process_id": "process-1" }),
        )
        .await
        .expect("idempotent duplicate send");
    server.await.expect("capture server");

    assert_eq!(invocation_id.as_str(), "inv_duplicate");
}

#[tokio::test]
async fn restate_ingress_client_calls_workflow_and_decodes_output() {
    let (base_url, captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "200 OK",
        body: r#"{"type":"success","value":{"ok":true}}"#,
    }])
    .await;
    let client = RestateIngressClient::new(base_url);

    let output: ProcessAwaitOutput = client
        .call_workflow_json(
            "LashProcessWorkflow",
            "process-1",
            "await_terminal",
            &RestateProcessAwaitRequest {
                process_id: "process-1".to_string(),
            },
        )
        .await
        .expect("call workflow");
    server.await.expect("capture server");

    assert_eq!(
        output,
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "ok": true }),
            control: None,
        }
    );
    let requests = captured.lock().expect("captured lock");
    assert!(
        requests[0].starts_with("POST /LashProcessWorkflow/process-1/await_terminal "),
        "unexpected request: {}",
        requests[0]
    );
    assert!(!requests[0].contains("/send "));
}

#[tokio::test]
async fn restate_process_attach_calls_await_terminal_ingress() {
    let (base_url, _captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "200 OK",
        body: r#"{"type":"success","value":"attached"}"#,
    }])
    .await;
    let runner = RestateProcessIngressRunner::new(base_url, process_registry());

    let output = runner
        .await_terminal("process-1")
        .await
        .expect("attach await");
    server.await.expect("capture server");

    assert_eq!(
        output,
        ProcessAwaitOutput::Success {
            value: serde_json::json!("attached"),
            control: None,
        }
    );
}

#[tokio::test]
async fn restate_process_attach_maps_ingress_error_to_plugin_error() {
    let (base_url, _captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "500 Internal Server Error",
        body: r#"{"message":"boom"}"#,
    }])
    .await;
    let runner = RestateProcessIngressRunner::new(base_url, process_registry());

    let err = runner
        .await_terminal("process-1")
        .await
        .expect_err("attach error");
    server.await.expect("capture server");

    assert!(
        err.to_string()
            .contains("ingress await for process `process-1` failed")
    );
    assert!(err.to_string().contains("500 Internal Server Error"));
}

/// Like [`spawn_restate_http_capture`], but holds each accepted connection open
/// for `delay` before responding, modeling a durable promise that resolves only
/// once the workflow's `run` completes.
async fn spawn_restate_http_capture_delayed(
    responses: Vec<MockHttpResponse>,
    delay: std::time::Duration,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let captured = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured_server = Arc::clone(&captured);
    let server = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_http_request(&mut socket).await;
            captured_server.lock().expect("captured lock").push(request);
            tokio::time::sleep(delay).await;
            let body = response.body.as_bytes();
            let header = format!(
                "HTTP/1.1 {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.status,
                body.len()
            );
            socket
                .write_all(header.as_bytes())
                .await
                .expect("write response headers");
            socket.write_all(body).await.expect("write response body");
            socket.flush().await.expect("flush");
        }
    });
    (format!("http://{addr}"), captured, server)
}

#[tokio::test]
async fn restate_attach_before_run_resolves_with_delayed_workflow_output() {
    // The ingress attach is a synchronous long-hold call issued while the
    // workflow's `run` is still in flight; it resolves only when the durable
    // promise does. A delayed mock stands in for that hold, and the eventual
    // output flows back through the driver's attach.
    let delay = std::time::Duration::from_millis(300);
    let (base_url, captured, server) = spawn_restate_http_capture_delayed(
        vec![MockHttpResponse {
            status: "200 OK",
            body: r#"{"type":"success","value":{"eventual":true}}"#,
        }],
        delay,
    )
    .await;
    let registry = process_registry();
    let deployment = RestateProcessDeployment::new(base_url, Arc::clone(&registry));
    let driver = deployment.process_work_driver();
    // A non-terminal process routes await_terminal through the ingress attach
    // rather than the registry short-circuit.
    driver
        .process_registry()
        .register_process(external_registration("process-1"))
        .await
        .expect("register non-terminal process");

    let started = std::time::Instant::now();
    let output = driver
        .await_terminal("process-1")
        .await
        .expect("attach await resolves with the eventual output");
    let elapsed = started.elapsed();
    server.await.expect("capture server");

    assert_eq!(
        output,
        ProcessAwaitOutput::Success {
            value: serde_json::json!({ "eventual": true }),
            control: None,
        }
    );
    assert!(
        elapsed >= delay,
        "the attach must block on the durable promise until run resolves (waited {elapsed:?})"
    );
    let requests = captured.lock().expect("captured lock");
    assert_eq!(
        requests.len(),
        1,
        "await_terminal issues exactly one ingress call"
    );
    assert!(
        requests[0].starts_with("POST /LashProcessWorkflow/process-1/await_terminal "),
        "unexpected request: {}",
        requests[0]
    );
}

#[tokio::test]
async fn restate_driver_short_circuits_terminal_without_ingress_call() {
    // Empty response set: the capture server accepts nothing, so any ingress
    // call would fail. The registry terminal short-circuit must fire first, so
    // the attach is never consulted for an already-terminal process.
    let (base_url, captured, server) = spawn_restate_http_capture(vec![]).await;
    let registry = process_registry();
    let deployment = RestateProcessDeployment::new(base_url, Arc::clone(&registry));
    let driver = deployment.process_work_driver();
    let output = ProcessAwaitOutput::Success {
        value: serde_json::json!("already-terminal"),
        control: None,
    };
    driver
        .process_registry()
        .register_process(external_registration("process-1"))
        .await
        .expect("register");
    driver
        .process_registry()
        .complete_process("process-1", output.clone())
        .await
        .expect("complete");

    let resolved = driver
        .await_terminal("process-1")
        .await
        .expect("terminal short-circuit resolves without ingress");
    server.await.expect("capture server");

    assert_eq!(resolved, output);
    assert!(
        captured.lock().expect("captured lock").is_empty(),
        "a terminal short-circuit must not issue any ingress call"
    );
}

#[tokio::test]
async fn restate_process_attach_maps_malformed_ingress_body_to_plugin_error() {
    // A 2xx response whose body does not decode into ProcessAwaitOutput must
    // surface as a PluginError, not a panic — complementing the non-2xx case.
    let (base_url, _captured, server) = spawn_restate_http_capture(vec![MockHttpResponse {
        status: "200 OK",
        body: "this-is-not-valid-json",
    }])
    .await;
    let runner = RestateProcessIngressRunner::new(base_url, process_registry());

    let err = runner
        .await_terminal("process-1")
        .await
        .expect_err("a malformed ingress body must surface as an error");
    server.await.expect("capture server");

    assert!(
        err.to_string()
            .contains("ingress await for process `process-1` failed"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn restate_process_attach_is_reentrant_across_sequential_awaits() {
    // The shared await_terminal handler is re-entrant: two sequential attaches
    // each issue an independent ingress call and both succeed.
    let (base_url, captured, server) = spawn_restate_http_capture(vec![
        MockHttpResponse {
            status: "200 OK",
            body: r#"{"type":"success","value":"first"}"#,
        },
        MockHttpResponse {
            status: "200 OK",
            body: r#"{"type":"success","value":"second"}"#,
        },
    ])
    .await;
    let runner = RestateProcessIngressRunner::new(base_url, process_registry());

    let first = runner
        .await_terminal("process-1")
        .await
        .expect("first attach await");
    let second = runner
        .await_terminal("process-1")
        .await
        .expect("second attach await");
    server.await.expect("capture server");

    assert_eq!(
        first,
        ProcessAwaitOutput::Success {
            value: serde_json::json!("first"),
            control: None,
        }
    );
    assert_eq!(
        second,
        ProcessAwaitOutput::Success {
            value: serde_json::json!("second"),
            control: None,
        }
    );
    assert_eq!(
        captured.lock().expect("captured lock").len(),
        2,
        "each await issues an independent ingress call"
    );
}

#[tokio::test]
async fn restate_admin_client_cancels_kills_and_queries_invocation_status() {
    let (base_url, captured, server) = spawn_restate_http_capture(vec![
        MockHttpResponse {
            status: "202 Accepted",
            body: "",
        },
        MockHttpResponse {
            status: "200 OK",
            body: "",
        },
        MockHttpResponse {
            status: "200 OK",
            body: r#"{"rows":[{"id":"inv_123","target":"WorkbenchTurnWorkflow/turn-1/run","target_service_name":"WorkbenchTurnWorkflow","target_service_key":"turn-1","target_handler_name":"run","status":"completed","completion_result":"success","completion_failure":null}]}"#,
        },
    ])
    .await;
    let client = RestateAdminClient::new(base_url);
    let invocation_id = RestateInvocationId::new("inv_123");

    client
        .cancel_invocation(&invocation_id)
        .await
        .expect("cancel");
    client
        .kill_invocation_for_test_cleanup(&invocation_id)
        .await
        .expect("kill");
    let status = client
        .invocation_status(&invocation_id)
        .await
        .expect("status")
        .expect("status row");
    server.await.expect("capture server");

    assert!(status.completed_successfully());
    assert_eq!(status.target_service_name, "WorkbenchTurnWorkflow");
    let requests = captured.lock().expect("captured lock");
    assert!(
        requests[0].starts_with("PATCH /invocations/inv_123/cancel "),
        "unexpected cancel request: {}",
        requests[0]
    );
    assert!(
        requests[1].starts_with("PATCH /invocations/inv_123/kill "),
        "unexpected kill request: {}",
        requests[1]
    );
    assert!(
        requests[2].starts_with("POST /query "),
        "unexpected query request: {}",
        requests[2]
    );
    assert!(requests[2].contains("FROM sys_invocation WHERE id = 'inv_123'"));
}
