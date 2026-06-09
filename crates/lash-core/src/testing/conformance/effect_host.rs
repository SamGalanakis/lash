//! [`EffectHost`] scope-factory and effect-controller replay conformance.

use super::*;

/// One scope selected by an [`EffectHost`] and one effect envelope executed
/// through the scoped controller.
#[derive(Clone, Debug)]
pub struct RecordingEffectHostRecord {
    pub runtime_scope: RuntimeScope,
    pub effect_scope: EffectScope,
    pub effect_id: String,
    pub effect_kind: RuntimeEffectKind,
    pub replay_key: Option<String>,
    pub envelope_hash: String,
}

#[derive(Clone)]
struct RecordingEffectHostController {
    effect_scope: EffectScope,
    records: Arc<Mutex<Vec<RecordingEffectHostRecord>>>,
}

#[async_trait::async_trait]
impl RuntimeEffectController for RecordingEffectHostController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        let envelope_hash = envelope.stable_hash()?;
        self.records
            .lock()
            .expect("effect host records")
            .push(RecordingEffectHostRecord {
                runtime_scope: envelope.invocation.scope.clone(),
                effect_scope: self.effect_scope.clone(),
                effect_id: envelope
                    .invocation
                    .effect_id()
                    .expect("effect invocation")
                    .to_string(),
                effect_kind: envelope.command.kind(),
                replay_key: envelope.invocation.replay_key().map(ToOwned::to_owned),
                envelope_hash,
            });
        match envelope.command {
            RuntimeEffectCommand::Sleep { .. } => Ok(RuntimeEffectOutcome::Sleep),
            command => Err(RuntimeEffectControllerError::new(
                "recording_effect_host_unsupported_command",
                format!(
                    "recording effect host cannot synthesize {} outcomes",
                    command.kind().as_str()
                ),
            )),
        }
    }
}

/// Test fixture that records every selected [`EffectScope`] and every effect
/// envelope executed through the returned scoped controller.
#[derive(Clone, Default)]
pub struct RecordingEffectHost {
    selected_scopes: Arc<Mutex<Vec<EffectScope>>>,
    records: Arc<Mutex<Vec<RecordingEffectHostRecord>>>,
}

impl RecordingEffectHost {
    pub fn selected_scopes(&self) -> Vec<EffectScope> {
        self.selected_scopes
            .lock()
            .expect("selected effect scopes")
            .clone()
    }

    pub fn records(&self) -> Vec<RecordingEffectHostRecord> {
        self.records.lock().expect("effect host records").clone()
    }

    fn scoped_for<'run>(
        &self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, crate::RuntimeError> {
        self.selected_scopes
            .lock()
            .expect("selected effect scopes")
            .push(scope.clone());
        ScopedEffectController::shared(
            Arc::new(RecordingEffectHostController {
                effect_scope: scope.clone(),
                records: Arc::clone(&self.records),
            }),
            scope,
        )
    }
}

impl EffectHost for RecordingEffectHost {
    fn scoped<'run>(
        &'run self,
        scope: EffectScope,
    ) -> Result<ScopedEffectController<'run>, crate::RuntimeError> {
        self.scoped_for(scope)
    }

    fn scoped_static(
        &self,
        scope: EffectScope,
    ) -> Result<Option<ScopedEffectController<'static>>, crate::RuntimeError> {
        Ok(Some(self.scoped_for(scope)?))
    }
}

/// Run the generic [`EffectHost`] scope-factory conformance suite.
///
/// This suite checks the deployment-level contract: effect scopes must carry
/// stable semantic identity, empty ids must fail loudly, and hosts that expose
/// a static scoped controller must preserve the same scope metadata. It does
/// not assert durability; that remains a property of each implementation.
/// Substrate-native hosts such as Restate complete the in-flight contract at
/// this effect-host/controller boundary; [`RuntimePersistence`] remains the
/// committed-state store contract, not a workflow-history contract.
pub async fn effect_host<F>(make: F)
where
    F: Fn() -> Arc<dyn EffectHost>,
{
    effect_host_preserves_scope_metadata(make()).await;
    effect_host_rejects_missing_scope_ids(make()).await;
    effect_host_static_scope_preserves_metadata_when_available(make()).await;
}

async fn effect_host_preserves_scope_metadata(host: Arc<dyn EffectHost>) {
    let scope = EffectScope::queue_drain("session-1", "drain-1");
    let scoped = host.scoped(scope.clone()).expect("queue drain scope");
    assert_eq!(
        scoped.effect_scope(),
        &scope,
        "scoped controller must retain the selected semantic scope"
    );
    assert_eq!(scoped.scope_id(), "drain-1");
    assert_eq!(scoped.turn_id(), None);

    let turn_scope = EffectScope::turn("session-1", "turn-1");
    let scoped_turn = host.scoped(turn_scope.clone()).expect("turn scope");
    assert_eq!(scoped_turn.effect_scope(), &turn_scope);
    assert_eq!(scoped_turn.scope_id(), "turn-1");
    assert_eq!(scoped_turn.turn_id(), Some("turn-1"));
}

async fn effect_host_rejects_missing_scope_ids(host: Arc<dyn EffectHost>) {
    let invalid_scopes = [
        EffectScope::turn("", "turn"),
        EffectScope::turn("session", ""),
        EffectScope::process(""),
        EffectScope::queue_drain("session", ""),
        EffectScope::session_delete(""),
        EffectScope::runtime_operation(""),
    ];

    for scope in invalid_scopes {
        let err = match host.scoped(scope) {
            Ok(_) => panic!("invalid effect scope must be rejected"),
            Err(err) => err,
        };
        assert_eq!(
            err.code,
            crate::RuntimeErrorCode::MissingEffectScopeId,
            "invalid scope ids must fail with the stable missing-scope code"
        );
    }
}

async fn effect_host_static_scope_preserves_metadata_when_available(host: Arc<dyn EffectHost>) {
    let scope = EffectScope::runtime_operation("static-runtime-op");
    let Some(scoped) = host
        .scoped_static(scope.clone())
        .expect("static scope factory")
    else {
        return;
    };
    assert_eq!(scoped.effect_scope(), &scope);
    assert_eq!(scoped.scope_id(), "static-runtime-op");
}

/// Run the concurrent recorded-effect replay conformance case for a
/// handler-scoped durable controller.
///
/// The first pass starts two recorded effects concurrently and intentionally
/// lets the second finish before the first. After `start_replay`, the same
/// effects are requested in the opposite order with local executors that fail
/// if called. A compliant controller returns the recorded outcomes by
/// `replay.key`, independent of local completion/request ordering.
#[cfg(any(test, feature = "testing"))]
pub async fn effect_controller_concurrent_replay_deterministic(
    controller: &dyn RuntimeEffectController,
    start_replay: impl FnOnce(),
) {
    let slow = replay_conformance_exec_envelope("effect-slow");
    let fast = replay_conformance_exec_envelope("effect-fast");
    let completion_order = Arc::new(Mutex::new(Vec::new()));
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let release_slow = Arc::new(tokio::sync::Notify::new());

    let first_pass = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(
            controller.execute_effect(
                slow.clone(),
                replay_conformance_recording_executor(
                    "effect-slow",
                    Arc::clone(&barrier),
                    Arc::clone(&release_slow),
                    Arc::clone(&completion_order),
                ),
            ),
            controller.execute_effect(
                fast.clone(),
                replay_conformance_recording_executor(
                    "effect-fast",
                    Arc::clone(&barrier),
                    Arc::clone(&release_slow),
                    Arc::clone(&completion_order),
                ),
            ),
        )
    })
    .await
    .expect("concurrent first-pass effects must both enter their local executors");
    let slow_first = first_pass.0.expect("slow first pass");
    let fast_first = first_pass.1.expect("fast first pass");
    assert_replay_conformance_exec_marker(slow_first, "effect-slow");
    assert_replay_conformance_exec_marker(fast_first, "effect-fast");
    assert_eq!(
        completion_order
            .lock()
            .expect("completion order")
            .as_slice(),
        &["effect-fast".to_string(), "effect-slow".to_string()],
        "first pass must prove local completion order can differ from effect request order"
    );

    start_replay();
    let replay_local_calls = Arc::new(Mutex::new(Vec::new()));
    let replay_pass = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(
            controller.execute_effect(
                fast,
                replay_conformance_failing_executor(Arc::clone(&replay_local_calls)),
            ),
            controller.execute_effect(
                slow,
                replay_conformance_failing_executor(Arc::clone(&replay_local_calls)),
            ),
        )
    })
    .await
    .expect("concurrent replay effects must resolve from host history");
    let fast_replay = replay_pass.0.expect("fast replay");
    let slow_replay = replay_pass.1.expect("slow replay");
    assert_replay_conformance_exec_marker(fast_replay, "effect-fast");
    assert_replay_conformance_exec_marker(slow_replay, "effect-slow");
    assert!(
        replay_local_calls
            .lock()
            .expect("replay local calls")
            .is_empty(),
        "replay must return recorded outcomes without invoking local executors"
    );
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_exec_envelope(effect_id: &'static str) -> RuntimeEffectEnvelope {
    RuntimeEffectEnvelope::new(
        RuntimeInvocation::effect(
            RuntimeScope::for_turn(
                "effect-conformance-session",
                "effect-conformance-turn",
                7,
                0,
            ),
            effect_id,
            RuntimeEffectKind::ExecCode,
            format!("effect-conformance:effect-conformance-turn:{effect_id}"),
        ),
        RuntimeEffectCommand::ExecCode {
            code: format!("emit {effect_id}"),
        },
    )
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_recording_executor(
    effect_id: &'static str,
    barrier: Arc<tokio::sync::Barrier>,
    release_slow: Arc<tokio::sync::Notify>,
    completion_order: Arc<Mutex<Vec<String>>>,
) -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(move |envelope| async move {
        assert_eq!(envelope.invocation.effect_id(), Some(effect_id));
        barrier.wait().await;
        if effect_id == "effect-slow" {
            release_slow.notified().await;
        } else {
            completion_order
                .lock()
                .expect("completion order")
                .push(effect_id.to_string());
            release_slow.notify_one();
        }
        if effect_id == "effect-slow" {
            completion_order
                .lock()
                .expect("completion order")
                .push(effect_id.to_string());
        }
        Ok(replay_conformance_exec_outcome(effect_id))
    })
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_failing_executor(
    replay_local_calls: Arc<Mutex<Vec<String>>>,
) -> RuntimeEffectLocalExecutor<'static> {
    RuntimeEffectLocalExecutor::testing(move |envelope| async move {
        replay_local_calls
            .lock()
            .expect("replay local calls")
            .push(envelope.invocation.effect_id().unwrap_or("").to_string());
        Err(RuntimeEffectControllerError::new(
            "conformance_replay_local_executor_called",
            "recorded replay must not invoke local effect execution",
        ))
    })
}

#[cfg(any(test, feature = "testing"))]
fn replay_conformance_exec_outcome(effect_id: &str) -> RuntimeEffectOutcome {
    RuntimeEffectOutcome::ExecCode {
        result: Ok(crate::ExecResponse {
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 0,
            terminal_finish: Some(serde_json::json!(effect_id)),
        }),
    }
}

#[cfg(any(test, feature = "testing"))]
fn assert_replay_conformance_exec_marker(outcome: RuntimeEffectOutcome, expected: &str) {
    let RuntimeEffectOutcome::ExecCode { result } = outcome else {
        panic!("expected exec-code effect outcome");
    };
    let response = result.expect("exec-code response");
    assert_eq!(
        response.terminal_finish,
        Some(serde_json::json!(expected)),
        "replayed outcome must come from the matching replay key"
    );
}
