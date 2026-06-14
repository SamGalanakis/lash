use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;

struct LinkedTestProcess {
    module_ref: lashlang::ModuleRef,
    required_surface_ref: lashlang::RequiredSurfaceRef,
    process_ref: lashlang::ProcessRef,
    process_name: String,
    signal_event_types: Vec<lash_core::ProcessEventType>,
}

impl LinkedTestProcess {
    async fn new(
        artifact_store: &dyn lash_core::LashlangArtifactStore,
        source: &str,
        process_name: &str,
    ) -> Self {
        let linked = lashlang::LinkedModule::link(
            lashlang::parse(source).expect("parse lashlang process"),
            lashlang::LashlangSurface::new(
                lashlang::ResourceCatalog::new(),
                lashlang::LashlangAbilities::default()
                    .with_processes()
                    .with_sleep()
                    .with_process_signals(),
            ),
        )
        .expect("link lashlang process");
        artifact_store
            .put_module_artifact(&linked.artifact)
            .await
            .expect("store lashlang process artifact");
        let process_ref = linked
            .artifact
            .process_ref(process_name)
            .unwrap_or_else(|| panic!("missing process ref `{process_name}`"))
            .clone();
        let signal_event_types = linked
            .artifact
            .canonical_ir
            .process(process_name)
            .map(lash_core::lashlang_process_signal_event_types)
            .unwrap_or_default();
        Self {
            module_ref: linked.module_ref,
            required_surface_ref: linked.required_surface_ref,
            process_ref,
            process_name: process_name.to_string(),
            signal_event_types,
        }
    }

    fn start_request(&self, process_id: &str) -> lash_core::ProcessStartRequest {
        lash_core::ProcessStartRequest::new(
            process_id,
            lash_core::ProcessInput::LashlangProcess {
                module_ref: self.module_ref.clone(),
                process_ref: self.process_ref.clone(),
                required_surface_ref: self.required_surface_ref.clone(),
                process_name: self.process_name.clone(),
                args: serde_json::Map::new(),
            },
            lash_core::ProcessOriginator::host(),
        )
        .with_env_spec(process_env_spec())
        .with_extra_event_types(
            lash_core::lashlang_process_event_types()
                .into_iter()
                .chain(self.signal_event_types.clone()),
        )
    }

    fn trigger_draft(
        &self,
        source_type: &str,
        source_key: String,
        env_ref: lash_core::ProcessExecutionEnvRef,
    ) -> lash_core::TriggerSubscriptionDraft {
        lash_core::TriggerSubscriptionDraft {
            registrant: lash_core::ProcessOriginator::host(),
            env_ref,
            wake_target: None,
            name: Some("host-owned-test-trigger".to_string()),
            source_type: source_type.to_string(),
            source_key,
            source: serde_json::json!({}),
            event_ty: lashlang::TypeExpr::Any,
            module_ref: self.module_ref.clone(),
            required_surface_ref: self.required_surface_ref.clone(),
            process_ref: self.process_ref.clone(),
            process_name: self.process_name.clone(),
            input_template: lashlang::TriggerInputTemplate::new(BTreeMap::new()),
        }
    }
}

fn process_env_spec() -> lash_core::ProcessExecutionEnvSpec {
    lash_core::ProcessExecutionEnvSpec::new(
        lash_core::PluginOptions::default(),
        lash_core::SessionPolicy {
            model: mock_model_spec(),
            ..lash_core::SessionPolicy::default()
        },
    )
}

async fn persist_process_env_ref(
    artifact_store: &dyn lash_core::LashlangArtifactStore,
) -> lash_core::ProcessExecutionEnvRef {
    let spec = process_env_spec();
    let env_ref = spec.stable_ref().expect("stable process env ref");
    let bytes = spec.to_store_bytes().expect("encode process env spec");
    artifact_store
        .put_artifact_bytes(env_ref.as_str(), "process_execution_env", &bytes)
        .await
        .expect("store process execution env");
    env_ref
}

fn signal_request(
    process_id: &str,
    signal_name: &str,
    signal_id: &str,
    payload: serde_json::Value,
) -> lash_core::ProcessEventAppendRequest {
    let event_type = lash_core::process_signal_event_type(signal_name).expect("signal event type");
    lash_core::ProcessEventAppendRequest::new(event_type, payload).with_replay_key(format!(
        "process:{process_id}:signal.{signal_name}:{signal_id}"
    ))
}

async fn wait_for_process(
    core: &LashCore,
    process_id: &str,
    label: &str,
    matches: impl Fn(&lash_core::ObservedProcess) -> bool,
) -> lash_core::ObservedProcess {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let Some(process) = core.processes().get(process_id).await.expect("get process")
                && matches(&process)
            {
                return process;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
    core.processes()
        .get(process_id)
        .await
        .expect("get final process state")
        .unwrap_or_else(|| panic!("process `{process_id}` disappeared"))
}

async fn wait_for_waiting_signal(
    core: &LashCore,
    process_id: &str,
    signal_name: &str,
) -> lash_core::ObservedProcess {
    wait_for_process(core, process_id, "process signal wait", |process| {
        matches!(
            process.wait.as_ref().map(|wait| &wait.kind),
            Some(lash_core::WaitKind::Signal { name, .. }) if name == signal_name
        )
    })
    .await
}

async fn wait_for_terminal(
    core: &LashCore,
    process_id: &str,
    status: lash_core::ProcessLifecycleStatus,
) -> lash_core::ObservedProcess {
    wait_for_process(core, process_id, "terminal process", |process| {
        process.lifecycle == status
    })
    .await
}

fn process_test_core(
    artifact_store: Arc<dyn lash_core::LashlangArtifactStore>,
    trigger_store: Arc<dyn lash_core::TriggerStore>,
    registry: Arc<dyn lash_core::ProcessRegistry>,
) -> Result<LashCore> {
    explicit_ephemeral_facets(LashCore::rlm())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .lashlang_artifact_store(artifact_store)
        .trigger_store(trigger_store)
        .process_registry(registry)
        .build()
}

#[tokio::test]
async fn host_owned_processes_run_without_application_session() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process main() signals { ready: any } {
          value = wait_signal("ready")
          finish value
        }
        "#,
        "main",
    )
    .await;

    core.processes()
        .start(
            process.start_request("sessionless-direct"),
            runtime_operation_scope("sessionless-direct-start"),
        )
        .await?;
    let waiting = wait_for_waiting_signal(&core, "sessionless-direct", "ready").await;
    assert!(matches!(
        waiting.originator,
        lash_core::ProcessOriginator::Host
    ));
    let waiting_events = core.processes().events("sessionless-direct", 0).await?;
    assert!(
        waiting_events
            .iter()
            .any(|event| event.event_type == "process.waiting")
    );

    let cancelled = core
        .processes()
        .cancel(
            "sessionless-direct",
            runtime_operation_scope("sessionless-direct-cancel"),
        )
        .await?;
    assert_eq!(cancelled.status, lash_core::ProcessLifecycleStatus::Running);
    wait_for_terminal(
        &core,
        "sessionless-direct",
        lash_core::ProcessLifecycleStatus::Cancelled,
    )
    .await;

    let source_type = "timer.tick";
    let source_key = trigger_store
        .source_key_for_subscription(source_type, &serde_json::json!({}))
        .await?;
    let env_ref = persist_process_env_ref(artifact_store.as_ref()).await;
    trigger_store
        .register_subscription(process.trigger_draft(source_type, source_key.clone(), env_ref))
        .await?;
    let report = core
        .triggers()
        .emit(
            lash_core::TriggerOccurrenceRequest::new(
                source_type,
                source_key,
                serde_json::json!({ "at": "2026-06-10T12:00:00Z" }),
                "sessionless-trigger-1",
            )
            .with_source(serde_json::json!({})),
            runtime_operation_scope("sessionless-trigger"),
        )
        .await?;
    assert_eq!(report.started_process_ids.len(), 1);
    let triggered_process_id = &report.started_process_ids[0];
    let triggered = wait_for_waiting_signal(&core, triggered_process_id, "ready").await;
    assert!(matches!(
        triggered.originator,
        lash_core::ProcessOriginator::Host
    ));
    let event = core
        .processes()
        .signal(
            triggered_process_id,
            "ready",
            "host-signal-1",
            signal_request(
                triggered_process_id,
                "ready",
                "host-signal-1",
                serde_json::json!({ "ok": true }),
            ),
            runtime_operation_scope("sessionless-host-signal"),
        )
        .await?;
    assert_eq!(event.event_type, "signal.ready");
    let output = core.processes().await_output(triggered_process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = output else {
        panic!("triggered process did not succeed: {output:#?}");
    };
    assert_eq!(value, serde_json::json!({ "ok": true }));
    let signal_events = core.processes().events(triggered_process_id, 0).await?;
    assert!(
        signal_events
            .iter()
            .any(|event| event.event_type == "signal.ready")
    );
    Ok(())
}

#[tokio::test]
async fn signal_validation_rejects_undeclared_names_and_mistyped_payloads() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process main() signals { ready: string } {
          value = wait_signal("ready")
          finish value
        }
        "#,
        "main",
    )
    .await;
    let process_id = "signal-validation";

    core.processes()
        .start(
            process.start_request(process_id),
            runtime_operation_scope("signal-validation-start"),
        )
        .await?;
    wait_for_waiting_signal(&core, process_id, "ready").await;

    let undeclared = core
        .processes()
        .signal(
            process_id,
            "nope",
            "undeclared-1",
            signal_request(process_id, "nope", "undeclared-1", serde_json::json!("x")),
            runtime_operation_scope("signal-validation-undeclared"),
        )
        .await;
    let undeclared_err = undeclared.expect_err("undeclared signal name must be rejected");
    assert!(
        undeclared_err.to_string().contains("undeclared"),
        "unexpected error: {undeclared_err}"
    );

    let mistyped = core
        .processes()
        .signal(
            process_id,
            "ready",
            "mistyped-1",
            signal_request(
                process_id,
                "ready",
                "mistyped-1",
                serde_json::json!({ "not": "a string" }),
            ),
            runtime_operation_scope("signal-validation-mistyped"),
        )
        .await;
    assert!(
        mistyped.is_err(),
        "schema-invalid signal payload must be rejected"
    );

    // Both rejected sends left the process parked with nothing consumed.
    let still_waiting = wait_for_waiting_signal(&core, process_id, "ready").await;
    assert_eq!(
        still_waiting.lifecycle,
        lash_core::ProcessLifecycleStatus::Running
    );
    assert!(
        core.processes()
            .events(process_id, 0)
            .await?
            .iter()
            .all(|event| event.event_type != "signal.ready" && event.event_type != "signal.nope")
    );

    core.processes()
        .signal(
            process_id,
            "ready",
            "valid-1",
            signal_request(process_id, "ready", "valid-1", serde_json::json!("done")),
            runtime_operation_scope("signal-validation-valid"),
        )
        .await?;
    let output = core.processes().await_output(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = output else {
        panic!("process did not succeed after valid signal: {output:#?}");
    };
    assert_eq!(value, serde_json::json!("done"));
    Ok(())
}

#[tokio::test]
async fn repeated_waits_on_one_signal_consume_in_order() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process main() signals { ready: any } {
          first = wait_signal("ready")
          second = wait_signal("ready")
          finish { first: first, second: second }
        }
        "#,
        "main",
    )
    .await;
    let process_id = "repeated-waits";

    core.processes()
        .start(
            process.start_request(process_id),
            runtime_operation_scope("repeated-waits-start"),
        )
        .await?;

    let first_wait = wait_for_waiting_signal(&core, process_id, "ready").await;
    let lash_core::WaitKind::Signal { ordinal, .. } =
        first_wait.wait.expect("first wait facet").kind;
    assert_eq!(ordinal, 1, "first wait must use ordinal 1");
    core.processes()
        .signal(
            process_id,
            "ready",
            "order-1",
            signal_request(process_id, "ready", "order-1", serde_json::json!(1)),
            runtime_operation_scope("repeated-waits-signal-1"),
        )
        .await?;

    let second_wait = wait_for_process(&core, process_id, "second signal wait", |process| {
        matches!(
            process.wait.as_ref().map(|wait| &wait.kind),
            Some(lash_core::WaitKind::Signal { ordinal, .. }) if *ordinal == 2
        )
    })
    .await;
    let lash_core::WaitKind::Signal {
        key: second_key, ..
    } = second_wait.wait.expect("second wait facet").kind;
    assert!(
        second_key.ends_with(":2"),
        "second wait key must carry ordinal 2: {second_key}"
    );
    core.processes()
        .signal(
            process_id,
            "ready",
            "order-2",
            signal_request(process_id, "ready", "order-2", serde_json::json!(2)),
            runtime_operation_scope("repeated-waits-signal-2"),
        )
        .await?;

    let output = core.processes().await_output(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = output else {
        panic!("process did not succeed: {output:#?}");
    };
    assert_eq!(value, serde_json::json!({ "first": 1, "second": 2 }));

    // The suspension history is on the event log: two waits, two resumes.
    let events = core.processes().events(process_id, 0).await?;
    let waiting = events
        .iter()
        .filter(|event| event.event_type == "process.waiting")
        .count();
    let resumed = events
        .iter()
        .filter(|event| event.event_type == "process.resumed")
        .count();
    assert_eq!((waiting, resumed), (2, 2));
    Ok(())
}

#[tokio::test]
async fn process_starts_and_awaits_child_process() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process child() {
          finish { from: "child" }
        }

        process main() {
          handle = start child()
          value = await handle
          finish { joined: value }
        }
        "#,
        "main",
    )
    .await;
    let process_id = "parent-joins-child";

    core.processes()
        .start(
            process.start_request(process_id),
            runtime_operation_scope("parent-joins-child-start"),
        )
        .await?;
    let output = core.processes().await_output(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = output else {
        panic!("parent process did not succeed: {output:#?}");
    };
    // `await handle` yields the await envelope: success flag plus the child's
    // finish value.
    assert_eq!(
        value,
        serde_json::json!({ "joined": { "ok": true, "value": { "from": "child" } } })
    );

    // Both parent and child are globally addressable, completed, and the
    // child INHERITS its parent's provenance chain: Host originator, no wake
    // target, no grants — the ephemeral execution scope appears nowhere.
    let all = core
        .processes()
        .list(&lash_core::ProcessListFilter {
            status: lash_core::ProcessStatusFilter::Completed,
            ..lash_core::ProcessListFilter::default()
        })
        .await?;
    assert_eq!(all.len(), 2, "parent and child should both be completed");
    assert!(
        all.iter()
            .all(|process| matches!(process.originator, lash_core::ProcessOriginator::Host)),
        "children of a host chain stay host-originated"
    );
    let child = all
        .iter()
        .find(|process| process.process_id != process_id)
        .expect("child process record");
    assert!(child.wake_target.is_none(), "host chain has no wake target");
    Ok(())
}

#[tokio::test]
async fn process_children_inherit_session_chain_provenance() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let session_id = "chain-session";
    let process_id = "chain-parent";
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process child() {
          finish { from: "child" }
        }

        process main() {
          handle = start child()
          value = await handle
          finish value
        }
        "#,
        "main",
    )
    .await;
    let session = core.session(session_id).open().await?;
    session
        .process_control()
        .start(
            {
                let mut request = process.start_request(process_id);
                request.originator =
                    lash_core::ProcessOriginator::session(lash_core::SessionScope::new(session_id));
                request
            }
            .with_wake_target(Some(lash_core::SessionScope::new(session_id)))
            .with_grant(Some(lash_core::ProcessStartGrant {
                session_scope: lash_core::SessionScope::new(session_id),
                descriptor: lash_core::ProcessHandleDescriptor::new(
                    Some("lashlang"),
                    Some("chain parent"),
                ),
            })),
            runtime_operation_scope("chain-parent-start"),
        )
        .await?;
    wait_for_terminal(
        &core,
        process_id,
        lash_core::ProcessLifecycleStatus::Completed,
    )
    .await;

    // The child inherited the chain: session originator, session wake target,
    // and a grant derived from the wake target — so the session's snapshot
    // shows the whole background tree, and the ephemeral execution scope
    // appears nowhere.
    let completed = core
        .processes()
        .list(&lash_core::ProcessListFilter {
            status: lash_core::ProcessStatusFilter::Completed,
            ..lash_core::ProcessListFilter::default()
        })
        .await?;
    assert_eq!(completed.len(), 2);
    for observed in &completed {
        match &observed.originator {
            lash_core::ProcessOriginator::Session { scope } => {
                assert_eq!(scope.session_id, session_id)
            }
            other => panic!("expected session originator, got {other:?}"),
        }
        assert_eq!(
            observed
                .wake_target
                .as_ref()
                .map(|scope| scope.session_id.as_str()),
            Some(session_id)
        );
    }
    let snapshot = core.processes().session_snapshot(session_id).await?;
    assert_eq!(
        snapshot.items.len(),
        2,
        "parent and child are both visible in the originating session"
    );
    Ok(())
}

#[tokio::test]
async fn process_outlives_deleted_session_and_resumes_from_host_signal() -> Result<()> {
    let artifact_store: Arc<dyn lash_core::LashlangArtifactStore> =
        Arc::new(lash_core::InMemoryLashlangArtifactStore::new());
    let trigger_store: Arc<dyn lash_core::TriggerStore> =
        Arc::new(lash_core::InMemoryTriggerStore::default());
    let registry: Arc<dyn lash_core::ProcessRegistry> =
        Arc::new(TestLocalProcessRegistry::default());
    let core = process_test_core(
        Arc::clone(&artifact_store),
        Arc::clone(&trigger_store),
        Arc::clone(&registry),
    )?;
    let session_id = "process-outlives-session";
    let process_id = "outliving-process";
    let process = LinkedTestProcess::new(
        artifact_store.as_ref(),
        r#"
        process main() signals { ready: any } {
          value = wait_signal("ready")
          finish { resumed: value }
        }
        "#,
        "main",
    )
    .await;
    let session = core.session(session_id).open().await?;
    session
        .process_control()
        .start(
            process
                .start_request(process_id)
                .with_grant(Some(lash_core::ProcessStartGrant {
                    session_scope: lash_core::SessionScope::new(session_id),
                    descriptor: lash_core::ProcessHandleDescriptor::new(
                        Some("lashlang"),
                        Some("outliving process"),
                    ),
                })),
            runtime_operation_scope("outliving-process-start"),
        )
        .await?;
    wait_for_waiting_signal(&core, process_id, "ready").await;
    drop(session);

    let report = core
        .delete_session(session_id, session_delete_scope(session_id))
        .await?;
    let process_report = report.process.expect("process delete report");
    assert_eq!(
        process_report.orphaned_process_ids,
        vec![process_id.to_string()]
    );
    assert!(process_report.preserved_process_ids.is_empty());
    assert_eq!(process_report.deleted_wake_count, 0);
    assert!(
        core.processes()
            .session_snapshot(session_id)
            .await?
            .items
            .is_empty()
    );
    let still_waiting = wait_for_waiting_signal(&core, process_id, "ready").await;
    assert!(still_waiting.env_ref.is_some());

    let wake_after_delete = registry
        .append_event(
            process_id,
            lash_core::ProcessEventAppendRequest::new(
                "process.wake",
                serde_json::json!({ "text": "wake after deleted session" }),
            )
            .with_wake_target_scope(lash_core::SessionScope::new(session_id)),
        )
        .await?;
    assert_eq!(wake_after_delete.event.event_type, "process.wake");
    assert!(
        core.processes()
            .events(process_id, 0)
            .await?
            .iter()
            .any(|event| event.payload["text"] == "wake after deleted session")
    );
    assert!(
        core.processes()
            .session_snapshot(session_id)
            .await?
            .items
            .is_empty()
    );

    core.processes()
        .signal(
            process_id,
            "ready",
            "outliving-host-signal",
            signal_request(
                process_id,
                "ready",
                "outliving-host-signal",
                serde_json::json!({ "after_delete": true }),
            ),
            runtime_operation_scope("outliving-process-signal"),
        )
        .await?;
    let output = core.processes().await_output(process_id).await?;
    let lash_core::ProcessAwaitOutput::Success { value, .. } = output else {
        panic!("outliving process did not succeed: {output:#?}");
    };
    assert_eq!(
        value,
        serde_json::json!({ "resumed": { "after_delete": true } })
    );
    wait_for_terminal(
        &core,
        process_id,
        lash_core::ProcessLifecycleStatus::Completed,
    )
    .await;
    Ok(())
}
