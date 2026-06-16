use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[derive(Default)]
struct CountingArtifactStore {
    inner: lashlang::InMemoryLashlangArtifactStore,
    puts: AtomicUsize,
    gets: AtomicUsize,
}

impl CountingArtifactStore {
    fn put_count(&self) -> usize {
        self.puts.load(AtomicOrdering::SeqCst)
    }

    fn get_count(&self) -> usize {
        self.gets.load(AtomicOrdering::SeqCst)
    }
}

#[async_trait::async_trait]
impl lashlang::LashlangArtifactStore for CountingArtifactStore {
    async fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        self.puts.fetch_add(1, AtomicOrdering::SeqCst);
        self.inner.put_module_artifact(artifact).await
    }

    async fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<Arc<lashlang::ModuleArtifact>>, lashlang::ArtifactStoreError> {
        self.gets.fetch_add(1, AtomicOrdering::SeqCst);
        self.inner.get_module_artifact(module_ref).await
    }

    async fn put_artifact_bytes(
        &self,
        artifact_ref: &str,
        descriptor: &str,
        bytes: &[u8],
    ) -> Result<(), lashlang::ArtifactStoreError> {
        self.inner
            .put_artifact_bytes(artifact_ref, descriptor, bytes)
            .await
    }

    async fn get_artifact_bytes(
        &self,
        artifact_ref: &str,
    ) -> Result<Option<Vec<u8>>, lashlang::ArtifactStoreError> {
        self.inner.get_artifact_bytes(artifact_ref).await
    }
}

struct TriggerRouteTestFactory;

impl crate::PluginFactory for TriggerRouteTestFactory {
    fn id(&self) -> &'static str {
        "trigger-route-test"
    }

    fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_sleep()
            .with_process_signals()
            .with_triggers()
    }

    fn lashlang_resources(&self) -> lashlang::LashlangHostCatalog {
        trigger_test_resources()
    }

    fn build(
        &self,
        _ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(TriggerRouteTestPlugin))
    }
}

struct TriggerRouteTestPlugin;

impl crate::SessionPlugin for TriggerRouteTestPlugin {
    fn id(&self) -> &'static str {
        "trigger-route-test"
    }

    fn register(&self, _reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        Ok(())
    }
}

fn trigger_test_resources() -> lashlang::LashlangHostCatalog {
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources
        .add_trigger_source_constructor(
            ["test", "Schedule"],
            lashlang::TypeExpr::Object(vec![
                lashlang::TypeField {
                    name: "expr".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                },
                lashlang::TypeField {
                    name: "tz".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: true,
                },
            ]),
            lashlang::NamedDataType::object(
                "test.Tick",
                vec![
                    lashlang::TypeField {
                        name: "id".into(),
                        ty: lashlang::TypeExpr::Str,
                        optional: false,
                    },
                    lashlang::TypeField {
                        name: "scheduled_at".into(),
                        ty: lashlang::TypeExpr::Str,
                        optional: false,
                    },
                ],
            )
            .expect("valid test tick type"),
        )
        .expect("valid test trigger source");
    resources
}

struct ClockTriggerFactory;

impl crate::PluginFactory for ClockTriggerFactory {
    fn id(&self) -> &'static str {
        "clock-trigger-test"
    }

    fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_triggers()
    }

    fn lashlang_resources(&self) -> lashlang::LashlangHostCatalog {
        let mut resources = lashlang::LashlangHostCatalog::new();
        resources
            .add_trigger_source_constructor(
                ["clock", "Alarm"],
                lashlang::TypeExpr::Object(vec![lashlang::TypeField {
                    name: "at".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                }]),
                lashlang::NamedDataType::object(
                    "clock.Tick",
                    vec![
                        lashlang::TypeField {
                            name: "id".into(),
                            ty: lashlang::TypeExpr::Str,
                            optional: false,
                        },
                        lashlang::TypeField {
                            name: "scheduled_at".into(),
                            ty: lashlang::TypeExpr::Str,
                            optional: false,
                        },
                    ],
                )
                .expect("valid clock tick type"),
            )
            .expect("valid clock trigger source");
        resources
    }

    fn build(
        &self,
        _ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(ClockTriggerPlugin))
    }
}

struct ClockTriggerPlugin;

impl crate::SessionPlugin for ClockTriggerPlugin {
    fn id(&self) -> &'static str {
        "clock-trigger-test"
    }

    fn register(&self, _reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        Ok(())
    }
}

struct ButtonTriggerFactory;

impl crate::PluginFactory for ButtonTriggerFactory {
    fn id(&self) -> &'static str {
        "button-trigger-test"
    }

    fn lashlang_abilities(&self) -> lashlang::LashlangAbilities {
        lashlang::LashlangAbilities::default()
            .with_processes()
            .with_process_signals()
            .with_triggers()
    }

    fn lashlang_resources(&self) -> lashlang::LashlangHostCatalog {
        let mut resources = lashlang::LashlangHostCatalog::new();
        resources
            .add_trigger_source_constructor(
                ["ui", "button", "pressed"],
                lashlang::TypeExpr::Object(vec![]),
                button_pressed_event_type(),
            )
            .expect("valid button trigger source");
        resources
    }

    fn build(
        &self,
        _ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(ButtonTriggerPlugin))
    }
}

struct ButtonTriggerPlugin;

impl crate::SessionPlugin for ButtonTriggerPlugin {
    fn id(&self) -> &'static str {
        "button-trigger-test"
    }

    fn register(&self, reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        reg.triggers().declare(crate::TriggerEvent::new(
            "Button",
            "ui.button",
            "pressed",
            button_pressed_event_type(),
        ))
    }
}

fn button_pressed_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Union(vec![
                    lashlang::TypeExpr::Enum(vec!["Red".into()]),
                    lashlang::TypeExpr::Enum(vec!["Blue".into()]),
                ]),
                optional: false,
            },
            lashlang::TypeField {
                name: "message".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "pressed_at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ],
    )
    .expect("valid button payload type")
}

fn schedule_trigger_test_source() -> &'static str {
    r#"
    process daily_digest(event: test.Tick) {
      wake { kind: "daily_digest_due", event: event }
      finish true
    }

    source = test.Schedule({
      expr: "0 8 * * *",
      tz: "UTC"
    })
    handle = await triggers.register({
      source: source,
      target: daily_digest,
      inputs: { event: trigger.event },
      name: "daily_digest"
    })?
    submit {
      handle: handle,
      registrations: await triggers.list({})?,
      target_registrations: await triggers.list({ target: daily_digest })?,
      named_registrations: await triggers.list({ name: "daily_digest" })?,
      source_registrations: await triggers.list({ source_type: "test.Schedule", enabled: true })?
    }
    "#
}

fn multi_input_trigger_test_source() -> &'static str {
    r#"
    process fanout(first: test.Tick, second: test.Tick, label: str) {
      finish { first: first.id, second: second.id, label: label }
    }

    source = test.Schedule({
      expr: "0 8 * * *",
      tz: "UTC"
    })
    handle = await triggers.register({
      source: source,
      target: fanout,
      inputs: {
        first: trigger.event,
        second: trigger.event,
        label: "fixed-label"
      },
      name: "fanout"
    })?
    submit handle
    "#
}

fn button_trigger_test_source() -> &'static str {
    r#"
    process on_button(event: ui.button.Pressed) {
      wake { kind: "button_pressed", button: event.button, message: event.message }
      finish true
    }

    handle = await triggers.register({
      source: ui.button.pressed({}),
      target: on_button,
      inputs: { event: trigger.event },
      name: "button watcher"
    })?
    submit handle
    "#
}

fn clock_trigger_test_source() -> &'static str {
    r#"
    process first(event: clock.Tick) {
      finish { name: "first", event: event }
    }

    process second(event: clock.Tick) {
      finish { name: "second", event: event }
    }

    first_source = clock.Alarm({ at: "08:00" })
    first_handle = await triggers.register({
      source: first_source,
      target: first,
      inputs: { event: trigger.event },
      name: "first_alarm"
    })?

    second_source = clock.Alarm({ at: "09:00" })
    second_handle = await triggers.register({
      source: second_source,
      target: second,
      inputs: { event: trigger.event },
      name: "second_alarm"
    })?

    submit {
      first: first_handle,
      second: second_handle,
      all_registrations: await triggers.list({})?,
      first_registrations: await triggers.list({ target: first })?,
      second_registrations: await triggers.list({ target: second })?,
      named_second_registrations: await triggers.list({ name: "second_alarm" })?
    }
    "#
}

struct TriggerRegistrationHost {
    session_id: String,
    store: Arc<dyn crate::TriggerStore>,
    artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
}

impl lashlang::ExecutionHost for TriggerRegistrationHost {
    async fn perform(
        &self,
        op: lashlang::AbilityOp,
    ) -> Result<lashlang::AbilityResult, lashlang::ExecutionHostError> {
        match op {
            lashlang::AbilityOp::ResourceOperation(operation) => {
                let payload = resource_operation_payload(operation.args)?;
                let value = match operation.operation.as_str() {
                    "triggers.register" | "register" => self.register_trigger(payload).await?,
                    "triggers.list" | "list" => self.list_triggers(payload).await?,
                    "triggers.cancel" | "cancel" => self.cancel_trigger(payload).await?,
                    other => {
                        return Err(lashlang::ExecutionHostError::new(format!(
                            "unsupported operation `{other}`"
                        )));
                    }
                };
                Ok(lashlang::AbilityResult::Value(lashlang::from_json(value)))
            }
            lashlang::AbilityOp::Submit(value)
            | lashlang::AbilityOp::Finish(value)
            | lashlang::AbilityOp::Fail(value) => Ok(lashlang::AbilityResult::Value(value)),
            _ => Err(lashlang::ExecutionHostError::new(
                "unsupported host ability in trigger registration test",
            )),
        }
    }
}

impl TriggerRegistrationHost {
    async fn register_trigger(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, lashlang::ExecutionHostError> {
        let request = lashlang::TriggerRegistrationRequest::decode(&payload)
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let source_type = request.source.source_type.clone();
        let source_value = request.source.value.clone();
        let source = request.source.to_json();
        let validation = crate::plugin::validate_target_process(
            &request.target,
            &source_type,
            &request.inputs,
            self.artifact_store.as_ref(),
        )
        .await
        .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let source_key = self
            .store
            .source_key_for_subscription(&source_type, &source_value)
            .await
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let env_ref = crate::persist_process_execution_env(
            self.artifact_store.as_ref(),
            &crate::ProcessExecutionEnvSpec::new(
                crate::PluginOptions::default(),
                crate::runtime::tests::helpers::standard_test_policy(),
            ),
        )
        .await
        .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let record = self
            .store
            .register_subscription(crate::TriggerSubscriptionDraft {
                registrant: crate::ProcessOriginator::session(crate::SessionScope::new(
                    self.session_id.clone(),
                )),
                env_ref,
                wake_target: Some(crate::SessionScope::new(self.session_id.clone())),
                name: request.name,
                source_type,
                source_key,
                source,
                event_ty: validation.resolved_event_type,
                module_ref: validation.definition.module_ref,
                host_requirements_ref: validation.definition.host_requirements_ref,
                process_ref: validation.definition.process_ref,
                process_name: validation.definition.process_name,
                input_template: validation.inputs,
            })
            .await
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(crate::plugin::trigger_handle_json(&record.handle))
    }

    async fn list_triggers(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, lashlang::ExecutionHostError> {
        let request = lashlang::TriggerListRequest::decode(&payload)
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let mut filter = crate::TriggerSubscriptionFilter::for_session(&self.session_id);
        filter.target = request.target;
        filter.name = request.name;
        filter.source_type = request.source_type;
        filter.enabled = request.enabled;
        let registrations = self
            .store
            .list_subscriptions(filter)
            .await
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?
            .iter()
            .map(crate::TriggerRegistration::from)
            .collect::<Vec<_>>();
        serde_json::to_value(registrations)
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))
    }

    async fn cancel_trigger(
        &self,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, lashlang::ExecutionHostError> {
        let request = lashlang::TriggerCancelRequest::decode(&payload)
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        let changed = self
            .store
            .cancel_subscription(&self.session_id, &request.handle)
            .await
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?;
        Ok(json!(changed))
    }
}

fn resource_operation_payload(
    args: Vec<lashlang::Value>,
) -> Result<serde_json::Value, lashlang::ExecutionHostError> {
    let payload = if let [lashlang::Value::Record(record)] = args.as_slice() {
        crate::lashlang_bridge::lashlang_value_to_json(&lashlang::Value::Record(Arc::clone(record)))
            .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?
    } else {
        serde_json::json!({
            "args": args
                .iter()
                .map(crate::lashlang_bridge::lashlang_value_to_json)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?,
        })
    };
    if payload.is_object() {
        Ok(payload)
    } else {
        Err(lashlang::ExecutionHostError::new(
            "module operation payload must be an object",
        ))
    }
}

async fn execute_trigger_registration(runtime: &mut LashRuntime, source: &str) -> lashlang::Value {
    let session = runtime.session.as_ref().expect("session");
    let surface = lashlang::LashlangHostEnvironment::new(
        session.plugins().lashlang_resources(),
        session.plugins().lashlang_abilities(),
    );
    let linked = lashlang::LinkedModule::link(
        lashlang::parse(source).expect("parse trigger registration"),
        surface,
    )
    .expect("link trigger registration");
    runtime
        .host
        .core
        .durability
        .lashlang_artifact_store
        .put_module_artifact(&linked.artifact)
        .await
        .expect("store module artifact");
    let compiled = lashlang::compile_linked(&linked);
    let host = TriggerRegistrationHost {
        session_id: runtime.session_id().to_string(),
        store: runtime
            .host
            .trigger_store
            .as_ref()
            .expect("trigger store")
            .clone(),
        artifact_store: Arc::clone(&runtime.host.core.durability.lashlang_artifact_store),
    };
    let mut state = lashlang::State::new();
    match lashlang::execute(&compiled, &mut state, &host)
        .await
        .expect("execute trigger registration")
    {
        lashlang::ExecutionOutcome::Finished(value) => value,
        outcome => panic!("expected finished trigger registration, got {outcome:?}"),
    }
}

fn plugin_node_count(runtime: &LashRuntime, expected_plugin_type: &str) -> usize {
    runtime
        .state
        .session_graph
        .active_path_nodes()
        .into_iter()
        .filter(|node| {
            matches!(
                &node.payload,
                crate::SessionNodePayload::Plugin { plugin_type, .. }
                    if plugin_type == expected_plugin_type
            )
        })
        .count()
}

async fn emit_test_occurrence(
    runtime: &LashRuntime,
    source_type: &str,
    source_key: String,
    payload: serde_json::Value,
    idempotency_key: &str,
) -> crate::TriggerEmitReport {
    let router = crate::TriggerRouter::new(
        runtime
            .host
            .trigger_store
            .as_ref()
            .expect("trigger store")
            .clone(),
        Arc::clone(&runtime.host.core.durability.lashlang_artifact_store),
        runtime.host.process_registry.clone(),
        runtime.host.process_work_poke.clone(),
    );
    let scoped = runtime
        .host
        .core
        .control
        .effect_host
        .scoped(crate::ExecutionScope::runtime_operation(format!(
            "test-trigger:{idempotency_key}"
        )))
        .expect("trigger occurrence execution scope");
    router
        .emit(
            crate::TriggerOccurrenceRequest::new(source_type, source_key, payload, idempotency_key),
            scoped.controller(),
        )
        .await
        .expect("emit trigger occurrence")
}

async fn try_emit_test_occurrence(
    runtime: &LashRuntime,
    source_type: &str,
    source_key: String,
    payload: serde_json::Value,
    idempotency_key: &str,
) -> Result<crate::TriggerEmitReport, crate::PluginError> {
    let router = crate::TriggerRouter::new(
        runtime
            .host
            .trigger_store
            .as_ref()
            .expect("trigger store")
            .clone(),
        Arc::clone(&runtime.host.core.durability.lashlang_artifact_store),
        runtime.host.process_registry.clone(),
        runtime.host.process_work_poke.clone(),
    );
    let scoped = runtime
        .host
        .core
        .control
        .effect_host
        .scoped(crate::ExecutionScope::runtime_operation(format!(
            "test-trigger:{idempotency_key}"
        )))
        .expect("trigger occurrence execution scope");
    router
        .emit(
            crate::TriggerOccurrenceRequest::new(source_type, source_key, payload, idempotency_key),
            scoped.controller(),
        )
        .await
}

async fn trigger_subscriptions(runtime: &LashRuntime) -> Vec<crate::TriggerSubscriptionRecord> {
    runtime
        .host
        .trigger_store
        .as_ref()
        .expect("trigger store")
        .list_subscriptions(crate::TriggerSubscriptionFilter::for_session(
            runtime.session_id(),
        ))
        .await
        .expect("list trigger subscriptions")
}

async fn source_key_for(
    runtime: &LashRuntime,
    source_type: &str,
    source: serde_json::Value,
) -> String {
    runtime
        .host
        .trigger_store
        .as_ref()
        .expect("trigger store")
        .source_key_for_subscription(source_type, &source)
        .await
        .expect("source key")
}

async fn cancel_trigger_subscription(runtime: &LashRuntime, handle: &str) -> bool {
    runtime
        .host
        .trigger_store
        .as_ref()
        .expect("trigger store")
        .cancel_subscription(runtime.session_id(), handle)
        .await
        .expect("cancel trigger subscription")
}

#[tokio::test]
async fn trigger_occurrence_activates_matching_button_subscription_without_graph_node() {
    let mut runtime = runtime_with_plugins_and_tools(
        vec![Arc::new(ButtonTriggerFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    let finish = execute_trigger_registration(&mut runtime, button_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    assert_eq!(finish_json["id"], json!("trigger:1"));

    let source_type = "ui.button.pressed";
    let source_key = source_key_for(&runtime, source_type, json!({})).await;
    let report = emit_test_occurrence(
        &runtime,
        source_type,
        source_key,
        json!({
            "button": "Red",
            "message": "user pressed the red button",
            "pressed_at": "2026-06-02T12:00:00Z"
        }),
        "button-red-1",
    )
    .await;

    assert_eq!(report.started_process_ids.len(), 1);
    let process_id = report
        .started_process_ids
        .first()
        .expect("started process id");
    let record = registry
        .get_process(process_id)
        .await
        .expect("button-triggered process record");
    let crate::ProcessInput::LashlangProcess {
        args, process_name, ..
    } = record.input.as_ref()
    else {
        panic!("button trigger should start a lashlang process");
    };
    assert_eq!(process_name, "on_button");
    assert_eq!(
        args.get("event").and_then(|event| event.get("button")),
        Some(&json!("Red"))
    );
    assert_eq!(
        args.get("event").and_then(|event| event.get("message")),
        Some(&json!("user pressed the red button"))
    );
    let process_caused_by = record
        .provenance
        .caused_by
        .clone()
        .expect("triggered process cause");
    let crate::CausalRef::TriggerOccurrence { occurrence_id } = &process_caused_by else {
        panic!("trigger-triggered process should be caused by the trigger occurrence");
    };
    assert_eq!(occurrence_id, &report.occurrence_id);
    assert_eq!(plugin_node_count(&runtime, "lash.trigger"), 0);
    assert_eq!(plugin_node_count(&runtime, "lash.trigger_activation"), 0);

    let replayed = emit_test_occurrence(
        &runtime,
        source_type,
        source_key_for(&runtime, source_type, json!({})).await,
        json!({
            "button": "Red",
            "message": "user pressed the red button",
            "pressed_at": "2026-06-02T12:00:00Z"
        }),
        "button-red-1",
    )
    .await;
    assert_eq!(replayed.occurrence_id, report.occurrence_id);
    assert!(replayed.started_process_ids.is_empty());

    let session_store_factory =
        Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default());
    let root_store = session_store_factory
        .create_store(&crate::SessionStoreCreateRequest {
            session_id: "root".to_string(),
            relation: crate::SessionRelation::default(),
            policy: crate::runtime::tests::helpers::standard_test_policy(),
        })
        .await
        .expect("create root target session store");
    let worker = crate::DurableProcessWorker::new(
        crate::DurableProcessWorkerConfig::new(
            Arc::new(crate::PluginHost::new(vec![Arc::new(ButtonTriggerFactory)])),
            RuntimeHostConfig::in_memory(),
            session_store_factory.clone(),
            Arc::clone(&registry),
        )
        .with_session_policy(crate::runtime::tests::helpers::standard_test_policy())
        .with_trigger_store(
            runtime
                .host
                .trigger_store
                .as_ref()
                .expect("trigger store")
                .clone(),
        ),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("drive button process");
    registry
        .await_process(process_id)
        .await
        .expect("button-triggered process finishes");
    let queued = crate::store::RuntimePersistence::list_queued_work(root_store.as_ref(), "root")
        .await
        .expect("queued wake");
    assert_eq!(queued.len(), 1);
    let crate::QueuedWorkPayload::ProcessWake { wake } = &queued[0].items[0].payload else {
        panic!("expected process wake queue payload");
    };
    assert_eq!(wake.process_caused_by, Some(process_caused_by));
    assert!(matches!(
        &wake.event_invocation.subject,
        crate::RuntimeSubject::ProcessEvent {
            process_id: wake_process_id,
            sequence: 1,
            event_type,
        } if wake_process_id == process_id && event_type == "process.wake"
    ));
}

#[tokio::test]
async fn trigger_occurrence_starts_valid_deliveries_when_stale_matching_subscription_fails() {
    let mut runtime = runtime_with_plugins_and_tools(
        vec![Arc::new(ButtonTriggerFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    execute_trigger_registration(&mut runtime, button_trigger_test_source()).await;
    let subscription = trigger_subscriptions(&runtime)
        .await
        .into_iter()
        .next()
        .expect("button subscription");
    runtime
        .host
        .trigger_store
        .as_ref()
        .expect("trigger store")
        .register_subscription(crate::TriggerSubscriptionDraft {
            registrant: crate::ProcessOriginator::session(crate::SessionScope::new(
                "stale-session",
            )),
            env_ref: subscription.env_ref.clone(),
            wake_target: Some(crate::SessionScope::new("stale-session")),
            name: Some("stale button watcher".to_string()),
            source_type: subscription.source_type.clone(),
            source_key: subscription.source_key.clone(),
            source: subscription.source.clone(),
            event_ty: subscription.event_ty.clone(),
            module_ref: lashlang::ModuleRef::new(&lashlang::ContentHash::new("missing-module")),
            host_requirements_ref: subscription.host_requirements_ref.clone(),
            process_ref: subscription.process_ref.clone(),
            process_name: subscription.process_name.clone(),
            input_template: subscription.input_template.clone(),
        })
        .await
        .expect("register stale subscription");

    let report = emit_test_occurrence(
        &runtime,
        "ui.button.pressed",
        subscription.source_key,
        json!({
            "button": "Blue",
            "message": "user pressed the blue button",
            "pressed_at": "2026-06-02T12:00:00Z"
        }),
        "button-blue-with-stale-subscriber",
    )
    .await;

    assert_eq!(report.started_process_ids.len(), 1);
    let record = registry
        .get_process(&report.started_process_ids[0])
        .await
        .expect("valid delivery process");
    let crate::ProcessInput::LashlangProcess { args, .. } = record.input.as_ref() else {
        panic!("valid trigger delivery should start a lashlang process");
    };
    assert_eq!(
        args.get("event").and_then(|event| event.get("button")),
        Some(&json!("Blue"))
    );
}

#[tokio::test]
async fn trigger_occurrence_materializes_event_and_fixed_input_templates() {
    let mut runtime = runtime_with_plugins_and_tools(
        vec![Arc::new(TriggerRouteTestFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    let finish =
        execute_trigger_registration(&mut runtime, multi_input_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    assert_eq!(finish_json["id"], json!("trigger:1"));
    let registrations = trigger_subscriptions(&runtime).await;
    assert_eq!(registrations.len(), 1);
    let subscription = &registrations[0];
    let registration = serde_json::to_value(crate::TriggerRegistration::from(subscription))
        .expect("registration json");
    assert_eq!(
        registration["target"]["inputs"]["first"]["kind"],
        json!("event")
    );
    assert_eq!(
        registration["target"]["inputs"]["second"]["kind"],
        json!("event")
    );
    assert_eq!(
        registration["target"]["inputs"]["label"],
        json!({ "kind": "fixed", "value": "fixed-label" })
    );

    let payload = json!({
        "id": "daily-2026-06-01",
        "scheduled_at": "2026-06-01T08:00:00Z"
    });
    let report = emit_test_occurrence(
        &runtime,
        "test.Schedule",
        subscription.source_key.clone(),
        payload.clone(),
        "daily-2026-06-01",
    )
    .await;
    assert_eq!(report.started_process_ids.len(), 1);
    let record = registry
        .get_process(&report.started_process_ids[0])
        .await
        .expect("triggered process record");
    let crate::ProcessInput::LashlangProcess {
        args, process_name, ..
    } = record.input.as_ref()
    else {
        panic!("trigger should start a lashlang process");
    };
    assert_eq!(process_name, "fanout");
    assert_eq!(args.get("first"), Some(&payload));
    assert_eq!(args.get("second"), Some(&payload));
    assert_eq!(args.get("label"), Some(&json!("fixed-label")));
}

#[tokio::test]
async fn trigger_router_handles_no_subscriptions_disabled_subscriptions_and_invalid_payloads() {
    let mut runtime = runtime_with_plugins_and_tools(
        vec![Arc::new(ButtonTriggerFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
    )
    .await;

    let source_type = "ui.button.pressed";
    let source_key = source_key_for(&runtime, source_type, json!({})).await;
    let no_routes = emit_test_occurrence(
        &runtime,
        source_type,
        source_key.clone(),
        json!({
            "button": "Blue",
            "message": "user pressed the blue button",
            "pressed_at": "2026-06-02T12:01:00Z"
        }),
        "button-blue-no-routes",
    )
    .await;
    assert!(no_routes.started_process_ids.is_empty());
    assert_eq!(plugin_node_count(&runtime, "lash.trigger"), 0);

    let finish = execute_trigger_registration(&mut runtime, button_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    let handle = finish_json["id"].as_str().expect("trigger handle");

    let invalid = try_emit_test_occurrence(
        &runtime,
        source_type,
        source_key.clone(),
        json!({
            "button": "Green",
            "message": "invalid button payload",
            "pressed_at": "2026-06-02T12:02:00Z"
        }),
        "button-green-invalid",
    )
    .await
    .expect_err("invalid payload should fail before process start");
    assert!(invalid.to_string().contains("invalid payload for trigger"));
    assert_eq!(plugin_node_count(&runtime, "lash.trigger"), 0);

    assert!(cancel_trigger_subscription(&runtime, handle).await);

    let disabled = emit_test_occurrence(
        &runtime,
        source_type,
        source_key,
        json!({
            "button": "Blue",
            "message": "user pressed the blue button",
            "pressed_at": "2026-06-02T12:03:00Z"
        }),
        "button-blue-disabled",
    )
    .await;
    assert!(disabled.started_process_ids.is_empty());
    assert_eq!(plugin_node_count(&runtime, "lash.trigger"), 0);
}

#[tokio::test]
async fn registering_schedule_source_stores_source_key_and_occurrence_reuses_artifact_refs() {
    let artifact_store = Arc::new(CountingArtifactStore::default());
    let artifact_store_for_host: Arc<dyn lashlang::LashlangArtifactStore> = artifact_store.clone();
    let host = EmbeddedRuntimeHost::new({
        let mut config = RuntimeHostConfig::in_memory();
        config.durability.lashlang_artifact_store = artifact_store_for_host;
        config
    });
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        vec![Arc::new(TriggerRouteTestFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        host,
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    let finish = execute_trigger_registration(&mut runtime, schedule_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    assert_eq!(
        finish_json["registrations"][0]["name"],
        json!("daily_digest")
    );
    assert_eq!(
        finish_json["target_registrations"][0]["name"],
        json!("daily_digest")
    );
    assert_eq!(
        finish_json["named_registrations"][0]["target"]["process_name"],
        json!("daily_digest")
    );
    assert_eq!(
        finish_json["source_registrations"][0]["source_type"],
        json!("test.Schedule")
    );
    assert_eq!(finish_json["registrations"][0]["enabled"], json!(true));
    assert!(
        finish_json["registrations"][0]["source_key"]
            .as_str()
            .expect("source key")
            .starts_with("source:test.Schedule:sha256:")
    );
    assert_eq!(
        finish_json["registrations"][0]["target"]["inputs"]["event"]["kind"],
        json!("event")
    );
    assert!(finish_json["registrations"][0].get("event_type").is_none());
    assert!(
        finish_json["registrations"][0]["target"]
            .get("module_ref")
            .is_none()
    );
    assert_eq!(artifact_store.put_count(), 1);

    let registrations = runtime
        .list_lashlang_trigger_registrations()
        .await
        .expect("registrations");
    assert_eq!(registrations.len(), 1);
    let registration = serde_json::to_value(&registrations[0]).expect("registration json");
    assert_eq!(registration["name"], json!("daily_digest"));
    assert_eq!(
        registration["target"]["process_name"],
        json!("daily_digest")
    );
    assert_eq!(
        registration["target"]["inputs"]["event"]["kind"],
        json!("event")
    );
    assert_eq!(registration["source_type"], json!("test.Schedule"));
    assert_eq!(
        registration["source"][lashlang::LASH_HOST_DESCRIPTOR_VALUE_KEY]["expr"],
        json!("0 8 * * *")
    );
    assert!(registration.get("event_type").is_none());
    assert!(registration["target"].get("module_ref").is_none());
    assert!(
        registration["target"]
            .get("host_requirements_ref")
            .is_none()
    );

    let snapshot = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .snapshot()
        .expect("plugin snapshot");
    let snapshot_json = serde_json::to_string(&snapshot).expect("snapshot json");
    assert!(!snapshot_json.contains("module_ref"));
    assert!(!snapshot_json.contains("process daily_digest"));
    assert!(!snapshot_json.contains("daily_digest_due"));

    let source_key = registrations[0].source_key.clone();
    let invalid = try_emit_test_occurrence(
        &runtime,
        "test.Schedule",
        source_key.clone(),
        json!({
            "id": "daily-2026-06-01"
        }),
        "daily-invalid",
    )
    .await
    .expect_err("missing scheduled_at should reject trigger payload");
    assert!(invalid.to_string().contains("invalid payload for trigger"));

    let report = emit_test_occurrence(
        &runtime,
        "test.Schedule",
        source_key,
        json!({
            "id": "daily-2026-06-01",
            "scheduled_at": "2026-06-01T08:00:00Z"
        }),
        "daily-2026-06-01",
    )
    .await;
    assert_eq!(report.started_process_ids.len(), 1);
    assert_eq!(artifact_store.put_count(), 1);

    // The inline controller now only registers the process row; the
    // lease-protected worker is the sole executor. Drive it over the same
    // registry + artifact store so the trigger-started process runs.
    let worker = crate::DurableProcessWorker::new(
        crate::DurableProcessWorkerConfig::new(
            Arc::new(crate::PluginHost::new(vec![Arc::new(
                TriggerRouteTestFactory,
            )])),
            {
                let mut config = RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store =
                    artifact_store.clone() as Arc<dyn lashlang::LashlangArtifactStore>;
                config
            },
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            Arc::clone(&registry),
        )
        .with_session_policy(crate::runtime::tests::helpers::standard_test_policy())
        .with_trigger_store(
            runtime
                .host
                .trigger_store
                .as_ref()
                .expect("trigger store")
                .clone(),
        ),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("drive pending processes");

    let process_id = &report.started_process_ids[0];
    registry
        .await_process(process_id)
        .await
        .expect("process finishes");
    assert!(artifact_store.get_count() > 0);
    let record = registry
        .get_process(process_id)
        .await
        .expect("process record");
    let Some(crate::CausalRef::TriggerOccurrence { occurrence_id }) = record.provenance.caused_by
    else {
        panic!("triggered process should be caused by the trigger occurrence");
    };
    assert_eq!(occurrence_id, report.occurrence_id);
    assert_eq!(plugin_node_count(&runtime, "lash.trigger_activation"), 0);
}

#[tokio::test]
async fn source_owner_lists_source_keys_and_emits_exact_key_without_crossfire() {
    let artifact_store = Arc::new(CountingArtifactStore::default());
    let artifact_store_for_host: Arc<dyn lashlang::LashlangArtifactStore> = artifact_store.clone();
    let host = EmbeddedRuntimeHost::new({
        let mut config = RuntimeHostConfig::in_memory();
        config.durability.lashlang_artifact_store = artifact_store_for_host;
        config
    });
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        vec![Arc::new(ClockTriggerFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        host,
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    let finish = execute_trigger_registration(&mut runtime, clock_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    assert_eq!(
        finish_json["first_registrations"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        finish_json["all_registrations"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        finish_json["first_registrations"][0]["target"]["process_name"],
        json!("first")
    );
    assert_eq!(
        finish_json["second_registrations"][0]["target"]["process_name"],
        json!("second")
    );
    assert_eq!(
        finish_json["named_second_registrations"][0]["target"]["process_name"],
        json!("second")
    );

    let first_handle = finish_json["first"]["id"].as_str().expect("first handle");
    let second_handle = finish_json["second"]["id"].as_str().expect("second handle");
    let source_registrations = runtime
        .lashlang_trigger_registrations_by_source_type(crate::TriggerEventType::from("clock.Alarm"))
        .await
        .expect("source registrations");
    assert_eq!(
        source_registrations
            .iter()
            .map(|registration| registration.handle.as_str())
            .collect::<Vec<_>>(),
        vec![first_handle, second_handle]
    );
    assert_ne!(
        source_registrations[0].source_key,
        source_registrations[1].source_key
    );
    let second_source_key = source_registrations
        .iter()
        .find(|registration| registration.handle == second_handle)
        .expect("second source registration")
        .source_key
        .clone();

    let report = emit_test_occurrence(
        &runtime,
        "clock.Alarm",
        second_source_key.clone(),
        json!({
            "id": "alarm-09",
            "scheduled_at": "2026-06-01T09:00:00Z"
        }),
        "alarm-09",
    )
    .await;
    assert_eq!(report.started_process_ids.len(), 1);
    let process_id = report.started_process_ids.first().expect("process id");
    let record = registry
        .get_process(process_id)
        .await
        .expect("activated process record");
    let crate::ProcessInput::LashlangProcess {
        args, process_name, ..
    } = record.input.as_ref()
    else {
        panic!("clock trigger should start a lashlang process");
    };
    assert_eq!(process_name, "second");
    assert_eq!(
        args.get("event").and_then(|event| event.get("id")),
        Some(&json!("alarm-09"))
    );

    assert!(cancel_trigger_subscription(&runtime, second_handle).await);
    let disabled = emit_test_occurrence(
        &runtime,
        "clock.Alarm",
        second_source_key,
        json!({
            "id": "alarm-disabled",
            "scheduled_at": "2026-06-01T09:05:00Z"
        }),
        "alarm-disabled",
    )
    .await;
    assert!(disabled.started_process_ids.is_empty());
    assert!(
        !runtime
            .host
            .trigger_store
            .as_ref()
            .expect("trigger store")
            .cancel_subscription("other-session", first_handle)
            .await
            .expect("cross-session cancel")
    );
    assert!(cancel_trigger_subscription(&runtime, first_handle).await);
}

#[tokio::test]
async fn cancel_disables_future_occurrences_without_canceling_started_runs() {
    let artifact_store = Arc::new(CountingArtifactStore::default());
    let artifact_store_for_host: Arc<dyn lashlang::LashlangArtifactStore> = artifact_store.clone();
    let host = EmbeddedRuntimeHost::new({
        let mut config = RuntimeHostConfig::in_memory();
        config.durability.lashlang_artifact_store = artifact_store_for_host;
        config
    });
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        vec![Arc::new(TriggerRouteTestFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        host,
    )
    .await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();

    let finish = execute_trigger_registration(&mut runtime, schedule_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    let handle = finish_json["handle"].clone();
    let handle_id = handle["id"].as_str().expect("handle id").to_string();
    let source_key = trigger_subscriptions(&runtime)
        .await
        .first()
        .expect("schedule subscription")
        .source_key
        .clone();

    let report = emit_test_occurrence(
        &runtime,
        "test.Schedule",
        source_key.clone(),
        json!({
            "id": "daily-2026-06-01",
            "scheduled_at": "2026-06-01T08:00:00Z"
        }),
        "daily-before-cancel",
    )
    .await;
    assert_eq!(report.started_process_ids.len(), 1);

    assert!(cancel_trigger_subscription(&runtime, &handle_id).await);
    let registrations = trigger_subscriptions(&runtime).await;
    assert!(!registrations[0].enabled);

    let after_cancel = emit_test_occurrence(
        &runtime,
        "test.Schedule",
        source_key,
        json!({
            "id": "daily-2026-06-02",
            "scheduled_at": "2026-06-02T08:00:00Z"
        }),
        "daily-after-cancel",
    )
    .await;
    assert!(after_cancel.started_process_ids.is_empty());

    let process_id = &report.started_process_ids[0];
    let record = registry
        .get_process(process_id)
        .await
        .expect("schedule-triggered process record");
    let crate::ProcessInput::LashlangProcess {
        args, process_name, ..
    } = record.input.as_ref()
    else {
        panic!("schedule trigger should start a lashlang process");
    };
    assert_eq!(process_name, "daily_digest");
    assert_eq!(
        args.get("event").and_then(|event| event.get("id")),
        Some(&json!("daily-2026-06-01"))
    );

    let worker = crate::DurableProcessWorker::new(
        crate::DurableProcessWorkerConfig::new(
            Arc::new(crate::PluginHost::new(vec![Arc::new(
                TriggerRouteTestFactory,
            )])),
            {
                let mut config = RuntimeHostConfig::in_memory();
                config.durability.lashlang_artifact_store =
                    artifact_store.clone() as Arc<dyn lashlang::LashlangArtifactStore>;
                config
            },
            Arc::new(crate::runtime::tests::helpers::RecordingSessionStoreFactory::default()),
            Arc::clone(&registry),
        )
        .with_session_policy(crate::runtime::tests::helpers::standard_test_policy())
        .with_trigger_store(
            runtime
                .host
                .trigger_store
                .as_ref()
                .expect("trigger store")
                .clone(),
        ),
    );
    worker
        .drive_pending_processes()
        .await
        .expect("drive pending processes");
    registry
        .await_process(process_id)
        .await
        .expect("schedule-triggered process finishes");
}

#[test]
fn session_node_causal_refs_are_exported_as_generic_trace_causality() {
    let context = crate::trace::trace_context_with_causal_ref(
        lash_trace::TraceContext::default(),
        &crate::CausalRef::SessionNode {
            session_id: "session".to_string(),
            node_id: "plugin-trigger".to_string(),
        },
    );

    assert_eq!(
        context.metadata["caused_by"],
        json!({
            "type": "session_node",
            "session_id": "session",
            "node_id": "plugin-trigger"
        })
    );
    assert_eq!(context.metadata.len(), 1);
}
