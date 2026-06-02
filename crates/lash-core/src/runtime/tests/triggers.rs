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

impl lashlang::LashlangArtifactStore for CountingArtifactStore {
    fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        self.puts.fetch_add(1, AtomicOrdering::SeqCst);
        self.inner.put_module_artifact(artifact)
    }

    fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<lashlang::ModuleArtifact>, lashlang::ArtifactStoreError> {
        self.gets.fetch_add(1, AtomicOrdering::SeqCst);
        self.inner.get_module_artifact(module_ref)
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

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
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

fn trigger_test_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_trigger_source_constructor(
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
        lashlang::TypeExpr::Ref("test.Tick".into()),
    );
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

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        let mut resources = lashlang::ResourceCatalog::new();
        resources.add_trigger_source_constructor(
            ["clock", "Alarm"],
            lashlang::TypeExpr::Object(vec![lashlang::TypeField {
                name: "at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            }]),
            lashlang::TypeExpr::Ref("clock.Tick".into()),
        );
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

    fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
        let mut resources = lashlang::ResourceCatalog::new();
        resources.add_trigger_source_constructor(
            ["ui", "button", "pressed"],
            lashlang::TypeExpr::Object(vec![]),
            button_pressed_event_type(),
        );
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
        reg.host_events().declare(
            crate::HostEvent::new("Button", "ui.button", "pressed")
                .payload(button_pressed_event_type()),
        )
    }
}

fn button_pressed_event_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![
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
    ])
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
      name: "daily_digest"
    })?
    submit {
      handle: handle,
      registrations: await triggers.list({ target: daily_digest })?
    }
    "#
}

fn button_trigger_test_source() -> &'static str {
    r#"
    type ButtonPressed = { button: "Red" | "Blue", message: str, pressed_at: str }

    process on_button(event: ButtonPressed) {
      wake { kind: "button_pressed", button: event.button, message: event.message }
      finish true
    }

    handle = await triggers.register({
      source: ui.button.pressed({}),
      target: on_button,
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
      name: "first_alarm"
    })?

    second_source = clock.Alarm({ at: "09:00" })
    second_handle = await triggers.register({
      source: second_source,
      target: second,
      name: "second_alarm"
    })?

    submit {
      first: first_handle,
      second: second_handle,
      first_registrations: await triggers.list({ target: first })?,
      second_registrations: await triggers.list({ target: second })?
    }
    "#
}

struct TriggerRegistrationHost {
    plugins: Arc<crate::PluginSession>,
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
                    "triggers.register" => self
                        .plugins
                        .register_lashlang_trigger(payload, Arc::clone(&self.artifact_store))
                        .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?,
                    "triggers.list" => self
                        .plugins
                        .list_lashlang_triggers(payload)
                        .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?,
                    "triggers.cancel" => self
                        .plugins
                        .cancel_lashlang_trigger(payload)
                        .map_err(|err| lashlang::ExecutionHostError::new(err.to_string()))?,
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
    let surface = lashlang::LashlangSurface::new(
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
        .expect("store module artifact");
    let compiled = lashlang::compile_linked(&linked);
    let host = TriggerRegistrationHost {
        plugins: Arc::clone(session.plugins()),
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

#[tokio::test]
async fn host_event_emission_activates_matching_button_trigger_routes() {
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

    let report = runtime
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            json!({
                "button": "Red",
                "message": "user pressed the red button",
                "pressed_at": "2026-06-02T12:00:00Z"
            }),
        )
        .await
        .expect("emit button host event");

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
    let Some(crate::CausalRef::SessionNode {
        session_id,
        node_id,
    }) = record.provenance.caused_by
    else {
        panic!("host-event-triggered process should be caused by the host event session node");
    };
    assert_eq!(session_id, runtime.session_id());
    let node = runtime
        .state
        .session_graph
        .find_node(&node_id)
        .expect("host event node");
    match &node.payload {
        crate::SessionNodePayload::Plugin { plugin_type, body } => {
            assert_eq!(plugin_type, "lash.host_event");
            assert_eq!(body.as_ref()["source_type"], json!("ui.button.pressed"));
            assert_eq!(body.as_ref()["payload"]["button"], json!("Red"));
        }
        _ => panic!("host event should append a plugin session node"),
    }
}

#[tokio::test]
async fn host_event_trigger_activation_handles_no_routes_disabled_routes_and_invalid_payloads() {
    let mut runtime = runtime_with_plugins_and_tools(
        vec![Arc::new(ButtonTriggerFactory)],
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
    )
    .await;

    let no_routes = runtime
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            json!({
                "button": "Blue",
                "message": "user pressed the blue button",
                "pressed_at": "2026-06-02T12:01:00Z"
            }),
        )
        .await
        .expect("emit button host event with no registrations");
    assert!(no_routes.started_process_ids.is_empty());
    assert_eq!(plugin_node_count(&runtime, "lash.host_event"), 1);

    let invalid = runtime
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            json!({
                "button": "Green",
                "message": "invalid button payload",
                "pressed_at": "2026-06-02T12:02:00Z"
            }),
        )
        .await
        .expect_err("invalid payload should fail before append");
    assert!(
        invalid
            .to_string()
            .contains("invalid payload for host event")
    );
    assert_eq!(plugin_node_count(&runtime, "lash.host_event"), 1);

    let finish = execute_trigger_registration(&mut runtime, button_trigger_test_source()).await;
    let finish_json = crate::lashlang_bridge::lashlang_value_to_json(&finish).expect("finish json");
    let handle = finish_json["id"].as_str().expect("trigger handle");
    let cancelled = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .cancel_lashlang_trigger(json!({ "handle": handle }))
        .expect("cancel trigger");
    assert_eq!(cancelled, json!(true));

    let disabled = runtime
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            json!({
                "button": "Blue",
                "message": "user pressed the blue button",
                "pressed_at": "2026-06-02T12:03:00Z"
            }),
        )
        .await
        .expect("disabled route should be skipped");
    assert!(disabled.started_process_ids.is_empty());
    assert_eq!(plugin_node_count(&runtime, "lash.host_event"), 2);
}

#[tokio::test]
async fn registering_schedule_source_persists_route_and_activation_reuses_artifact_refs() {
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
    assert_eq!(finish_json["registrations"][0]["enabled"], json!(true));
    assert_eq!(
        finish_json["registrations"][0]["target"]["input_name"],
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
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .list_all_lashlang_triggers()
        .expect("registrations");
    assert_eq!(registrations.len(), 1);
    let registration = serde_json::to_value(&registrations[0]).expect("registration json");
    assert_eq!(registration["name"], json!("daily_digest"));
    assert_eq!(
        registration["target"]["process_name"],
        json!("daily_digest")
    );
    assert_eq!(registration["target"]["input_name"], json!("event"));
    assert_eq!(registration["source_type"], json!("test.Schedule"));
    assert_eq!(registration["source"]["value"]["expr"], json!("0 8 * * *"));
    assert!(registration.get("event_type").is_none());
    assert!(registration["target"].get("module_ref").is_none());
    assert!(registration["target"].get("required_surface_ref").is_none());

    let snapshot = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .snapshot()
        .expect("plugin snapshot");
    let snapshot_json = serde_json::to_string(&snapshot).expect("snapshot json");
    assert!(snapshot_json.contains("module_ref"));
    assert!(!snapshot_json.contains("process daily_digest"));
    assert!(!snapshot_json.contains("daily_digest_due"));

    let handle = finish_json["handle"]["id"].as_str().expect("handle id");
    let report = runtime
        .activate_lashlang_trigger(
            handle,
            json!({
                "id": "daily-2026-06-01",
                "scheduled_at": "2026-06-01T08:00:00Z"
            }),
        )
        .await
        .expect("activate schedule trigger");
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
        .with_session_policy(crate::runtime::tests::helpers::standard_test_policy()),
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
    let Some(crate::CausalRef::SessionNode {
        session_id,
        node_id,
    }) = record.provenance.caused_by
    else {
        panic!("triggered process should be caused by the trigger activation session node");
    };
    assert_eq!(session_id, runtime.session_id());
    let node = runtime
        .state
        .session_graph
        .find_node(&node_id)
        .expect("trigger activation node");
    match &node.payload {
        crate::SessionNodePayload::Plugin { plugin_type, body } => {
            assert_eq!(plugin_type, "lash.trigger_activation");
            assert_eq!(body.as_ref()["handle"], handle);
            assert_eq!(body.as_ref()["payload"]["id"], "daily-2026-06-01");
        }
        _ => panic!("trigger activation should append a plugin session node"),
    }
}

#[tokio::test]
async fn source_owner_lists_source_routes_and_activates_exact_handles_without_crossfire() {
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
        finish_json["first_registrations"][0]["target"]["process_name"],
        json!("first")
    );
    assert_eq!(
        finish_json["second_registrations"][0]["target"]["process_name"],
        json!("second")
    );

    let first_handle = finish_json["first"]["id"].as_str().expect("first handle");
    let second_handle = finish_json["second"]["id"].as_str().expect("second handle");
    let source_registrations = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .lashlang_trigger_registrations_by_source_type(crate::TriggerSourceType::from(
            "clock.Alarm",
        ))
        .expect("source registrations");
    assert_eq!(
        source_registrations
            .iter()
            .map(|registration| registration.handle.as_str())
            .collect::<Vec<_>>(),
        vec![first_handle, second_handle]
    );

    let report = runtime
        .activate_lashlang_trigger(
            second_handle,
            json!({
                "id": "alarm-09",
                "scheduled_at": "2026-06-01T09:00:00Z"
            }),
        )
        .await
        .expect("activate second handle");
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

    let cancelled = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .cancel_lashlang_trigger(json!({ "handle": { "id": second_handle } }))
        .expect("cancel second handle");
    assert_eq!(cancelled, json!(true));
    let disabled = runtime
        .activate_lashlang_trigger(
            second_handle,
            json!({
                "id": "alarm-disabled",
                "scheduled_at": "2026-06-01T09:05:00Z"
            }),
        )
        .await
        .expect("disabled activation is not an error");
    assert!(disabled.started_process_ids.is_empty());
}

#[tokio::test]
async fn cancel_disables_future_activations_without_canceling_started_runs() {
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

    let report = runtime
        .activate_lashlang_trigger(
            &handle_id,
            json!({
                "id": "daily-2026-06-01",
                "scheduled_at": "2026-06-01T08:00:00Z"
            }),
        )
        .await
        .expect("activate schedule trigger");
    assert_eq!(report.started_process_ids.len(), 1);

    let cancelled = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .cancel_lashlang_trigger(json!({ "handle": handle }))
        .expect("cancel trigger");
    assert_eq!(cancelled, json!(true));
    let registrations = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .list_all_lashlang_triggers()
        .expect("list triggers");
    assert!(!registrations[0].enabled);

    let after_cancel = runtime
        .activate_lashlang_trigger(
            &handle_id,
            json!({
                "id": "daily-2026-06-02",
                "scheduled_at": "2026-06-02T08:00:00Z"
            }),
        )
        .await
        .expect("activate schedule trigger after cancel");
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
        .with_session_policy(crate::runtime::tests::helpers::standard_test_policy()),
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
            node_id: "plugin-host-event".to_string(),
        },
    );

    assert_eq!(
        context.metadata["caused_by"],
        json!({
            "type": "session_node",
            "session_id": "session",
            "node_id": "plugin-host-event"
        })
    );
    assert_eq!(context.metadata.len(), 1);
}
