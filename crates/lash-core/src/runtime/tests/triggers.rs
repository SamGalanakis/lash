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

    fn register(&self, reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        reg.host_events().declare(
            crate::HostEvent::new("Button", "ui.button", "pressed")
                .payload(trigger_test_payload_type()),
        )
    }
}

fn trigger_test_resources() -> lashlang::ResourceCatalog {
    let mut resources = lashlang::ResourceCatalog::new();
    resources.add_module_instance(["ui", "button"], "Button");
    resources.add_trigger_event("Button", "pressed", trigger_test_payload_type());
    resources
}

fn trigger_test_payload_type() -> lashlang::TypeExpr {
    lashlang::TypeExpr::Object(vec![
        lashlang::TypeField {
            name: "button".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        },
        lashlang::TypeField {
            name: "message".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        },
    ])
}

fn trigger_test_source() -> &'static str {
    r#"
    process remember(event: any) {
      finish event.message
    }

    trigger remembered on ui.button.pressed as event
      -> remember(event: event)
    "#
}

#[tokio::test]
async fn trigger_install_stores_activation_routes_and_emit_reuses_artifact_refs() {
    let artifact_store = Arc::new(CountingArtifactStore::default());
    let artifact_store_for_host: Arc<dyn lashlang::LashlangArtifactStore> = artifact_store.clone();
    let host = EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::in_memory().with_lashlang_artifact_store(artifact_store_for_host),
    );
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

    let install = runtime
        .install_lashlang_trigger_source(trigger_test_source())
        .await
        .expect("install trigger");
    assert_eq!(install.installed, vec!["remembered"]);
    assert_eq!(artifact_store.put_count(), 1);

    let routes = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .installed_lashlang_trigger_routes()
        .expect("routes");
    assert_eq!(routes.len(), 1);
    let route = &routes[0];
    assert_eq!(route.name, "remembered");
    assert_eq!(route.process_name, "remember");
    assert!(route.module_ref.as_str().starts_with("lashlang:v1:sha256:"));
    assert!(
        route
            .required_surface_ref
            .as_str()
            .starts_with("lashlang-surface:v1:sha256:")
    );
    assert!(route.source_sha256.is_some());

    let snapshot = runtime
        .session
        .as_ref()
        .expect("session")
        .plugins()
        .snapshot()
        .expect("plugin snapshot");
    let snapshot_json = serde_json::to_string(&snapshot).expect("snapshot json");
    assert!(snapshot_json.contains("module_ref"));
    assert!(!snapshot_json.contains("process remember"));
    assert!(!snapshot_json.contains("finish event.message"));

    let report = runtime
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            json!({ "button": "Blue", "message": "user pressed blue" }),
        )
        .await
        .expect("emit host event");
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
            RuntimeCoreConfig::in_memory().with_lashlang_artifact_store(
                artifact_store.clone() as Arc<dyn lashlang::LashlangArtifactStore>
            ),
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
        panic!("triggered process should be caused by the host event session node");
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
            assert_eq!(body.as_ref()["event"], "pressed");
            assert_eq!(body.as_ref()["payload"]["message"], "user pressed blue");
        }
        _ => panic!("host event should append a plugin session node"),
    }
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
