fn main() -> AnyhowResult<()> {
    let stack_bytes = std::env::var("AGENT_WORKBENCH_TOKIO_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOKIO_THREAD_STACK_BYTES);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(stack_bytes)
        .build()
        .context("build agent-workbench tokio runtime")?
        .block_on(async_main())
}

async fn async_main() -> AnyhowResult<()> {
    let _ = dotenvy::dotenv();

    let addr: SocketAddr = std::env::var("AGENT_WORKBENCH_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3030".to_string())
        .parse()
        .context("invalid AGENT_WORKBENCH_ADDR")?;
    let restate_endpoint_addr: SocketAddr = std::env::var("AGENT_WORKBENCH_RESTATE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9081".to_string())
        .parse()
        .context("invalid AGENT_WORKBENCH_RESTATE_ADDR")?;
    let restate_ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let restate_admin_url =
        std::env::var("RESTATE_ADMIN_URL").unwrap_or_else(|_| "http://127.0.0.1:19070".to_string());
    let data_dir = std::env::var("AGENT_WORKBENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".agent-workbench"));
    std::fs::create_dir_all(&data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    let trace_path = std::env::var("AGENT_WORKBENCH_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("trace.jsonl"));
    eprintln!("agent-workbench trace: {}", trace_path.display());
    let trace_path_display = trace_path.display().to_string();
    let trace_sink = Arc::new(TeeTraceSink::new([
        Arc::new(StderrTraceSink::default()) as Arc<dyn TraceSink>,
        Arc::new(JsonlTraceSink::new(trace_path)),
    ])) as Arc<dyn TraceSink>;
    let lashlang_execution_path = std::env::var("AGENT_WORKBENCH_LASHLANG_EXECUTION_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("lashlang-execution.jsonl"));
    eprintln!(
        "agent-workbench Lashlang execution trace: {}",
        lashlang_execution_path.display()
    );
    let lashlang_execution = Arc::new(TraceLashlangGraphStore::default());
    let lashlang_execution_sink = Arc::new(TeeTraceSink::new([
        Arc::clone(&lashlang_execution) as Arc<dyn TraceSink>,
        Arc::new(JsonlTraceSink::new(lashlang_execution_path.clone())) as Arc<dyn TraceSink>,
    ])) as Arc<dyn TraceSink>;

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    if api_key.trim().is_empty() {
        eprintln!("warning: OPENROUTER_API_KEY is empty; turns will fail until it is set");
    }
    let tavily_api_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    if tavily_api_key.trim().is_empty() {
        eprintln!("warning: TAVILY_API_KEY is empty; web tools will return configuration errors");
    }
    let model = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.6".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "high".to_string());

    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL)
            .with_options(ProviderOptions {
                expose_thinking: true,
                ..ProviderOptions::default()
            })
            .into_components(),
    );
    let model_spec = lash::ModelSpec::from_token_limits(
        model.clone(),
        lash::provider::ReasoningSelection::Effort(model_variant.clone()),
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
    )
    .map_err(|err| anyhow!("invalid OPENROUTER_MODEL metadata: {err}"))?;
    let model_spec = with_workbench_model_capability(model_spec);
    let session_store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        data_dir.join("lash-sessions"),
    ));
    let core_store_factory: Arc<dyn lash::persistence::SessionStoreFactory> =
        session_store_factory.clone();
    let process_registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
            .await
            .context("open process registry")?,
    ) as Arc<dyn lash::process::ProcessRegistry>;
    let trigger_store = Arc::new(
        lash_sqlite_store::SqliteTriggerStore::open(&data_dir.join("triggers.db"))
            .await
            .context("open trigger store")?,
    );
    // Deployment-level Lashlang artifact/process-env store (compiled
    // trigger/process modules plus durable process environment specs), shared
    // across the session tree. SQLite keeps installed triggers and process
    // rebuild metadata durable across restarts.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .await
            .context("open lashlang artifact store")?,
    );
    let subagent_registry = Arc::new(lash_subagents::default_registry(&BTreeMap::new()));
    let mail_world = mail::MailWorld::new();
    let session_ids = WorkbenchSessionIds::fresh();
    let (event_tx, _) = broadcast::channel(1024);
    let restate_http = reqwest::Client::new();
    let active_restate_invocations = ActiveRestateInvocations::default();
    // Best-effort freshness feed for appended process events (ADR 0017). The
    // sink is a freshness overlay on the durable event log, never truth: no
    // delivery guarantee, terminal events never arrive here (they ride
    // `await_terminal`), and a consumer needing completeness reconciles from
    // `events_after`. `emit` must be fast, so it only hands each event to this
    // channel; the consumer task does the projection off the append path.
    let (process_event_tx, mut process_event_rx) =
        mpsc::channel::<lash::process::ProcessEvent>(256);
    tokio::spawn(async move {
        while let Some(event) = process_event_rx.recv().await {
            eprintln!(
                "agent-workbench process event: process={} seq={} type={}",
                event.process_id, event.sequence, event.event_type
            );
        }
    });
    let process_event_sink = Arc::new(ChannelProcessEventSink::new(process_event_tx))
        as Arc<dyn lash::process::ProcessEventSink>;
    let process_deployment = lash_restate::RestateProcessDeployment::new_with_sink(
        restate_ingress_url.clone(),
        process_registry,
        Some(process_event_sink),
    );
    // Retained so a host-facing "wait for the work item" flow can route through
    // `ProcessWorkDriver::await_terminal` (see the `/api/work/{id}/await` route).
    let process_work_driver = process_deployment.process_work_driver();
    let queued_work_driver =
        lash::runtime::QueuedWorkDriver::new(Arc::new(WorkbenchQueuedWorkSubmitter {
            session_ids: session_ids.clone(),
            store_factory: Arc::clone(&core_store_factory),
            restate_ingress_url: restate_ingress_url.clone(),
            restate_http: restate_http.clone(),
            active_restate_invocations: active_restate_invocations.clone(),
        }));

    let runtime_host_config = lash::durability::RuntimeHostConfig::new(
        Arc::new(lash_restate::RestateEffectHost::with_ingress_url(
            restate_ingress_url.clone(),
        )),
        Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )),
        artifact_store.clone(),
    );

    let factory = lash_protocol_rlm::RlmProtocolPluginFactory::new(
        lash::rlm::RlmProtocolPluginConfig::default()
            .with_lashlang_abilities(workbench_lashlang_abilities()),
        Arc::clone(&artifact_store) as Arc<dyn lash::persistence::LashlangArtifactStore>,
    )
    .with_lashlang_execution_sink(Arc::clone(&lashlang_execution_sink));
    let core = LashCore::rlm_builder(factory)
        .provider(provider)
        .model(model_spec)
        .store_factory(Arc::clone(&core_store_factory))
        .trigger_store(trigger_store)
        .trace_sink(Arc::clone(&trace_sink))
        .trace_level(TraceLevel::Extended)
        .configure_plugins(|plugins| {
            plugins.push(Arc::new(
                WorkbenchPluginFactory::new(tavily_api_key.clone())
                    .with_mail_world(mail_world.clone()),
            ));
            plugins.push(Arc::new(
                lash_plugin_process_controls::SessionProcessAdminPluginFactory::new(),
            ));
            plugins.push(Arc::new(
                lash_subagents::SubagentsPluginFactory::new(subagent_registry)
                    .with_session_spec(SessionSpec::inherit()),
            ));
        })
        .process_work_driver(process_work_driver.clone())
        .queued_work_driver(queued_work_driver.clone())
        .advanced()
        .runtime_host_config(runtime_host_config)
        .build()
        .context("build Lash core")?;
    let process_worker = lash::durability::DurableProcessWorker::new(
        core.durable_process_worker_config()
            .context("build Restate process worker config")?,
    );
    let process_observer = core
        .processes()
        .observer()
        .expect("process observer configured");

    let state = AppState {
        core,
        process_observer,
        process_work_driver,
        session_ids,
        messages: Arc::new(Mutex::new(Vec::new())),
        selected_model: Arc::new(Mutex::new(ModelSelection {
            model,
            model_variant: Some(model_variant),
        })),
        web_configured: !tavily_api_key.trim().is_empty(),
        trace_sink: Some(Arc::clone(&trace_sink)),
        lashlang_execution,
        event_tx,
        queued_work_driver,
        restate_ingress_url,
        restate_admin_url,
        restate_http,
        restate_cron_job_keys: Arc::new(Mutex::new(BTreeSet::new())),
        mail_world,
        active_restate_invocations,
    };
    restate::spawn_restate_endpoint(
        restate_endpoint_addr,
        state.clone(),
        process_deployment,
        process_worker,
    );
    emit_workbench_trace(
        &state.trace_sink,
        None,
        "startup",
        json!({
            "addr": addr.to_string(),
            "data_dir": data_dir.display().to_string(),
            "trace_path": trace_path_display,
            "lashlang_execution_path": lashlang_execution_path.display().to_string(),
            "model": serde_json::to_value(state.selected_model()).unwrap_or(Value::Null),
            "web_configured": state.web_configured,
            "restate_endpoint_addr": restate_endpoint_addr.to_string(),
            "restate_ingress_url": state.restate_ingress_url,
        }),
    );

    let app = Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/state", get(app_state))
        .route("/api/events", get(session_events))
        .route("/api/turn", post(send_turn))
        .route("/api/turn/cancel", post(cancel_turn))
        .route("/api/reset", post(reset_chat))
        .route("/api/button-trigger", post(button_trigger))
        .route("/api/accounts", get(list_accounts).post(add_account))
        .route("/api/accounts/{slug}", delete(delete_account))
        .route("/api/accounts/{slug}/messages", post(inject_message))
        .route("/api/accounts/{slug}/messages/{id}", delete(delete_message))
        .route("/api/accounts/{slug}/inbox", get(account_inbox))
        .route("/api/work", get(list_work))
        .route("/api/work/{process_id}/await", get(await_work))
        .route("/api/lashlang-graphs", get(list_lashlang_graphs))
        .route("/api/lashlang-graph/{graph_key}", get(lashlang_graph))
        .with_state(state);

    println!("agent-workbench listening on http://{addr}");
    println!("agent-workbench Restate endpoint listening on http://{restate_endpoint_addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind listener")?;
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}
