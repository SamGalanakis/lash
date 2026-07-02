use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "restate")]
use std::sync::Mutex;

use axum::Router;
use axum::routing::get;
#[cfg(feature = "restate")]
use lash::PluginBinding;
use lash::{
    durability::InlineEffectHost,
    provider::{ProviderHandle, ProviderOptions},
    tracing::{JsonlTraceSink, StderrTraceSink, TeeTraceSink, TraceLevel, TraceSink},
};
use lash_provider_openai::{OPENROUTER_BASE_URL, OpenAiCompatibleProvider};

mod board;
mod db;
mod demo_plugin;
#[cfg(feature = "restate")]
mod restate;
mod routes;
mod state;
mod ui;

use crate::db::AppDb;
#[cfg(feature = "restate")]
use crate::demo_plugin::{DemoPlugin, DemoPluginConfig};
#[cfg(feature = "restate")]
use crate::restate::{AgentServiceTurnWorkflow, AgentServiceTurnWorkflowImpl};
use crate::routes::{
    create_chat, index, list_chats, list_messages, send_message, settings, update_chat_model,
};
use crate::state::{AgentServiceDurability, AppStateData, anyhow_like};
#[cfg(feature = "restate")]
use lash::durability::DurableProcessWorker;
#[cfg(feature = "restate")]
use lash_restate::{LashProcessWorkflow, RestateEffectHost, RestateProcessDeployment};

const DEFAULT_TOKIO_THREAD_STACK_BYTES: usize = 2 * 1024 * 1024;

fn main() -> anyhow_like::Result<()> {
    let stack_bytes = std::env::var("AGENT_SERVICE_TOKIO_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOKIO_THREAD_STACK_BYTES);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(stack_bytes)
        .build()
        .map_err(|err| format!("build agent-service Tokio runtime: {err}"))?
        .block_on(async_main())
}

async fn async_main() -> anyhow_like::Result<()> {
    let _ = dotenvy::dotenv();

    let durability = AgentServiceDurability::configured()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY is required".to_string())?;
    let model = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| "anthropic/claude-sonnet-4.6".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "high".to_string());
    let addr: SocketAddr = std::env::var("AGENT_SERVICE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string())
        .parse()
        .map_err(|err| format!("invalid AGENT_SERVICE_ADDR: {err}"))?;
    #[cfg(feature = "restate")]
    let restate_endpoint_addr: SocketAddr = std::env::var("AGENT_SERVICE_RESTATE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9080".to_string())
        .parse()
        .map_err(|err| format!("invalid AGENT_SERVICE_RESTATE_ADDR: {err}"))?;
    #[cfg(feature = "restate")]
    let restate_ingress_url = std::env::var("RESTATE_INGRESS_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    #[cfg(not(feature = "restate"))]
    if durability == AgentServiceDurability::Restate {
        return Err(
            "AGENT_SERVICE_DURABILITY=restate requires `cargo run -p agent-service --features restate`"
                .to_string(),
        );
    }
    let data_dir = std::env::var("AGENT_SERVICE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".agent-service"));
    std::fs::create_dir_all(&data_dir).map_err(|err| err.to_string())?;
    let trace_path = std::env::var("AGENT_SERVICE_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir.join("trace.jsonl"));
    eprintln!("agent-service trace: {}", trace_path.display());

    let provider = ProviderHandle::new(
        OpenAiCompatibleProvider::new(api_key, OPENROUTER_BASE_URL)
            .with_options(ProviderOptions {
                expose_thinking: true,
                ..ProviderOptions::default()
            })
            .into_components(),
    );
    // Retain a clone for the shutdown drain: the core owns the working copy, but
    // the host is what calls `close()` to release transports on the way out.
    let drain_provider = provider.clone();

    // Worker identity for durable session-execution leases. WORKER_ID is stable
    // across restarts (set one per replica in a fleet); the incarnation is
    // bumped every boot. Failover consequence: if this process crashes and a new
    // boot (or a same-host peer) reopens a session whose lease this boot still
    // holds, the local-process liveness metadata proves the dead pid gone and
    // reclaims the lease before its TTL instead of waiting the full window. A
    // machine reboot changes the kernel boot id, so that path falls back to the
    // TTL backstop. The identity is stable within a boot, so keep at most one
    // in-flight turn per chat; the fenced head commit is the last-resort
    // single-writer backstop.
    let worker_id = std::env::var("WORKER_ID").unwrap_or_else(|_| "agent-service-1".to_string());
    let worker_host = std::env::var("HOSTNAME").unwrap_or_else(|_| worker_id.clone());
    let worker_incarnation = std::env::var("AGENT_SERVICE_INCARNATION").unwrap_or_else(|_| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_millis().to_string())
            .unwrap_or_else(|_| "0".to_string())
    });
    let session_owner = lash::persistence::LeaseOwnerIdentity::local_process(
        worker_id,
        worker_incarnation,
        worker_host,
    );
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        data_dir.join("lash-sessions"),
    ));
    // Deployment-level Lashlang artifact store (compiled module cache), shared
    // across the session tree and durable in SQLite.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .await
            .map_err(|err| err.to_string())?,
    ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
    let process_env_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("process-env.db"))
            .await
            .map_err(|err| err.to_string())?,
    );
    let trigger_store = Arc::new(
        lash_sqlite_store::SqliteTriggerStore::open(&data_dir.join("triggers.db"))
            .await
            .map_err(|err| err.to_string())?,
    );
    let app_db = AppDb::open(&data_dir.join("app.db")).map_err(|err| err.to_string())?;
    #[cfg(feature = "restate")]
    let shared_db = Arc::new(Mutex::new(app_db));
    let model_spec = lash::ModelSpec::from_token_limits(
        model.clone(),
        Some(model_variant.clone()),
        200_000,
        None,
    )
    .map_err(|err| format!("invalid OPENROUTER_MODEL metadata: {err}"))?;
    let factory = lash_protocol_rlm::RlmProtocolPluginFactory::new(
        lash_protocol_rlm::RlmProtocolPluginConfig::default(),
        artifact_store,
    );
    let core_builder = lash::LashCore::rlm_builder(factory)
        .provider(provider)
        .model(model_spec)
        .store_factory(store_factory)
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .process_env_store(process_env_store)
        .trace_sink(Arc::new(TeeTraceSink::new([
            Arc::new(StderrTraceSink::default()) as Arc<dyn TraceSink>,
            Arc::new(JsonlTraceSink::new(trace_path)),
        ])))
        .trace_level(TraceLevel::Extended)
        .trigger_store(trigger_store);
    let process_registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
            .await
            .map_err(|err| err.to_string())?,
    ) as Arc<dyn lash::process::ProcessRegistry>;
    #[cfg(feature = "restate")]
    let process_deployment = (durability == AgentServiceDurability::Restate).then(|| {
        RestateProcessDeployment::new(restate_ingress_url.clone(), Arc::clone(&process_registry))
    });
    let core = match durability {
        AgentServiceDurability::Local => core_builder
            .effect_host(Arc::new(InlineEffectHost::default()))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .map_err(|err| err.to_string())?,
        AgentServiceDurability::Restate => {
            #[cfg(feature = "restate")]
            {
                // Deployment host for paths outside a Restate workflow scope;
                // it fails loudly if an effect tries to execute without a
                // handler. Restate-backed turns pass a handler-scoped
                // controller per turn via `.effects(&controller)`. The
                // Restate ingress runner is the sole executor of
                // out-of-turn/background processes.
                core_builder
                    .effect_host(Arc::new(RestateEffectHost::with_ingress_url(
                        restate_ingress_url.clone(),
                    )))
                    .process_work_driver(
                        process_deployment
                            .as_ref()
                            .expect("process deployment configured for Restate")
                            .process_work_driver(),
                    )
                    .build()
                    .map_err(|err| err.to_string())?
            }
            #[cfg(not(feature = "restate"))]
            unreachable!("restate mode is rejected before core construction");
        }
    };

    #[cfg(feature = "restate")]
    let process_worker = if durability == AgentServiceDurability::Restate {
        let demo_factory = DemoPlugin::factory(&DemoPluginConfig {
            db: Arc::clone(&shared_db),
        });
        Some(DurableProcessWorker::new(
            core.durable_process_worker_config_with_plugins([demo_factory])
                .map_err(|err| err.to_string())?,
        ))
    } else {
        None
    };
    #[cfg(feature = "restate")]
    let restate_ingress_url =
        (durability == AgentServiceDurability::Restate).then_some(restate_ingress_url);
    #[cfg(feature = "restate")]
    let state = AppStateData::from_shared_db(
        core,
        Arc::clone(&shared_db),
        session_owner,
        model,
        Some(model_variant),
        durability,
        restate_ingress_url,
    );
    #[cfg(not(feature = "restate"))]
    let state = AppStateData::new(
        core,
        app_db,
        session_owner,
        model,
        Some(model_variant),
        durability,
    );

    #[cfg(feature = "restate")]
    if durability == AgentServiceDurability::Restate {
        let process_deployment = process_deployment.expect("process deployment configured");
        let endpoint = restate_sdk::endpoint::Endpoint::builder()
            .bind(AgentServiceTurnWorkflowImpl::new(state.clone()).serve())
            .bind(
                process_deployment
                    .workflow(process_worker.expect("process worker configured for Restate"))
                    .serve(),
            )
            .build();
        tokio::spawn(async move {
            restate_sdk::http_server::HttpServer::new(endpoint)
                .listen_and_serve(restate_endpoint_addr)
                .await;
        });
        process_deployment
            .process_work_driver()
            .claim_and_run_pending("agent_service_startup")
            .await
            .map_err(|err| err.to_string())?;
        println!("agent-service Restate endpoint listening on http://{restate_endpoint_addr}");
    }

    // Keep a state clone for the drain; the router consumes the original.
    let drain_state = state.clone();
    let app = Router::new()
        .route("/", get(index))
        .route("/api/settings", get(settings))
        .route("/api/chats", get(list_chats).post(create_chat))
        .route(
            "/api/chats/{chat_id}/model",
            axum::routing::post(update_chat_model),
        )
        .route(
            "/api/chats/{chat_id}/messages",
            get(list_messages).post(send_message),
        )
        .with_state(state);

    println!(
        "agent-service listening on http://{addr} (durability: {})",
        durability.as_str()
    );
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|err| err.to_string())?;
    // Step 1 of the drain (see docs/operations.html): stop admitting. Axum's
    // graceful shutdown stops accepting connections and lets in-flight requests
    // finish once a signal arrives.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| err.to_string())?;
    // Admission has stopped; run the teardown levers this process owns.
    drain(&drain_state, &drain_provider).await;
    Ok(())
}

/// Resolve when the process receives Ctrl-C or SIGTERM — the host-owned signal
/// that begins the drain. lash has no opinion on which signal means "drain".
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    println!("agent-service draining");
}

/// Host-composed teardown. lash ships no drain orchestrator (ADR-0014): each
/// step is an explicit lever the host calls in its own order.
///
/// This service opens a fresh session per request and detaches the turn task,
/// so it holds no long-lived sessions to `park()`/`close()` here and no external
/// queued-work claims to hand back. A host that caches live sessions would, at
/// this point, `cancel_running_turns()`, then `park()` (or `close()`) each one,
/// and `abandon_queued_work_claim` / `revoke_durable_waits` for any driver it
/// stopped mid-claim. See docs/operations.html for the full lever list.
async fn drain(state: &AppStateData, provider: &ProviderHandle) {
    // Release provider transports (the Codex provider sends WebSocket Close
    // frames; the default provider close is a no-op).
    if let Err(err) = provider.close().await {
        eprintln!("agent-service: provider close failed: {err}");
    }
    // Flush the trace sink (fsync the JSONL). An OTel host would also flush its
    // own TracerProvider here, which lash cannot do for it.
    if let Err(err) = state.core().flush_trace_sink() {
        eprintln!("agent-service: trace flush failed: {err}");
    }
}
