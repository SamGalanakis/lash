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
    LashCore, ModeId, ModePreset,
    advanced::InlineEffectHost,
    provider::{ProviderHandle, ProviderOptions, ProviderThinkingPolicy},
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
use lash::advanced::ProcessWorkRunner;
#[cfg(feature = "restate")]
use lash_restate::{
    LashProcessWorkflow, LashProcessWorkflowImpl, RestateCoreProcessRunner,
    RestateProcessIngressRunner,
};

#[tokio::main]
async fn main() -> anyhow_like::Result<()> {
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
                thinking: ProviderThinkingPolicy { expose: true },
                ..ProviderOptions::default()
            })
            .into_components(),
    );
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        data_dir.join("lash-sessions"),
    ));
    // Deployment-level Lashlang artifact store (compiled module cache), shared
    // across the session tree and durable in SQLite.
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
            .map_err(|err| err.to_string())?,
    ) as Arc<dyn lash::persistence::LashlangArtifactStore>;
    let app_db = AppDb::open(&data_dir.join("app.db")).map_err(|err| err.to_string())?;
    #[cfg(feature = "restate")]
    let shared_db = Arc::new(Mutex::new(app_db));
    let model_spec = lash::ModelSpec::from_token_limits(
        model.clone(),
        Some(model_variant.clone()),
        200_000,
        None,
        None,
    )
    .map_err(|err| format!("invalid OPENROUTER_MODEL metadata: {err}"))?;
    let core_builder = LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model_spec)
        .store_factory(store_factory)
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .lashlang_artifact_store(artifact_store)
        .trace_sink(Some(Arc::new(TeeTraceSink::new([
            Arc::new(StderrTraceSink::default()) as Arc<dyn TraceSink>,
            Arc::new(JsonlTraceSink::new(trace_path)),
        ]))))
        .trace_level(TraceLevel::Extended);
    let process_registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&data_dir.join("processes.db"))
            .map_err(|err| err.to_string())?,
    ) as Arc<dyn lash::persistence::ProcessRegistry>;
    // Durable tier: a ProcessWorkRunner over a Restate ingress-client run handle
    // drives the registry's non-terminal rows to terminal by ingress-submitting
    // their LashProcessWorkflow (folding in the former startup-only recovery
    // sweep as its first tick). The Local tier gets the default inline runner
    // for free — LashCore lazily spawns it on first session open.
    #[cfg(feature = "restate")]
    let process_work_runner = (durability == AgentServiceDurability::Restate).then(|| {
        ProcessWorkRunner::new(Arc::new(RestateProcessIngressRunner::new(
            restate_ingress_url.clone(),
            Arc::clone(&process_registry),
        )))
    });
    let core = match durability {
        AgentServiceDurability::Local => core_builder
            .advanced()
            .effect_host(Arc::new(InlineEffectHost::default()))
            .process_registry(Arc::clone(&process_registry))
            .build()
            .map_err(|err| err.to_string())?,
        AgentServiceDurability::Restate => {
            #[cfg(feature = "restate")]
            {
                // Base host for turns that run outside a Restate workflow
                // scope; durable turns pass a handler-scoped controller per
                // turn via `stream_with_effect_scope`. The Restate ingress
                // runner is the sole executor of out-of-turn/background
                // processes, so disable the default inline runner and hand the
                // core its poke (fired after every successful process start).
                core_builder
                    .advanced()
                    .effect_host(Arc::new(InlineEffectHost::default()))
                    .process_registry(Arc::clone(&process_registry))
                    .disable_default_process_work_runner()
                    .with_process_work_runner(
                        process_work_runner
                            .as_ref()
                            .expect("process work runner configured for Restate")
                            .poke_handle(),
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
        Some(lash::advanced::DurableProcessWorker::new(
            core.durable_process_worker_config_with_plugins(
                Arc::clone(&process_registry),
                [demo_factory],
            )
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
        model,
        Some(model_variant),
        durability,
        restate_ingress_url,
    );
    #[cfg(not(feature = "restate"))]
    let state = AppStateData::new(core, app_db, model, Some(model_variant), durability);

    #[cfg(feature = "restate")]
    if durability == AgentServiceDurability::Restate {
        let process_runner = Arc::new(RestateCoreProcessRunner::new(
            process_worker.expect("process worker configured for Restate"),
        ));
        let endpoint = restate_sdk::endpoint::Endpoint::builder()
            .bind(AgentServiceTurnWorkflowImpl::new(state.clone()).serve())
            .bind(
                LashProcessWorkflowImpl::new(
                    Arc::clone(&process_runner),
                    Arc::clone(&process_registry),
                )
                .serve(),
            )
            .build();
        tokio::spawn(async move {
            restate_sdk::http_server::HttpServer::new(endpoint)
                .listen_and_serve(restate_endpoint_addr)
                .await;
        });
        // Spawn the wake-driven runner after the workflow endpoint is bound so
        // its first tick (which folds in the former startup-only recovery
        // sweep) ingress-submits onto a registered LashProcessWorkflow. The
        // runner then drives the registry's non-terminal rows on every poke and
        // poll tick, lease-fencing each submit so a process runs exactly once.
        process_work_runner
            .expect("process work runner configured for Restate")
            .spawn();
        println!("agent-service Restate endpoint listening on http://{restate_endpoint_addr}");
    }

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
    axum::serve(listener, app)
        .await
        .map_err(|err| err.to_string())?;
    Ok(())
}
