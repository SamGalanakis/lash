use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::routing::get;
use lash::{
    LashCore, ModeId, ModePreset,
    advanced::{InlineRuntimeEffectController, LocalProcessRegistry},
    provider::{ProviderHandle, ProviderOptions, ProviderThinkingPolicy},
    tracing::{JsonlTraceSink, TraceLevel, TraceRecord, TraceSink, TraceSinkError},
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
use crate::restate::{AgentServiceTurnWorkflow, AgentServiceTurnWorkflowImpl};
use crate::routes::{
    create_chat, index, list_chats, list_messages, send_message, settings, update_chat_model,
};
use crate::state::{AgentServiceDurability, AppStateData, anyhow_like};

#[derive(Default)]
struct StderrTraceSink {
    lock: Mutex<()>,
}

impl TraceSink for StderrTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        let line = serde_json::to_string(record)?;
        let _guard = self.lock.lock().map_err(|_| TraceSinkError::LockPoisoned)?;
        eprintln!("{line}");
        Ok(())
    }
}

struct FanoutTraceSink {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl TraceSink for FanoutTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        for sink in &self.sinks {
            sink.append(record)?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow_like::Result<()> {
    let durability = AgentServiceDurability::configured()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY is required".to_string())?;
    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-5.5".to_string());
    let model_variant =
        std::env::var("OPENROUTER_MODEL_VARIANT").unwrap_or_else(|_| "medium".to_string());
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
    let core_builder = LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model.clone(), Some(model_variant.clone()))
        .max_context_tokens(200_000)
        .store_factory(store_factory)
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .trace_sink(Some(Arc::new(FanoutTraceSink {
            sinks: vec![
                Arc::new(StderrTraceSink::default()),
                Arc::new(JsonlTraceSink::new(trace_path)),
            ],
        })))
        .trace_level(TraceLevel::Extended);
    let core = match durability {
        AgentServiceDurability::Local => core_builder
            .advanced()
            .effect_controller(Arc::new(InlineRuntimeEffectController::default()))
            .process_registry(Arc::new(LocalProcessRegistry::default()))
            .build()
            .map_err(|err| err.to_string())?,
        AgentServiceDurability::Restate => {
            #[cfg(feature = "restate")]
            {
                core_builder
                    .advanced()
                    .process_registry(Arc::new(LocalProcessRegistry::default()))
                    .build()
                    .map_err(|err| err.to_string())?
            }
            #[cfg(not(feature = "restate"))]
            unreachable!("restate mode is rejected before core construction");
        }
    };

    let app_db = AppDb::open(&data_dir.join("app.db")).map_err(|err| err.to_string())?;
    #[cfg(feature = "restate")]
    let restate_ingress_url =
        (durability == AgentServiceDurability::Restate).then_some(restate_ingress_url);
    #[cfg(feature = "restate")]
    let state = AppStateData::new(
        core,
        app_db,
        model,
        Some(model_variant),
        durability,
        restate_ingress_url,
    );
    #[cfg(not(feature = "restate"))]
    let state = AppStateData::new(core, app_db, model, Some(model_variant), durability);

    #[cfg(feature = "restate")]
    if durability == AgentServiceDurability::Restate {
        let endpoint = restate_sdk::endpoint::Endpoint::builder()
            .bind(AgentServiceTurnWorkflowImpl::new(state.clone()).serve())
            .build();
        tokio::spawn(async move {
            restate_sdk::http_server::HttpServer::new(endpoint)
                .listen_and_serve(restate_endpoint_addr)
                .await;
        });
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
