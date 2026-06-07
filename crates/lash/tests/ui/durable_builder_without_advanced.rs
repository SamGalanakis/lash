use std::{path::Path, sync::Arc};

use lash::{LashCore, ModeId, ModePreset};

async fn durable_core_without_advanced(
    provider: lash::provider::ProviderHandle,
    data_dir: &Path,
) -> lash::Result<lash::LashCore> {
    let model = lash::ModelSpec::from_token_limits("compile-only", None, 4096, None)
        .expect("valid model metadata");

    LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider)
        .model(model)
        .store_factory(Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
            data_dir.join("sessions"),
        )))
        .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
            data_dir.join("attachments"),
        )))
        .lashlang_artifact_store(Arc::new(
            lash_sqlite_store::Store::open(&data_dir.join("artifacts.db"))
                .await
                .expect("turso artifact store"),
        ))
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .residency(lash::durability::Residency::ActivePathOnly)
        .termination(lash::durability::TerminationPolicy::default())
        .build()
}

fn main() {
    let _ = durable_core_without_advanced;
}
