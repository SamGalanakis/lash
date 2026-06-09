//! Compiled sources for the Rust snippets on `docs/tracing.html`.

use std::sync::Arc;

use lash::provider::ProviderHandle;
use lash::tracing::{TraceRecord, TraceSink, TraceSinkError};
use lash::{LashCore, ModelSpec};

async fn jsonl_trace_core(provider: ProviderHandle, model: String) -> anyhow::Result<()> {
    // docs:start:jsonl-trace-core
    use std::sync::Arc;

    use lash::{
        LashCore,
        tracing::{JsonlTraceSink, TraceLevel, TraceSink},
    };

    let trace_sink: Arc<dyn TraceSink> = Arc::new(JsonlTraceSink::new("./.lash-data/trace.jsonl"));

    let core = LashCore::standard()
        .provider(provider)
        .model(
            lash::ModelSpec::from_token_limits(model.clone(), None, 200_000, None)
                .expect("valid model metadata"),
        )
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .trace_sink(trace_sink)
        .trace_level(TraceLevel::Extended)
        .build()?;
    // docs:end:jsonl-trace-core
    Ok(())
}

async fn lashlang_execution_jsonl(
    provider: ProviderHandle,
    model: ModelSpec,
) -> anyhow::Result<()> {
    // docs:start:lashlang-execution-jsonl
    let core = LashCore::rlm()
        .provider(provider)
        .model(model)
        .effect_host(std::sync::Arc::new(
            lash::durability::InlineEffectHost::default(),
        ))
        .lashlang_artifact_store(std::sync::Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(std::sync::Arc::new(
            lash::persistence::InMemoryAttachmentStore::new(),
        ))
        .lashlang_execution_jsonl_path("./.lash-data/lashlang-execution.jsonl")
        .build()?;
    // docs:end:lashlang-execution-jsonl
    Ok(())
}

async fn lashlang_graph_store(provider: ProviderHandle, model: ModelSpec) -> anyhow::Result<()> {
    // docs:start:lashlang-graph-store
    use std::sync::Arc;

    use lash::tracing::{JsonlTraceSink, TeeTraceSink, TraceLashlangGraphStore, TraceSink};

    let lashlang_graphs = Arc::new(TraceLashlangGraphStore::default());
    let lashlang_execution_sink = Arc::new(TeeTraceSink::new([
        Arc::clone(&lashlang_graphs) as Arc<dyn TraceSink>,
        Arc::new(JsonlTraceSink::new("./.lash-data/lashlang-execution.jsonl"))
            as Arc<dyn TraceSink>,
    ]));

    let core = LashCore::rlm()
        .provider(provider)
        .model(model)
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .lashlang_execution_sink(lashlang_execution_sink)
        .build()?;

    let graph = lashlang_graphs.graph("process:process-id");
    // docs:end:lashlang-graph-store
    Ok(())
}

// docs:start:fanout-trace-sink
struct FanoutTraceSink {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl TraceSink for FanoutTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        for sink in &self.sinks {
            // Treat errors per-sink; one failing destination shouldn't take the others down.
            let _ = sink.append(record);
        }
        Ok(())
    }
}
// docs:end:fanout-trace-sink

async fn otel_trace_core() -> anyhow::Result<()> {
    // docs:start:otel-trace-core
    use std::sync::Arc;

    use lash::{
        LashCore,
        tracing::{TraceLevel, TraceSink},
    };
    use lash_core::OtelTraceSink;

    // Exporter/provider setup stays with the host; this reads the
    // process-global OpenTelemetry tracer provider.
    let sink: Arc<dyn TraceSink> = Arc::new(OtelTraceSink::from_global_provider());
    let core = LashCore::standard()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .trace_sink(sink)
        .trace_level(TraceLevel::Extended)
        .build()?;
    // docs:end:otel-trace-core
    Ok(())
}
