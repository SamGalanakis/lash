//! Compiled sources for the Rust snippets on `docs/plugins-tools.html`.

use std::sync::Arc;

use lash::plugins::PluginFactory;
use lash::tools::{ToolCall, ToolResult};

// docs:start:direct-completion-tool
use lash::direct::{DirectOutputSpec, DirectRequest};

async fn rank(call: ToolCall<'_>) -> ToolResult {
    let model = match call.context.sessions().model().await {
        Ok(model) => model,
        Err(err) => return ToolResult::err_fmt(format_args!("{err}")),
    };

    let request = DirectRequest {
        model: model.model,
        model_variant: model.model_variant,
        messages: vec![/* ... */],
        attachments: Vec::new(),
        output: DirectOutputSpec::Text,
        generation: Default::default(),
        stream_events: None,
        session_id: None, // filled by ToolContext
        caused_by: None,  // filled by ToolContext
        replay: None,
    };

    match call
        .context
        .direct_completions()
        .complete(request, "my_tool")
        .await
    {
        Ok(completion) => ToolResult::ok(serde_json::json!({ "text": completion.text })),
        Err(err) => ToolResult::err_fmt(format_args!("{err}")),
    }
}
// docs:end:direct-completion-tool

fn serial_tool(
    input_schema: serde_json::Value,
    output_schema: serde_json::Value,
) -> lash::tools::ToolDefinition {
    // docs:start:serial-tool
    use lash::tools::{ToolDefinition, ToolScheduling};

    ToolDefinition::raw_named(
        "write_file",
        "Replace a file's contents.",
        input_schema,
        output_schema,
    )
    .with_scheduling(ToolScheduling::Serial)
    // docs:end:serial-tool
}

fn budget_stack() -> lash::PluginStack {
    // docs:start:budget-stack
    use lash::plugins::{
        ToolOutputBudgetConfig, ToolOutputBudgetPluginFactory, runtime_plugin_stack,
    };

    let config = ToolOutputBudgetConfig {
        limit: 32 * 1024, // default: 16 * 1024 bytes
        max_lines: 800,   // default: 400
        ..ToolOutputBudgetConfig::default()
    };

    let plugins = runtime_plugin_stack().configure(|plugins| {
        plugins.replace(Arc::new(ToolOutputBudgetPluginFactory::new(config)));
    });
    // docs:end:budget-stack
    plugins
}
