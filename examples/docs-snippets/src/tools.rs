//! Compiled sources for the Rust snippets on `docs/tools.html`.

// docs:start:simple-fixed-tool
use std::sync::Arc;

use async_trait::async_trait;
use lash::tools::{
    StaticToolExecute, StaticToolProvider, ToolCall, ToolDefinition, ToolProvider, ToolResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
    units: Option<String>,
}

#[derive(Serialize, JsonSchema)]
struct WeatherReport {
    summary: String,
    temperature_c: f32,
}

struct WeatherTools;

#[async_trait]
impl StaticToolExecute for WeatherTools {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "weather_lookup" => {
                let args: WeatherArgs = match serde_json::from_value(call.args.clone()) {
                    Ok(args) => args,
                    Err(err) => return ToolResult::err_fmt(format_args!("invalid args: {err}")),
                };

                let report = lookup_weather(args).await;
                match serde_json::to_value(report) {
                    Ok(value) => ToolResult::ok(value),
                    Err(err) => ToolResult::err_fmt(format_args!("serialize output: {err}")),
                }
            }
            other => ToolResult::err_fmt(format_args!("unknown tool: {other}")),
        }
    }
}

pub fn weather_provider() -> Arc<dyn ToolProvider> {
    let definition = ToolDefinition::typed::<WeatherArgs, WeatherReport>(
        "tool:weather_lookup",
        "weather_lookup",
        "Look up the current weather for a city.",
    );

    Arc::new(StaticToolProvider::new(vec![definition], WeatherTools)) as Arc<dyn ToolProvider>
}

async fn lookup_weather(args: WeatherArgs) -> WeatherReport {
    let units = args.units.unwrap_or_else(|| "metric".to_string());
    WeatherReport {
        summary: format!("{} weather in {}", units, args.city),
        temperature_c: 21.0,
    }
}
// docs:end:simple-fixed-tool
