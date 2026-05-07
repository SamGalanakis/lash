use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lash::{
    LlmOutputPart, LlmResponse, ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult,
};
use lash_embed::{EmbedError, EmbedPlugin, Input, LashCore, ModePreset};
use serde_json::json;

#[derive(Clone, Debug)]
struct TestPlugin;

#[derive(Clone)]
struct TestPluginConfig {
    required: bool,
    prompt_seen: Arc<Mutex<Vec<String>>>,
    tool_seen: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone, Debug)]
struct TestTurnInput {
    label: String,
}

impl EmbedPlugin for TestPlugin {
    const ID: &'static str = "test_typed";
    type SessionConfig = TestPluginConfig;
    type TurnInput = TestTurnInput;

    fn factory(config: &Self::SessionConfig) -> Arc<dyn lash::PluginFactory> {
        Arc::new(TestPluginFactory {
            config: config.clone(),
        })
    }

    fn requires_turn_input(config: &Self::SessionConfig) -> bool {
        config.required
    }
}

struct TestPluginFactory {
    config: TestPluginConfig,
}

impl lash::PluginFactory for TestPluginFactory {
    fn id(&self) -> &'static str {
        TestPlugin::ID
    }

    fn build(
        &self,
        _ctx: &lash::PluginSessionContext,
    ) -> Result<Arc<dyn lash::SessionPlugin>, lash::PluginError> {
        Ok(Arc::new(TestSessionPlugin {
            config: self.config.clone(),
        }))
    }
}

struct TestSessionPlugin {
    config: TestPluginConfig,
}

impl lash::SessionPlugin for TestSessionPlugin {
    fn id(&self) -> &'static str {
        TestPlugin::ID
    }

    fn register(&self, reg: &mut lash::PluginRegistrar) -> Result<(), lash::PluginError> {
        let prompt_seen = Arc::clone(&self.config.prompt_seen);
        reg.prompt().contribute(Arc::new(move |ctx| {
            let prompt_seen = Arc::clone(&prompt_seen);
            Box::pin(async move {
                if let Some(input) = ctx
                    .turn_context
                    .plugin_input::<TestTurnInput>(TestPlugin::ID)
                {
                    prompt_seen
                        .lock()
                        .expect("prompt seen lock")
                        .push(input.label.clone());
                }
                Ok(Vec::new())
            })
        }));
        reg.tools().provider(Arc::new(TestTools {
            seen: Arc::clone(&self.config.tool_seen),
        }))
    }
}

struct TestTools {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ToolProvider for TestTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "typed_probe",
            "Probe typed turn input.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            json!({ "type": "object" }),
        )]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt("typed_probe requires context")
    }

    async fn execute_with_context(
        &self,
        name: &str,
        _args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        assert_eq!(name, "typed_probe");
        let Some(input) = context
            .turn_context
            .plugin_input::<TestTurnInput>(TestPlugin::ID)
        else {
            return ToolResult::err_fmt("missing typed input");
        };
        self.seen
            .lock()
            .expect("tool seen lock")
            .push(input.label.clone());
        ToolResult::ok(json!({ "label": input.label }))
    }
}

fn response_text(text: &str) -> LlmResponse {
    LlmResponse {
        full_text: text.to_string(),
        parts: vec![LlmOutputPart::Text {
            text: text.to_string(),
            response_meta: None,
        }],
        ..LlmResponse::default()
    }
}

fn response_tool_call() -> LlmResponse {
    LlmResponse {
        parts: vec![LlmOutputPart::ToolCall {
            call_id: "tool-1".to_string(),
            tool_name: "typed_probe".to_string(),
            input_json: "{}".to_string(),
            item_id: None,
            signature: None,
        }],
        ..LlmResponse::default()
    }
}

fn core_with_responses(responses: Vec<LlmResponse>) -> LashCore {
    let responses = Arc::new(Mutex::new(responses.into_iter()));
    let provider = lash::testing::TestProvider::builder()
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move {
                Ok(responses
                    .lock()
                    .expect("responses lock")
                    .next()
                    .unwrap_or_else(|| response_text("fallback")))
            }
        })
        .build()
        .into_handle();
    LashCore::builder()
        .install_mode(ModePreset::standard())
        .register_plugin::<TestPlugin>()
        .provider(provider)
        .max_context_tokens(16_000)
        .build()
        .expect("core")
}

#[tokio::test]
async fn required_turn_input_missing_is_validated_before_execution() {
    let config = TestPluginConfig {
        required: true,
        prompt_seen: Arc::new(Mutex::new(Vec::new())),
        tool_seen: Arc::new(Mutex::new(Vec::new())),
    };
    let core = core_with_responses(vec![response_text("should not run")]);
    let session = core
        .session("required-missing")
        .use_plugin::<TestPlugin>(config)
        .expect("use plugin")
        .open()
        .await
        .expect("session");

    let err = session
        .run(Input::text("hello"))
        .await
        .expect_err("missing required input");
    assert!(matches!(
        err,
        EmbedError::MissingPluginTurnInput {
            plugin_id: TestPlugin::ID
        }
    ));
}

#[tokio::test]
async fn prompt_hook_and_tool_provider_read_typed_turn_context() {
    let prompt_seen = Arc::new(Mutex::new(Vec::new()));
    let tool_seen = Arc::new(Mutex::new(Vec::new()));
    let config = TestPluginConfig {
        required: true,
        prompt_seen: Arc::clone(&prompt_seen),
        tool_seen: Arc::clone(&tool_seen),
    };
    let core = core_with_responses(vec![response_tool_call(), response_text("done")]);
    let session = core
        .session("typed-context")
        .use_plugin::<TestPlugin>(config)
        .expect("use plugin")
        .open()
        .await
        .expect("session");

    let result = session
        .turn(Input::text("probe"))
        .with_plugin_input::<TestPlugin>(TestTurnInput {
            label: "page-a".to_string(),
        })
        .run()
        .await
        .expect("turn");

    assert_eq!(result.assistant_prose, "done");
    assert_eq!(
        prompt_seen.lock().expect("prompt seen lock").as_slice(),
        ["page-a", "page-a"]
    );
    assert_eq!(
        tool_seen.lock().expect("tool seen lock").as_slice(),
        ["page-a"]
    );
}

#[tokio::test]
async fn optional_turn_input_can_be_absent() {
    let config = TestPluginConfig {
        required: false,
        prompt_seen: Arc::new(Mutex::new(Vec::new())),
        tool_seen: Arc::new(Mutex::new(Vec::new())),
    };
    let core = core_with_responses(vec![response_text("ok")]);
    let session = core
        .session("optional-absent")
        .use_plugin::<TestPlugin>(config)
        .expect("use plugin")
        .open()
        .await
        .expect("session");

    let result = session.run(Input::text("hello")).await.expect("turn");
    assert_eq!(result.assistant_prose, "ok");
}
