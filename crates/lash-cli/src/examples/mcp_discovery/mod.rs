//! Reference MCP tool-discovery example.
//!
//! This is host policy, not a lash primitive: lash ships no tool discovery. The
//! example shows the recommended way to make a large MCP tool set discoverable
//! under the flat Tool Catalog + RLM deferred-resolution model:
//!
//! 1. Enumerate MCP tools and build a ranking index ([`ranking`]).
//! 2. Advertise them through a catalogue-preview prompt contribution
//!    ([`lash_lashlang_runtime::catalogue_preview_contribution`]).
//! 3. Expose a `search_tools` host tool over the index ([`definitions`],
//!    [`service`]).
//! 4. Register a [`DeferredToolResolver`](lash_lashlang_runtime::DeferredToolResolver)
//!    that resolves chosen MCP call-paths into a Tool Grant + Tool Execution
//!    Binding ([`resolver`]).
//!
//! The index ranking (BM25 / optional semantic / RRF) lives here as a reference
//! example hosts can copy or adapt. The catalogue-preview formatter is a public
//! helper in `lash-lashlang-runtime`.

mod catalog;
mod common;
mod definitions;
mod plugin;
mod ranking;
mod rerank;
mod resolver;
mod schema_index;
mod service;

pub use plugin::ToolDiscoveryPluginFactory;
pub use resolver::{McpCatalogedTool, McpDeferredToolResolver};

use std::sync::Arc;

use lash_core::ToolProvider;
use serde_json::json;

/// Enumerate provider tools into the shared MCP discovery/deferred record.
pub fn mcp_cataloged_tools(server: &str, provider: &dyn ToolProvider) -> Vec<McpCatalogedTool> {
    provider
        .tool_manifests()
        .into_iter()
        .filter_map(|manifest| {
            let contract = provider.resolve_contract(&manifest.name)?;
            Some(McpCatalogedTool {
                server: server.to_string(),
                definition: lash_core::ToolDefinition::from_parts(
                    manifest,
                    Arc::unwrap_or_clone(contract),
                ),
            })
        })
        .collect()
}

/// Project enumerated MCP tools to the JSON catalog consumed by `search_tools`
/// and the catalogue-preview prompt contribution.
pub fn mcp_catalog_records(tools: &[McpCatalogedTool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            let manifest = tool.definition.manifest();
            json!({
                "id": manifest.id,
                "name": manifest.name,
                "description": manifest.description,
                "bindings": manifest.bindings,
                "activation": manifest.activation,
                "contract": manifest.compact_contract,
            })
        })
        .collect()
}

/// End-to-end test of the reference MCP-discovery loop: discover -> preview ->
/// search_tools -> call -> resolve -> execute. The per-module unit tests cover
/// the pieces in isolation; this proves they compose through one real RLM turn
/// over the exact plugin stack, prompt layer, and deferred resolver the CLI
/// startup wires (`startup::cli_prompt_config`, `ToolDiscoveryPluginFactory`,
/// and an `McpDeferredToolResolver`).
///
/// The model's program searches the non-resident MCP catalog, then references
/// `appworld.venmo_send`, a call-path the resident host environment does not
/// provide. Linking defers to the resolver, which folds the grant for tool id
/// `tool:mcp/venmo_send`; execution then routes by grant id to the hidden MCP
/// provider without adding the tool to the catalog.
///
/// Hard constraint / compromise: the test builds the session with an ephemeral
/// `InlineEffectHost` rather than `CliSessionOpener`'s durable `SqliteEffectHost`.
/// A bare in-process turn cannot drive the durable effect host to completion —
/// its replay/journal lifecycle is pumped by the CLI's autonomous/interactive
/// runner, not by a unit test awaiting a single turn. Swapping only the effect
/// host (the supported in-process turn path) keeps every flat-catalog and
/// deferred-resolution code path under test.
#[cfg(all(test, feature = "test-provider"))]
mod gate4_e2e {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lash::PromptLayerSink;
    use lash::durability::InlineEffectHost;
    use lash::persistence::{
        InMemoryAttachmentStore, InMemoryLashlangArtifactStore, InMemoryProcessExecutionEnvStore,
    };
    use lash::tools::SharedDeferredToolResolver;
    use lash_core::plugin::{PluginSpec, StaticPluginFactory};
    use lash_core::{
        InMemorySessionStoreFactory, ModelSpec, ToolCall, ToolContract, ToolDefinition,
        ToolManifest, ToolProvider, ToolResult,
    };
    use lash_tool_support::{LashlangToolBinding, ToolDefinitionLashlangExt};
    use serde_json::json;

    use crate::examples::mcp_discovery::{
        McpCatalogedTool, McpDeferredToolResolver, ToolDiscoveryPluginFactory,
    };
    use crate::execution_settings::ExecutionMode;

    /// A stand-in MCP server tool: `appworld.venmo_send`, reachable by its
    /// Lashlang call-path. It counts executions so the test can prove the
    /// deferred-resolved call actually ran.
    struct HiddenVenmoProvider {
        definition: ToolDefinition,
        executions: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for HiddenVenmoProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest_by_id(&self, id: &lash_core::ToolId) -> Option<ToolManifest> {
            (id == &self.definition.manifest.id).then(|| self.definition.manifest())
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            (name == self.definition.name()).then(|| Arc::new(self.definition.contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            assert_eq!(call.name, "venmo_send");
            self.executions.fetch_add(1, Ordering::SeqCst);
            let amount = call.args.get("amount").cloned().unwrap_or(json!(0));
            ToolResult::ok(json!({ "sent": true, "amount": amount }))
        }
    }

    /// One venmo tool definition (always `tool:mcp/venmo_send`) bound under the
    /// deferred `appworld.venmo_send` call-path the model invents.
    fn venmo_definition(module: &str, operation: &str) -> ToolDefinition {
        ToolDefinition::raw(
            "tool:mcp/venmo_send",
            "venmo_send",
            "Send a Venmo payment to a receiver.",
            ToolContract::default_input_schema(),
            json!({ "type": "object" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new([module], operation))
    }

    /// Scripted RLM provider: a single Lashlang block that searches the catalog
    /// (`tools.search`), calls the deferred MCP call-path
    /// (`appworld.venmo_send`, resolved on demand), and finishs the routed
    /// result. Every model request is captured so the test can assert the
    /// catalogue-preview prompt and the `search_tools` host tool reached the
    /// model.
    fn scripted_provider(
        prompts: Arc<std::sync::Mutex<Vec<String>>>,
    ) -> lash_core::provider::ProviderHandle {
        lash_core::testing::TestProvider::builder()
            .kind("test")
            .complete(move |request| {
                let prompts = Arc::clone(&prompts);
                async move {
                    let joined = request
                        .messages
                        .iter()
                        .flat_map(|message| message.blocks.iter())
                        .filter_map(|block| match block {
                            lash_core::llm::types::LlmContentBlock::Text { text, .. } => {
                                Some(text.to_string())
                            }
                            _ => None,
                        })
                        .collect::<Vec<String>>()
                        .join("\n");
                    prompts.lock().expect("prompts lock").push(joined.clone());
                    // `search_tools` reranks its candidates through a direct LLM
                    // completion; that request flows through this same provider.
                    // Answer it with the JSON tool-name schema the reranker
                    // expects so search succeeds and the turn proceeds.
                    let response = if joined.contains("select API tools")
                        || joined.contains("Select tools from candidates")
                    {
                        "{\"tool_names\":[\"venmo_send\"]}".to_string()
                    } else {
                        // search_tools (discover) -> deferred resolve + execute of
                        // `appworld.venmo_send` -> finish, all in one turn so the
                        // loop is deterministic and terminates immediately.
                        r#"<lashlang>
hits = await tools.search({ query: "venmo send money", module: "appworld" })?
if hits[0].call != "appworld.venmo_send" {
    finish { value: "search-miss", discovered: hits }
}
result = await appworld.venmo_send({ amount: 25 })?
finish { value: "venmo-routed-ok", discovered: hits[0].call, sent: result.sent }
</lashlang>"#
                            .to_string()
                    };
                    Ok(lash_core::LlmResponse {
                        full_text: response.clone(),
                        parts: vec![lash_core::llm::types::LlmOutputPart::Text {
                            text: response,
                            response_meta: None,
                        }],
                        ..Default::default()
                    })
                }
            })
            .build()
            .into_handle()
    }

    #[tokio::test]
    async fn cli_mcp_discovery_loop_discovers_previews_searches_resolves_and_executes() {
        let executions = Arc::new(AtomicUsize::new(0));
        let venmo_definition = venmo_definition("appworld", "venmo_send");
        // Non-resident MCP provider for `tool:mcp/venmo_send`: it advertises no
        // catalog members but can resolve and execute the deferred grant's id.
        let venmo_provider = Arc::new(HiddenVenmoProvider {
            definition: venmo_definition.clone(),
            executions: Arc::clone(&executions),
        }) as Arc<dyn ToolProvider>;

        // discover: build the deferred resolver the same shape `startup::run`
        // does. The path the model invents is absent from the resident host
        // environment, so linking must defer to the resolver.
        let cataloged_tools = vec![McpCatalogedTool {
            server: "appworld".to_string(),
            definition: venmo_definition,
        }];
        let discovery_catalog =
            crate::examples::mcp_discovery::mcp_catalog_records(&cataloged_tools);
        let resolver: SharedDeferredToolResolver =
            Arc::new(McpDeferredToolResolver::new(cataloged_tools));

        // The live CLI plugin surface: the reference `search_tools` discovery
        // plugin (preview prompt + host tool) plus a non-resident MCP provider
        // that can execute only through the deferred grant route.
        let mut plugin_stack = lash::PluginStack::new();
        plugin_stack.push(Arc::new(ToolDiscoveryPluginFactory::with_catalog(
            discovery_catalog,
        )));
        plugin_stack.push(Arc::new(StaticPluginFactory::new(
            "appworld_mcp",
            PluginSpec::new().with_tool_provider(Arc::clone(&venmo_provider)),
        )));

        let prompts = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let factory = lash_protocol_rlm::RlmProtocolPluginFactory::new(
            lash_protocol_rlm::RlmProtocolPluginConfig::default(),
            Arc::new(InMemoryLashlangArtifactStore::new()),
        )
        .with_deferred_tool_resolver(resolver);
        let core = lash::LashCore::rlm_builder(factory)
            .provider(scripted_provider(Arc::clone(&prompts)))
            .model(
                ModelSpec::from_token_limits(
                    "test/cli-e2e-model",
                    Default::default(),
                    200_000,
                    None,
                )
                .expect("model spec"),
            )
            // Safety bound: the scripted program finishs on its first turn, so a
            // healthy run terminates well within this; it stops a regression from
            // re-prompting forever.
            .max_turns(2)
            .plugins(plugin_stack)
            // The CLI's own RLM prompt layer; the catalogue-preview prompt slots
            // into its Execution section.
            .prompt_layer(crate::startup::cli_prompt_config(true, &ExecutionMode::Rlm))
            .store_factory(Arc::new(InMemorySessionStoreFactory::new()))
            .effect_host(Arc::new(InlineEffectHost::default()))
            .attachment_store(Arc::new(InMemoryAttachmentStore::new()))
            .process_env_store(Arc::new(InMemoryProcessExecutionEnvStore::new()))
            .build()
            .expect("build ephemeral RLM core");
        let session = core
            .session("gate4-mcp-discovery")
            .open()
            .await
            .expect("open RLM session");

        // Refresh the catalog so the `search_tools` provider and catalogue
        // preview see the members, as the startup pipeline does pre-first-turn.
        session
            .admin()
            .commands()
            .refresh_tool_catalog("gate4", "gate4-refresh")
            .await
            .expect("refresh tool catalog");
        assert!(
            !session
                .tools()
                .active_manifests()
                .await
                .expect("active manifests before turn")
                .iter()
                .any(|manifest| manifest.id == lash_core::ToolId::from("tool:mcp/venmo_send")),
            "deferred MCP tool must not be a catalog member before execution"
        );

        let output = session
            .turn(lash::TurnInput::text("Send 25 on venmo"))
            .run()
            .await
            .expect("run RLM turn");

        // resolve + execute: linking deferred `appworld.venmo_send` to the
        // resolver's grant, which dispatched by id to the non-resident venmo tool.
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "deferred-resolved venmo tool should execute exactly once; outcome: {:?}",
            output.result.outcome
        );
        // call -> resolve -> execute -> finish closed the loop, and the routed
        // tool's output (`sent`) made it back into the final value.
        assert_eq!(
            output.final_value(),
            Some(&json!({
                "value": "venmo-routed-ok",
                "discovered": "appworld.venmo_send",
                "sent": true
            })),
            "turn should finish the routed value; outcome: {:?}",
            output.result.outcome
        );
        assert!(
            !session
                .tools()
                .active_manifests()
                .await
                .expect("active manifests after turn")
                .iter()
                .any(|manifest| manifest.id == lash_core::ToolId::from("tool:mcp/venmo_send")),
            "deferred MCP execution must not mutate catalog membership"
        );

        let prompts = prompts.lock().expect("prompts lock");
        let first_prompt = prompts.first().expect("at least one model request");
        // preview: the catalogue-preview prompt advertised the indexed tail.
        assert!(
            first_prompt.contains("Catalogued Capabilities"),
            "catalogue-preview prompt contribution did not reach the model"
        );
        assert!(
            first_prompt.contains("appworld.venmo_send"),
            "catalogue-preview prompt did not advertise the deferred MCP call"
        );
        // discover/search_tools: the host `search_tools` tool was presented.
        assert!(
            first_prompt.contains("search_tools") || first_prompt.contains("tools.search"),
            "search_tools host tool was not advertised to the model"
        );
    }
}
