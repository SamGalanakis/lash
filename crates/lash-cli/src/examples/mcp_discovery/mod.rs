//! Reference MCP tool-discovery example.
//!
//! This is host policy, not a lash primitive: lash ships no tool discovery. The
//! example shows the recommended way to make a large MCP tool set discoverable
//! under the flat Tool Catalog + RLM deferred-resolution model:
//!
//! 1. Enumerate MCP tools and build a ranking index ([`ranking`]).
//! 2. Advertise them through a catalogue-preview prompt contribution
//!    ([`catalogue_preview`]).
//! 3. Expose a `search_tools` host tool over the index ([`definitions`],
//!    [`service`]).
//! 4. Register a [`DeferredToolResolver`](lash_lashlang_runtime::DeferredToolResolver)
//!    that resolves chosen MCP call-paths into a Tool Grant + Tool Execution
//!    Binding ([`resolver`]).
//!
//! The index ranking (BM25 / optional semantic / RRF) and the catalogue-preview
//! formatter were formerly the `lash-plugin-tool-discovery` crate; they live
//! here now as the reference example.

mod catalog;
mod catalogue_preview;
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

/// Build the reference [`McpDeferredToolResolver`] from a tool provider that
/// enumerates MCP tools (e.g. `lash_plugin_mcp::McpToolProvider`). The resolver
/// maps each tool's Lashlang call-path to a Tool Grant + Tool Execution
/// Binding, so RLM linking can resolve a deferred call-path on demand.
pub fn mcp_deferred_tool_resolver(
    server: &str,
    provider: &dyn ToolProvider,
) -> McpDeferredToolResolver {
    let tools = provider.tool_manifests().into_iter().filter_map(|manifest| {
        let contract = provider.resolve_contract(&manifest.name)?;
        Some(McpCatalogedTool {
            server: server.to_string(),
            definition: lash_core::ToolDefinition::from_parts(
                manifest,
                Arc::unwrap_or_clone(contract),
            ),
        })
    });
    McpDeferredToolResolver::new(tools)
}

/// End-to-end test of the reference MCP-discovery loop, satisfying
/// flat-tool-catalog-cutover plan verification gate 4: discover -> preview ->
/// search_tools -> call -> resolve -> execute. The per-module unit tests cover
/// the pieces in isolation; this proves they compose through one real RLM turn
/// over the exact plugin stack, prompt layer, and deferred resolver the CLI
/// startup wires (`startup::cli_prompt_config`, `ToolDiscoveryPluginFactory`,
/// and an `McpDeferredToolResolver`).
///
/// The model's program references `appworld.venmo_send`, a call-path the
/// resident host environment does not provide, so linking defers to the
/// resolver, which folds the grant for tool id `tool:mcp/venmo_send`. The same
/// tool id is a resident, callable catalog member (advertised under
/// `payments.send`), so dispatch-by-id then executes it — membership is
/// callability, and a deferred grant only augments linking, not the registry.
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

    use lash::durability::InlineEffectHost;
    use lash::persistence::{
        InMemoryAttachmentStore, InMemoryLashlangArtifactStore, InMemoryProcessExecutionEnvStore,
    };
    use lash::PromptLayerSink;
    use lash::tools::SharedDeferredToolResolver;
    use lash_core::plugin::{PluginSpec, StaticPluginFactory};
    use lash_core::{
        InMemorySessionStoreFactory, ModelSpec, ToolCall, ToolContract, ToolDefinition,
        ToolProvider, ToolResult,
    };
    use lash_tool_support::{
        LashlangToolBinding, StaticToolExecute, StaticToolProvider, ToolDefinitionLashlangExt,
    };
    use serde_json::json;

    use crate::examples::mcp_discovery::{
        McpCatalogedTool, McpDeferredToolResolver, ToolDiscoveryPluginFactory,
    };
    use crate::execution_settings::ExecutionMode;

    /// A stand-in MCP server tool: `appworld.venmo_send`, reachable by its
    /// Lashlang call-path. It counts executions so the test can prove the
    /// deferred-resolved call actually ran.
    struct VenmoTool {
        executions: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl StaticToolExecute for VenmoTool {
        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            assert_eq!(call.name, "venmo_send");
            self.executions.fetch_add(1, Ordering::SeqCst);
            let amount = call.args.get("amount").cloned().unwrap_or(json!(0));
            ToolResult::ok(json!({ "sent": true, "amount": amount }))
        }
    }

    /// One venmo tool definition (always `tool:mcp/venmo_send`) bound under a
    /// chosen Lashlang call-path. The resident provider advertises it under
    /// `payments.send`; the deferred resolver advertises the *same tool id*
    /// under `appworld.venmo_send`, the call-path the model actually invents.
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
    /// (`appworld.venmo_send`, resolved on demand), and submits the routed
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
                        // `appworld.venmo_send` -> submit, all in one turn so the
                        // loop is deterministic and terminates immediately.
                        r#"<lashlang>
hits = await tools.search({ query: "venmo send money" })?
result = await appworld.venmo_send({ amount: 25 })?
submit { value: "venmo-routed-ok", sent: result.sent }
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
        // Resident, callable provider for `tool:mcp/venmo_send`, advertised under
        // the `payments.send` call-path. Membership is callability: this is what
        // lets the resolved call dispatch and execute.
        let venmo_provider = Arc::new(StaticToolProvider::new(
            vec![venmo_definition("payments", "send")],
            VenmoTool {
                executions: Arc::clone(&executions),
            },
        )) as Arc<dyn ToolProvider>;

        // discover: build the deferred resolver the same shape `startup::run`
        // does, but advertising the *same tool id* under the `appworld.venmo_send`
        // call-path — the path the model invents and that the resident host
        // environment does not provide, so linking must defer to the resolver.
        let resolver: SharedDeferredToolResolver =
            Arc::new(McpDeferredToolResolver::new([McpCatalogedTool {
                server: "appworld".to_string(),
                definition: venmo_definition("appworld", "venmo_send"),
            }]));

        // The live CLI plugin surface: the reference `search_tools` discovery
        // plugin (preview prompt + host tool) plus the resident venmo provider
        // so the resolved tool id is a callable catalog member.
        let mut plugin_stack = lash::PluginStack::new();
        plugin_stack.push(Arc::new(ToolDiscoveryPluginFactory::new()));
        plugin_stack.push(Arc::new(StaticPluginFactory::new(
            "appworld_mcp",
            PluginSpec::new().with_tool_provider(Arc::clone(&venmo_provider)),
        )));

        let prompts = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let core = lash::RlmCore::builder()
            .provider(scripted_provider(Arc::clone(&prompts)))
            .model(
                ModelSpec::from_token_limits("test/cli-e2e-model", None, 200_000, None)
                    .expect("model spec"),
            )
            // Safety bound: the scripted program submits on its first turn, so a
            // healthy run terminates well within this; it stops a regression from
            // re-prompting forever.
            .max_turns(2)
            .plugins(plugin_stack)
            // The CLI's own RLM prompt layer; the catalogue-preview prompt slots
            // into its Execution section.
            .prompt_layer(crate::startup::cli_prompt_config(true, &ExecutionMode::Rlm))
            .deferred_tool_resolver(resolver)
            .store_factory(Arc::new(InMemorySessionStoreFactory::new()))
            .effect_host(Arc::new(InlineEffectHost::default()))
            .lashlang_artifact_store(Arc::new(InMemoryLashlangArtifactStore::new()))
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

        let output = session
            .turn(lash::TurnInput::text("Send 25 on venmo"))
            .run()
            .await
            .expect("run RLM turn");

        // resolve + execute: linking deferred `appworld.venmo_send` to the
        // resolver's grant, which dispatched by id to the resident venmo tool.
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "deferred-resolved venmo tool should execute exactly once; outcome: {:?}",
            output.result.outcome
        );
        // call -> resolve -> execute -> submit closed the loop, and the routed
        // tool's output (`sent`) made it back into the submitted value.
        assert_eq!(
            output.submitted_value(),
            Some(&json!({ "value": "venmo-routed-ok", "sent": true })),
            "turn should submit the routed value; outcome: {:?}",
            output.result.outcome
        );

        let prompts = prompts.lock().expect("prompts lock");
        let first_prompt = prompts.first().expect("at least one model request");
        // preview: the catalogue-preview prompt advertised the searchable members.
        assert!(
            first_prompt.contains("Catalogued capabilities"),
            "catalogue-preview prompt contribution did not reach the model"
        );
        // discover/search_tools: the host `search_tools` tool was presented.
        assert!(
            first_prompt.contains("search_tools") || first_prompt.contains("tools.search"),
            "search_tools host tool was not advertised to the model"
        );
    }
}
