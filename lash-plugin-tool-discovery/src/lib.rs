use std::{collections::BTreeMap, fmt::Write as _, sync::Arc};

use lash::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolSurfaceContext,
};
use lash::{ToolAvailability, ToolProvider, ToolSurfaceContribution, ToolSurfaceOverride};

mod catalog;
mod common;
mod definitions;
mod provider;
mod ranking;
mod rerank;
mod schema_index;

use provider::ToolDiscoveryToolsProvider;

#[cfg(test)]
use common::{DEFAULT_LIMIT, DEFAULT_LLM_RERANK_MODEL, MAX_LIMIT};
#[cfg(test)]
use definitions::{load_tools_definition, search_tools_definition};
#[cfg(test)]
use lash::{
    DirectOutputSpec, DirectPart, ToolActivation, ToolAvailabilityConfig, ToolDefinition,
    ToolExecutionMode,
};
#[cfg(test)]
use ranking::{RankedCandidate, ToolDiscoveryIndex, reciprocal_rank_fusion};
#[cfg(test)]
use rerank::{llm_rerank_request, merge_llm_selection};
#[cfg(test)]
use serde_json::{Value, json};

#[derive(Default)]
pub struct ToolDiscoveryPluginFactory;

impl ToolDiscoveryPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for ToolDiscoveryPluginFactory {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ToolDiscoveryPlugin {
            provider: Arc::new(ToolDiscoveryToolsProvider::new()),
        }))
    }
}

struct ToolDiscoveryPlugin {
    provider: Arc<ToolDiscoveryToolsProvider>,
}

impl SessionPlugin for ToolDiscoveryPlugin {
    fn id(&self) -> &'static str {
        "tool_discovery"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.surface().contribute(Arc::new(rlm_tool_surface));
        Ok(())
    }
}

fn rlm_tool_surface(ctx: ToolSurfaceContext) -> Result<ToolSurfaceContribution, PluginError> {
    if ctx.mode.plugin_id() != "rlm" {
        return Ok(ToolSurfaceContribution::default());
    }

    let has_catalogued_tools = has_catalogued_tools(&ctx);
    let overrides = ctx
        .tools
        .iter()
        .filter_map(|tool| {
            if tool.name == "load_tools" {
                return Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Hidden),
                });
            }
            if tool.name == "search_tools" && !has_catalogued_tools {
                return Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Hidden),
                });
            }
            let availability = tool.effective_availability(&ctx.mode);
            if availability == ToolAvailability::Discoverable {
                Some(ToolSurfaceOverride {
                    tool_name: tool.name.clone(),
                    availability: Some(ToolAvailability::Callable),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(ToolSurfaceContribution {
        overrides,
        tool_list_notes: catalogue_notes(&ctx, has_catalogued_tools),
    })
}

fn has_catalogued_tools(ctx: &ToolSurfaceContext) -> bool {
    ctx.tools.iter().any(|tool| {
        !matches!(tool.name.as_str(), "search_tools" | "load_tools")
            && tool.effective_availability(&ctx.mode).is_discoverable()
            && !tool.effective_availability(&ctx.mode).is_documented()
    })
}

const CATALOGUE_NAMESPACE_LIMIT: usize = 100;
const CATALOGUE_TOOL_NAME_LIMIT: usize = 50;

fn catalogue_notes(ctx: &ToolSurfaceContext, has_catalogued_tools: bool) -> Vec<String> {
    if !has_catalogued_tools {
        return Vec::new();
    }

    let mut by_namespace: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut omitted_tool_count = 0usize;
    for tool in &ctx.tools {
        if matches!(tool.name.as_str(), "search_tools" | "load_tools") {
            continue;
        }
        let availability = tool.effective_availability(&ctx.mode);
        if !availability.is_discoverable() || availability.is_documented() {
            continue;
        }
        omitted_tool_count += 1;
        let namespace = tool
            .discovery
            .namespace
            .as_deref()
            .filter(|namespace| !namespace.trim().is_empty())
            .unwrap_or("default");
        by_namespace
            .entry(namespace)
            .or_default()
            .push(tool.name.as_str());
    }
    for names in by_namespace.values_mut() {
        names.sort_unstable();
    }

    let mut rendered = format!(
        "Catalogued tools: {omitted_tool_count} not showcased here; searchable with `search_tools`.\n\
         When a task needs a tool not showcased here, run `search_tools(query=...)` and call the relevant result by name. \
         Results use the same compact contract shape as showcased tools: signature, description, and capped examples."
    );

    if by_namespace.len() <= CATALOGUE_NAMESPACE_LIMIT {
        rendered.push_str("\n\nNamespaces: ");
        for (index, (namespace, names)) in by_namespace.iter().enumerate() {
            if index > 0 {
                rendered.push_str(", ");
            }
            let _ = write!(rendered, "{namespace}({})", names.len());
        }
    } else {
        let _ = write!(
            rendered,
            "\n\nNamespaces: {} total; use `search_tools` to narrow them.",
            by_namespace.len()
        );
    }

    if omitted_tool_count <= CATALOGUE_TOOL_NAME_LIMIT {
        rendered.push_str("\n\nCatalogued names:");
        for (namespace, names) in by_namespace {
            rendered.push('\n');
            let _ = write!(rendered, "{namespace}: {}", names.join(", "));
        }
    }

    vec![rendered]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PluginError, PromptHost,
        SessionGraphHost, SessionHandle, SessionLifecycleHost, SessionSnapshot,
        SessionSnapshotHost, SessionTurnHandle, TaskHost, ToolCatalogHost, TraceHost, TurnHost,
    };
    use lash::{
        AssembledTurn, DirectCompletion, ExecutionMode, TokenUsage, ToolExecutionContext,
        ToolSurfaceBuildInput, TurnInput, build_tool_surface,
    };
    use std::sync::Mutex;

    fn catalog_tool(name: &str, description: &str) -> Value {
        json!({
            "name": name,
            "description": description,
            "params": [],
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": true
            },
            "output_schema": {},
            "examples": [],
            "aliases": [],
            "availability": "discoverable",
            "callable": false,
            "documented": false,
            "discoverable": true,
            "activation": "loadable",
            "loadable": true,
            "activation_hint": "",
        })
    }

    fn callable_undocumented_tool(name: &str) -> Value {
        let mut tool = catalog_tool(name, "callable omitted tool");
        let obj = tool.as_object_mut().unwrap();
        obj.insert("callable".to_string(), json!(true));
        obj.insert("documented".to_string(), json!(false));
        obj.insert("loadable".to_string(), json!(false));
        tool
    }

    #[derive(Default)]
    struct FakeSessionManager {
        catalog: Vec<Value>,
        promoted: Mutex<Vec<String>>,
        direct_response: Mutex<Option<String>>,
        direct_requests: Mutex<Vec<lash::DirectRequest>>,
    }

    #[async_trait::async_trait]
    impl SessionSnapshotHost for FakeSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }
    }

    #[async_trait::async_trait]
    impl ToolCatalogHost for FakeSessionManager {
        async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<Value>, PluginError> {
            Ok(self.catalog.clone())
        }
    }

    #[async_trait::async_trait]
    impl DynamicToolHost for FakeSessionManager {
        async fn set_tools_availability(
            &self,
            _session_id: &str,
            tool_names: &[String],
            _availability: Option<ToolAvailability>,
        ) -> Result<u64, PluginError> {
            self.promoted
                .lock()
                .expect("promoted lock poisoned")
                .extend(tool_names.iter().cloned());
            Ok(2)
        }
    }

    #[async_trait::async_trait]
    impl DirectCompletionHost for FakeSessionManager {
        async fn direct_completion(
            &self,
            request: lash::DirectRequest,
            _usage_source: &str,
        ) -> Result<DirectCompletion, PluginError> {
            self.direct_requests
                .lock()
                .expect("direct requests lock poisoned")
                .push(request);
            let text = self
                .direct_response
                .lock()
                .expect("direct response lock poisoned")
                .clone()
                .unwrap_or_else(|| "{\"tool_names\":[]}".to_string());
            Ok(DirectCompletion {
                text,
                usage: TokenUsage::default(),
            })
        }
    }

    #[async_trait::async_trait]
    impl SessionLifecycleHost for FakeSessionManager {
        async fn create_session(
            &self,
            _request: lash::plugin::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl TurnHost for FakeSessionManager {
        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    impl TaskHost for FakeSessionManager {}
    impl MonitorHost for FakeSessionManager {}
    impl SessionGraphHost for FakeSessionManager {}
    impl PromptHost for FakeSessionManager {}
    impl TraceHost for FakeSessionManager {}

    fn catalog_tool_with_metadata(
        name: &str,
        description: &str,
        namespace: Option<&str>,
        aliases: Vec<&str>,
    ) -> Value {
        let mut tool = catalog_tool(name, description);
        let obj = tool.as_object_mut().unwrap();
        if let Some(namespace) = namespace {
            obj.insert("namespace".to_string(), json!(namespace));
        }
        obj.insert("aliases".to_string(), json!(aliases));
        tool
    }

    fn ranked_names(results: &[Value]) -> Vec<String> {
        results
            .iter()
            .map(|result| {
                result
                    .get("name")
                    .and_then(Value::as_str)
                    .expect("ranked result name")
                    .to_string()
            })
            .collect()
    }

    fn venmo_payment_tools() -> Vec<Value> {
        let mut create_transaction = catalog_tool_with_metadata(
            "mcp__appworld__venmo_create_transaction",
            "Create a Venmo transaction to send money to another user.",
            Some("appworld"),
            vec!["venmo_send_money", "pay_user"],
        );
        create_transaction.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "receiver_email": {
                        "type": "string",
                        "description": "Email of the person receiving the money."
                    },
                    "payment_card_id": {
                        "type": "integer",
                        "description": "Payment card to fund the transaction."
                    },
                    "private": {
                        "type": "boolean",
                        "description": "Whether the transaction is private."
                    }
                },
                "required": ["access_token", "receiver_email", "payment_card_id"]
            }),
        );

        let mut create_request = catalog_tool_with_metadata(
            "mcp__appworld__venmo_create_payment_request",
            "Create a Venmo payment request asking another user to pay you.",
            Some("appworld"),
            vec!["venmo_request_payment"],
        );
        create_request.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "payer_email": {
                        "type": "string",
                        "description": "Email of the person who should pay the request."
                    },
                    "amount": {"type": "number"}
                },
                "required": ["access_token", "payer_email", "amount"]
            }),
        );

        let mut remind_request = catalog_tool_with_metadata(
            "mcp__appworld__venmo_remind_payment_request",
            "Send a reminder for a pending Venmo payment request.",
            Some("appworld"),
            vec!["venmo_remind_request"],
        );
        remind_request.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "request_id": {"type": "integer"}
                },
                "required": ["access_token", "request_id"]
            }),
        );

        let mut add_balance = catalog_tool_with_metadata(
            "mcp__appworld__venmo_add_to_venmo_balance",
            "Add money from a funding source to your Venmo balance.",
            Some("appworld"),
            vec!["venmo_balance_transfer"],
        );
        add_balance.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "amount": {"type": "number"},
                    "payment_card_id": {"type": "integer"}
                },
                "required": ["access_token", "amount", "payment_card_id"]
            }),
        );

        vec![
            create_request,
            remind_request,
            add_balance,
            create_transaction,
        ]
    }

    #[test]
    fn exact_name_beats_fuzzy_typo() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool("spotify_search_songs", "Find songs in Spotify"),
                catalog_tool("spotty_notes", "Scratch notes"),
            ],
        );
        let results = index.search(&json!({ "query": "spotify songs" }));
        assert_eq!(results[0]["name"], json!("spotify_search_songs"));
        let typo = index.search(&json!({ "query": "spotfy songs" }));
        assert_eq!(typo[0]["name"], json!("spotify_search_songs"));
    }

    #[test]
    fn short_query_words_do_not_create_substring_matches() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_album",
                    "Show details for a Spotify album.",
                    Some("appworld"),
                    vec!["spotify_album_details"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_song_library",
                    "Show songs saved in your collection.",
                    Some("appworld"),
                    vec!["spotify_song_library"],
                ),
            ],
        );

        let results = index.search(&json!({
            "query": "tracks in collection",
            "namespace": "appworld",
            "limit": 5
        }));

        assert_eq!(
            ranked_names(&results),
            vec!["mcp__appworld__spotify_show_song_library"]
        );
    }

    #[test]
    fn default_limit_is_bounded_for_empty_query() {
        let catalog = (0..75)
            .map(|idx| catalog_tool(&format!("tool_{idx:02}"), "tool"))
            .collect();
        let index = ToolDiscoveryIndex::build(1, catalog);
        assert_eq!(index.search(&json!({})).len(), DEFAULT_LIMIT);
    }

    #[test]
    fn limit_is_capped_at_max_limit() {
        let catalog = (0..150)
            .map(|idx| catalog_tool(&format!("tool_{idx:03}"), "tool"))
            .collect();
        let index = ToolDiscoveryIndex::build(1, catalog);
        assert_eq!(index.search(&json!({ "limit": 500 })).len(), MAX_LIMIT);
    }

    #[test]
    fn ranking_finds_core_tool_categories() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "spawn_agent",
                    "Delegate work to subagents",
                    Some("agents"),
                    vec!["subagent"],
                ),
                catalog_tool_with_metadata(
                    "search_web",
                    "Search the web for current sources",
                    Some("web"),
                    vec!["web_search"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_search_songs",
                    "Find songs in Spotify",
                    Some("appworld"),
                    vec!["spotify_search_songs"],
                ),
            ],
        );
        assert_eq!(
            index.search(&json!({ "query": "read files" }))[0]["name"],
            json!("read_file")
        );
        assert_eq!(
            index.search(&json!({ "query": "delegate agent" }))[0]["name"],
            json!("spawn_agent")
        );
        assert_eq!(
            index.search(&json!({ "query": "web search" }))[0]["name"],
            json!("search_web")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify songs", "namespace": "appworld" }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify", "namespace": ["web", "appworld"] }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
        assert_eq!(
            index.search(&json!({ "query": "spotify", "namespace": "web, appworld" }))[0]["name"],
            json!("mcp__appworld__spotify_search_songs")
        );
    }

    #[test]
    fn ranking_prefers_name_matches_over_parameter_only_matches() {
        let mut login = catalog_tool_with_metadata(
            "mcp__appworld__spotify_login",
            "Login to your account.",
            Some("appworld"),
            vec!["spotify_login"],
        );
        login.as_object_mut().unwrap().insert(
            "params".to_string(),
            json!([
                {"name": "username", "type": "str", "required": true},
                {"name": "password", "type": "str", "required": true}
            ]),
        );

        let mut logout = catalog_tool_with_metadata(
            "mcp__appworld__spotify_logout",
            "Logout from your account.",
            Some("appworld"),
            vec!["spotify_logout"],
        );
        logout.as_object_mut().unwrap().insert(
            "params".to_string(),
            json!([
                {"name": "access_token", "type": "str", "required": true}
            ]),
        );

        let index = ToolDiscoveryIndex::build(1, vec![logout, login]);
        let results = index.search(&json!({
            "query": "spotify login access token",
            "namespace": "appworld"
        }));

        assert_eq!(results[0]["name"], json!("mcp__appworld__spotify_login"));
    }

    #[test]
    fn ranking_prefers_output_fields_over_input_filter_matches() {
        let mut filter_songs = catalog_tool_with_metadata(
            "mcp__appworld__spotify_filter_songs",
            "Search Spotify songs by filters.",
            Some("appworld"),
            vec!["spotify_filter_songs"],
        );
        filter_songs.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "genre": {
                        "type": "string",
                        "description": "Genre filter."
                    },
                    "play_count": {
                        "type": "integer",
                        "description": "Minimum play count filter."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title filter."
                    }
                },
                "required": ["access_token"]
            }),
        );
        filter_songs.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song_id": {"type": "integer"}
                            },
                            "required": ["song_id"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let mut show_song = catalog_tool_with_metadata(
            "mcp__appworld__spotify_show_song",
            "Get a Spotify song record.",
            Some("appworld"),
            vec!["spotify_show_song", "song_details"],
        );
        show_song.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "song_id": {"type": "integer"}
                },
                "required": ["access_token", "song_id"]
            }),
        );
        show_song.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "description": "Detailed song record.",
                        "properties": {
                            "genre": {
                                "type": "string",
                                "description": "Song genre."
                            },
                            "play_count": {
                                "type": "integer",
                                "description": "Number of times the song was played."
                            },
                            "title": {
                                "type": "string",
                                "description": "Song title."
                            }
                        },
                        "required": ["genre", "play_count", "title"]
                    }
                },
                "required": ["response"]
            }),
        );

        let index = ToolDiscoveryIndex::build(1, vec![filter_songs, show_song]);
        let results = index.search(&json!({
            "query": "play_count genre title",
            "namespace": "appworld"
        }));

        assert_eq!(
            results[0]["name"],
            json!("mcp__appworld__spotify_show_song")
        );
    }

    #[test]
    fn ranking_orders_tools_by_expected_spotify_field_utility() {
        let mut search_songs = catalog_tool_with_metadata(
            "mcp__appworld__spotify_search_songs",
            "Search Spotify songs by filters.",
            Some("appworld"),
            vec!["spotify_search_songs"],
        );
        search_songs.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "genre": {"type": "string"},
                    "play_count": {"type": "integer"},
                    "title": {"type": "string"}
                },
                "required": ["access_token"]
            }),
        );
        search_songs.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song_id": {"type": "integer"}
                            },
                            "required": ["song_id"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let mut show_song = catalog_tool_with_metadata(
            "mcp__appworld__spotify_show_song",
            "Get a Spotify song record.",
            Some("appworld"),
            vec!["spotify_show_song", "song_details"],
        );
        show_song.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "song_id": {"type": "integer"}
                },
                "required": ["access_token", "song_id"]
            }),
        );
        show_song.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "description": "Detailed song record.",
                        "properties": {
                            "genre": {"type": "string"},
                            "play_count": {"type": "integer"},
                            "title": {"type": "string"}
                        },
                        "required": ["genre", "play_count", "title"]
                    }
                },
                "required": ["response"]
            }),
        );

        let mut list_albums = catalog_tool_with_metadata(
            "mcp__appworld__spotify_list_albums",
            "List Spotify albums.",
            Some("appworld"),
            vec!["spotify_albums"],
        );
        list_albums.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "album_id": {"type": "integer"},
                                "title": {"type": "string"}
                            },
                            "required": ["album_id", "title"]
                        }
                    }
                },
                "required": ["response"]
            }),
        );

        let index = ToolDiscoveryIndex::build(1, vec![search_songs, show_song, list_albums]);
        let results = index.search(&json!({
            "query": "spotify song details play_count genre title",
            "namespace": "appworld",
            "limit": 3
        }));

        assert_eq!(
            ranked_names(&results),
            vec![
                "mcp__appworld__spotify_show_song",
                "mcp__appworld__spotify_search_songs",
                "mcp__appworld__spotify_list_albums",
            ]
        );
    }

    #[test]
    fn ranking_orders_venmo_tools_by_send_money_intent() {
        let index = ToolDiscoveryIndex::build(1, venmo_payment_tools());
        let results = index.search(&json!({
            "query": "venmo send money private payment_card receiver_email",
            "namespace": "appworld",
            "limit": 3
        }));

        assert_eq!(
            ranked_names(&results),
            vec![
                "mcp__appworld__venmo_create_transaction",
                "mcp__appworld__venmo_create_payment_request",
                "mcp__appworld__venmo_add_to_venmo_balance",
            ]
        );
    }

    #[test]
    fn ranking_orders_venmo_short_payment_queries_by_action_intent() {
        for query in [
            "venmo send payment",
            "venmo send payment to user",
            "venmo make payment transfer money",
        ] {
            let index = ToolDiscoveryIndex::build(1, venmo_payment_tools());
            let results = index.search(&json!({
                "query": query,
                "namespace": "appworld",
                "limit": 4
            }));

            assert_eq!(
                results[0]["name"],
                json!("mcp__appworld__venmo_create_transaction"),
                "unexpected ranking for query {query:?}"
            );
        }
    }

    #[test]
    fn semantic_fusion_adds_recall_candidates_without_replacing_exact_matches() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__music_collection",
                    "Show saved tracks.",
                    Some("appworld"),
                    vec!["music_collection"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_liked_songs",
                    "Show songs that you have favorited.",
                    Some("appworld"),
                    vec!["spotify_favorite_songs"],
                ),
            ],
        );

        let lexical_only = index.search(&json!({
            "query": "spotify liked songs library",
            "namespace": "appworld",
            "limit": 2
        }));
        assert_eq!(
            ranked_names(&lexical_only),
            vec!["mcp__appworld__spotify_show_liked_songs"]
        );

        let semantic = index.search_with_semantic_scores(
            &json!({
                "query": "spotify liked songs library",
                "namespace": "appworld",
                "limit": 2
            }),
            Some(&[0.0, 0.92, 0.86]),
        );
        assert_eq!(
            ranked_names(&semantic),
            vec![
                "mcp__appworld__spotify_show_liked_songs",
                "mcp__appworld__music_collection",
            ]
        );

        let exact = index.search_with_semantic_scores(
            &json!({ "query": "read file", "limit": 2 }),
            Some(&[0.45, 0.9, 0.8]),
        );
        assert_eq!(exact[0]["name"], json!("read_file"));
    }

    #[test]
    fn semantic_fusion_promotes_complementary_tools_for_mixed_qualifier_queries() {
        let catalog = vec![
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_liked_songs",
                "Show songs that you have liked.",
                Some("appworld"),
                vec!["spotify_liked_songs"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_song_library",
                "Show songs in your saved Spotify library.",
                Some("appworld"),
                vec!["spotify_song_library"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_liked_playlists",
                "Show Spotify playlists that you have liked.",
                Some("appworld"),
                vec!["spotify_liked_playlists"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_playlist_library",
                "Show playlists in your Spotify library.",
                Some("appworld"),
                vec!["spotify_playlist_library"],
            ),
            catalog_tool_with_metadata(
                "mcp__appworld__spotify_show_playlist",
                "Show songs from a Spotify playlist.",
                Some("appworld"),
                vec!["spotify_playlist_songs"],
            ),
        ];
        let index = ToolDiscoveryIndex::build(1, catalog);

        let library_and_liked = index.search_with_semantic_scores(
            &json!({
                "query": "spotify liked songs library",
                "namespace": "appworld",
                "limit": 3
            }),
            Some(&[0.92, 0.91, 0.76, 0.74, 0.68]),
        );
        let names = ranked_names(&library_and_liked);
        assert_eq!(names.len(), 3);
        assert!(names[..2].contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names[..2].contains(&"mcp__appworld__spotify_show_song_library".to_string()));

        let playlist_and_liked = index.search_with_semantic_scores(
            &json!({
                "query": "spotify playlist songs liked",
                "namespace": "appworld",
                "limit": 4
            }),
            Some(&[0.88, 0.62, 0.91, 0.84, 0.87]),
        );
        let names = ranked_names(&playlist_and_liked);
        assert!(names.contains(&"mcp__appworld__spotify_show_playlist".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_playlists".to_string()));
    }

    #[cfg(feature = "semantic-tool-search")]
    #[test]
    #[ignore = "downloads and loads an external embedding model"]
    fn semantic_model_smoke_retrieves_paraphrased_tool_matches() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_liked_songs",
                    "Show songs that you have liked.",
                    Some("appworld"),
                    vec!["spotify_liked_songs"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_song_library",
                    "Show songs saved in your Spotify library.",
                    Some("appworld"),
                    vec!["spotify_song_library"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__spotify_show_album",
                    "Show details for a Spotify album.",
                    Some("appworld"),
                    vec!["spotify_album_details"],
                ),
                catalog_tool_with_metadata(
                    "mcp__appworld__venmo_create_transaction",
                    "Send money to another Venmo user.",
                    Some("appworld"),
                    vec!["venmo_send_money"],
                ),
            ],
        );

        let lexical = index.search(&json!({
            "query": "favorite tracks in my saved collection",
            "namespace": "appworld",
            "limit": 3
        }));
        let semantic = index.search(&json!({
            "query": "favorite tracks in my saved collection",
            "namespace": "appworld",
            "semantic": true,
            "limit": 3
        }));

        eprintln!("lexical={:?}", ranked_names(&lexical));
        eprintln!("semantic={:?}", ranked_names(&semantic));

        let names = ranked_names(&semantic);
        assert!(names.contains(&"mcp__appworld__spotify_show_liked_songs".to_string()));
        assert!(names.contains(&"mcp__appworld__spotify_show_song_library".to_string()));
        assert!(!names.contains(&"mcp__appworld__venmo_create_transaction".to_string()));
    }

    #[test]
    fn reciprocal_rank_fusion_keeps_cross_list_hits_ahead_of_single_list_noise() {
        let fused = reciprocal_rank_fusion(
            vec![
                RankedCandidate {
                    idx: 0,
                    lexical_score: 10.0,
                    semantic_score: None,
                },
                RankedCandidate {
                    idx: 1,
                    lexical_score: 8.0,
                    semantic_score: None,
                },
                RankedCandidate {
                    idx: 2,
                    lexical_score: 6.0,
                    semantic_score: None,
                },
            ],
            vec![
                RankedCandidate {
                    idx: 3,
                    lexical_score: 0.0,
                    semantic_score: Some(0.99),
                },
                RankedCandidate {
                    idx: 1,
                    lexical_score: 0.0,
                    semantic_score: Some(0.88),
                },
                RankedCandidate {
                    idx: 4,
                    lexical_score: 0.0,
                    semantic_score: Some(0.87),
                },
            ],
        );

        let names = fused
            .iter()
            .map(|candidate| candidate.idx)
            .collect::<Vec<_>>();
        assert_eq!(names[..3], [1, 0, 3]);
    }

    #[test]
    fn search_results_include_compact_schema_parameter_restrictions() {
        let mut spotify = catalog_tool("mcp__appworld__spotify_search_songs", "Find songs");
        spotify.as_object_mut().unwrap().insert(
            "examples".to_string(),
            json!(["search songs by genre", "search songs by play count"]),
        );
        spotify.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "Access token obtained from spotify app login."
                    },
                    "genre": {
                        "type": ["string", "null"],
                        "description": "Only include songs from this genre.",
                        "default": null
                    },
                    "page_limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 20
                    }
                },
                "required": ["access_token"]
            }),
        );
        spotify.as_object_mut().unwrap().insert(
            "output_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "description": "Matched songs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "album_id": {"type": ["integer", "null"]},
                                "duration": {"type": "integer"},
                                "genre": {"type": "string"},
                                "like_count": {"type": "integer"},
                                "play_count": {
                                    "type": "integer",
                                    "description": "Number of times the song was played.",
                                    "minimum": 0
                                },
                                "rating": {"type": "number"},
                                "release_date": {"type": "string"},
                                "song_id": {
                                    "type": "integer",
                                    "description": "Stable song identifier."
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Song title."
                                }
                            },
                            "required": [
                                "album_id",
                                "duration",
                                "genre",
                                "like_count",
                                "play_count",
                                "rating",
                                "release_date",
                                "song_id",
                                "title"
                            ]
                        }
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message when search fails."
                    }
                },
                "required": ["response"]
            }),
        );
        let index = ToolDiscoveryIndex::build(1, vec![spotify]);

        let results = index.search(&json!({ "query": "spotify" }));
        assert_eq!(
            results[0]["signature"],
            json!(
                "mcp__appworld__spotify_search_songs(access_token: str, genre?: str | null = null, page_limit?: int >= 1 <= 20 = 20) -> record{error?: str, response: list[record{album_id: int | null, duration: int, genre: str, like_count: int, play_count: int, rating: float, release_date: str, song_id: int, title: str}]}\nParameters:\n- `access_token: str` — Access token obtained from spotify app login.\n- `genre?: str | null = null` — Only include songs from this genre.\n- `page_limit?: int >= 1 <= 20 = 20` — Maximum number of results to return.\nReturn fields:\n- `error?: str` — Error message when search fails.\n- `response: list[record]` — Matched songs.\n- `response[].album_id: int | null`\n- `response[].duration: int`\n- `response[].genre: str`\n- `response[].like_count: int`\n- `response[].play_count: int >= 0` — Number of times the song was played.\n- `response[].rating: float`\n- `response[].release_date: str`\n- `response[].song_id: int` — Stable song identifier.\n- `response[].title: str` — Song title."
            )
        );
        assert!(results[0].get("returns").is_none());
        assert!(results[0].get("params").is_none());
        assert!(results[0].get("parameters").is_none());
        assert!(results[0].get("return_fields").is_none());
        assert_eq!(
            results[0]["examples"],
            json!(["search songs by genre", "search songs by play count"])
        );
        assert!(results[0].get("input_schema").is_none());
        assert!(results[0].get("output_schema").is_none());
        assert!(results[0].get("matched_fields").is_none());
        assert!(results[0].get("score").is_none());
    }

    #[tokio::test]
    async fn search_tools_uses_host_catalog_and_projects_compact_contract() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "search_web",
                    "Search the web",
                    Some("web"),
                    vec!["web_search"],
                ),
            ],
            promoted: Mutex::default(),
            ..Default::default()
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host,
            cancellation_token: None,
            async_task_id: None,
            turn_context: lash::TurnContext::default(),
            tool_call_id: None,
        };

        let result = provider
            .execute_streaming_with_context(
                "search_tools",
                &json!({
                    "query": "cat",
                    "namespace": "filesystem",
                    "limit": 1,
                }),
                &context,
                None,
            )
            .await;

        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("search result list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["name"], json!("read_file"));
        assert_eq!(results[0]["signature"], json!("read_file() -> any"));
        assert!(results[0].get("returns").is_none());
        assert_eq!(results[0]["description"], json!("Read file contents"));
        assert!(results[0].get("namespace").is_none());
        assert!(results[0].get("matched_fields").is_none());
        assert!(results[0].get("score").is_none());
    }

    #[tokio::test]
    async fn search_tools_projects_dynamic_output_contracts() {
        let mut spawn_agent = catalog_tool("spawn_agent", "Run a subagent");
        spawn_agent.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "agent_name": { "type": "string" },
                    "task": { "type": "string" },
                    "output": { "type": "object", "additionalProperties": true }
                },
                "required": ["agent_name", "task"]
            }),
        );
        spawn_agent.as_object_mut().unwrap().insert(
            "output_contract".to_string(),
            json!({ "kind": "from_input_schema", "input_field": "output" }),
        );

        let mut llm_query = catalog_tool("llm_query", "Run one lightweight LLM call");
        llm_query.as_object_mut().unwrap().insert(
            "input_schema".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "task": { "type": "string" },
                    "output": { "type": "object", "additionalProperties": true }
                },
                "required": ["task"]
            }),
        );
        llm_query.as_object_mut().unwrap().insert(
            "output_contract".to_string(),
            json!({
                "kind": "from_input_schema",
                "input_field": "output",
                "default_schema": { "type": "string" }
            }),
        );

        let host = Arc::new(FakeSessionManager {
            catalog: vec![spawn_agent, llm_query],
            promoted: Mutex::default(),
            ..Default::default()
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host,
            cancellation_token: None,
            async_task_id: None,
            turn_context: lash::TurnContext::default(),
            tool_call_id: None,
        };

        let result = provider
            .execute_streaming_with_context(
                "search_tools",
                &json!({ "query": "spawn_agent", "limit": 1 }),
                &context,
                None,
            )
            .await;
        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("spawn results");
        assert_eq!(results[0]["name"], json!("spawn_agent"));
        assert_eq!(
            results[0]["signature"],
            json!(
                "spawn_agent<T = any>(agent_name: str, task: str, output?: TypeSpec<T>) -> T\nParameters:\n- `agent_name: str`\n- `task: str`\n- `output?: TypeSpec<T>`"
            )
        );
        assert!(results[0].get("returns").is_none());

        let result = provider
            .execute_streaming_with_context(
                "search_tools",
                &json!({ "query": "llm_query", "limit": 1 }),
                &context,
                None,
            )
            .await;
        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("llm results");
        assert_eq!(results[0]["name"], json!("llm_query"));
        assert_eq!(
            results[0]["signature"],
            json!(
                "llm_query<T = str>(task: str, output?: TypeSpec<T>) -> T\nParameters:\n- `task: str`\n- `output?: TypeSpec<T>`"
            )
        );
        assert!(results[0].get("returns").is_none());
    }

    #[test]
    fn debug_search_includes_minimal_score() {
        let index = ToolDiscoveryIndex::build(1, vec![catalog_tool("read_file", "Read files")]);

        let results = index.search(&json!({ "query": "read", "debug": true }));

        assert!(results[0]["score"].as_f64().is_some());
        assert!(results[0].get("matched_fields").is_none());
    }

    #[test]
    fn exclude_filter_removes_exact_tool_names() {
        let index = ToolDiscoveryIndex::build(
            1,
            vec![
                catalog_tool("read_file", "Read files"),
                catalog_tool("search_web", "Search the web"),
            ],
        );

        let results = index.search(&json!({
            "query": "",
            "exclude": ["read_file"],
        }));

        assert_eq!(ranked_names(&results), vec!["search_web"]);
    }

    #[test]
    fn llm_rerank_request_uses_structured_name_enum_schema() {
        let candidates = vec![
            json!({"name": "read_file", "signature": "read_file() -> str", "description": "Read file"}),
            json!({"name": "search_web", "signature": "search_web(query: str) -> record", "description": "Search web"}),
        ];

        let request = llm_rerank_request(&json!({"query": "find docs"}), &candidates, 2);

        assert_eq!(
            request.model,
            std::env::var("LASH_TOOL_SEARCH_LLM_MODEL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_LLM_RERANK_MODEL.to_string())
        );
        let DirectOutputSpec::JsonSchema(schema) = request.output else {
            panic!("expected json schema output");
        };
        assert_eq!(schema.name, "tool_search_rerank");
        assert!(
            schema.schema["properties"]["tool_names"]
                .get("uniqueItems")
                .is_none()
        );
        assert!(
            schema.schema["properties"]["tool_names"]
                .get("maxItems")
                .is_none()
        );
        assert_eq!(
            schema.schema["properties"]["tool_names"]["items"]["enum"],
            json!(["read_file", "search_web"])
        );
    }

    #[test]
    fn merge_llm_selection_dedupes_and_fills_from_deterministic_order() {
        let candidates = vec![
            json!({"name": "a"}),
            json!({"name": "b"}),
            json!({"name": "c"}),
        ];

        let merged = merge_llm_selection(
            candidates,
            vec!["b".to_string(), "b".to_string(), "missing".to_string()],
            3,
        );

        assert_eq!(ranked_names(&merged), vec!["b", "a", "c"]);
    }

    #[tokio::test]
    async fn search_tools_reranks_candidates_with_direct_completion() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                catalog_tool_with_metadata("read_file", "Read file contents", None, vec!["cat"]),
                catalog_tool_with_metadata("search_web", "Search the web", None, vec!["web"]),
            ],
            promoted: Mutex::default(),
            direct_response: Mutex::new(Some(
                "{\"tool_names\":[\"search_web\",\"search_web\",\"unknown\"]}".to_string(),
            )),
            ..Default::default()
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host: host.clone(),
            cancellation_token: None,
            async_task_id: None,
            turn_context: lash::TurnContext::default(),
            tool_call_id: None,
        };

        let result = provider
            .execute_with_context(
                "search_tools",
                &json!({
                    "query": "",
                    "exclude": ["read_file"],
                    "limit": 2,
                }),
                &context,
            )
            .await;

        assert!(result.success, "{result:?}");
        let results = result.result.as_array().expect("search result list");
        assert_eq!(ranked_names(results), vec!["search_web"]);
        let requests = host
            .direct_requests
            .lock()
            .expect("direct requests lock poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "medium");
        assert!(
            requests[0]
                .messages
                .iter()
                .flat_map(|message| message.parts.iter())
                .any(|part| matches!(
                    part,
                    DirectPart::Text(text)
                        if text.contains("\"exclude\":[\"read_file\"]")
                            && !text.contains("\"name\":\"read_file\"")
                ))
        );
        let DirectOutputSpec::JsonSchema(schema) = &requests[0].output else {
            panic!("expected json schema output");
        };
        assert_eq!(
            schema.schema["properties"]["tool_names"]["items"]["enum"],
            json!(["search_web"])
        );
    }

    #[tokio::test]
    async fn load_tools_reports_callable_undocumented_as_already_callable() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                callable_undocumented_tool("mcp__appworld__spotify_search_songs"),
                catalog_tool("fetch_url", "Fetch a URL"),
            ],
            promoted: Mutex::default(),
            ..Default::default()
        });
        let provider = ToolDiscoveryToolsProvider::new();
        let context = ToolExecutionContext {
            session_id: "session".to_string(),
            host: host.clone(),
            cancellation_token: None,
            async_task_id: None,
            turn_context: lash::TurnContext::default(),
            tool_call_id: None,
        };
        let result = provider
            .execute_with_context(
                "load_tools",
                &json!({
                    "names": ["mcp__appworld__spotify_search_songs", "fetch_url"]
                }),
                &context,
            )
            .await;
        assert!(result.success, "{result:?}");
        assert_eq!(
            result.result["already_callable"],
            json!(["mcp__appworld__spotify_search_songs"])
        );
        assert_eq!(result.result["loaded"], json!(["fetch_url"]));
        assert_eq!(
            *host.promoted.lock().expect("promoted lock poisoned"),
            vec!["fetch_url".to_string()]
        );
    }

    #[test]
    fn search_tools_has_typed_result_schema() {
        let definition = search_tools_definition();

        assert_eq!(definition.output_schema["type"], json!("array"));
        let item = &definition.output_schema["items"];
        assert_eq!(item["type"], json!("object"));
        assert_eq!(item["required"], json!(["name", "signature"]));
        let properties = item["properties"].as_object().expect("properties");
        assert!(properties.contains_key("description"));
        assert!(properties.contains_key("examples"));
        assert!(!properties.contains_key("returns"));

        let rendered_signature = definition.compact_contract().render_signature();
        assert!(
            rendered_signature.starts_with("search_tools(query: str"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("-> list[record{"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("name: str"),
            "{rendered_signature}"
        );
        assert!(
            rendered_signature.contains("signature: str"),
            "{rendered_signature}"
        );
        assert!(
            !rendered_signature.contains("returns: str"),
            "{rendered_signature}"
        );
    }

    #[test]
    fn rlm_surface_hides_load_tools_and_promotes_discoverable_tools() {
        let tools = vec![
            search_tools_definition(),
            load_tools_definition(),
            ToolDefinition::new(
                "fetch_url",
                "Fetch URL",
                ToolDefinition::default_input_schema(),
                serde_json::json!({ "type": "string" }),
            )
            .with_availability(ToolAvailabilityConfig::same(ToolAvailability::Discoverable))
            .with_activation(ToolActivation::Loadable)
            .with_execution_mode(ToolExecutionMode::Parallel),
        ];
        let mode = ExecutionMode::new("rlm");
        let contribution = rlm_tool_surface(ToolSurfaceContext {
            session_id: "session".to_string(),
            mode: mode.clone(),
            tools: tools.clone(),
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
        })
        .unwrap();
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools,
            mode,
            contributions: vec![contribution],
        });

        assert_eq!(
            surface.tool_availability("load_tools"),
            Some(ToolAvailability::Hidden)
        );
        assert_eq!(
            surface.tool_availability("search_tools"),
            Some(ToolAvailability::Documented)
        );
        assert_eq!(
            surface.tool_availability("fetch_url"),
            Some(ToolAvailability::Callable)
        );
        let docs = surface.prompt_tool_docs();
        assert!(docs.contains("Catalogued tools: 1 not showcased here"));
        assert!(docs.contains("default: fetch_url"));
    }

    #[test]
    fn rlm_surface_hides_search_tools_when_there_is_no_catalogue() {
        let tools = vec![search_tools_definition(), load_tools_definition()];
        let mode = ExecutionMode::new("rlm");
        let contribution = rlm_tool_surface(ToolSurfaceContext {
            session_id: "session".to_string(),
            mode: mode.clone(),
            tools: tools.clone(),
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
        })
        .unwrap();
        let surface = build_tool_surface(ToolSurfaceBuildInput {
            tools,
            mode,
            contributions: vec![contribution],
        });

        assert_eq!(
            surface.tool_availability("search_tools"),
            Some(ToolAvailability::Hidden)
        );
        assert_eq!(
            surface.tool_availability("load_tools"),
            Some(ToolAvailability::Hidden)
        );
        assert!(!surface.prompt_tool_docs().contains("search_tools"));
        assert_eq!(surface.omitted_tool_count(), 0);
    }
}
