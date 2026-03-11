use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use serde_json::json;

use crate::dynamic::DynamicCapabilityDef;
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs, truncate_preview};
use crate::store::{HistoryTurnRecord, MemRecord, Store};
use crate::tools::run_blocking;
use crate::{ExecutionMode, ToolDefinition, ToolParam, ToolText};

use super::*;

fn agent_id(args: &serde_json::Value) -> String {
    args.get("__agent_id__")
        .and_then(|v| v.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root")
        .to_string()
}

fn current_turn_index(turn: &AssembledTurn) -> i64 {
    turn.state
        .messages
        .iter()
        .filter(|msg| matches!(msg.role, crate::MessageRole::User))
        .count() as i64
}

fn history_capability_def() -> DynamicCapabilityDef {
    DynamicCapabilityDef {
        id: "history".to_string(),
        name: "History".to_string(),
        description: "Persistent turn history and retrieval.".to_string(),
        prompt_section: Some(
            "## History\n\nPrior turns can be searched with `search_history(...)` when earlier context is actually needed."
                .to_string(),
        ),
        helper_bindings: BTreeSet::from(["search_history".to_string()]),
        tool_names: BTreeSet::from(["search_history".to_string()]),
        enabled_by_default: true,
    }
}

fn memory_capability_def() -> DynamicCapabilityDef {
    DynamicCapabilityDef {
        id: "memory".to_string(),
        name: "Memory".to_string(),
        description: "Persistent key-value memory and retrieval.".to_string(),
        prompt_section: Some(
            "## Memory\n\nUse `mem_set`, `mem_get`, `mem_delete`, `mem_all`, and `search_mem(...)` for durable decisions that should survive context pruning."
                .to_string(),
        ),
        helper_bindings: BTreeSet::from([
            "search_mem".to_string(),
            "mem_set".to_string(),
            "mem_get".to_string(),
            "mem_delete".to_string(),
            "mem_all".to_string(),
        ]),
        tool_names: BTreeSet::from([
            "search_mem".to_string(),
            "mem_set".to_string(),
            "mem_get".to_string(),
            "mem_delete".to_string(),
            "mem_all".to_string(),
        ]),
        enabled_by_default: true,
    }
}

pub fn builtin_dynamic_capability_defs() -> BTreeMap<String, DynamicCapabilityDef> {
    let mut defs = BTreeMap::new();
    for def in [history_capability_def(), memory_capability_def()] {
        defs.insert(def.id.clone(), def);
    }
    defs
}

#[derive(Clone)]
struct HistoryTools {
    store: Arc<Store>,
}

impl HistoryTools {
    fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    fn search_history(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = limit_from_args(args);
        let since_turn = args.get("since_turn").and_then(|v| v.as_i64());

        let selected_fields: HashSet<String> = args
            .get("fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_else(|| {
                [
                    "user_message".to_string(),
                    "code".to_string(),
                    "prose".to_string(),
                    "output".to_string(),
                    "tool_calls".to_string(),
                ]
                .into_iter()
                .collect()
            });

        let turns: Vec<HistoryTurnRecord> = self
            .store
            .history_export(&agent_id)
            .into_iter()
            .filter(|turn| since_turn.is_none_or(|min_turn| turn.index >= min_turn))
            .collect();

        let docs: Vec<SearchDoc> = turns
            .iter()
            .map(|turn| {
                let mut fields = HashMap::new();
                if selected_fields.contains("user_message") {
                    fields.insert("user_message", turn.user_message.clone());
                }
                if selected_fields.contains("code") {
                    fields.insert("code", turn.code.clone());
                }
                if selected_fields.contains("prose") {
                    fields.insert("prose", turn.prose.clone());
                }
                if selected_fields.contains("output") {
                    fields.insert("output", turn.output.clone());
                }
                if selected_fields.contains("tool_calls") {
                    let tool_calls = serde_json::to_string(&turn.tool_calls).unwrap_or_default();
                    fields.insert("tool_calls", tool_calls);
                }
                SearchDoc { fields }
            })
            .collect();

        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[
                ("user_message", 3.0),
                ("prose", 2.0),
                ("code", 2.0),
                ("output", 1.0),
                ("tool_calls", 1.0),
            ],
        );

        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, hits)| {
                let turn = &turns[idx];
                let preview_source = if !turn.user_message.trim().is_empty() {
                    &turn.user_message
                } else if !turn.prose.trim().is_empty() {
                    &turn.prose
                } else {
                    &turn.output
                };
                json!({
                    "turn": turn.index,
                    "score": score,
                    "field_hits": hits,
                    "preview": truncate_preview(preview_source, 220),
                    "tool_calls": turn.tool_calls,
                    "files_read": turn.files_read,
                    "files_written": turn.files_written,
                })
            })
            .collect();
        ToolResult::ok(json!(items))
    }
}

#[async_trait::async_trait]
impl ToolProvider for HistoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search_history".into(),
            description: vec![ToolText::new(
                "Search turn history using hybrid/literal/regex matching.",
                [ExecutionMode::Repl, ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam::typed("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("fields", "list"),
                ToolParam::optional("since_turn", "int"),
            ],
            returns: "list".into(),
            examples: vec![],
            hidden: false,
            inject_into_prompt: false,
        }]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let tools = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "search_history" => tools.search_history(&args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

#[derive(Clone)]
struct MemoryTools {
    store: Arc<Store>,
}

impl MemoryTools {
    fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    fn search_mem(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = limit_from_args(args);
        let key_filter: Option<HashSet<String>> =
            args.get("keys").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect::<HashSet<_>>()
            });

        let entries: Vec<MemRecord> = self
            .store
            .mem_export(&agent_id)
            .into_iter()
            .filter(|entry| {
                key_filter
                    .as_ref()
                    .is_none_or(|keys| keys.contains(&entry.key))
            })
            .collect();

        let docs: Vec<SearchDoc> = entries
            .iter()
            .map(|entry| {
                let mut fields = HashMap::new();
                fields.insert("key", entry.key.clone());
                fields.insert("description", entry.description.clone());
                fields.insert("value", entry.value.clone());
                SearchDoc { fields }
            })
            .collect();
        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[("key", 4.0), ("description", 2.0), ("value", 1.0)],
        );
        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, field_hits)| {
                let entry = &entries[idx];
                json!({
                    "key": entry.key,
                    "description": entry.description,
                    "value": entry.value,
                    "turn": entry.turn,
                    "score": score,
                    "field_hits": field_hits,
                })
            })
            .collect();
        ToolResult::ok(json!(items))
    }

    fn mem_set(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let key = args.get("key").and_then(|v| v.as_str()).unwrap_or_default();
        if key.is_empty() {
            return ToolResult::err(json!("Missing required parameter: key"));
        }
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or(description);
        self.store.mem_set(&agent_id, key, description, value);
        ToolResult::ok(json!(null))
    }

    fn mem_get(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let key = args.get("key").and_then(|v| v.as_str()).unwrap_or_default();
        if key.is_empty() {
            return ToolResult::err(json!("Missing required parameter: key"));
        }
        match self.store.mem_get(&agent_id, key) {
            Some(entry) => ToolResult::ok(json!({
                "key": entry.key,
                "description": entry.description,
                "value": entry.value,
                "turn": entry.turn,
            })),
            None => ToolResult::ok(json!(null)),
        }
    }

    fn mem_delete(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let key = args.get("key").and_then(|v| v.as_str()).unwrap_or_default();
        if key.is_empty() {
            return ToolResult::err(json!("Missing required parameter: key"));
        }
        let _ = self.store.mem_delete(&agent_id, key);
        ToolResult::ok(json!(null))
    }

    fn mem_all(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let entries = self
            .store
            .mem_export(&agent_id)
            .into_iter()
            .map(|entry| {
                json!({
                    "key": entry.key,
                    "description": entry.description,
                    "value": entry.value,
                    "turn": entry.turn,
                })
            })
            .collect::<Vec<_>>();
        ToolResult::ok(json!(entries))
    }
}

#[async_trait::async_trait]
impl ToolProvider for MemoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "search_mem".into(),
                description: vec![ToolText::new(
                    "Search persistent memory using hybrid/literal/regex matching.",
                    [ExecutionMode::Repl, ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("query", "str"),
                    ToolParam::optional("mode", "str"),
                    ToolParam::optional("regex", "str"),
                    ToolParam::optional("limit", "int"),
                    ToolParam::optional("keys", "list"),
                ],
                returns: "list".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "mem_set".into(),
                description: vec![ToolText::new(
                    "Store or update a persistent memory entry.",
                    [ExecutionMode::Repl, ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("key", "str"),
                    ToolParam::typed("description", "str"),
                    ToolParam::optional("value", "str"),
                ],
                returns: "None".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "mem_get".into(),
                description: vec![ToolText::new(
                    "Fetch a persistent memory entry by key.",
                    [ExecutionMode::Repl, ExecutionMode::Standard],
                )],
                params: vec![ToolParam::typed("key", "str")],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "mem_delete".into(),
                description: vec![ToolText::new(
                    "Delete a persistent memory entry by key.",
                    [ExecutionMode::Repl, ExecutionMode::Standard],
                )],
                params: vec![ToolParam::typed("key", "str")],
                returns: "None".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "mem_all".into(),
                description: vec![ToolText::new(
                    "List all persistent memory entries.",
                    [ExecutionMode::Repl, ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "list".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let tools = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "search_mem" => tools.search_mem(&args),
            "mem_set" => tools.mem_set(&args),
            "mem_get" => tools.mem_get(&args),
            "mem_delete" => tools.mem_delete(&args),
            "mem_all" => tools.mem_all(&args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

pub struct HistoryPluginFactory {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
}

impl HistoryPluginFactory {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            tools: Arc::new(HistoryTools::new(Arc::clone(&store))),
            store,
        }
    }
}

impl PluginFactory for HistoryPluginFactory {
    fn id(&self) -> &'static str {
        "history"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(HistoryPlugin {
            store: Arc::clone(&self.store),
            tools: Arc::clone(&self.tools),
            agent_id: ctx.agent_id.clone(),
        }))
    }
}

struct HistoryPlugin {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
    agent_id: String,
}

impl SessionPlugin for HistoryPlugin {
    fn id(&self) -> &'static str {
        "history"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.register_tool_provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)?;
        reg.register_capability(history_capability_def())?;

        let store = Arc::clone(&self.store);
        let agent_id = self.agent_id.clone();
        reg.on_turn_committed(Arc::new(move |turn| {
            let store = Arc::clone(&store);
            let agent_id = agent_id.clone();
            Box::pin(async move {
                store.history_add_turn(
                    &agent_id,
                    &crate::runtime::latest_turn_history_payload(&turn),
                );
                Ok(())
            })
        }));

        let store = Arc::clone(&self.store);
        reg.register_external_op(
            ExternalOpDef {
                name: "history.summary".to_string(),
                description: "Summarize the recent turn history for a session.".to_string(),
                kind: ExternalOpKind::Query,
                session_param: SessionParam::Required,
                input_schema: json!({
                    "type":"object",
                    "properties":{"limit":{"type":"integer","minimum":1}},
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type":"object",
                    "properties":{
                        "session_id":{"type":"string"},
                        "turn_count":{"type":"integer"},
                        "latest_user_message":{"type":["string","null"]},
                        "recent_turns":{"type":"array"}
                    },
                    "required":["session_id","turn_count","recent_turns"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, args| {
                let store = Arc::clone(&store);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("history.summary requires session_id"));
                    };
                    let limit = args
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(5)
                        .clamp(1, 20);
                    let turns = store.history_export(&session_id);
                    let latest_user_message = turns.iter().rev().find_map(|turn| {
                        (!turn.user_message.trim().is_empty()).then_some(turn.user_message.clone())
                    });
                    let recent_turns = turns
                        .iter()
                        .rev()
                        .take(limit)
                        .map(|turn| {
                            let preview_source = if !turn.user_message.trim().is_empty() {
                                &turn.user_message
                            } else if !turn.prose.trim().is_empty() {
                                &turn.prose
                            } else {
                                &turn.output
                            };
                            json!({
                                "turn": turn.index,
                                "preview": truncate_preview(preview_source, 180),
                                "tool_calls": turn.tool_calls.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    ToolResult::ok(json!({
                        "session_id": session_id,
                        "turn_count": turns.len(),
                        "latest_user_message": latest_user_message,
                        "recent_turns": recent_turns,
                    }))
                })
            }),
        )?;
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let turns = self
            .store
            .history_export(&self.agent_id)
            .into_iter()
            .map(|turn| {
                json!({
                    "index": turn.index,
                    "user_message": turn.user_message,
                    "prose": turn.prose,
                    "code": turn.code,
                    "output": turn.output,
                    "error": turn.error,
                    "tool_calls": turn.tool_calls,
                    "files_read": turn.files_read,
                    "files_written": turn.files_written,
                })
            })
            .collect::<Vec<_>>();
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({ "turns": turns })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let turns = meta
            .state
            .as_ref()
            .and_then(|state| state.get("turns"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        self.store.history_load(&self.agent_id, &turns);
        Ok(())
    }
}

pub struct MemoryPluginFactory {
    store: Arc<Store>,
    tools: Arc<MemoryTools>,
}

impl MemoryPluginFactory {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            tools: Arc::new(MemoryTools::new(Arc::clone(&store))),
            store,
        }
    }
}

impl PluginFactory for MemoryPluginFactory {
    fn id(&self) -> &'static str {
        "memory"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(MemoryPlugin {
            store: Arc::clone(&self.store),
            tools: Arc::clone(&self.tools),
            agent_id: ctx.agent_id.clone(),
        }))
    }
}

struct MemoryPlugin {
    store: Arc<Store>,
    tools: Arc<MemoryTools>,
    agent_id: String,
}

impl SessionPlugin for MemoryPlugin {
    fn id(&self) -> &'static str {
        "memory"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.register_tool_provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)?;
        reg.register_capability(memory_capability_def())?;

        let store = Arc::clone(&self.store);
        let agent_id = self.agent_id.clone();
        reg.on_turn_committed(Arc::new(move |turn| {
            let store = Arc::clone(&store);
            let agent_id = agent_id.clone();
            Box::pin(async move {
                store.mem_set_turn(&agent_id, current_turn_index(&turn));
                Ok(())
            })
        }));
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let entries = self
            .store
            .mem_export(&self.agent_id)
            .into_iter()
            .map(|entry| {
                json!({
                    "key": entry.key,
                    "description": entry.description,
                    "value": entry.value,
                    "turn": entry.turn,
                })
            })
            .collect::<Vec<_>>();
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({ "entries": entries })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let entries = meta
            .state
            .as_ref()
            .and_then(|state| state.get("entries"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        self.store.mem_load(&self.agent_id, &entries);
        Ok(())
    }
}

pub use HistoryPluginFactory as BuiltinHistoryPluginFactory;
pub use MemoryPluginFactory as BuiltinMemoryPluginFactory;
