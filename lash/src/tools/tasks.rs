use std::sync::Arc;

use serde_json::json;

use crate::store::Store;
use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

pub struct TaskStore {
    store: Arc<Store>,
}

impl TaskStore {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    fn err(msg: impl Into<String>) -> ToolResult {
        ToolResult::err(json!(msg.into()))
    }

    fn agent_id(args: &serde_json::Value) -> String {
        args.get("__agent_id__")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("root")
            .to_string()
    }

    fn task_to_json(t: &crate::store::TaskEntry, caller: &str) -> serde_json::Value {
        let owner_display = if !t.owner.is_empty() && t.owner == caller {
            "you".to_string()
        } else {
            t.owner.clone()
        };
        let mut obj = serde_json::Map::from_iter([
            ("__type__".to_string(), json!("task")),
            ("id".to_string(), json!(t.id)),
            ("subject".to_string(), json!(t.subject)),
            ("status".to_string(), json!(t.status)),
            ("priority".to_string(), json!(t.priority)),
        ]);
        if !t.description.is_empty() {
            obj.insert("description".to_string(), json!(t.description));
        }
        if !t.active_form.is_empty() {
            obj.insert("active_form".to_string(), json!(t.active_form));
        }
        if !owner_display.is_empty() {
            obj.insert("owner".to_string(), json!(owner_display));
        }
        if !t.blocks.is_empty() {
            obj.insert("blocks".to_string(), json!(t.blocks));
        }
        if !t.blocked_by.is_empty() {
            obj.insert("blocked_by".to_string(), json!(t.blocked_by));
        }
        if !t.metadata.as_object().is_some_and(|map| map.is_empty()) && !t.metadata.is_null() {
            obj.insert("metadata".to_string(), t.metadata.clone());
        }
        serde_json::Value::Object(obj)
    }

    fn execute_create(&self, args: &serde_json::Value) -> ToolResult {
        let caller = Self::agent_id(args);
        let subject = match args.get("subject").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s,
            _ => return Self::err("Missing required parameter: subject"),
        };

        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let status = args
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");
        let priority = args
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");
        let active_form = args
            .get("active_form")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

        if !matches!(priority, "high" | "medium" | "low") {
            return Self::err(format!(
                "Invalid priority '{}': must be high, medium, or low",
                priority
            ));
        }
        if !matches!(
            status,
            "pending" | "in_progress" | "completed" | "cancelled"
        ) {
            return Self::err(format!(
                "Invalid status '{}': must be pending, in_progress, completed, or cancelled",
                status
            ));
        }

        let id = self.store.next_task_id();
        let owner = if status == "in_progress" {
            caller.as_str()
        } else {
            ""
        };
        let entry = self.store.create_task_with_state(
            &id,
            subject,
            description,
            status,
            priority,
            active_form,
            owner,
            &metadata,
        );

        ToolResult::ok(Self::task_to_json(&entry, &caller))
    }

    fn execute_start(&self, args: &serde_json::Value) -> ToolResult {
        let mut payload = args.clone();
        let Some(obj) = payload.as_object_mut() else {
            return Self::err("Invalid parameters: expected object");
        };
        obj.insert("status".to_string(), json!("in_progress"));
        self.execute_create(&payload)
    }

    fn execute_tasks(&self, args: &serde_json::Value) -> ToolResult {
        let caller = Self::agent_id(args);
        let status_filter = args.get("status").and_then(|v| v.as_str());
        let blocked_filter = args.get("blocked").and_then(|v| v.as_bool());

        let items: Vec<serde_json::Value> = self
            .store
            .list_tasks(status_filter, blocked_filter)
            .iter()
            .map(|t| Self::task_to_json(t, &caller))
            .collect();

        ToolResult::ok(json!({ "__type__": "task_list", "items": items }))
    }

    fn execute_get(&self, args: &serde_json::Value) -> ToolResult {
        let caller = Self::agent_id(args);
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Self::err("Missing required parameter: id"),
        };

        match self.store.get_task(id) {
            Some(entry) => ToolResult::ok(Self::task_to_json(&entry, &caller)),
            None => Self::err(format!("Task not found: {}", id)),
        }
    }

    fn execute_update(&self, args: &serde_json::Value) -> ToolResult {
        let caller = Self::agent_id(args);
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Self::err("Missing required parameter: id"),
        };

        if let Some(status) = args.get("status").and_then(|v| v.as_str())
            && !matches!(
                status,
                "pending" | "in_progress" | "completed" | "cancelled"
            )
        {
            return Self::err(format!(
                "Invalid status '{}': must be pending, in_progress, completed, or cancelled",
                status
            ));
        }
        if let Some(priority) = args.get("priority").and_then(|v| v.as_str())
            && !matches!(priority, "high" | "medium" | "low")
        {
            return Self::err(format!(
                "Invalid priority '{}': must be high, medium, or low",
                priority
            ));
        }

        let metadata_merge = args.get("metadata").and_then(|v| v.as_object());

        let entry = self.store.update_task(
            id,
            args.get("subject").and_then(|v| v.as_str()),
            args.get("description").and_then(|v| v.as_str()),
            args.get("status").and_then(|v| v.as_str()),
            args.get("priority").and_then(|v| v.as_str()),
            args.get("active_form").and_then(|v| v.as_str()),
            metadata_merge,
        );

        let Some(_entry) = entry else {
            return Self::err(format!("Task not found: {}", id));
        };

        if let Some(ids) = args.get("add_blocks").and_then(|v| v.as_array()) {
            for id_val in ids {
                if let Some(bid) = id_val.as_str() {
                    self.store.add_dep(id, bid);
                }
            }
        }
        if let Some(ids) = args.get("add_blocked_by").and_then(|v| v.as_array()) {
            for id_val in ids {
                if let Some(bid) = id_val.as_str() {
                    self.store.add_dep(bid, id);
                }
            }
        }
        if let Some(ids) = args.get("remove_blocks").and_then(|v| v.as_array()) {
            for id_val in ids {
                if let Some(bid) = id_val.as_str() {
                    self.store.remove_dep(id, bid);
                }
            }
        }
        if let Some(ids) = args.get("remove_blocked_by").and_then(|v| v.as_array()) {
            for id_val in ids {
                if let Some(bid) = id_val.as_str() {
                    self.store.remove_dep(bid, id);
                }
            }
        }

        match self.store.get_task(id) {
            Some(updated) => ToolResult::ok(Self::task_to_json(&updated, &caller)),
            None => Self::err(format!("Task not found after update: {}", id)),
        }
    }

    fn execute_claim(&self, args: &serde_json::Value) -> ToolResult {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let owner = Self::agent_id(args);

        match self.store.claim_task(id, &owner) {
            Ok(entry) => ToolResult::ok(Self::task_to_json(&entry, &owner)),
            Err(msg) => Self::err(msg),
        }
    }

    fn execute_delete(&self, args: &serde_json::Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Self::err("Missing required parameter: id"),
        };

        if !self.store.delete_task(id) {
            return Self::err(format!("Task not found: {}", id));
        }

        ToolResult::ok(json!(null))
    }

    fn execute_summary(&self) -> ToolResult {
        ToolResult::ok(json!(self.store.task_summary()))
    }
}

#[async_trait::async_trait]
impl ToolProvider for TaskStore {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "create_task".into(),
                description: vec![crate::ToolText::new(
                    "Create a new task. Required field: `subject` (not `title`). Use this for queued or not-yet-started work; for work you are starting now, prefer `start_task(...)`. Returns the Task object.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("subject", "str"),
                    ToolParam {
                        name: "description".into(),
                        r#type: "str".into(),
                        description: "Detailed description of what needs to be done".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "status".into(),
                        r#type: "str".into(),
                        description:
                            "pending, in_progress, completed, or cancelled (default: pending)"
                                .into(),
                        required: false,
                    },
                    ToolParam {
                        name: "priority".into(),
                        r#type: "str".into(),
                        description: "high, medium, or low (default: medium)".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "active_form".into(),
                        r#type: "str".into(),
                        description:
                            "Present-continuous label shown in spinner (e.g. 'Fixing auth')".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "metadata".into(),
                        r#type: "dict".into(),
                        description: "Arbitrary key/value metadata".into(),
                        required: false,
                    },
                ],
                returns: "Task".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "start_task".into(),
                description: vec![crate::ToolText::new(
                    "Create a new task and immediately mark it `in_progress` for the current agent. Use this as the default way to begin substantial work.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("subject", "str"),
                    ToolParam {
                        name: "description".into(),
                        r#type: "str".into(),
                        description: "Detailed description of what needs to be done".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "priority".into(),
                        r#type: "str".into(),
                        description: "high, medium, or low (default: medium)".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "active_form".into(),
                        r#type: "str".into(),
                        description:
                            "Present-continuous label shown in spinner (e.g. 'Fixing auth')".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "metadata".into(),
                        r#type: "dict".into(),
                        description: "Arbitrary key/value metadata".into(),
                        required: false,
                    },
                ],
                returns: "Task".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "tasks".into(),
                description: vec![crate::ToolText::new(
                    "List tasks. Returns `{items: [...]}`. Optional filters: `status`, `blocked`.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam {
                        name: "status".into(),
                        r#type: "str".into(),
                        description: "Filter by status: pending, in_progress, completed, cancelled"
                            .into(),
                        required: false,
                    },
                    ToolParam {
                        name: "blocked".into(),
                        r#type: "bool".into(),
                        description:
                            "Filter by blocked state: True for blocked tasks, False for unblocked"
                                .into(),
                        required: false,
                    },
                ],
                returns: "dict".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "get_task".into(),
                description: vec![crate::ToolText::new(
                    "Get a single task by ID.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![ToolParam::typed("id", "str")],
                returns: "Task".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "update_task".into(),
                description: vec![crate::ToolText::new(
                    "Update a task. Pass only changed fields. Valid `status`: `pending`, `in_progress`, `completed`, `cancelled`.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("subject", "str"),
                    ToolParam::optional("description", "str"),
                    ToolParam {
                        name: "status".into(),
                        r#type: "str".into(),
                        description: "pending, in_progress, completed, or cancelled".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "priority".into(),
                        r#type: "str".into(),
                        description: "high, medium, or low".into(),
                        required: false,
                    },
                    ToolParam::optional("active_form", "str"),
                    ToolParam {
                        name: "metadata".into(),
                        r#type: "dict".into(),
                        description: "Merge into existing metadata (set key to null to delete)"
                            .into(),
                        required: false,
                    },
                    ToolParam {
                        name: "add_blocks".into(),
                        r#type: "list".into(),
                        description: "Task IDs that this task blocks".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "add_blocked_by".into(),
                        r#type: "list".into(),
                        description: "Task IDs that block this task".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "remove_blocks".into(),
                        r#type: "list".into(),
                        description: "Task IDs to remove from blocks".into(),
                        required: false,
                    },
                    ToolParam {
                        name: "remove_blocked_by".into(),
                        r#type: "list".into(),
                        description: "Task IDs to remove from blocked_by".into(),
                        required: false,
                    },
                ],
                returns: "Task".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "claim_task".into(),
                description: vec![crate::ToolText::new(
                    "Claim an existing unclaimed task and mark it `in_progress`. Use this for adopting queued work; for a brand-new task, prefer `start_task(...)`.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![ToolParam {
                    name: "id".into(),
                    r#type: "str".into(),
                    description: "Task ID to claim (omit to auto-pick next available)".into(),
                    required: false,
                }],
                returns: "Task".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "delete_task".into(),
                description: vec![crate::ToolText::new(
                    "Permanently remove a task. This is mainly for cleanup or tiny smoke-test hygiene; normal workflow should usually keep the task and mark it completed or cancelled.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![ToolParam::typed("id", "str")],
                returns: "None".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "tasks_summary".into(),
                description: vec![crate::ToolText::new(
                    "Get a formatted task summary: counts by status, blocked tasks, and high-priority items.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![],
                returns: "str".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "create_task" => self.execute_create(args),
            "start_task" => self.execute_start(args),
            "tasks" => self.execute_tasks(args),
            "get_task" => self.execute_get(args),
            "update_task" => self.execute_update(args),
            "claim_task" => self.execute_claim(args),
            "delete_task" => self.execute_delete(args),
            "tasks_summary" => self.execute_summary(),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolProvider;
    use serde_json::json;

    fn make_store() -> TaskStore {
        TaskStore::new(Arc::new(Store::memory().unwrap()))
    }

    #[tokio::test]
    async fn test_create_task() {
        let store = make_store();
        let result = store
            .execute("create_task", &json!({"subject": "Fix bug"}))
            .await;
        assert!(result.success);
        let task = &result.result;
        assert_eq!(task["subject"], "Fix bug");
        assert_eq!(task["status"], "pending");
    }

    #[tokio::test]
    async fn test_create_task_can_start_in_progress() {
        let store = make_store();
        let result = store
            .execute(
                "create_task",
                &json!({
                    "__agent_id__": "root",
                    "subject": "Fix bug",
                    "status": "in_progress"
                }),
            )
            .await;
        assert!(result.success);
        let task = &result.result;
        assert_eq!(task["subject"], "Fix bug");
        assert_eq!(task["status"], "in_progress");
        assert_eq!(task["owner"], "you");
    }

    #[tokio::test]
    async fn test_start_task_starts_in_progress() {
        let store = make_store();
        let result = store
            .execute(
                "start_task",
                &json!({
                    "__agent_id__": "root",
                    "subject": "Fix bug"
                }),
            )
            .await;
        assert!(result.success);
        let task = &result.result;
        assert_eq!(task["subject"], "Fix bug");
        assert_eq!(task["status"], "in_progress");
        assert_eq!(task["owner"], "you");
    }

    #[tokio::test]
    async fn test_task_json_omits_empty_fields() {
        let store = make_store();
        let result = store
            .execute("create_task", &json!({"subject": "Fix bug"}))
            .await;
        assert!(result.success);
        let task = result.result.as_object().expect("task object");
        assert!(!task.contains_key("owner"));
        assert!(!task.contains_key("blocks"));
        assert!(!task.contains_key("blocked_by"));
        assert!(!task.contains_key("metadata"));
        assert!(!task.contains_key("description"));
        assert!(!task.contains_key("active_form"));
    }

    #[tokio::test]
    async fn test_list_tasks() {
        let store = make_store();
        store
            .execute("create_task", &json!({"subject": "Task A"}))
            .await;
        store
            .execute("create_task", &json!({"subject": "Task B"}))
            .await;
        let result = store.execute("tasks", &json!({})).await;
        assert!(result.success);
        let items = result.result["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn test_update_task() {
        let store = make_store();
        let created = store
            .execute("create_task", &json!({"subject": "Task"}))
            .await;
        let id = created.result["id"].as_str().unwrap().to_string();
        let result = store
            .execute("update_task", &json!({"id": id, "status": "completed"}))
            .await;
        assert!(result.success);
        assert_eq!(result.result["status"], "completed");
    }

    #[tokio::test]
    async fn test_delete_task() {
        let store = make_store();
        let created = store
            .execute("create_task", &json!({"subject": "Doomed"}))
            .await;
        let id = created.result["id"].as_str().unwrap().to_string();
        let del = store.execute("delete_task", &json!({"id": id})).await;
        assert!(del.success);
        let get = store.execute("get_task", &json!({"id": id})).await;
        assert!(!get.success);
    }

    #[tokio::test]
    async fn test_tasks_summary() {
        let store = make_store();
        store.execute("create_task", &json!({"subject": "A"})).await;
        let result = store.execute("tasks_summary", &json!({})).await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("pending"));
    }

    // ── claim_task ──

    #[tokio::test]
    async fn test_claim_task() {
        let store = make_store();
        let created = store.execute("create_task", &json!({"subject": "T"})).await;
        let id = created.result["id"].as_str().unwrap().to_string();
        let result = store
            .execute("claim_task", &json!({"id": id, "__agent_id__": "agent-1"}))
            .await;
        assert!(result.success);
        assert_eq!(result.result["owner"], "you");
        assert_eq!(result.result["status"], "in_progress");
    }

    #[tokio::test]
    async fn test_claim_task_double_claim_different_owner() {
        let store = make_store();
        let created = store.execute("create_task", &json!({"subject": "T"})).await;
        let id = created.result["id"].as_str().unwrap().to_string();
        store
            .execute("claim_task", &json!({"id": id, "__agent_id__": "agent-1"}))
            .await;
        let result = store
            .execute("claim_task", &json!({"id": id, "__agent_id__": "agent-2"}))
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_claim_blocked_task() {
        let store = make_store();
        let t1 = store
            .execute("create_task", &json!({"subject": "Blocker"}))
            .await;
        let t2 = store
            .execute("create_task", &json!({"subject": "Blocked"}))
            .await;
        let id1 = t1.result["id"].as_str().unwrap().to_string();
        let id2 = t2.result["id"].as_str().unwrap().to_string();
        store
            .execute("update_task", &json!({"id": id2, "add_blocked_by": [id1]}))
            .await;
        let result = store
            .execute("claim_task", &json!({"id": id2, "owner": "a"}))
            .await;
        assert!(!result.success);
    }

    // ── Dependency management via update_task ──

    #[tokio::test]
    async fn test_add_and_remove_deps() {
        let store = make_store();
        let t1 = store.execute("create_task", &json!({"subject": "A"})).await;
        let t2 = store.execute("create_task", &json!({"subject": "B"})).await;
        let id1 = t1.result["id"].as_str().unwrap().to_string();
        let id2 = t2.result["id"].as_str().unwrap().to_string();

        // add_blocks: t1 blocks t2
        let updated = store
            .execute("update_task", &json!({"id": id1, "add_blocks": [id2]}))
            .await;
        assert!(
            updated.result["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == &json!(id2))
        );

        // remove_blocks
        let updated = store
            .execute("update_task", &json!({"id": id1, "remove_blocks": [id2]}))
            .await;
        assert!(
            updated.result["blocks"].is_null()
                || updated.result["blocks"]
                    .as_array()
                    .is_some_and(|items| items.is_empty())
        );

        // add_blocked_by
        let updated = store
            .execute("update_task", &json!({"id": id2, "add_blocked_by": [id1]}))
            .await;
        assert!(
            updated.result["blocked_by"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == &json!(id1))
        );

        // remove_blocked_by
        let updated = store
            .execute(
                "update_task",
                &json!({"id": id2, "remove_blocked_by": [id1]}),
            )
            .await;
        assert!(
            updated.result["blocked_by"].is_null()
                || updated.result["blocked_by"]
                    .as_array()
                    .is_some_and(|items| items.is_empty())
        );
    }

    // ── Owner auto-clear via tool ──

    #[tokio::test]
    async fn test_owner_cleared_on_completed() {
        let store = make_store();
        let created = store.execute("create_task", &json!({"subject": "T"})).await;
        let id = created.result["id"].as_str().unwrap().to_string();
        store
            .execute("claim_task", &json!({"id": id, "owner": "agent-1"}))
            .await;
        let result = store
            .execute("update_task", &json!({"id": id, "status": "completed"}))
            .await;
        assert!(result.success);
        assert!(result.result["owner"].is_null());
    }

    // ── Validation ──

    #[tokio::test]
    async fn test_invalid_priority() {
        let store = make_store();
        let result = store
            .execute(
                "create_task",
                &json!({"subject": "T", "priority": "urgent"}),
            )
            .await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_invalid_status() {
        let store = make_store();
        let created = store.execute("create_task", &json!({"subject": "T"})).await;
        let id = created.result["id"].as_str().unwrap().to_string();
        let result = store
            .execute("update_task", &json!({"id": id, "status": "invalid"}))
            .await;
        assert!(!result.success);
    }

    // ── Summary content ──

    #[tokio::test]
    async fn test_summary_counts() {
        let store = make_store();
        store.execute("create_task", &json!({"subject": "A"})).await;
        store.execute("create_task", &json!({"subject": "B"})).await;
        let b = store.execute("create_task", &json!({"subject": "C"})).await;
        let id_c = b.result["id"].as_str().unwrap().to_string();
        store
            .execute("update_task", &json!({"id": id_c, "status": "completed"}))
            .await;
        let result = store.execute("tasks_summary", &json!({})).await;
        let text = result.result.as_str().unwrap();
        assert!(text.contains("3 total"));
        assert!(text.contains("2 pending"));
        assert!(text.contains("1 completed"));
    }
}
