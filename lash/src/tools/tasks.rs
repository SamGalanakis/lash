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

    fn task_to_json(t: &crate::store::TaskEntry) -> serde_json::Value {
        json!({
            "__type__": "task",
            "id": t.id,
            "subject": t.subject,
            "description": t.description,
            "status": t.status,
            "priority": t.priority,
            "active_form": t.active_form,
            "blocks": t.blocks,
            "blocked_by": t.blocked_by,
            "metadata": t.metadata,
        })
    }

    fn execute_create(&self, args: &serde_json::Value) -> ToolResult {
        let subject = match args.get("subject").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s,
            _ => return Self::err("Missing required parameter: subject"),
        };

        let description = args.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let priority = args.get("priority").and_then(|v| v.as_str()).unwrap_or("medium");
        let active_form = args.get("active_form").and_then(|v| v.as_str()).unwrap_or("");
        let metadata = args.get("metadata").cloned().unwrap_or_else(|| json!({}));

        if !matches!(priority, "high" | "medium" | "low") {
            return Self::err(format!("Invalid priority '{}': must be high, medium, or low", priority));
        }

        let id = self.store.next_task_id();
        let entry = self.store.create_task(&id, subject, description, priority, active_form, &metadata);

        ToolResult::ok(Self::task_to_json(&entry))
    }

    fn execute_tasks(&self, args: &serde_json::Value) -> ToolResult {
        let status_filter = args.get("status").and_then(|v| v.as_str());
        let blocked_filter = args.get("blocked").and_then(|v| v.as_bool());

        let items: Vec<serde_json::Value> = self
            .store
            .list_tasks(status_filter, blocked_filter)
            .iter()
            .map(Self::task_to_json)
            .collect();

        ToolResult::ok(json!({ "__type__": "task_list", "items": items }))
    }

    fn execute_get(&self, args: &serde_json::Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Self::err("Missing required parameter: id"),
        };

        match self.store.get_task(id) {
            Some(entry) => ToolResult::ok(Self::task_to_json(&entry)),
            None => Self::err(format!("Task not found: {}", id)),
        }
    }

    fn execute_update(&self, args: &serde_json::Value) -> ToolResult {
        let id = match args.get("id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Self::err("Missing required parameter: id"),
        };

        if let Some(status) = args.get("status").and_then(|v| v.as_str()) {
            if !matches!(status, "pending" | "in_progress" | "completed" | "cancelled") {
                return Self::err(format!(
                    "Invalid status '{}': must be pending, in_progress, completed, or cancelled",
                    status
                ));
            }
        }
        if let Some(priority) = args.get("priority").and_then(|v| v.as_str()) {
            if !matches!(priority, "high" | "medium" | "low") {
                return Self::err(format!(
                    "Invalid priority '{}': must be high, medium, or low",
                    priority
                ));
            }
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
            Some(updated) => ToolResult::ok(Self::task_to_json(&updated)),
            None => Self::err(format!("Task not found after update: {}", id)),
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
                description: "Create a new task. Returns the Task object.".into(),
                params: vec![
                    ToolParam::typed("subject", "str"),
                    ToolParam { name: "description".into(), r#type: "str".into(), description: "Detailed description of what needs to be done".into(), required: false },
                    ToolParam { name: "priority".into(), r#type: "str".into(), description: "high, medium, or low (default: medium)".into(), required: false },
                    ToolParam { name: "active_form".into(), r#type: "str".into(), description: "Present-continuous label shown in spinner (e.g. 'Fixing auth')".into(), required: false },
                    ToolParam { name: "metadata".into(), r#type: "dict".into(), description: "Arbitrary key/value metadata".into(), required: false },
                ],
                returns: "Task".into(),
                hidden: false,
            },
            ToolDefinition {
                name: "tasks".into(),
                description: "List tasks. Returns a list of Task objects. Optional filters: status, blocked.".into(),
                params: vec![
                    ToolParam { name: "status".into(), r#type: "str".into(), description: "Filter by status: pending, in_progress, completed, cancelled".into(), required: false },
                    ToolParam { name: "blocked".into(), r#type: "bool".into(), description: "Filter by blocked state: True for blocked tasks, False for unblocked".into(), required: false },
                ],
                returns: "list[Task]".into(),
                hidden: false,
            },
            ToolDefinition {
                name: "get_task".into(),
                description: "Get a single task by ID.".into(),
                params: vec![ToolParam::typed("id", "str")],
                returns: "Task".into(),
                hidden: false,
            },
            ToolDefinition {
                name: "update_task".into(),
                description: "Update a task. Pass only the fields to change. Returns updated Task.".into(),
                params: vec![
                    ToolParam::typed("id", "str"),
                    ToolParam::optional("subject", "str"),
                    ToolParam::optional("description", "str"),
                    ToolParam { name: "status".into(), r#type: "str".into(), description: "pending, in_progress, completed, or cancelled".into(), required: false },
                    ToolParam { name: "priority".into(), r#type: "str".into(), description: "high, medium, or low".into(), required: false },
                    ToolParam::optional("active_form", "str"),
                    ToolParam { name: "metadata".into(), r#type: "dict".into(), description: "Merge into existing metadata (set key to null to delete)".into(), required: false },
                    ToolParam { name: "add_blocks".into(), r#type: "list".into(), description: "Task IDs that this task blocks".into(), required: false },
                    ToolParam { name: "add_blocked_by".into(), r#type: "list".into(), description: "Task IDs that block this task".into(), required: false },
                    ToolParam { name: "remove_blocks".into(), r#type: "list".into(), description: "Task IDs to remove from blocks".into(), required: false },
                    ToolParam { name: "remove_blocked_by".into(), r#type: "list".into(), description: "Task IDs to remove from blocked_by".into(), required: false },
                ],
                returns: "Task".into(),
                hidden: false,
            },
            ToolDefinition {
                name: "delete_task".into(),
                description: "Permanently remove a task.".into(),
                params: vec![ToolParam::typed("id", "str")],
                returns: "None".into(),
                hidden: true,
            },
            ToolDefinition {
                name: "tasks_summary".into(),
                description: "Get a formatted overview of all tasks: counts by status, blocked tasks, high-priority items.".into(),
                params: vec![],
                returns: "str".into(),
                hidden: false,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "create_task" => self.execute_create(args),
            "tasks" => self.execute_tasks(args),
            "get_task" => self.execute_get(args),
            "update_task" => self.execute_update(args),
            "delete_task" => self.execute_delete(args),
            "tasks_summary" => self.execute_summary(),
            _ => ToolResult::err(json!(format!("Unknown tool: {}", name))),
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
    async fn test_list_tasks() {
        let store = make_store();
        store.execute("create_task", &json!({"subject": "Task A"})).await;
        store.execute("create_task", &json!({"subject": "Task B"})).await;
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
        let del = store
            .execute("delete_task", &json!({"id": id}))
            .await;
        assert!(del.success);
        let get = store
            .execute("get_task", &json!({"id": id}))
            .await;
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
}
