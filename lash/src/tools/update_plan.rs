use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PlanItem {
    step: String,
    status: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PlanSnapshot {
    explanation: Option<String>,
    plan: Vec<PlanItem>,
}

#[derive(Default)]
struct PlanState {
    explanation: Option<String>,
    items: Vec<PlanItem>,
}

pub struct UpdatePlanTool {
    state: Arc<Mutex<PlanState>>,
}

impl UpdatePlanTool {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(PlanState::default())),
        }
    }

    pub fn snapshot(&self) -> Result<PlanSnapshot, String> {
        let guard = self
            .state
            .lock()
            .map_err(|_| "plan state poisoned".to_string())?;
        Ok(PlanSnapshot {
            explanation: guard.explanation.clone(),
            plan: guard.items.clone(),
        })
    }

    pub fn restore(&self, snapshot: PlanSnapshot) -> Result<(), String> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| "plan state poisoned".to_string())?;
        guard.explanation = snapshot.explanation;
        guard.items = snapshot.plan;
        Ok(())
    }
}

impl Default for UpdatePlanTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ToolProvider for UpdatePlanTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "update_plan".into(),
            description: vec![crate::ToolText::new(
                "Update the task plan for substantial multi-step work. Provide an optional explanation and a list of short steps with statuses. Valid statuses: pending, in_progress, completed. At most one step can be in_progress at a time.",
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam::optional("explanation", "str"),
                ToolParam::typed("steps", "list"),
            ],
            returns: "str".into(),
            examples: vec![
                crate::ToolText::new(
                    "call update_plan { explanation: \"I found the main renderer.\", steps: [{ label: \"Inspect renderer\", status: \"completed\" }, { label: \"Patch layout\", status: \"in_progress\" }, { label: \"Run tests\", status: \"pending\" }] }",
                    [crate::ExecutionMode::Repl],
                ),
                crate::ToolText::new(
                    "{\"explanation\":\"I found the main renderer.\",\"steps\":[{\"label\":\"Inspect renderer\",\"status\":\"completed\"},{\"label\":\"Patch layout\",\"status\":\"in_progress\"},{\"label\":\"Run tests\",\"status\":\"pending\"}]}",
                    [crate::ExecutionMode::Standard],
                ),
            ],
            hidden: false,
            inject_into_prompt: true,
        }]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "update_plan" => execute_update_plan(&self.state, args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

fn execute_update_plan(state: &Arc<Mutex<PlanState>>, args: &serde_json::Value) -> ToolResult {
    let explanation = args
        .get("explanation")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let Some(raw_plan) = args.get("steps").and_then(|value| value.as_array()) else {
        return ToolResult::err_fmt("Missing required parameter: steps");
    };
    if raw_plan.is_empty() {
        return ToolResult::err_fmt("Plan must contain at least one step");
    }

    let mut items = Vec::with_capacity(raw_plan.len());
    for (idx, item) in raw_plan.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return ToolResult::err_fmt(format_args!(
                "Invalid steps[{idx}]: expected object with label and status"
            ));
        };
        let Some(step) = object
            .get("label")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid steps[{idx}].label: expected non-empty string"
            ));
        };
        let Some(status) = object
            .get("status")
            .and_then(|value| value.as_str())
            .map(str::trim)
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid steps[{idx}].status: expected string"
            ));
        };
        if !matches!(status, "pending" | "in_progress" | "completed") {
            return ToolResult::err_fmt(format_args!(
                "Invalid steps[{idx}].status: expected pending, in_progress, or completed"
            ));
        }
        items.push(PlanItem {
            step: step.to_string(),
            status: status.to_string(),
        });
    }

    let in_progress = items
        .iter()
        .filter(|item| item.status == "in_progress")
        .count();
    if in_progress > 1 {
        return ToolResult::err_fmt("Plan may contain at most one in_progress step");
    }

    let mut guard = state.lock().unwrap();
    guard.explanation = explanation.clone();
    guard.items = items.clone();
    ToolResult::ok(json!("Plan updated"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn update_plan_validates_shape() {
        let tool = UpdatePlanTool::new();
        let result = tool
            .execute(
                "update_plan",
                &json!({"steps":[{"label":"","status":"pending"}]}),
            )
            .await;
        assert!(!result.success);
        assert!(
            result
                .result
                .as_str()
                .is_some_and(|value| value.contains("steps[0].label"))
        );
    }

    #[tokio::test]
    async fn update_plan_returns_text_acknowledgement() {
        let tool = UpdatePlanTool::new();
        let result = tool
            .execute(
                "update_plan",
                &json!({
                    "explanation":"done reading",
                    "steps":[
                        {"label":"Inspect code","status":"completed"},
                        {"label":"Patch UI","status":"in_progress"}
                    ]
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(result.result.as_str(), Some("Plan updated"));
    }

    #[tokio::test]
    async fn update_plan_rejects_multiple_in_progress_steps() {
        let tool = UpdatePlanTool::new();
        let result = tool
            .execute(
                "update_plan",
                &json!({
                    "steps":[
                        {"label":"Inspect UI","status":"in_progress"},
                        {"label":"Patch layout","status":"in_progress"}
                    ]
                }),
            )
            .await;
        assert!(!result.success);
        assert!(
            result
                .result
                .as_str()
                .is_some_and(|value| value.contains("at most one in_progress"))
        );
    }

    #[test]
    fn update_plan_snapshot_round_trip() {
        let tool = UpdatePlanTool::new();
        tool.restore(PlanSnapshot {
            explanation: Some("Found the entry point.".to_string()),
            plan: vec![PlanItem {
                step: "Inspect renderer".to_string(),
                status: "completed".to_string(),
            }],
        })
        .expect("restore");
        let snapshot = tool.snapshot().expect("snapshot");
        assert_eq!(
            snapshot,
            PlanSnapshot {
                explanation: Some("Found the entry point.".to_string()),
                plan: vec![PlanItem {
                    step: "Inspect renderer".to_string(),
                    status: "completed".to_string(),
                }],
            }
        );
    }
}
