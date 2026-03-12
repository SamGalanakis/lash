use std::sync::Mutex;

use serde_json::json;

use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

#[derive(Clone, Debug, serde::Serialize)]
struct PlanItem {
    step: String,
    status: String,
}

#[derive(Default)]
struct PlanState {
    explanation: Option<String>,
    items: Vec<PlanItem>,
}

pub struct UpdatePlanTool {
    state: Mutex<PlanState>,
}

impl UpdatePlanTool {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(PlanState::default()),
        }
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
                "Update the working plan with short steps and statuses. Use for substantial multi-step work. Valid statuses: pending, in_progress, completed.",
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam::optional("explanation", "str"),
                ToolParam::typed("plan", "list"),
            ],
            returns: "dict".into(),
            examples: vec![
                crate::ToolText::new(
                    "update_plan(explanation=\"I found the main renderer.\", plan=[{\"step\":\"Inspect renderer\", \"status\":\"completed\"}, {\"step\":\"Patch layout\", \"status\":\"in_progress\"}, {\"step\":\"Run tests\", \"status\":\"pending\"}])",
                    [crate::ExecutionMode::Repl],
                ),
                crate::ToolText::new(
                    "{\"explanation\":\"I found the main renderer.\",\"plan\":[{\"step\":\"Inspect renderer\",\"status\":\"completed\"},{\"step\":\"Patch layout\",\"status\":\"in_progress\"},{\"step\":\"Run tests\",\"status\":\"pending\"}]}",
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

fn execute_update_plan(state: &Mutex<PlanState>, args: &serde_json::Value) -> ToolResult {
    let explanation = args
        .get("explanation")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let Some(raw_plan) = args.get("plan").and_then(|value| value.as_array()) else {
        return ToolResult::err_fmt("Missing required parameter: plan");
    };
    if raw_plan.is_empty() {
        return ToolResult::err_fmt("Plan must contain at least one step");
    }

    let mut items = Vec::with_capacity(raw_plan.len());
    for (idx, item) in raw_plan.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}]: expected object with step and status"
            ));
        };
        let Some(step) = object
            .get("step")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].step: expected non-empty string"
            ));
        };
        let Some(status) = object
            .get("status")
            .and_then(|value| value.as_str())
            .map(str::trim)
        else {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].status: expected string"
            ));
        };
        if !matches!(status, "pending" | "in_progress" | "completed") {
            return ToolResult::err_fmt(format_args!(
                "Invalid plan[{idx}].status: expected pending, in_progress, or completed"
            ));
        }
        items.push(PlanItem {
            step: step.to_string(),
            status: status.to_string(),
        });
    }

    let mut guard = state.lock().unwrap();
    guard.explanation = explanation.clone();
    guard.items = items.clone();

    let completed = items
        .iter()
        .filter(|item| item.status == "completed")
        .count();
    let in_progress = items
        .iter()
        .filter(|item| item.status == "in_progress")
        .count();
    let summary = if in_progress > 0 {
        format!(
            "updated plan · {} steps, {} completed, {} in progress",
            items.len(),
            completed,
            in_progress
        )
    } else {
        format!(
            "updated plan · {} steps, {} completed",
            items.len(),
            completed
        )
    };

    ToolResult::ok(json!({
        "__type__": "plan_update",
        "summary": summary,
        "explanation": explanation,
        "plan": items,
    }))
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
                &json!({"plan":[{"step":"","status":"pending"}]}),
            )
            .await;
        assert!(!result.success);
        assert!(
            result
                .result
                .as_str()
                .is_some_and(|value| value.contains("plan[0].step"))
        );
    }

    #[tokio::test]
    async fn update_plan_returns_structured_payload() {
        let tool = UpdatePlanTool::new();
        let result = tool
            .execute(
                "update_plan",
                &json!({
                    "explanation":"done reading",
                    "plan":[
                        {"step":"Inspect code","status":"completed"},
                        {"step":"Patch UI","status":"in_progress"}
                    ]
                }),
            )
            .await;
        assert!(result.success);
        assert_eq!(
            result
                .result
                .get("__type__")
                .and_then(|value| value.as_str()),
            Some("plan_update")
        );
        assert_eq!(
            result
                .result
                .get("plan")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
    }
}
