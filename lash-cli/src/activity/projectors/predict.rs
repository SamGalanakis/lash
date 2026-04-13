//! Predict projector: typed sub-session calls via `predict`.
//!
//! Shows the task, bound variable names + types, and the declared
//! output schema in a compact summary line with detail lines. On
//! success, renders the returned typed record as a compact preview.

use serde_json::Value;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, ProjectCtx, ToolProjector,
    shared::{inline_snippet, tool_arg_str},
};

pub(crate) struct PredictProjector;

impl ToolProjector for PredictProjector {
    fn tool_names(&self) -> &'static [&'static str] {
        &["predict"]
    }

    fn project(&self, ctx: &mut ProjectCtx<'_>) -> Vec<ActivityBlock> {
        let task = tool_arg_str(&ctx.args, "task")
            .unwrap_or("typed sub-session")
            .to_string();

        let status = if ctx.success {
            ActivityStatus::Completed
        } else {
            ActivityStatus::Failed
        };

        let summary = format!("predict · {}", inline_snippet(&task, 60));

        let mut detail_lines = Vec::new();

        // Vars line: show bound variable names + types.
        if let Some(vars) = ctx.args.get("vars").and_then(Value::as_object)
            && !vars.is_empty()
        {
            let var_entries: Vec<String> = vars
                .iter()
                .map(|(name, value)| format!("{name}: {}", json_type_short(value)))
                .collect();
            detail_lines.push(format!("vars  {}", var_entries.join(", ")));
        }

        // Output schema line: show declared field names → types.
        if let Some(output) = ctx.args.get("output").and_then(Value::as_object)
            && !output.is_empty()
        {
            let field_entries: Vec<String> = output
                .iter()
                .map(|(name, ty)| {
                    let ty_str = ty.as_str().unwrap_or("any");
                    format!("{name}: {ty_str}")
                })
                .collect();
            detail_lines.push(format!("out   {{ {} }}", field_entries.join(", ")));
        }

        // Result preview on success.
        let artifact = if ctx.success {
            let result_text = match &ctx.result {
                Value::Object(obj) => {
                    let entries: Vec<String> = obj
                        .iter()
                        .map(|(k, v)| {
                            let v_str = match v {
                                Value::String(s) => inline_snippet(s, 80),
                                other => {
                                    let s = other.to_string();
                                    inline_snippet(&s, 80)
                                }
                            };
                            format!("{k}: {v_str}")
                        })
                        .collect();
                    format!("{{ {} }}", entries.join(", "))
                }
                Value::String(s) => inline_snippet(s, 120),
                other => inline_snippet(&other.to_string(), 120),
            };
            Some(ActivityArtifact::TextPreview {
                title: Some("Result".to_string()),
                text: result_text,
            })
        } else {
            let error_text = ctx
                .result
                .as_str()
                .or_else(|| ctx.result.get("error").and_then(Value::as_str))
                .unwrap_or("predict failed");
            Some(ActivityArtifact::TextPreview {
                title: Some("Error".to_string()),
                text: error_text.to_string(),
            })
        };

        let args = std::mem::replace(&mut ctx.args, Value::Null);
        let result = std::mem::replace(&mut ctx.result, Value::Null);
        vec![
            ActivityBlock::new(
                ActivityKind::Delegate,
                ctx.name,
                args,
                summary,
                status,
                result,
                ctx.duration_ms,
            )
            .with_detail_lines(detail_lines)
            .with_artifact(artifact),
        ]
    }
}

fn json_type_short(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() && n.as_f64().is_some_and(|v| v.fract() != 0.0) {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "record",
    }
}
