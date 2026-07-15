use lashlang::{ExecutionHostError, Record, Value};

use crate::{DisplayDelta, DisplayState};

pub(crate) fn apply_tool(
    display: &mut DisplayState,
    operation: &str,
    args: &[Value],
) -> Result<(Value, DisplayDelta), ExecutionHostError> {
    let args = args.first().and_then(Value::as_record).ok_or_else(|| {
        ExecutionHostError::new(format!("{operation} expects one record argument"))
    })?;
    let mut delta = DisplayDelta::default();
    match operation {
        "show_message" => {
            let text = string_arg(args, "text")?;
            display.messages.push(text.clone());
            delta.messages_appended.push(text);
        }
        "set_status" => {
            let key = string_arg(args, "key")?;
            let value = string_arg(args, "value")?;
            display.statuses.insert(key.clone(), value.clone());
            delta.statuses.insert(key, value);
        }
        "add_item" => {
            let list = string_arg(args, "list")?;
            let item = scalar_text_arg(args, "item")?;
            display
                .lists
                .entry(list.clone())
                .or_default()
                .push(item.clone());
            delta
                .list_items_appended
                .entry(list)
                .or_default()
                .push(item);
        }
        "set_light" => {
            let name = string_arg(args, "name")?;
            let state = scalar_text_arg(args, "state")?;
            display.lights.insert(name.clone(), state.clone());
            delta.lights.insert(name, state);
        }
        "set_progress" => {
            let pct = number_arg(args, "pct")?.clamp(0.0, 100.0);
            display.progress = pct;
            delta.progress = Some(pct);
        }
        "highlight" => {
            let target = string_arg(args, "target")?;
            display.highlighted = Some(target.clone());
            delta.highlighted = Some(target);
        }
        _ => {
            return Err(ExecutionHostError::new(format!(
                "unknown display operation `{operation}`"
            )));
        }
    }
    Ok((Value::Null, delta))
}

fn string_arg(args: &Record, key: &str) -> Result<String, ExecutionHostError> {
    match args.get(key) {
        Some(Value::String(value)) => Ok(value.to_string()),
        _ => Err(ExecutionHostError::new(format!(
            "missing string argument `{key}`"
        ))),
    }
}

fn number_arg(args: &Record, key: &str) -> Result<f64, ExecutionHostError> {
    match args.get(key) {
        Some(Value::Number(value)) => Ok(*value),
        _ => Err(ExecutionHostError::new(format!(
            "missing number argument `{key}`"
        ))),
    }
}

fn scalar_text_arg(args: &Record, key: &str) -> Result<String, ExecutionHostError> {
    match args.get(key) {
        Some(Value::String(value)) => Ok(value.to_string()),
        Some(Value::Number(value)) => Ok(value.to_string()),
        Some(Value::Bool(value)) => Ok(value.to_string()),
        _ => Err(ExecutionHostError::new(format!(
            "missing scalar argument `{key}`"
        ))),
    }
}
