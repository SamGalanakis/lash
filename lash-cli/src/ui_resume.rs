use lash::Store;

use crate::app::UiResumeState;

pub(crate) const UI_RESUME_STATE_KEY: &str = "ui_resume_state";

pub(crate) fn with_ui_resume_state(
    mut config: serde_json::Value,
    ui_state: &UiResumeState,
) -> serde_json::Value {
    let resume_value = serde_json::to_value(ui_state).unwrap_or(serde_json::Value::Null);
    if let Some(object) = config.as_object_mut() {
        object.insert(UI_RESUME_STATE_KEY.to_string(), resume_value);
        config
    } else {
        serde_json::json!({
            UI_RESUME_STATE_KEY: ui_state,
        })
    }
}

pub(crate) fn load_ui_resume_state(store: &Store) -> UiResumeState {
    store
        .load_session_state()
        .and_then(|state| serde_json::from_str::<serde_json::Value>(&state.config_json).ok())
        .and_then(|config| {
            config
                .get(UI_RESUME_STATE_KEY)
                .cloned()
                .and_then(|value| serde_json::from_value::<UiResumeState>(value).ok())
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_ui_resume_state_merges_into_existing_config() {
        let config = serde_json::json!({
            "manifest": { "configured_model": "gpt-5" },
            "dynamic_state": { "base_generation": 1 }
        });
        let ui_state = UiResumeState {
            streaming_output: vec!["tail".to_string()],
            streaming_output_hidden: 2,
            ..UiResumeState::default()
        };

        let merged = with_ui_resume_state(config, &ui_state);

        assert_eq!(merged["manifest"]["configured_model"], "gpt-5");
        assert_eq!(merged[UI_RESUME_STATE_KEY]["streaming_output_hidden"], 2);
        assert_eq!(merged[UI_RESUME_STATE_KEY]["streaming_output"][0], "tail");
    }
}
