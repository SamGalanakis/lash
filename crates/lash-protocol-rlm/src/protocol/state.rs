use lash_core::AttachmentRef;
use serde_json::Value;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(super) struct RlmDriverState {
    pub(super) reasoning: String,
    pub(super) tool_call_ids: Vec<String>,
    pub(super) images: Vec<AttachmentRef>,
    /// One entry per `print` from the executed lashlang block (plus any
    /// raw stdout-style emission). Replaces the old split between a
    /// concatenated `combined_output: String` and a sibling
    /// `observations: Vec<String>` — the two carried the same content.
    pub(super) output: Vec<String>,
    pub(super) exec_error: Option<String>,
    pub(super) executed_code: Option<String>,
    pub(super) terminal_finish: Option<Value>,
}

pub(super) fn rlm_driver_state(state: RlmDriverState) -> lash_core::ProtocolDriverState {
    lash_core::ProtocolDriverState::new(
        crate::plugin::RLM_PROTOCOL_PLUGIN_ID,
        serde_json::to_value(state).expect("RLM driver state must serialize"),
    )
}

pub(super) fn decode_rlm_driver_state(
    state: lash_core::ProtocolDriverState,
) -> Result<RlmDriverState, String> {
    if state.plugin_id != crate::plugin::RLM_PROTOCOL_PLUGIN_ID {
        return Err(format!(
            "driver state belongs to plugin `{}`, expected `{}`",
            state.plugin_id,
            crate::plugin::RLM_PROTOCOL_PLUGIN_ID
        ));
    }
    serde_json::from_value(state.payload)
        .map_err(|err| format!("invalid RLM driver state payload: {err}"))
}
