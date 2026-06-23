use lashlang::State as FlowState;

pub(super) const RLM_SNAPSHOT_VERSION: u32 = 4;

pub(super) fn snapshot_runtime(rlm: &FlowState) -> Result<String, String> {
    serde_json::to_string(&rlm.snapshot()).map_err(|err| format!("failed to snapshot RLM: {err}"))
}

pub(super) fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: lashlang::Snapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore RLM: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
}
