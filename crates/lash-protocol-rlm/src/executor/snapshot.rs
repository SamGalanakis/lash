use lashlang::State as FlowState;

// v5 adds the required MIME field to persisted Lashlang image descriptors.
// Older snapshots are rejected at restore rather than silently degrading an
// image into a plain record.
pub(super) const RLM_SNAPSHOT_VERSION: u32 = 5;

pub(super) fn snapshot_runtime(rlm: &FlowState) -> Result<String, String> {
    serde_json::to_string(&rlm.snapshot()).map_err(|err| format!("failed to snapshot RLM: {err}"))
}

pub(super) fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: lashlang::Snapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore RLM: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
}
