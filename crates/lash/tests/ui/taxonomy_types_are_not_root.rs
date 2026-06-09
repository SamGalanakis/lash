fn main() {
    // Observation, host-event/trigger, process, and tool-state taxonomy types
    // are single-homed in their domain modules (`lash::observe`,
    // `lash::host_events`, `lash::process`, `lash::tools`), not at the root.
    let _: Option<lash::SessionObservation> = None;
    let _: Option<lash::SessionCursor> = None;
    let _: Option<lash::SessionResume> = None;
    let _: Option<lash::HostEvent> = None;
    let _: Option<lash::TriggerRegistration> = None;
    let _: Option<lash::ProcessHandleSummary> = None;
    let _: Option<lash::ToolState> = None;
    let _ = lash::empty_host_event_source_key("ui.button.pressed");
}
