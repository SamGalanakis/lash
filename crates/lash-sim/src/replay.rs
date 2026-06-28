use std::fmt;
use std::path::Path;

use serde_json::Value;

use crate::oracles::replay_determinism;
use crate::scheduler::BoundaryScheduler;
use crate::store::ModelStore;
use crate::trace::{
    ReplayReport, SimulationTrace, TRACE_SCHEMA, TraceIoError, read_trace, write_replay_report,
};

#[derive(Debug)]
pub enum ReplayError {
    TraceIo(TraceIoError),
    IncompatibleTrace(String),
    MissingBoundary(String),
    Divergence(String),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TraceIo(err) => write!(f, "{err}"),
            Self::IncompatibleTrace(message) => write!(f, "incompatible replay trace: {message}"),
            Self::MissingBoundary(id) => write!(f, "replay boundary `{id}` was not scheduled"),
            Self::Divergence(message) => write!(f, "replay diverged: {message}"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<TraceIoError> for ReplayError {
    fn from(value: TraceIoError) -> Self {
        Self::TraceIo(value)
    }
}

pub fn replay_trace_file(
    trace_path: &Path,
    report_path: Option<&Path>,
) -> Result<ReplayReport, ReplayError> {
    let trace = read_trace(trace_path)?;
    let report = replay_trace(trace_path, &trace)?;
    if let Some(report_path) = report_path {
        write_replay_report(report_path, &report)?;
    }
    Ok(report)
}

pub fn replay_trace(
    trace_path: &Path,
    trace: &SimulationTrace,
) -> Result<ReplayReport, ReplayError> {
    if trace.schema != TRACE_SCHEMA {
        return Err(ReplayError::IncompatibleTrace(format!(
            "expected schema `{TRACE_SCHEMA}`, got `{}`",
            trace.schema
        )));
    }
    let mut scheduler = BoundaryScheduler::with_events(
        trace.seed,
        trace.events.iter().map(|event| event.as_event()),
    );
    let mut store = ModelStore::default();
    let mut sequence = Vec::new();

    for expected in &trace.events {
        let observed = store.apply_boundary(&expected.as_event());
        let delivered = scheduler
            .deliver_boundary(&expected.boundary_id, observed)
            .ok_or_else(|| ReplayError::MissingBoundary(expected.boundary_id.clone()))?;
        let actual_observed = normalize(&delivered.observed);
        let expected_observed = normalize(&expected.observed);
        if actual_observed != expected_observed {
            return Err(ReplayError::Divergence(format!(
                "boundary `{}` observed payload changed; expected={}; actual={}",
                expected.boundary_id, expected_observed, actual_observed
            )));
        }
        sequence.push(delivered.boundary_id);
    }

    if !scheduler.is_empty() {
        return Err(ReplayError::Divergence(format!(
            "{} boundaries remained pending after replay",
            scheduler.pending_len()
        )));
    }

    let final_summary = store.summary();
    let terminal_verdict = replay_determinism(&trace.final_summary, &final_summary);
    if !terminal_verdict.is_passed() {
        return Err(ReplayError::Divergence(terminal_verdict.message.clone()));
    }

    Ok(ReplayReport::new(
        trace_path,
        terminal_verdict,
        sequence,
        final_summary,
    ))
}

fn normalize(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("provider_parser_matrix");
        object.remove("runtime_effect");
        object.remove("runtime_effect_outcome");
        object.remove("runtime_tool_record");
        object.remove("runtime_queued_work");
        object.remove("runtime_worker_store");
        object.remove("runtime_active_lease");
        object.remove("runtime_stale_completion");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::generate_workload;
    use crate::runner::run_generated_workload_for_fixture;

    #[tokio::test]
    async fn replay_reproduces_boundary_sequence_and_summary() {
        let workload = generate_workload(5, "fast-random", 24).expect("workload");
        let trace = run_generated_workload_for_fixture(workload, "bundle")
            .await
            .expect("trace");
        let report = replay_trace(Path::new("trace.json"), &trace).expect("replay");

        assert_eq!(report.delivered_event_count, trace.events.len());
        assert_eq!(report.final_summary, trace.final_summary);
        assert!(report.terminal_verdict.is_passed());
    }
}
