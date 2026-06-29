use std::collections::{BTreeMap, BTreeSet};

use lash::scenario_contracts::AGENT_SCENARIO_CONTRACTS;
use lash_core::runtime::{RUNTIME_SCENARIO_CONTRACTS, ScenarioContractSpec};
use lash_protocol_rlm::scenario_contracts::RLM_PROTOCOL_SCENARIO_CONTRACTS;
use lash_protocol_standard::scenario_contracts::STANDARD_PROTOCOL_SCENARIO_CONTRACTS;
use serde_json::Value;

use crate::runtime_contracts::{
    RuntimeAgentFrameInvariantFacts, RuntimeGraphInvariantFacts, RuntimeUsageInvariantFacts,
};
use crate::provider_mutations::is_transport_provider_mutation;
use crate::runtime_providers::MIGRATED_RUNTIME_PROVIDER_KINDS;
use crate::scheduler::{BoundaryKind, DeliveredBoundary};
use crate::trace::{AbstractWorldSummary, OracleVerdict};

pub const CROSS_SESSION_ISOLATION_ORACLE: &str = "sim.oracle.cross-session-isolation.v1";
pub const BACKEND_FAILURE_ORACLE: &str = "sim.oracle.backend-failure-observed.v1";
pub const CANCELLATION_ORACLE: &str = "sim.oracle.cancellation-observed.v1";
pub const DURABLE_EFFECT_EXACTLY_ONCE_ORACLE: &str = "sim.oracle.durable-effect-exactly-once.v1";
pub const EXEC_CODE_ORACLE: &str = "sim.oracle.exec-code-observed.v1";
pub const INGRESS_SESSION_OPENED_ORACLE: &str = "sim.oracle.ingress-session-opened.v1";
pub const LEASE_TIME_MONOTONIC_ORACLE: &str = "sim.oracle.lease-time-monotonic.v1";
pub const OBSERVER_CONVERGENCE_ORACLE: &str = "sim.oracle.observer-convergence.v1";
pub const OBSERVER_RECONNECT_ORACLE: &str = "sim.oracle.observer-reconnect.v1";
pub const OPERATIONAL_COVERAGE_ORACLE: &str = "sim.oracle.operational-coverage.v1";
pub const PROCESS_WAKE_ORACLE: &str = "sim.oracle.process-wake-observed.v1";
pub const PROVIDER_MUTATION_ORACLE: &str = "sim.oracle.provider-mutation-rejected.v1";
pub const QUEUED_INGRESS_ORACLE: &str = "sim.oracle.queued-ingress-observed.v1";
pub const REPLAY_DETERMINISM_ORACLE: &str = "sim.oracle.replay-determinism.v1";
pub const RUNTIME_PROVIDER_TURN_ORACLE: &str = "sim.oracle.runtime-provider-turn.v1";
pub const PENDING_TOOL_COMPLETION_ORACLE: &str =
    "sim.oracle.pending-tool-completion-through-turn.v1";
pub const RUNTIME_GRAPH_ACYCLIC_ORACLE: &str = "sim.oracle.runtime-graph-acyclic.v1";
pub const RUNTIME_SINGLE_ACTIVE_AGENT_FRAME_ORACLE: &str =
    "sim.oracle.runtime-single-active-agent-frame.v1";
pub const RUNTIME_USAGE_MONOTONIC_ORACLE: &str = "sim.oracle.runtime-usage-monotonic.v1";
pub const RUNTIME_FINAL_VALUE_SEMANTIC_ORACLE: &str =
    "sim.oracle.runtime-final-value-semantic-channel.v1";
pub const GENERATED_PROVIDER_MATRIX_ORACLE: &str =
    "sim.oracle.generated-runtime-provider-matrix.v1";
pub const PROVIDER_TURN_INTERLEAVING_ORACLE: &str =
    "sim.oracle.provider-turn-interleaving-depth.v1";
pub const PROVIDER_TRANSPORT_MUTATION_ORACLE: &str =
    "sim.oracle.provider-transport-mutation-classified.v1";
pub const RUNTIME_SESSION_GRAPH_ORACLE: &str = "sim.oracle.runtime-session-graph.v1";
pub const SCHEDULER_CONTROLLED_DELIVERY_ORACLE: &str =
    "sim.oracle.scheduler-controlled-delivery.v1";
pub const SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE: &str =
    "sim.oracle.scheduler-owned-runtime-completions.v1";
pub const STATE_MACHINE_SEMANTIC_INVARIANTS_ORACLE: &str =
    "sim.oracle.state-machine-semantic-invariants.v1";
pub const SCENARIO_AGENT_CONTRACT_ORACLE: &str = "sim.oracle.scenario.agent-contract.v1";
pub const SCENARIO_RLM_CONTRACT_ORACLE: &str = "sim.oracle.scenario.rlm-contract.v1";
pub const SCENARIO_RUNTIME_CONTRACT_ORACLE: &str = "sim.oracle.scenario.runtime-contract.v1";
pub const SCENARIO_STANDARD_CONTRACT_ORACLE: &str = "sim.oracle.scenario.standard-contract.v1";
pub const SCENARIO_MINI_RUNTIME_QUEUED_HIDDEN_ORACLE: &str =
    "sim.oracle.scenario-mini.runtime.queued-input-hidden-while-live.v1";
pub const SCENARIO_MINI_RUNTIME_CANCEL_IDLE_ORACLE: &str =
    "sim.oracle.scenario-mini.runtime.cancellation-prevents-idle-claim.v1";
pub const SCENARIO_MINI_RUNTIME_PROCESS_WAKE_DEDUPE_ORACLE: &str =
    "sim.oracle.scenario-mini.runtime.process-wake-duplicate-rejected.v1";
pub const SCENARIO_MINI_RUNTIME_STALE_LEASE_ORACLE: &str =
    "sim.oracle.scenario-mini.runtime.stale-lease-commit-rejected.v1";
pub const SCENARIO_MINI_STANDARD_STREAM_FINALIZE_ORACLE: &str =
    "sim.oracle.scenario-mini.standard.streamed-text-finalizes-once.v1";
pub const SCENARIO_MINI_STANDARD_PROVIDER_ERROR_ORACLE: &str =
    "sim.oracle.scenario-mini.standard.provider-error-without-checkpoint.v1";
pub const SCENARIO_MINI_STANDARD_TOOL_REENTRY_ORACLE: &str =
    "sim.oracle.scenario-mini.standard.tool-loop-reenters-after-checkpoint.v1";
pub const SCENARIO_MINI_RLM_FINISH_REPAIR_ORACLE: &str =
    "sim.oracle.scenario-mini.rlm.finish-required-prose-repair.v1";
pub const SCENARIO_MINI_RLM_SCHEMA_REPAIR_ORACLE: &str =
    "sim.oracle.scenario-mini.rlm.schema-mismatch-repair.v1";
pub const SCENARIO_MINI_RLM_CELL_EXEC_ORACLE: &str =
    "sim.oracle.scenario-mini.rlm.lashlang-cell-exec-continues.v1";
pub const SCENARIO_MINI_AGENT_DURABLE_INPUT_ORACLE: &str =
    "sim.oracle.scenario-mini.agent.durable-input-resolution.v1";
pub const SCENARIO_MINI_AGENT_CHILD_FAILURE_ORACLE: &str =
    "sim.oracle.scenario-mini.agent.child-failure-graph.v1";
pub const SCENARIO_MINI_AGENT_PARALLEL_JOIN_ORACLE: &str =
    "sim.oracle.scenario-mini.agent.parallel-spawn-join-determinism.v1";
pub const TOOL_BOUNDARY_ORACLE: &str = "sim.oracle.tool-boundary-observed.v1";
pub const TRIGGER_ORACLE: &str = "sim.oracle.trigger-delivery-observed.v1";
pub const WORKER_STALE_COMPLETION_ORACLE: &str = "sim.oracle.worker-stale-completion-rejected.v1";
pub const GENERATED_SUSPEND_RESUME_ORACLE: &str = "sim.oracle.generated-suspend-resume.v1";
pub const GENERATED_FINAL_VALUE_ORACLE: &str =
    "sim.oracle.generated-final-value-semantic-channel.v1";

fn is_suspend_resume(event: &DeliveredBoundary) -> bool {
    event
        .payload
        .get("suspend_resume")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Assert that every generated suspend turn genuinely parked mid-flight and was
/// resumed only by the scheduler-delivered completion boundary — never ran
/// synchronously. This generalizes the fixed pending-tool proof into the live
/// generated search. Workloads with no suspend boundary (e.g. minimized
/// fixtures) pass vacuously.
pub fn generated_suspend_resume(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut checked = 0usize;
    for event in events.iter().filter(|event| is_suspend_resume(event)) {
        let Some(suspend) = event.observed.get("runtime_suspend") else {
            return OracleVerdict::failed(
                GENERATED_SUSPEND_RESUME_ORACLE,
                format!(
                    "suspend resume `{}` recorded no runtime_suspend evidence",
                    event.boundary_id
                ),
            );
        };
        let suspended_before = suspend
            .get("turn_suspended_before_completion")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let scheduler_delivered = suspend
            .get("scheduler_delivered_completion")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let resolve_accepted = suspend
            .get("resolve_accepted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let resumed_after = suspend
            .get("resumed_after_completion")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let before = suspend
            .get("completed_event_count_before_resolution")
            .and_then(Value::as_u64)
            .unwrap_or(u64::MAX);
        let after = suspend
            .get("completed_event_count_after_resolution")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if !(suspended_before
            && scheduler_delivered
            && resolve_accepted
            && resumed_after
            && before == 0
            && after > before)
        {
            return OracleVerdict::failed(
                GENERATED_SUSPEND_RESUME_ORACLE,
                format!(
                    "suspend `{}` ({}) did not park-then-resume: suspended_before={suspended_before} scheduler_delivered={scheduler_delivered} resolve_accepted={resolve_accepted} resumed_after={resumed_after} completed_before={before} completed_after={after}",
                    event.boundary_id,
                    suspend.get("suspend_kind").and_then(Value::as_str).unwrap_or("?")
                ),
            );
        }
        checked += 1;
    }
    OracleVerdict::passed(
        GENERATED_SUSPEND_RESUME_ORACLE,
        format!(
            "{checked} generated suspend turn(s) parked mid-flight and resumed only after a scheduler-delivered completion"
        ),
    )
}

/// Run the final-value (semantic-channel / distinct-from-transcript) invariant
/// on every generated provider turn, not just the fixed RLM proof. Generated
/// turns are assistant-message turns, so the invariant asserts the dual of the
/// proof: the assistant prose was NOT mis-projected as a semantic final value —
/// `outcome_kind` is `assistant_message`, no `semantic_value` leaked, and no
/// terminal FinalValue/ToolValue event was emitted. A turn that smuggled a
/// final value through transcript inference would fail.
pub fn generated_final_value_semantic_channel(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut checked = 0usize;
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
    {
        let Some(facts) = event.observed.get("runtime_final_value_facts") else {
            return OracleVerdict::failed(
                GENERATED_FINAL_VALUE_ORACLE,
                format!(
                    "provider turn `{}` recorded no runtime_final_value_facts",
                    event.boundary_id
                ),
            );
        };
        let outcome_kind = facts.get("outcome_kind").and_then(Value::as_str).unwrap_or("");
        let has_semantic_value = facts
            .get("semantic_value")
            .map(|value| !value.is_null())
            .unwrap_or(false);
        let terminal_event_count = facts
            .get("terminal_event_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let semantic_channel_observed = facts
            .get("semantic_channel_observed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if outcome_kind != "assistant_message"
            || has_semantic_value
            || terminal_event_count != 0
            || semantic_channel_observed
        {
            return OracleVerdict::failed(
                GENERATED_FINAL_VALUE_ORACLE,
                format!(
                    "provider turn `{}` leaked a semantic final value: outcome_kind={outcome_kind} has_semantic_value={has_semantic_value} terminal_events={terminal_event_count} semantic_channel_observed={semantic_channel_observed}",
                    event.boundary_id
                ),
            );
        }
        checked += 1;
    }
    OracleVerdict::passed(
        GENERATED_FINAL_VALUE_ORACLE,
        format!(
            "{checked} generated assistant-message turn(s) kept the semantic final-value channel empty (no transcript-inferred final values)"
        ),
    )
}


pub fn cross_session_isolation(summary: &AbstractWorldSummary) -> OracleVerdict {
    if summary.session_count < 2 {
        return OracleVerdict::failed(
            CROSS_SESSION_ISOLATION_ORACLE,
            "workload did not contain at least two sessions",
        );
    }
    let aliases = summary
        .sessions
        .iter()
        .map(|session| session.alias.as_str())
        .collect::<BTreeSet<_>>();
    for session in &summary.sessions {
        if !session.opened {
            return OracleVerdict::failed(
                CROSS_SESSION_ISOLATION_ORACLE,
                format!("session `{}` was never opened", session.alias),
            );
        }
        for other_alias in aliases
            .iter()
            .copied()
            .filter(|alias| *alias != session.alias)
        {
            let leaked_provider = session
                .provider_outputs
                .iter()
                .any(|output| output.contains(other_alias));
            let leaked_tool = session
                .tool_outputs
                .iter()
                .any(|output| output.contains(other_alias));
            if leaked_provider || leaked_tool {
                return OracleVerdict::failed(
                    CROSS_SESSION_ISOLATION_ORACLE,
                    format!(
                        "session `{}` observed output from `{other_alias}`",
                        session.alias
                    ),
                );
            }
        }
    }
    OracleVerdict::passed(
        CROSS_SESSION_ISOLATION_ORACLE,
        "all generated session outputs stayed scoped to their session alias",
    )
}

pub fn ingress_sessions_opened(summary: &AbstractWorldSummary) -> OracleVerdict {
    for session in &summary.sessions {
        if !session.opened || session.ingress_count != 1 {
            return OracleVerdict::failed(
                INGRESS_SESSION_OPENED_ORACLE,
                format!(
                    "session `{}` expected exactly one ingress opening, got opened={} ingress_count={}",
                    session.alias, session.opened, session.ingress_count
                ),
            );
        }
    }
    OracleVerdict::passed(
        INGRESS_SESSION_OPENED_ORACLE,
        "each generated session opened through an ingress boundary exactly once",
    )
}

pub fn observer_convergence(summary: &AbstractWorldSummary) -> OracleVerdict {
    for session in &summary.sessions {
        let expected_turns = session.provider_outputs.len();
        let Some(last_observed_turn) = session.observer_turn_indices.last().copied() else {
            return OracleVerdict::failed(
                OBSERVER_CONVERGENCE_ORACLE,
                format!("session `{}` had no observer snapshot", session.alias),
            );
        };
        if last_observed_turn != expected_turns {
            return OracleVerdict::failed(
                OBSERVER_CONVERGENCE_ORACLE,
                format!(
                    "session `{}` observer saw turn {}, expected {}",
                    session.alias, last_observed_turn, expected_turns
                ),
            );
        }
    }
    OracleVerdict::passed(
        OBSERVER_CONVERGENCE_ORACLE,
        "observer snapshots converged to the generated runtime turn count",
    )
}

/// Build a failing-capable coverage verdict: the boundary kind must be present
/// AND the runtime invariant its reason claims must actually hold in the events.
/// A present-but-broken boundary fails loudly instead of passing on presence.
fn coverage_invariant_verdict(
    oracle_id: &'static str,
    present: bool,
    missing_reason: &'static str,
    invariant_holds: bool,
    invariant_failed_reason: &'static str,
    passed_reason: &'static str,
) -> OracleVerdict {
    if !present {
        OracleVerdict::failed(oracle_id, missing_reason)
    } else if !invariant_holds {
        OracleVerdict::failed(oracle_id, invariant_failed_reason)
    } else {
        OracleVerdict::passed(oracle_id, passed_reason)
    }
}

pub fn queued_ingress_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        QUEUED_INGRESS_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.queued_ingress_count > 0),
        "no queued ingress boundary was observed",
        queued_ingress_has_source_keys(events),
        "queued ingress boundary was observed but it lacked a stable payload/observed source key",
        "generated workload queued turn input through an explicit ingress boundary carrying a stable source key",
    )
}

pub fn cancellation_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        CANCELLATION_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.cancellation_count > 0),
        "no cancellation boundary was observed",
        cancellation_terminalizes_pending_input(events),
        "cancellation boundary was observed but it did not terminalize the pending queued input",
        "generated workload cancelled a pending boundary and terminalized its queued input",
    )
}

pub fn trigger_delivery_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        TRIGGER_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.trigger_count > 0),
        "no trigger boundary was observed",
        trigger_delivery_runtime_observed(events),
        "trigger boundary was observed but it did not route through a runtime trigger DTO with a stable source identity",
        "generated workload delivered a trigger boundary routed through a runtime trigger DTO with stable source identity",
    )
}

pub fn observer_reconnect_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        OBSERVER_RECONNECT_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.observer_reconnects > 0),
        "no observer reconnect boundary was observed",
        observer_reconnect_has_matching_turn(events, summary),
        "observer reconnect boundary was observed but its replayed snapshot did not converge to the session's final provider turn",
        "observer reconnect boundary converged on the same session state (final provider turn)",
    )
}

pub fn backend_failure_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        BACKEND_FAILURE_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.backend_failure_count > 0),
        "no backend failure boundary was observed",
        backend_retry_terminalization_semantics(events),
        "backend failure boundary was observed but it did not terminalize through the retry/terminalization path",
        "generated workload injected a backend failure boundary that terminalized through the retry path",
    )
}

pub fn provider_mutation_rejected(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        PROVIDER_MUTATION_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.provider_mutation_count > 0),
        "no provider/script mutation boundary was observed",
        provider_mutation_parser_matrix_observed(events),
        "provider/script mutation boundary was observed but it did not run through the real provider parser matrix",
        "provider/script mutation boundary was rejected through the real provider parser matrix without changing runtime state",
    )
}

pub fn generated_runtime_provider_matrix(events: &[DeliveredBoundary]) -> OracleVerdict {
    let expected = MIGRATED_RUNTIME_PROVIDER_KINDS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
    {
        let payload_kind = event
            .payload
            .get("provider_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        let observed_kind = event
            .observed
            .get("provider_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        if payload_kind.is_empty() || observed_kind.is_empty() || payload_kind != observed_kind {
            return OracleVerdict::failed(
                GENERATED_PROVIDER_MATRIX_ORACLE,
                format!(
                    "provider event `{}` payload kind `{payload_kind}` did not match observed kind `{observed_kind}`",
                    event.boundary_id
                ),
            );
        }
        observed.insert(observed_kind);
    }
    if observed != expected {
        return OracleVerdict::failed(
            GENERATED_PROVIDER_MATRIX_ORACLE,
            format!(
                "generated runtime provider kinds {:?} did not cover {:?}",
                observed, expected
            ),
        );
    }
    OracleVerdict::passed(
        GENERATED_PROVIDER_MATRIX_ORACLE,
        "generated runtime turns covered OpenAI-compatible, direct OpenAI, Anthropic, and Google provider crates through ScriptedLlmHttpTransport",
    )
}

/// Peak number of provider turns that were simultaneously live during the run,
/// derived purely from the delivered boundary stream so the value is identical
/// on the generation path and on cross-backend replay.
///
/// A provider turn is "live" in the delivered event stream from the first
/// `ProviderEvent` released for its turn (the turn future is parked on the
/// scripted transport gate at that point) until its `Provider` completion
/// boundary is delivered. The number of distinct turns simultaneously in that
/// window is the interleaving depth; its maximum is the highwater.
pub fn peak_concurrent_live_turns(events: &[DeliveredBoundary]) -> usize {
    let mut live = BTreeSet::<String>::new();
    let mut peak = 0usize;
    for event in events {
        match event.kind {
            BoundaryKind::ProviderEvent => {
                if let Some(turn_boundary_id) = event
                    .payload
                    .get("turn_boundary_id")
                    .and_then(Value::as_str)
                {
                    live.insert(turn_boundary_id.to_string());
                    peak = peak.max(live.len());
                }
            }
            BoundaryKind::Provider => {
                live.remove(&event.boundary_id);
            }
            _ => {}
        }
    }
    peak
}

/// Count of distinct sessions that ran at least one provider turn. Interleaving
/// is structurally impossible below two such sessions, so the interleaving
/// oracle treats those workloads as vacuously satisfied.
fn provider_turn_session_count(events: &[DeliveredBoundary]) -> usize {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
        .map(|event| event.actor_alias.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

/// Make interleaving load-bearing: whenever a workload runs provider turns in at
/// least two sessions, the scheduler must actually drive at least two of those
/// turns concurrently. A multi-session workload that never interleaves is a real
/// scheduling regression and fails this oracle; single-session workloads (and
/// minimized fixtures that collapse to one session) pass vacuously.
pub fn provider_turn_interleaving_depth(events: &[DeliveredBoundary]) -> OracleVerdict {
    let peak = peak_concurrent_live_turns(events);
    let sessions = provider_turn_session_count(events);
    if sessions < 2 {
        return OracleVerdict::passed(
            PROVIDER_TURN_INTERLEAVING_ORACLE,
            format!(
                "interleaving not required: only {sessions} session(s) ran provider turns (peak concurrent live turns {peak})"
            ),
        );
    }
    if peak >= 2 {
        OracleVerdict::passed(
            PROVIDER_TURN_INTERLEAVING_ORACLE,
            format!(
                "scheduler drove {peak} provider turns concurrently across {sessions} sessions"
            ),
        )
    } else {
        OracleVerdict::failed(
            PROVIDER_TURN_INTERLEAVING_ORACLE,
            format!(
                "{sessions} sessions ran provider turns but peak concurrent live turns was {peak}; the scheduler never interleaved live provider turns"
            ),
        )
    }
}

/// Each generator-driven transport/HTTP mutation must land on its own named
/// failure path through the real provider: a mid-stream disconnect classifies
/// as a retryable stream fault, response-start and chunk timeouts as retryable
/// timeouts, and a 5xx as a retryable HTTP status carrying its status code.
/// This proves the new mutation classes drive distinct, executable behaviors
/// rather than collapsing into a single generic parser error. Workloads that
/// happen to contain no transport mutation pass vacuously (the class is still
/// covered by the seeded anchor in every full run).
pub fn provider_transport_mutation_classified(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut observed_classes = BTreeSet::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProviderMutation)
    {
        let Some(mutation) = event
            .observed
            .get("mutation")
            .or_else(|| event.payload.get("mutation"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        if !is_transport_provider_mutation(mutation) {
            continue;
        }
        let Some(proof) = event
            .observed
            .pointer("/provider_parser_matrix/matrix/proofs")
            .and_then(Value::as_array)
            .and_then(|proofs| proofs.first())
        else {
            return OracleVerdict::failed(
                PROVIDER_TRANSPORT_MUTATION_ORACLE,
                format!(
                    "transport mutation `{mutation}` on `{}` recorded no real provider parser proof",
                    event.boundary_id
                ),
            );
        };
        let raw_kind = proof.get("kind").and_then(Value::as_str).unwrap_or("");
        let retryable = proof
            .pointer("/classification/retryable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let status = proof.get("status").and_then(Value::as_u64);
        let classified = match mutation {
            "mid_stream_disconnect" => raw_kind == "Stream" && retryable,
            "response_start_timeout" | "stream_chunk_timeout" => raw_kind == "Timeout" && retryable,
            "retryable_server_error_sequence" => status == Some(503) && retryable,
            _ => false,
        };
        if !classified {
            return OracleVerdict::failed(
                PROVIDER_TRANSPORT_MUTATION_ORACLE,
                format!(
                    "transport mutation `{mutation}` on `{}` classified incorrectly: raw_kind={raw_kind} retryable={retryable} status={status:?}",
                    event.boundary_id
                ),
            );
        }
        observed_classes.insert(mutation.to_string());
    }
    OracleVerdict::passed(
        PROVIDER_TRANSPORT_MUTATION_ORACLE,
        format!(
            "transport mutation classes classified on distinct failure paths: {:?}",
            observed_classes
        ),
    )
}

pub fn process_wake_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        PROCESS_WAKE_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| session.process_wake_count > 0),
        "no process wake boundary was observed",
        process_wake_runtime_dto_observed(events) && duplicate_process_wake_rejected(events),
        "process wake boundary was observed but it did not materialize a runtime DTO with a deduped duplicate wake",
        "process wake boundary materialized a runtime DTO and the duplicate wake was rejected (delivered exactly once)",
    )
}

pub fn exec_code_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        EXEC_CODE_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| !session.exec_code_outputs.is_empty()),
        "no exec-code boundary was observed",
        exec_runtime_outcome_observed(events),
        "exec-code boundary was observed but it did not produce a captured RuntimeEffectOutcome",
        "exec-code boundary produced a captured RuntimeEffectOutcome result",
    )
}

pub fn tool_boundary_observed(
    summary: &AbstractWorldSummary,
    events: &[DeliveredBoundary],
) -> OracleVerdict {
    coverage_invariant_verdict(
        TOOL_BOUNDARY_ORACLE,
        summary
            .sessions
            .iter()
            .any(|session| !session.tool_outputs.is_empty()),
        "no tool boundary was observed",
        tool_runtime_output_observed(events),
        "tool boundary was observed but it did not produce a captured runtime tool-output DTO",
        "tool boundary produced a captured runtime tool-output DTO",
    )
}

pub fn runtime_session_graph_contract(summary: &AbstractWorldSummary) -> OracleVerdict {
    for session in &summary.sessions {
        if session.provider_outputs.len() < 2 {
            return OracleVerdict::failed(
                RUNTIME_SESSION_GRAPH_ORACLE,
                format!(
                    "session `{}` ran only {} provider turns",
                    session.alias,
                    session.provider_outputs.len()
                ),
            );
        }
        for (index, output) in session.provider_outputs.iter().enumerate() {
            let turn_index = index + 1;
            let expected_exchange_count = turn_index;
            if session.provider_exchange_counts.get(index).copied() != Some(expected_exchange_count)
            {
                return OracleVerdict::failed(
                    RUNTIME_SESSION_GRAPH_ORACLE,
                    format!(
                        "session `{}` turn {turn_index} provider exchanges did not converge to {expected_exchange_count}",
                        session.alias
                    ),
                );
            }
            let expected_min_count = turn_index * 2;
            if session.graph_node_counts.get(index).copied().unwrap_or(0) < expected_min_count {
                return OracleVerdict::failed(
                    RUNTIME_SESSION_GRAPH_ORACLE,
                    format!(
                        "session `{}` turn {turn_index} graph had fewer than {expected_min_count} nodes",
                        session.alias
                    ),
                );
            }
            if session
                .transcript_message_counts
                .get(index)
                .copied()
                .unwrap_or(0)
                < expected_min_count
            {
                return OracleVerdict::failed(
                    RUNTIME_SESSION_GRAPH_ORACLE,
                    format!(
                        "session `{}` turn {turn_index} transcript had fewer than {expected_min_count} messages",
                        session.alias
                    ),
                );
            }
            if !output.contains(&session.alias) {
                return OracleVerdict::failed(
                    RUNTIME_SESSION_GRAPH_ORACLE,
                    format!(
                        "session `{}` turn {turn_index} provider output did not identify its session",
                        session.alias
                    ),
                );
            }
        }
    }
    OracleVerdict::passed(
        RUNTIME_SESSION_GRAPH_ORACLE,
        "runtime-backed generated sessions advanced provider exchanges, graph nodes, and transcript messages across multiple turns",
    )
}

pub fn runtime_graph_acyclic(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut checked = 0;
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
    {
        let Some(facts) = runtime_observed_fact::<RuntimeGraphInvariantFacts>(event, "graph")
        else {
            return OracleVerdict::failed(
                RUNTIME_GRAPH_ACYCLIC_ORACLE,
                format!(
                    "provider boundary `{}` did not expose real graph invariant facts",
                    event.boundary_id
                ),
            );
        };
        checked += 1;
        if !facts.passed {
            return OracleVerdict::failed(
                RUNTIME_GRAPH_ACYCLIC_ORACLE,
                format!(
                    "provider boundary `{}` observed graph duplicates={:?} missing_parents={:?} cycles={:?} leaf_exists={}",
                    event.boundary_id,
                    facts.duplicate_node_ids,
                    facts.missing_parent_links,
                    facts.cycle_node_ids,
                    facts.leaf_exists
                ),
            );
        }
    }
    if checked == 0 {
        return OracleVerdict::failed(
            RUNTIME_GRAPH_ACYCLIC_ORACLE,
            "no provider turn exposed runtime graph invariant facts",
        );
    }
    OracleVerdict::passed(
        RUNTIME_GRAPH_ACYCLIC_ORACLE,
        format!(
            "{checked} real provider turn graphs had unique nodes, valid parents, and no cycles"
        ),
    )
}

pub fn runtime_single_active_agent_frame(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut checked = 0;
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
    {
        let Some(facts) =
            runtime_observed_fact::<RuntimeAgentFrameInvariantFacts>(event, "agent_frame")
        else {
            return OracleVerdict::failed(
                RUNTIME_SINGLE_ACTIVE_AGENT_FRAME_ORACLE,
                format!(
                    "provider boundary `{}` did not expose real Agent Frame facts",
                    event.boundary_id
                ),
            );
        };
        checked += 1;
        if !facts.passed {
            return OracleVerdict::failed(
                RUNTIME_SINGLE_ACTIVE_AGENT_FRAME_ORACLE,
                format!(
                    "provider boundary `{}` observed current_frame=`{}` active={:?} current_exists={} current_active={} unknown_node_frames={:?}",
                    event.boundary_id,
                    facts.current_agent_frame_id,
                    facts.active_frame_ids,
                    facts.current_frame_exists,
                    facts.current_frame_active,
                    facts.node_agent_frame_ids_without_record
                ),
            );
        }
    }
    if checked == 0 {
        return OracleVerdict::failed(
            RUNTIME_SINGLE_ACTIVE_AGENT_FRAME_ORACLE,
            "no provider turn exposed runtime Agent Frame facts",
        );
    }
    OracleVerdict::passed(
        RUNTIME_SINGLE_ACTIVE_AGENT_FRAME_ORACLE,
        format!(
            "{checked} real provider turn snapshots had exactly one active current Agent Frame"
        ),
    )
}

pub fn runtime_usage_monotonic(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut checked = 0;
    let mut last_context_total_by_session = BTreeMap::<String, i64>::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
    {
        let Some(facts) = runtime_observed_fact::<RuntimeUsageInvariantFacts>(event, "usage")
        else {
            return OracleVerdict::failed(
                RUNTIME_USAGE_MONOTONIC_ORACLE,
                format!(
                    "provider boundary `{}` did not expose real usage facts",
                    event.boundary_id
                ),
            );
        };
        checked += 1;
        if !facts.non_negative {
            return OracleVerdict::failed(
                RUNTIME_USAGE_MONOTONIC_ORACLE,
                format!(
                    "provider boundary `{}` observed negative token fields {:?}",
                    event.boundary_id, facts.negative_fields
                ),
            );
        }
        if !facts.usage_events_monotonic {
            return OracleVerdict::failed(
                RUNTIME_USAGE_MONOTONIC_ORACLE,
                format!(
                    "provider boundary `{}` emitted non-monotonic usage activity cumulatives",
                    event.boundary_id
                ),
            );
        }
        let current = facts.token_ledger_total.context_total_tokens;
        if let Some(previous) =
            last_context_total_by_session.insert(event.actor_alias.clone(), current)
            && current < previous
        {
            return OracleVerdict::failed(
                RUNTIME_USAGE_MONOTONIC_ORACLE,
                format!(
                    "session `{}` token ledger context total regressed from {previous} to {current} at `{}`",
                    event.actor_alias, event.boundary_id
                ),
            );
        }
    }
    if checked == 0 {
        return OracleVerdict::failed(
            RUNTIME_USAGE_MONOTONIC_ORACLE,
            "no provider turn exposed runtime usage facts",
        );
    }
    OracleVerdict::passed(
        RUNTIME_USAGE_MONOTONIC_ORACLE,
        format!(
            "{checked} real provider turns had non-negative usage and non-decreasing session ledger totals"
        ),
    )
}

pub fn durable_effect_exactly_once(summary: &AbstractWorldSummary) -> OracleVerdict {
    if summary.durable_effects.is_empty() {
        return OracleVerdict::failed(
            DURABLE_EFFECT_EXACTLY_ONCE_ORACLE,
            "workload did not execute a durable effect boundary",
        );
    }
    for effect in &summary.durable_effects {
        if effect.execution_count != 1 {
            return OracleVerdict::failed(
                DURABLE_EFFECT_EXACTLY_ONCE_ORACLE,
                format!(
                    "durable key `{}` executed {} times",
                    effect.durable_key, effect.execution_count
                ),
            );
        }
        if effect.replay_count == 0 {
            return OracleVerdict::failed(
                DURABLE_EFFECT_EXACTLY_ONCE_ORACLE,
                format!("durable key `{}` was never replayed", effect.durable_key),
            );
        }
    }
    OracleVerdict::passed(
        DURABLE_EFFECT_EXACTLY_ONCE_ORACLE,
        "durable effect replay reused the first semantic result for each durable key",
    )
}

pub fn worker_stale_completion_rejected(summary: &AbstractWorldSummary) -> OracleVerdict {
    if summary.workers.iter().any(|worker| {
        worker.lease_owner_changes > 0
            && worker.stale_completion_rejections > 0
            && !worker.active_incarnation_id.is_empty()
            && worker.active_fencing_token > 1
    }) {
        return OracleVerdict::passed(
            WORKER_STALE_COMPLETION_ORACLE,
            "worker topology rejected a stale completion after an incarnation change",
        );
    }
    OracleVerdict::failed(
        WORKER_STALE_COMPLETION_ORACLE,
        "no worker boundary proved stale completion rejection after lease owner change",
    )
}

pub const WORKER_FAILOVER_CONTINUATION_ORACLE: &str =
    "sim.oracle.worker-failover-continues-work.v1";

/// Real worker FAILOVER CONTINUATION: a second worker incarnation reclaimed the
/// crashed first owner's session-execution lease at a strictly higher fencing
/// token and COMMITTED (continued) the queued work the dead owner could not,
/// while the dead owner's stale work completion was rejected. These facts are
/// produced by the real lease/queued-work store in
/// `runtime_boundaries::run_worker_stale_completion` (start_worker_owned_work +
/// resume_crashed_worker_work), so this oracle can only pass when a real
/// successor actually continued real work — not merely when stale-completion
/// evidence was observed.
pub fn worker_failover_continues_work(events: &[DeliveredBoundary]) -> OracleVerdict {
    if events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Worker)
        .any(worker_owned_work_continued_by_successor)
    {
        return OracleVerdict::passed(
            WORKER_FAILOVER_CONTINUATION_ORACLE,
            "a second worker incarnation reclaimed the dead owner's lease at a strictly higher fence and committed (continued) the queued work the crashed first owner could not, rejecting its stale completion",
        );
    }
    OracleVerdict::failed(
        WORKER_FAILOVER_CONTINUATION_ORACLE,
        "no worker boundary proved second-owner failover CONTINUATION of the first owner's in-flight work (claimed -> reclaimed at higher fence -> continued -> stale rejected)",
    )
}

fn worker_owned_work_continued_by_successor(event: &DeliveredBoundary) -> bool {
    let Some(work) = event
        .observed
        .get("runtime_worker_store")
        .and_then(|store| store.get("worker_owned_work"))
    else {
        return false;
    };
    let flag = |key: &str| work.get(key).and_then(Value::as_bool).unwrap_or(false);
    flag("first_owner_claimed_work")
        && flag("second_owner_resumed_work")
        && flag("second_owner_outranks_first")
        && flag("stale_work_completion_rejected")
}

/// Assert that the real session-execution-lease fencing tokens recorded by the
/// lease-time boundaries strictly increase per session. Unlike the old
/// generator-fed tick (which was monotonic by construction and could never
/// fail), this reads the ground-truth fencing token the in-memory/SQLite lease
/// store handed back, so a broken fencing implementation that reissued or
/// regressed a token would fail this oracle.
pub fn lease_time_monotonic(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut last_by_session: BTreeMap<&str, u64> = BTreeMap::new();
    let mut grounded = 0usize;
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::LeaseTime)
    {
        let Some(fencing_token) = event
            .observed
            .pointer("/runtime_lease_probe/session_execution_lease_fencing_token")
            .and_then(Value::as_u64)
        else {
            return OracleVerdict::failed(
                LEASE_TIME_MONOTONIC_ORACLE,
                format!(
                    "lease-time boundary `{}` recorded no real lease fencing token",
                    event.boundary_id
                ),
            );
        };
        grounded += 1;
        if let Some(previous) = last_by_session.insert(event.actor_alias.as_str(), fencing_token)
            && fencing_token <= previous
        {
            return OracleVerdict::failed(
                LEASE_TIME_MONOTONIC_ORACLE,
                format!(
                    "session `{}` lease fencing token did not advance: {previous} -> {fencing_token}",
                    event.actor_alias
                ),
            );
        }
    }
    OracleVerdict::passed(
        LEASE_TIME_MONOTONIC_ORACLE,
        format!(
            "{grounded} lease-time boundaries advanced a real session-execution-lease fencing token monotonically per session"
        ),
    )
}

pub fn scheduler_controlled_delivery(events: &[DeliveredBoundary]) -> OracleVerdict {
    if events.is_empty() {
        return OracleVerdict::failed(
            SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
            "generated workload delivered no scheduler boundaries",
        );
    }
    for (sequence, event) in events.iter().enumerate() {
        if event.sequence != sequence {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` had sequence {}, expected {}",
                    event.boundary_id, event.sequence, sequence
                ),
            );
        }
        if !event.scheduler.scheduler_controlled {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` was delivered outside scheduler control",
                    event.boundary_id
                ),
            );
        }
        if event.scheduler.delivered_at != event.at {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` delivery tick {} diverged from scheduled tick {}",
                    event.boundary_id, event.scheduler.delivered_at, event.at
                ),
            );
        }
        if event.scheduler.min_scheduled_at > event.scheduler.delivered_at {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` was delivered before the scheduler's minimum tick",
                    event.boundary_id
                ),
            );
        }
        if event.scheduler.candidate_count_at_tick == 0 {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` recorded zero scheduler candidates",
                    event.boundary_id
                ),
            );
        }
        if event.scheduler.selected_candidate_index >= event.scheduler.candidate_count_at_tick {
            return OracleVerdict::failed(
                SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
                format!(
                    "boundary `{}` selected candidate {} out of {}",
                    event.boundary_id,
                    event.scheduler.selected_candidate_index,
                    event.scheduler.candidate_count_at_tick
                ),
            );
        }
    }
    OracleVerdict::passed(
        SCHEDULER_CONTROLLED_DELIVERY_ORACLE,
        "generated trace records scheduler-owned delivery order, timing, and tie-break evidence",
    )
}

pub fn scheduler_owned_runtime_completions(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut missing = Vec::new();
    for kind in [
        BoundaryKind::Provider,
        BoundaryKind::Cancellation,
        BoundaryKind::BackendFailure,
        BoundaryKind::ProviderMutation,
        BoundaryKind::Tool,
        BoundaryKind::ExecCode,
        BoundaryKind::DurableEffect,
        BoundaryKind::Worker,
    ] {
        let mut saw_kind = false;
        for event in events
            .iter()
            .filter(|event| event.kind == kind && !is_suspend_resume(event))
        {
            saw_kind = true;
            let Some(completion) = event.payload.get("runtime_completion") else {
                return OracleVerdict::failed(
                    SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE,
                    format!(
                        "runtime completion `{}` for {kind:?} was delivered without pending runtime boundary evidence",
                        event.boundary_id
                    ),
                );
            };
            if !event.scheduler.scheduler_controlled {
                return OracleVerdict::failed(
                    SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE,
                    format!(
                        "runtime completion `{}` for {kind:?} bypassed scheduler control",
                        event.boundary_id
                    ),
                );
            }
            let family = completion
                .get("completion_family")
                .and_then(Value::as_str)
                .unwrap_or("");
            let units = completion
                .get("completion_units")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            let ready_at = completion
                .get("ready_at")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let registered_after = completion
                .get("registered_after")
                .and_then(Value::as_str)
                .unwrap_or("");
            if family.is_empty()
                || units == 0
                || ready_at != event.at
                || registered_after.is_empty()
            {
                return OracleVerdict::failed(
                    SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE,
                    format!(
                        "runtime completion `{}` for {kind:?} had incomplete pending evidence: family=`{family}` units={units} ready_at={ready_at} registered_after=`{registered_after}`",
                        event.boundary_id
                    ),
                );
            }
        }
        if !saw_kind {
            missing.push(format!("{kind:?}"));
        }
    }
    if !missing.is_empty() {
        return OracleVerdict::failed(
            SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE,
            format!(
                "generated trace did not include scheduler-owned runtime completion kinds: {}",
                missing.join(", ")
            ),
        );
    }
    OracleVerdict::passed(
        SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE,
        "provider chunks/retries, cancellation, tool returns, exec results, durable completions, provider mutations, and worker completions were registered as pending runtime boundaries and delivered by the scheduler",
    )
}

pub fn operational_coverage(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let mut missing = Vec::new();
    if !summary
        .sessions
        .iter()
        .any(|session| session.queued_ingress_count > 0)
    {
        missing.push("queueing inputs");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| session.trigger_count > 0)
    {
        missing.push("triggers");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| session.cancellation_count > 0)
    {
        missing.push("cancellation");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| session.observer_reconnects > 0)
    {
        missing.push("observer reconnects");
    }
    let provider_mutations = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProviderMutation)
        .filter_map(|event| {
            event
                .observed
                .get("mutation")
                .or_else(|| event.payload.get("mutation"))
                .and_then(serde_json::Value::as_str)
        })
        .collect::<BTreeSet<_>>();
    if !provider_mutations.contains("malformed_sse_chunk")
        || !provider_mutations.contains("rate_limit_error_envelope")
    {
        missing.push("provider failures/mutations");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| session.process_wake_count > 0)
    {
        missing.push("process wakes");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| !session.tool_outputs.is_empty())
    {
        missing.push("tool results");
    }
    if !summary
        .sessions
        .iter()
        .any(|session| !session.exec_code_outputs.is_empty())
    {
        missing.push("exec-code results");
    }
    if !summary
        .durable_effects
        .iter()
        .any(|effect| effect.execution_count == 1 && effect.replay_count > 0)
    {
        missing.push("durable effects");
    }
    if !summary
        .workers
        .iter()
        .any(|worker| worker.lease_owner_changes > 0 && worker.stale_completion_rejections > 0)
    {
        missing.push("worker lease/failover");
    }
    let backend_retryable = events.iter().any(|event| {
        event.kind == BoundaryKind::BackendFailure
            && event
                .observed
                .get("retryable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
    });
    let backend_terminal = events.iter().any(|event| {
        event.kind == BoundaryKind::BackendFailure
            && !event
                .observed
                .get("retryable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true)
    });
    if !backend_retryable || !backend_terminal {
        missing.push("backend choices");
    }
    let backend_retry_attempt = events.iter().any(|event| {
        event.kind == BoundaryKind::BackendFailure
            && event
                .observed
                .get("attempt")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 1
    });
    let duplicate_wake_rejected = events.iter().any(|event| {
        event.kind == BoundaryKind::ProcessWake
            && !event
                .observed
                .get("claimed_once")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true)
    });
    if !backend_retry_attempt || !duplicate_wake_rejected {
        missing.push("retries/duplicates");
    }

    if missing.is_empty() {
        OracleVerdict::passed(
            OPERATIONAL_COVERAGE_ORACLE,
            "generated DST trace hit queueing, triggers, cancellation, observer reconnects, provider failure/mutation, process wake, tool/exec, durable effect, worker failover, backend choice, retry, and duplicate cases",
        )
    } else {
        OracleVerdict::failed(
            OPERATIONAL_COVERAGE_ORACLE,
            format!(
                "generated DST trace missed operational cases: {}",
                missing.join(", ")
            ),
        )
    }
}

pub fn state_machine_semantic_invariants(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let mut missing = Vec::new();
    if !queued_active_turn_input_hidden_semantics(events) {
        missing.push("queued active-turn input hidden from live provider turns");
    }
    if !cancellation_terminalizes_pending_input(events) {
        missing.push("cancellation terminalizes a pending queued input");
    }
    if !trigger_wakeup_route_semantics(events) {
        missing.push("trigger wakeup routes through TriggerStore reservation");
    }
    if !backend_retry_terminalization_semantics(events) {
        missing.push("backend retry terminalization");
    }
    if !duplicate_delivery_semantics(events, summary) {
        missing.push("duplicate delivery/replay semantics");
    }
    if !protocol_terminal_state_semantics(events, summary) {
        missing.push("provider/protocol terminal state semantics");
    }

    if missing.is_empty() {
        OracleVerdict::passed(
            STATE_MACHINE_SEMANTIC_INVARIANTS_ORACLE,
            "queued input, cancellation, trigger wakeup, retry terminalization, duplicate delivery/replay, and protocol terminal-state invariants held",
        )
    } else {
        OracleVerdict::failed(
            STATE_MACHINE_SEMANTIC_INVARIANTS_ORACLE,
            format!(
                "generated DST trace violated semantic state-machine invariants: {}",
                missing.join(", ")
            ),
        )
    }
}

/// Scenario-contract oracle vector. The RUNTIME suite is genuinely exercised
/// per-contract by the generated runtime state machine, so each runtime contract
/// emits its OWN distinct, failing-capable verdict. The STANDARD/RLM/AGENT
/// protocol suites are NOT distinctly exercised per-contract by the generated
/// workload (extending it to per-contract evidence is out of scope this pass), so
/// rather than stamp ~36 shared-bucket verdicts and overstate the oracle count,
/// each of those suites collapses to ONE explicit, failing-capable SUITE-LEVEL
/// coverage-manifest verdict (see `scenario_suite_coverage_manifest`). The real
/// per-behavior failing oracles for those suites are the mini-oracles
/// (`scenario_contract_mini_oracles`); the per-contract protocol semantics are
/// validated upstream by each protocol crate's own unit tests.
pub fn scenario_contract_oracles(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> Vec<OracleVerdict> {
    let mut verdicts = RUNTIME_SCENARIO_CONTRACTS
        .iter()
        .map(|contract| scenario_contract_oracle(contract, events, summary))
        .collect::<Vec<_>>();
    verdicts.push(scenario_suite_coverage_manifest(
        STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
        events,
        summary,
    ));
    verdicts.push(scenario_suite_coverage_manifest(
        RLM_PROTOCOL_SCENARIO_CONTRACTS,
        events,
        summary,
    ));
    verdicts.push(scenario_suite_coverage_manifest(
        AGENT_SCENARIO_CONTRACTS,
        events,
        summary,
    ));
    verdicts
}

/// The suite-level coverage-manifest oracle id for a protocol suite (e.g.
/// `sim.oracle.scenario.standard-contract.v1:coverage-manifest`). Stable so the
/// generated slice/package machinery can attach this verdict to every contract
/// in the suite as its coverage backing.
pub fn scenario_suite_coverage_manifest_oracle_id(contract: &ScenarioContractSpec) -> String {
    format!("{}:coverage-manifest", contract.oracle_id)
}

/// One failing-capable, suite-level coverage manifest for a protocol suite. This
/// is explicitly a COVERAGE MANIFEST, not a per-contract oracle: it enumerates
/// the protocol contracts the generated lane covers at the suite level and
/// asserts the suite-level evidence the generated workload genuinely produces. If
/// that evidence regresses the manifest fails loudly; the per-contract semantics
/// remain validated upstream by the protocol crate and by the suite mini-oracles.
fn scenario_suite_coverage_manifest(
    contracts: &'static [ScenarioContractSpec],
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let first = contracts
        .first()
        .expect("protocol suite must declare at least one contract");
    let suite = first.suite;
    let oracle_id = scenario_suite_coverage_manifest_oracle_id(first);
    let covered = contracts
        .iter()
        .map(|contract| contract.test_name)
        .collect::<Vec<_>>()
        .join(", ");
    let evidence = suite_coverage_evidence(suite, events, summary);
    if evidence.passed {
        OracleVerdict::passed(
            oracle_id,
            format!(
                "{suite} suite coverage manifest: the generated lane covers {} protocol contracts [{covered}] at the suite level; suite evidence held ({}). Per-contract semantics are validated upstream by the {suite} protocol crate's own tests and by the {suite} mini-oracles; not re-derived per-contract here.",
                contracts.len(),
                evidence.reason
            ),
        )
    } else {
        OracleVerdict::failed(
            oracle_id,
            format!(
                "{suite} suite coverage manifest FAILED: suite-level evidence missing ({}) while covering contracts [{covered}]",
                evidence.reason
            ),
        )
    }
}

/// Suite-level evidence the generated workload must produce for a protocol suite,
/// expressed as the conjunction of the distinct runtime DTO/projection facts the
/// suite's contracts depend on (no `|`-grouped per-contract stamping, no global
/// `any-session>0` presence bucket). Failing-capable: a regressed fact fails the
/// suite coverage manifest.
fn suite_coverage_evidence(
    suite: &str,
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> ScenarioSemanticVerdict {
    match suite {
        "standard" => assert_semantic(
            provider_mutation_parser_matrix_observed(events)
                && tool_provider_reentry_observed(events, summary)
                && tool_runtime_output_observed(events)
                && tool_effects_execute_once(events)
                && duplicate_free_stream_finalization(events, summary)
                && provider_exchange_counts_are_turn_indexed(summary)
                && observer_convergence(summary).is_passed(),
            "provider parser matrix, tool-loop re-entry + single tool effect, duplicate-free streamed finalization, and turn-indexed exchanges with observer convergence",
        ),
        "rlm" => assert_semantic(
            exec_runtime_outcome_observed(events)
                && exec_effects_execute_once(events)
                && exec_outcome_has_no_tool_call_replay(events)
                && trigger_delivery_runtime_observed(events)
                && provider_mutation_parser_matrix_observed(events)
                && provider_mutation_classes_observed(events)
                && provider_exchange_counts_are_turn_indexed(summary)
                && observer_convergence(summary).is_passed(),
            "exec-code RuntimeEffectOutcome (single execution, no replayed tool calls), trigger delivery, provider mutation parser matrix + classes, and turn-indexed exchanges with observer convergence",
        ),
        "agent" => assert_semantic(
            process_wake_runtime_dto_observed(events)
                && durable_runtime_effect_observed(events)
                && exec_runtime_outcome_observed(events)
                && tool_runtime_output_observed(events)
                && observer_reconnect_has_matching_turn(events, summary)
                && provider_exchange_counts_are_turn_indexed(summary)
                && observer_convergence(summary).is_passed()
                && summary.session_count >= 2,
            "process wake DTO, durable effect, exec + tool facade outputs, observer reconnect convergence, turn-indexed exchanges, and a multi-session graph",
        ),
        other => ScenarioSemanticVerdict::failed(format!(
            "unknown protocol suite `{other}` has no suite coverage evidence"
        )),
    }
}

pub fn scenario_contract_mini_oracles(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> Vec<OracleVerdict> {
    vec![
        mini_runtime_queued_input_hidden(events),
        mini_runtime_cancellation_prevents_idle_claim(events),
        mini_runtime_process_wake_duplicate_rejected(events),
        mini_runtime_stale_lease_commit_rejected(events, summary),
        mini_standard_streamed_text_finalizes_once(events),
        mini_standard_provider_error_without_checkpoint(events),
        mini_standard_tool_loop_reenters(events),
        mini_rlm_finish_required_prose_repair(events, summary),
        mini_rlm_schema_mismatch_repair(events),
        mini_rlm_lashlang_cell_exec_continues(events),
        mini_agent_durable_input_resolution(events),
        mini_agent_child_failure_graph(events, summary),
        mini_agent_parallel_spawn_join(events, summary),
    ]
}

fn mini_runtime_queued_input_hidden(events: &[DeliveredBoundary]) -> OracleVerdict {
    let Some(queued) = first_event(events, BoundaryKind::QueuedIngress) else {
        return OracleVerdict::failed(
            SCENARIO_MINI_RUNTIME_QUEUED_HIDDEN_ORACLE,
            "no queued input boundary was generated",
        );
    };
    let queued_text = queued
        .payload
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("");
    let leaked = events.iter().any(|event| {
        event.kind == BoundaryKind::Provider
            && event.actor_alias == queued.actor_alias
            && event
                .observed
                .get("provider_output")
                .and_then(Value::as_str)
                .is_some_and(|output| !queued_text.is_empty() && output.contains(queued_text))
    });
    if leaked {
        OracleVerdict::failed(
            SCENARIO_MINI_RUNTIME_QUEUED_HIDDEN_ORACLE,
            format!(
                "queued input `{}` leaked into a provider turn before explicit claim",
                queued.boundary_id
            ),
        )
    } else {
        OracleVerdict::passed(
            SCENARIO_MINI_RUNTIME_QUEUED_HIDDEN_ORACLE,
            "queued input stayed hidden from provider turns while the generated live turn continued",
        )
    }
}

fn mini_runtime_cancellation_prevents_idle_claim(events: &[DeliveredBoundary]) -> OracleVerdict {
    let queued = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::QueuedIngress)
        .map(|event| event.boundary_id.as_str())
        .collect::<BTreeSet<_>>();
    let Some(cancel) = events.iter().find(|event| {
        event.kind == BoundaryKind::Cancellation
            && event
                .observed
                .get("cancelled")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    }) else {
        let outcomes = events
            .iter()
            .filter(|event| event.kind == BoundaryKind::Cancellation)
            .map(|event| {
                format!(
                    "{}={}",
                    event.boundary_id,
                    event
                        .observed
                        .get("cancel_outcome")
                        .and_then(Value::as_str)
                        .unwrap_or("missing")
                )
            })
            .collect::<Vec<_>>();
        return OracleVerdict::failed(
            SCENARIO_MINI_RUNTIME_CANCEL_IDLE_ORACLE,
            format!(
                "no cancellation boundary reported a real runtime cancellation; outcomes={}",
                outcomes.join(",")
            ),
        );
    };
    let target = cancel
        .observed
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("");
    if queued.contains(target) {
        OracleVerdict::passed(
            SCENARIO_MINI_RUNTIME_CANCEL_IDLE_ORACLE,
            "cancelled queued input targets a generated queued boundary and prevents later idle claim",
        )
    } else {
        OracleVerdict::failed(
            SCENARIO_MINI_RUNTIME_CANCEL_IDLE_ORACLE,
            format!("cancellation target `{target}` was not a generated queued boundary"),
        )
    }
}

fn mini_runtime_process_wake_duplicate_rejected(events: &[DeliveredBoundary]) -> OracleVerdict {
    let mut dedupe_events: BTreeMap<String, Vec<&DeliveredBoundary>> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProcessWake)
    {
        let key = event
            .observed
            .get("dedupe_key")
            .or_else(|| event.payload.get("dedupe_key"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        dedupe_events.entry(key).or_default().push(event);
    }
    if dedupe_events.values().any(|events| {
        let claims = events
            .iter()
            .filter_map(|event| event.observed.get("claimed_once").and_then(Value::as_bool))
            .collect::<Vec<_>>();
        let strict_claim_dedupe =
            claims.iter().filter(|claimed| **claimed).count() == 1 && claims.contains(&false);
        let in_flight_rejection = events.iter().any(|event| {
            event
                .observed
                .get("lease_busy")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && event
                    .observed
                    .pointer("/runtime_queued_work/enqueued")
                    .and_then(Value::as_bool)
                    == Some(false)
        }) && claims.contains(&false);
        strict_claim_dedupe || in_flight_rejection
    }) {
        OracleVerdict::passed(
            SCENARIO_MINI_RUNTIME_PROCESS_WAKE_DEDUPE_ORACLE,
            "duplicate process wake used the same dedupe key and was rejected by runtime queued-work claims or by an in-flight session lease",
        )
    } else {
        OracleVerdict::failed(
            SCENARIO_MINI_RUNTIME_PROCESS_WAKE_DEDUPE_ORACLE,
            "no process wake dedupe key showed a claim/rejection pair or in-flight lease rejection",
        )
    }
}

fn mini_runtime_stale_lease_commit_rejected(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let stale_rejected = worker_runtime_lease_dto_observed(events)
        && summary.workers.iter().any(|worker| {
            worker.active_fencing_token > 1 && worker.stale_completion_rejections > 0
        });
    verdict_from_bool(
        SCENARIO_MINI_RUNTIME_STALE_LEASE_ORACLE,
        stale_rejected,
        "stale worker lease completion was rejected while the reclaimed live lease stayed renewable",
        "worker lease evidence did not prove stale completion rejection",
    )
}

fn mini_standard_streamed_text_finalizes_once(events: &[DeliveredBoundary]) -> OracleVerdict {
    let Some(provider) = events.iter().find(|event| {
        event.kind == BoundaryKind::Provider
            && provider_completion_units(event)
                .iter()
                .filter(|unit| unit.contains("sse"))
                .count()
                >= 2
    }) else {
        return OracleVerdict::failed(
            SCENARIO_MINI_STANDARD_STREAM_FINALIZE_ORACLE,
            "no streamed provider completion with multiple scheduler-owned SSE units was observed",
        );
    };
    let expected = provider
        .payload
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("");
    let output = provider
        .observed
        .get("provider_output")
        .and_then(Value::as_str)
        .unwrap_or("");
    let occurrences = if expected.is_empty() {
        0
    } else {
        output.matches(expected).count()
    };
    verdict_from_bool(
        SCENARIO_MINI_STANDARD_STREAM_FINALIZE_ORACLE,
        occurrences == 1,
        "streamed provider output contains exactly one final assistant text projection",
        "streamed provider output did not finalize exactly once",
    )
}

fn mini_standard_provider_error_without_checkpoint(events: &[DeliveredBoundary]) -> OracleVerdict {
    let failure_before_provider = events.iter().any(|event| {
        matches!(
            event.kind,
            BoundaryKind::ProviderMutation | BoundaryKind::BackendFailure
        ) && event.payload.get("runtime_completion").is_some()
            && event.sequence
                < next_provider_sequence(events, &event.actor_alias).unwrap_or(usize::MAX)
    });
    verdict_from_bool(
        SCENARIO_MINI_STANDARD_PROVIDER_ERROR_ORACLE,
        failure_before_provider && provider_mutation_parser_matrix_observed(events),
        "provider failure/mutation completed through scheduler before any later successful checkpointed provider turn",
        "no scheduler-owned provider failure before checkpoint/next provider turn was observed",
    )
}

fn mini_standard_tool_loop_reenters(events: &[DeliveredBoundary]) -> OracleVerdict {
    let Some(tool) = first_event(events, BoundaryKind::Tool) else {
        return OracleVerdict::failed(
            SCENARIO_MINI_STANDARD_TOOL_REENTRY_ORACLE,
            "no tool boundary was generated",
        );
    };
    let reentered = events.iter().any(|event| {
        event.kind == BoundaryKind::Provider
            && event.actor_alias == tool.actor_alias
            && event.sequence > tool.sequence
    });
    verdict_from_bool(
        SCENARIO_MINI_STANDARD_TOOL_REENTRY_ORACLE,
        reentered && tool_runtime_output_observed(events),
        "tool result was captured and a later provider turn re-entered the model loop for the same session",
        "tool result did not have a later provider re-entry",
    )
}

fn mini_rlm_finish_required_prose_repair(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    verdict_from_bool(
        SCENARIO_MINI_RLM_FINISH_REPAIR_ORACLE,
        provider_exchange_counts_are_turn_indexed(summary)
            && observer_reconnect_has_matching_turn(events, summary)
            && events
                .iter()
                .filter(|event| event.kind == BoundaryKind::Provider)
                .any(|event| event.payload.get("runtime_completion").is_some()),
        "finish-required repair mini-replay observed repeated provider completions and observer convergence",
        "finish-required repair mini-replay lacked repeated provider/observer convergence",
    )
}

fn mini_rlm_schema_mismatch_repair(events: &[DeliveredBoundary]) -> OracleVerdict {
    let saw_schema_mutation = events.iter().any(|event| {
        event.kind == BoundaryKind::ProviderMutation
            && event
                .observed
                .pointer("/provider_parser_matrix/matrix/real_provider_parser_execution")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });
    verdict_from_bool(
        SCENARIO_MINI_RLM_SCHEMA_REPAIR_ORACLE,
        saw_schema_mutation && provider_mutation_classes_observed(events),
        "schema mismatch repair mini-replay used mutated provider scripts through real provider parsers",
        "schema mismatch repair mini-replay lacked provider parser mutation evidence",
    )
}

fn mini_rlm_lashlang_cell_exec_continues(events: &[DeliveredBoundary]) -> OracleVerdict {
    let Some(exec) = first_event(events, BoundaryKind::ExecCode) else {
        return OracleVerdict::failed(
            SCENARIO_MINI_RLM_CELL_EXEC_ORACLE,
            "no exec-code boundary was generated",
        );
    };
    let continued = events.iter().any(|event| {
        event.kind == BoundaryKind::Provider
            && event.actor_alias == exec.actor_alias
            && event.sequence > exec.sequence
    });
    verdict_from_bool(
        SCENARIO_MINI_RLM_CELL_EXEC_ORACLE,
        continued && exec_runtime_outcome_observed(events),
        "lashlang cell exec mini-replay produced an exec outcome and continued to a later provider turn",
        "lashlang cell exec mini-replay did not continue after exec",
    )
}

fn mini_agent_durable_input_resolution(events: &[DeliveredBoundary]) -> OracleVerdict {
    let durable = events.iter().any(|event| {
        event.kind == BoundaryKind::DurableEffect
            && event
                .observed
                .get("replayed")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });
    verdict_from_bool(
        SCENARIO_MINI_AGENT_DURABLE_INPUT_ORACLE,
        durable && process_wake_runtime_dto_observed(events) && observer_reconnect_has_any(events),
        "durable input mini-replay observed durable replay, process wake, and observer reconnect resolution",
        "durable input mini-replay lacked durable replay/process wake/observer reconnect evidence",
    )
}

fn mini_agent_child_failure_graph(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    verdict_from_bool(
        SCENARIO_MINI_AGENT_CHILD_FAILURE_ORACLE,
        summary.session_count >= 2
            && worker_runtime_lease_dto_observed(events)
            && events
                .iter()
                .any(|event| event.kind == BoundaryKind::BackendFailure),
        "child failure mini-replay kept multi-session graph evidence while worker/backend failure boundaries executed",
        "child failure mini-replay lacked multi-session worker/backend failure evidence",
    )
}

fn mini_agent_parallel_spawn_join(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let wake_sessions = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProcessWake)
        .filter_map(|event| event.observed.get("session").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let ordered_sequences = events
        .iter()
        .filter(|event| matches!(event.kind, BoundaryKind::ProcessWake | BoundaryKind::Worker))
        .map(|event| event.sequence)
        .collect::<Vec<_>>();
    let deterministic_order = ordered_sequences.windows(2).all(|pair| pair[0] < pair[1]);
    verdict_from_bool(
        SCENARIO_MINI_AGENT_PARALLEL_JOIN_ORACLE,
        summary.session_count >= 2 && !wake_sessions.is_empty() && deterministic_order,
        "parallel spawn/join mini-replay recorded deterministic process/worker sequence ordering",
        "parallel spawn/join mini-replay did not record deterministic process/worker ordering",
    )
}

fn verdict_from_bool(
    oracle_id: &'static str,
    condition: bool,
    passed: &'static str,
    failed: &'static str,
) -> OracleVerdict {
    if condition {
        OracleVerdict::passed(oracle_id, passed)
    } else {
        OracleVerdict::failed(oracle_id, failed)
    }
}

fn first_event(events: &[DeliveredBoundary], kind: BoundaryKind) -> Option<&DeliveredBoundary> {
    events.iter().find(|event| event.kind == kind)
}

fn provider_completion_units(event: &DeliveredBoundary) -> Vec<String> {
    event
        .payload
        .pointer("/runtime_completion/completion_units")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|unit| unit.get("unit").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn next_provider_sequence(events: &[DeliveredBoundary], actor_alias: &str) -> Option<usize> {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider && event.actor_alias == actor_alias)
        .map(|event| event.sequence)
        .min()
}

fn observer_reconnect_has_any(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::Observer
            && event
                .observed
                .get("reconnected")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    })
}

fn scenario_contract_oracle(
    contract: &ScenarioContractSpec,
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> OracleVerdict {
    let missing = contract
        .required_sim_evidence
        .iter()
        .copied()
        .filter(|evidence| !scenario_evidence_satisfied(evidence, events, summary))
        .collect::<Vec<_>>();
    let oracle_id = scenario_contract_oracle_id(contract);
    let semantic = scenario_contract_semantics(contract, events, summary);
    if missing.is_empty() && semantic.passed {
        OracleVerdict::passed(
            oracle_id,
            format!(
                "{} contract `{}` passed semantic `{}`: {}. Evidence: {}; adapter: {}",
                contract.suite,
                contract.test_name,
                contract.semantic_oracle,
                contract.owned_invariant,
                contract.required_sim_evidence.join(", "),
                semantic.reason
            ),
        )
    } else {
        let mut failures = Vec::new();
        if !missing.is_empty() {
            failures.push(format!("missing evidence [{}]", missing.join(", ")));
        }
        if !semantic.passed {
            failures.push(format!("semantic adapter failed: {}", semantic.reason));
        }
        OracleVerdict::failed(
            oracle_id,
            format!(
                "{} contract `{}` failed {} for invariant: {}",
                contract.suite,
                contract.test_name,
                failures.join("; "),
                contract.owned_invariant
            ),
        )
    }
}

fn scenario_contract_oracle_id(contract: &ScenarioContractSpec) -> String {
    format!("{}:{}", contract.oracle_id, contract.test_name)
}

fn scenario_evidence_satisfied(
    evidence: &str,
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    match evidence {
        "queued_ingress" => summary
            .sessions
            .iter()
            .any(|session| session.queued_ingress_count > 0),
        "cancellation" => summary
            .sessions
            .iter()
            .any(|session| session.cancellation_count > 0),
        "process_wake" => {
            summary
                .sessions
                .iter()
                .any(|session| session.process_wake_count > 0)
                && process_wake_runtime_dto_observed(events)
        }
        "worker_stale_completion" => {
            summary.workers.iter().any(|worker| {
                worker.lease_owner_changes > 0
                    && worker.stale_completion_rejections > 0
                    && worker.active_fencing_token > 1
            }) && worker_runtime_lease_dto_observed(events)
        }
        "lease_time" => summary.sessions.iter().any(|session| {
            !session.lease_time_ticks.is_empty()
                && session
                    .lease_time_ticks
                    .windows(2)
                    .all(|ticks| ticks[0] <= ticks[1])
        }),
        "provider_turn" => {
            let provider_turns = summary
                .sessions
                .iter()
                .map(|session| session.provider_outputs.len())
                .sum::<usize>();
            summary.session_count > 0 && provider_turns >= summary.session_count * 2
        }
        "tool_result" => {
            summary
                .sessions
                .iter()
                .any(|session| !session.tool_outputs.is_empty())
                && tool_runtime_output_observed(events)
        }
        "observer_convergence" => {
            summary.session_count > 0
                && summary.sessions.iter().all(|session| {
                    session.observer_turn_indices.last().copied()
                        == Some(session.provider_outputs.len())
                })
        }
        "runtime_session_graph" => runtime_session_graph_contract(summary).is_passed(),
        "exec_code" => {
            summary
                .sessions
                .iter()
                .any(|session| !session.exec_code_outputs.is_empty())
                && exec_runtime_outcome_observed(events)
        }
        "trigger" => summary
            .sessions
            .iter()
            .any(|session| session.trigger_count > 0),
        "durable_effect" => {
            summary
                .durable_effects
                .iter()
                .any(|effect| effect.execution_count == 1 && effect.replay_count > 0)
                && durable_runtime_effect_observed(events)
        }
        "multi_session" => summary.session_count >= 2,
        "observer_reconnect" => summary
            .sessions
            .iter()
            .any(|session| session.observer_reconnects > 0),
        _ => false,
    }
}

struct ScenarioSemanticVerdict {
    passed: bool,
    reason: String,
}

impl ScenarioSemanticVerdict {
    fn passed(reason: impl Into<String>) -> Self {
        Self {
            passed: true,
            reason: reason.into(),
        }
    }

    fn failed(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: reason.into(),
        }
    }
}

/// Per-contract semantic adapter. Only the RUNTIME suite emits per-contract
/// oracles (the STANDARD/RLM/AGENT suites collapse to suite coverage manifests in
/// `scenario_contract_oracles`), so only runtime contracts reach this dispatcher.
fn scenario_contract_semantics(
    contract: &ScenarioContractSpec,
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> ScenarioSemanticVerdict {
    match contract.suite {
        "runtime" => runtime_contract_semantics(contract.semantic_oracle, events, summary),
        other => ScenarioSemanticVerdict::failed(format!(
            "suite `{other}` does not emit per-contract oracles; it is covered by a suite coverage manifest"
        )),
    }
}

fn runtime_contract_semantics(
    semantic_oracle: &str,
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> ScenarioSemanticVerdict {
    // Each runtime contract owns a DISTINCT semantic adapter keyed on the
    // specific runtime boundary it governs — no `|`-grouped shared arms and no
    // single global condition standing in for per-contract evidence.
    match semantic_oracle {
        "runtime.process_wake_claim" => assert_semantic(
            process_wake_runtime_dto_observed(events) && duplicate_process_wake_rejected(events),
            "process wake DTO materialized and the duplicate wake was deduped/rejected",
        ),
        // The dead owner's lease release/completion is rejected (commit fenced out).
        "runtime.lease_release_rejects_commit" => assert_semantic(
            worker_runtime_lease_dto_observed(events)
                && summary
                    .workers
                    .iter()
                    .any(|worker| worker.stale_completion_rejections > 0),
            "the stale lease holder's completion was rejected by the live fence",
        ),
        // A new incarnation reclaims the dead lease and the fence strictly advances.
        "runtime.dead_lease_reclaim_rejects_stale" => assert_semantic(
            worker_runtime_lease_dto_observed(events)
                && summary
                    .workers
                    .iter()
                    .any(|worker| worker.lease_owner_changes > 0 && worker.active_fencing_token > 1),
            "a second incarnation reclaimed the dead lease at a strictly higher fence",
        ),
        "runtime.checkpoint_redrive_cancel" => assert_semantic(
            queued_inputs_have_cancel_targets(events) && observer_convergence(summary).is_passed(),
            "queued input cancellation targeted a source key and observers converged",
        ),
        // The queued (next-turn) input stays pending/hidden while the live turn runs.
        "runtime.queued_work_keeps_pending_input" => assert_semantic(
            queued_ingress_has_source_keys(events) && provider_turns_after_queue(summary),
            "queued ingress kept a stable source key while live provider turns continued",
        ),
        // A queued turn input is eventually completed into a subsequent turn.
        "runtime.queued_turn_input_completion" => assert_semantic(
            queued_ingress_has_source_keys(events)
                && summary.sessions.iter().any(|session| {
                    session.queued_ingress_count > 0 && session.provider_outputs.len() >= 2
                }),
            "a queued turn input source key was followed by a completed subsequent turn",
        ),
        // A command-only queue drains against monotonic lease fencing tokens.
        "runtime.command_only_queue_drain" => assert_semantic(
            queued_ingress_has_source_keys(events) && lease_time_monotonic(events).is_passed(),
            "command queue source keys drained against monotonically advancing lease fences",
        ),
        // A command applied before turn work still lets later provider turns run.
        "runtime.command_before_turn_work" => assert_semantic(
            queued_ingress_has_source_keys(events)
                && provider_turns_after_queue(summary)
                && lease_time_monotonic(events).is_passed(),
            "a command queued before turn work preserved later turns and lease ordering",
        ),
        "runtime.observation_replay_preserves_input" => assert_semantic(
            observer_reconnect_has_matching_turn(events, summary),
            "observer reconnect replay converged to the final provider turn",
        ),
        other => unmapped_scenario_semantic("runtime", other),
    }
}

/// An unmapped scenario-contract semantic oracle. Each contract must own a
/// distinct semantic adapter; a new or renamed contract that reaches here fails
/// loudly rather than passing through a decorative shared fallback.
fn unmapped_scenario_semantic(suite: &str, semantic_oracle: &str) -> ScenarioSemanticVerdict {
    ScenarioSemanticVerdict::failed(format!(
        "{suite} scenario contract `{semantic_oracle}` has no per-contract semantic adapter; add distinct evidence instead of a generic fallback"
    ))
}

fn assert_semantic(condition: bool, reason: &'static str) -> ScenarioSemanticVerdict {
    if condition {
        ScenarioSemanticVerdict::passed(reason)
    } else {
        ScenarioSemanticVerdict::failed(reason)
    }
}

fn queued_active_turn_input_hidden_semantics(events: &[DeliveredBoundary]) -> bool {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::QueuedIngress)
        .any(|queued| {
            let source_key = queued.payload.get("source_key").and_then(Value::as_str);
            let observed_source_key = queued.observed.get("source_key").and_then(Value::as_str);
            let text = queued
                .payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("");
            let active_turn_queued = queued.payload.get("ingress_mode").and_then(Value::as_str)
                == Some("active_turn")
                && queued.observed.get("ingress_mode").and_then(Value::as_str)
                    == Some("active_turn")
                && queued
                    .observed
                    .get("input_state")
                    .and_then(Value::as_str)
                    .is_some_and(|state| state.starts_with("pending"))
                && source_key.is_some()
                && source_key == observed_source_key
                && queued
                    .observed
                    .get("active_turn_id")
                    .or_else(|| queued.payload.get("active_turn_id"))
                    .and_then(Value::as_str)
                    .is_some_and(|turn_id| !turn_id.is_empty());
            let active_turn_id = queued
                .observed
                .get("active_turn_id")
                .or_else(|| queued.payload.get("active_turn_id"))
                .and_then(Value::as_str);
            if !active_turn_queued {
                return false;
            }
            let live_turn_in_flight = active_turn_id.is_some_and(|turn_id| {
                let provider_completed_after_queue = events.iter().any(|event| {
                    event.kind == BoundaryKind::Provider
                        && event.boundary_id == turn_id
                        && event.actor_alias == queued.actor_alias
                        && event.sequence > queued.sequence
                        && event
                            .observed
                            .get("success")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                });
                let provider_release_after_queue = events.iter().any(|event| {
                    event.kind == BoundaryKind::ProviderEvent
                        && event.actor_alias == queued.actor_alias
                        && event.sequence > queued.sequence
                        && event
                            .payload
                            .get("turn_boundary_id")
                            .and_then(Value::as_str)
                            == Some(turn_id)
                        && event
                            .observed
                            .get("released_while_turn_pending")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                });
                provider_completed_after_queue && provider_release_after_queue
            });
            let leaked = events.iter().any(|event| {
                event.kind == BoundaryKind::Provider
                    && event.actor_alias == queued.actor_alias
                    && event.sequence > queued.sequence
                    && event
                        .observed
                        .get("provider_output")
                        .and_then(Value::as_str)
                        .is_some_and(|output| !text.is_empty() && output.contains(text))
            });
            live_turn_in_flight && !leaked
        })
}

fn cancellation_terminalizes_pending_input(events: &[DeliveredBoundary]) -> bool {
    let queued = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::QueuedIngress)
        .map(|event| (event.boundary_id.as_str(), event))
        .collect::<BTreeMap<_, _>>();
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Cancellation)
        .any(|cancel| {
            let Some(target) = cancel.observed.get("target").and_then(Value::as_str) else {
                return false;
            };
            let Some(queued) = queued.get(target).copied() else {
                return false;
            };
            let queued_text = queued
                .payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("");
            let registered_after = cancel
                .payload
                .pointer("/runtime_completion/registered_after")
                .and_then(Value::as_str);
            let cancelled = cancel
                .observed
                .get("cancelled")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && cancel
                    .observed
                    .get("cancel_outcome")
                    .and_then(Value::as_str)
                    == Some("cancelled")
                && cancel.sequence > queued.sequence
                && registered_after == Some(target);
            let leaked_after_cancel = events.iter().any(|event| {
                event.kind == BoundaryKind::Provider
                    && event.actor_alias == cancel.actor_alias
                    && event.sequence > cancel.sequence
                    && event
                        .observed
                        .get("provider_output")
                        .and_then(Value::as_str)
                        .is_some_and(|output| {
                            !queued_text.is_empty() && output.contains(queued_text)
                        })
            });
            cancelled && !leaked_after_cancel
        })
}

fn trigger_wakeup_route_semantics(events: &[DeliveredBoundary]) -> bool {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Trigger)
        .any(|event| {
            let payload_source_key = event.payload.get("source_key").and_then(Value::as_str);
            event
                .observed
                .get("trigger_delivered")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && event
                    .observed
                    .get("started_process")
                    .and_then(Value::as_bool)
                    == Some(true)
                && event.observed.get("session").and_then(Value::as_str)
                    == Some(event.actor_alias.as_str())
                && event
                    .observed
                    .get("occurrence_id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id.starts_with("trigger:"))
                && event
                    .observed
                    .get("reservation_count")
                    .and_then(Value::as_u64)
                    .is_some_and(|count| count > 0)
                && payload_source_key.is_some()
                && event.observed.get("source_key").and_then(Value::as_str) == payload_source_key
        })
}

fn backend_retry_terminalization_semantics(events: &[DeliveredBoundary]) -> bool {
    let mut by_operation: BTreeMap<String, Vec<&DeliveredBoundary>> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::BackendFailure)
    {
        let operation = event
            .observed
            .get("operation")
            .or_else(|| event.payload.get("operation"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        by_operation.entry(operation).or_default().push(event);
    }
    by_operation.values().any(|events| {
        let mut events = events.clone();
        events.sort_by_key(|event| event.sequence);
        let mut saw_retryable = false;
        let mut last_attempt = 0;
        for event in events {
            let attempt = event
                .observed
                .get("attempt")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if attempt == 0 || attempt <= last_attempt {
                return false;
            }
            last_attempt = attempt;
            let retryable = event
                .observed
                .get("retryable")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let store_error_retryable = event
                .observed
                .pointer("/production_store_error/retryable_class")
                .and_then(Value::as_bool)
                .unwrap_or(retryable);
            if retryable && store_error_retryable {
                saw_retryable = true;
                continue;
            }
            if saw_retryable
                && !retryable
                && !store_error_retryable
                && event
                    .observed
                    .get("store_error_class")
                    .and_then(Value::as_str)
                    == Some("terminal_backend_error")
            {
                return true;
            }
        }
        false
    })
}

fn duplicate_delivery_semantics(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    duplicate_process_wake_claim_semantics(events)
        && durable_effect_replay_semantics(events, summary)
}

fn duplicate_process_wake_claim_semantics(events: &[DeliveredBoundary]) -> bool {
    let mut by_dedupe: BTreeMap<String, Vec<&DeliveredBoundary>> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProcessWake)
    {
        let Some(dedupe_key) = event
            .observed
            .get("dedupe_key")
            .or_else(|| event.payload.get("dedupe_key"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        by_dedupe
            .entry(dedupe_key.to_string())
            .or_default()
            .push(event);
    }
    by_dedupe.values().any(|events| {
        let claims = events
            .iter()
            .filter_map(|event| event.observed.get("claimed_once").and_then(Value::as_bool))
            .collect::<Vec<_>>();
        let queued_claims = events
            .iter()
            .filter_map(|event| {
                event
                    .observed
                    .pointer("/runtime_queued_work/claimed")
                    .and_then(Value::as_bool)
            })
            .collect::<Vec<_>>();
        let strict_claim_dedupe = claims.iter().filter(|claimed| **claimed).count() == 1
            && claims.contains(&false)
            && queued_claims.iter().filter(|claimed| **claimed).count() == 1
            && queued_claims.contains(&false);
        let in_flight_rejection = events
            .iter()
            .filter(|event| {
                event
                    .observed
                    .get("lease_busy")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    && event.observed.get("runtime_process_wake").is_some()
                    && event
                        .observed
                        .pointer("/runtime_queued_work/enqueued")
                        .and_then(Value::as_bool)
                        == Some(false)
            })
            .count()
            > 0
            && claims.contains(&false);
        strict_claim_dedupe || in_flight_rejection
    })
}

fn durable_effect_replay_semantics(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    summary
        .durable_effects
        .iter()
        .any(|effect| effect.execution_count == 1 && effect.replay_count > 0)
        && {
            let mut by_key: BTreeMap<String, Vec<&DeliveredBoundary>> = BTreeMap::new();
            for event in events
                .iter()
                .filter(|event| event.kind == BoundaryKind::DurableEffect)
            {
                let Some(key) = event.observed.get("durable_key").and_then(Value::as_str) else {
                    continue;
                };
                by_key.entry(key.to_string()).or_default().push(event);
            }
            by_key.values().any(|events| {
                let first = events.iter().any(|event| {
                    event.observed.get("replayed").and_then(Value::as_bool) == Some(false)
                        && event
                            .observed
                            .pointer("/runtime_effect/local_executor_called")
                            .and_then(Value::as_bool)
                            == Some(true)
                });
                let replay = events.iter().any(|event| {
                    event.observed.get("replayed").and_then(Value::as_bool) == Some(true)
                        && event
                            .observed
                            .pointer("/runtime_effect/local_executor_called")
                            .and_then(Value::as_bool)
                            == Some(false)
                        && event
                            .observed
                            .get("execution_count")
                            .and_then(Value::as_u64)
                            == Some(1)
                        && event.observed.get("replay_count").and_then(Value::as_u64) == Some(1)
                });
                first && replay
            })
        }
}

fn protocol_terminal_state_semantics(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    duplicate_free_stream_finalization(events, summary)
        && provider_rate_limit_terminalized_by_all_migrated_parsers(events)
        && provider_dropped_terminal_event_classified(events)
}

fn queued_ingress_has_source_keys(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::QueuedIngress
            && event
                .observed
                .get("source_key")
                .and_then(Value::as_str)
                .is_some_and(|source_key| !source_key.is_empty())
    })
}

fn queued_inputs_have_cancel_targets(events: &[DeliveredBoundary]) -> bool {
    let queued = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::QueuedIngress)
        .map(|event| event.boundary_id.as_str())
        .collect::<BTreeSet<_>>();
    events.iter().any(|event| {
        event.kind == BoundaryKind::Cancellation
            && event
                .observed
                .get("target")
                .and_then(Value::as_str)
                .is_some_and(|target| queued.contains(target))
    })
}

fn provider_turns_after_queue(summary: &AbstractWorldSummary) -> bool {
    summary
        .sessions
        .iter()
        .any(|session| session.queued_ingress_count > 0 && session.provider_outputs.len() >= 2)
}

fn process_wake_runtime_dto_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::ProcessWake
            && event.observed.get("runtime_process_wake").is_some()
            && event
                .observed
                .pointer("/runtime_process_wake/event_invocation/subject/process_id")
                .and_then(Value::as_str)
                .is_some()
    })
}

fn duplicate_process_wake_rejected(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::ProcessWake
            && !event
                .observed
                .get("claimed_once")
                .and_then(Value::as_bool)
                .unwrap_or(true)
    })
}

fn worker_runtime_lease_dto_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::Worker
            && event.observed.get("runtime_active_lease").is_some()
            && event.observed.get("runtime_stale_completion").is_some()
            && event
                .observed
                .get("stale_completion_rejected")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    })
}

fn durable_runtime_effect_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::DurableEffect
            && event.observed.get("runtime_effect").is_some()
            && event
                .observed
                .get("execution_count")
                .and_then(Value::as_u64)
                == Some(1)
    })
}

fn tool_runtime_output_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::Tool
            && event.observed.get("runtime_tool_output").is_some()
            && event
                .observed
                .get("execution_count")
                .and_then(Value::as_u64)
                == Some(1)
    })
}

fn tool_effects_execute_once(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::Tool
            && event
                .observed
                .get("execution_count")
                .and_then(Value::as_u64)
                == Some(1)
            && event.observed.get("runtime_tool_record").is_some()
    })
}

fn tool_provider_reentry_observed(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    tool_runtime_output_observed(events)
        && runtime_session_graph_contract(summary).is_passed()
        && summary.sessions.iter().any(|session| {
            !session.tool_outputs.is_empty()
                && session.provider_outputs.len() >= 2
                && session.provider_exchange_counts.len() >= 2
        })
}

fn exec_runtime_outcome_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::ExecCode
            && event.observed.get("runtime_effect_outcome").is_some()
            && event
                .observed
                .get("execution_count")
                .and_then(Value::as_u64)
                == Some(1)
    })
}

fn exec_effects_execute_once(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::ExecCode
            && event
                .observed
                .get("execution_count")
                .and_then(Value::as_u64)
                == Some(1)
            && event.observed.get("runtime_effect_outcome").is_some()
    })
}

fn exec_outcome_has_no_tool_call_replay(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        if event.kind != BoundaryKind::ExecCode {
            return false;
        }
        let Some(outcome) = event.observed.get("runtime_effect_outcome") else {
            return false;
        };
        if outcome.get("type").and_then(Value::as_str) != Some("exec_code") {
            return false;
        }
        let serialized = outcome.to_string();
        serialized.contains("\"tool_calls\":[]") || !serialized.contains("\"tool_calls\"")
    })
}

fn duplicate_free_stream_finalization(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    let provider_count = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::Provider)
        .count();
    let observed_turns = summary
        .sessions
        .iter()
        .map(|session| session.provider_outputs.len())
        .sum::<usize>();
    provider_count == observed_turns
        && provider_exchange_counts_are_turn_indexed(summary)
        && observer_convergence(summary).is_passed()
}

fn provider_mutation_parser_matrix_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::ProviderMutation
            && event
                .observed
                .pointer("/provider_parser_matrix/matrix/real_provider_parser_execution")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && event
                .observed
                .pointer("/provider_parser_matrix/matrix/provider_kinds")
                .and_then(Value::as_array)
                .is_some_and(|providers| {
                    let providers = providers
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<BTreeSet<_>>();
                    ["openai-compatible", "openai", "anthropic", "google_oauth"]
                        .into_iter()
                        .all(|provider| providers.contains(provider))
                })
    })
}

fn provider_mutation_classes_observed(events: &[DeliveredBoundary]) -> bool {
    let mutations = events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProviderMutation)
        .filter_map(|event| {
            event
                .observed
                .get("mutation")
                .or_else(|| event.payload.get("mutation"))
                .and_then(Value::as_str)
        })
        .collect::<BTreeSet<_>>();
    mutations.contains("malformed_sse_chunk") && mutations.contains("rate_limit_error_envelope")
}

fn provider_rate_limit_terminalized_by_all_migrated_parsers(events: &[DeliveredBoundary]) -> bool {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProviderMutation)
        .filter(|event| {
            event
                .observed
                .get("mutation")
                .or_else(|| event.payload.get("mutation"))
                .and_then(Value::as_str)
                == Some("rate_limit_error_envelope")
        })
        .any(|event| {
            let Some(proofs) = event
                .observed
                .pointer("/provider_parser_matrix/matrix/proofs")
                .and_then(Value::as_array)
            else {
                return false;
            };
            let mut providers = BTreeSet::new();
            let mut retryable_provider_observed = false;
            for proof in proofs {
                let Some(provider_kind) = proof.get("provider_kind").and_then(Value::as_str) else {
                    continue;
                };
                let terminal_reason = proof.get("terminal_reason").and_then(Value::as_str);
                let status = proof
                    .get("classification")
                    .and_then(|classification| classification.get("status"))
                    .and_then(Value::as_u64)
                    .or_else(|| proof.get("status").and_then(Value::as_u64));
                let retryable = proof
                    .get("classification")
                    .and_then(|classification| classification.get("retryable"))
                    .and_then(Value::as_bool);
                let kind = proof
                    .get("classification")
                    .and_then(|classification| classification.get("kind"))
                    .and_then(Value::as_str);
                if terminal_reason == Some("provider_error")
                    && status == Some(429)
                    && retryable.is_some()
                    && kind.is_some()
                {
                    retryable_provider_observed |= retryable == Some(true);
                    providers.insert(provider_kind);
                }
            }
            MIGRATED_RUNTIME_PROVIDER_KINDS
                .iter()
                .all(|provider| providers.contains(*provider))
                && retryable_provider_observed
        })
}

fn provider_dropped_terminal_event_classified(events: &[DeliveredBoundary]) -> bool {
    events
        .iter()
        .filter(|event| event.kind == BoundaryKind::ProviderMutation)
        .filter(|event| {
            event
                .observed
                .get("mutation")
                .or_else(|| event.payload.get("mutation"))
                .and_then(Value::as_str)
                == Some("dropped_terminal_event")
        })
        .any(|event| {
            let Some(proofs) = event
                .observed
                .pointer("/provider_parser_matrix/matrix/proofs")
                .and_then(Value::as_array)
            else {
                return false;
            };
            let providers = proofs
                .iter()
                .filter(|proof| {
                    proof.get("terminal_reason").and_then(Value::as_str) == Some("provider_error")
                        && proof
                            .get("classification")
                            .and_then(|classification| classification.get("retryable"))
                            .and_then(Value::as_bool)
                            == Some(false)
                })
                .filter_map(|proof| proof.get("provider_kind").and_then(Value::as_str))
                .collect::<BTreeSet<_>>();
            MIGRATED_RUNTIME_PROVIDER_KINDS
                .iter()
                .all(|provider| providers.contains(*provider))
        })
}

fn trigger_delivery_runtime_observed(events: &[DeliveredBoundary]) -> bool {
    events.iter().any(|event| {
        event.kind == BoundaryKind::Trigger
            && event
                .observed
                .get("occurrence_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("trigger:"))
            && event
                .observed
                .get("reservation_count")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                > 0
    })
}

fn provider_exchange_counts_are_turn_indexed(summary: &AbstractWorldSummary) -> bool {
    summary.sessions.iter().all(|session| {
        !session.provider_exchange_counts.is_empty()
            && session
                .provider_exchange_counts
                .iter()
                .enumerate()
                .all(|(index, count)| *count == index + 1)
    })
}

fn observer_reconnect_has_matching_turn(
    events: &[DeliveredBoundary],
    summary: &AbstractWorldSummary,
) -> bool {
    let reconnect_seen = events.iter().any(|event| {
        event.kind == BoundaryKind::Observer
            && event
                .observed
                .get("reconnected")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });
    reconnect_seen
        && summary.sessions.iter().all(|session| {
            session.observer_turn_indices.last().copied() == Some(session.provider_outputs.len())
        })
}

pub fn replay_determinism(
    expected: &AbstractWorldSummary,
    actual: &AbstractWorldSummary,
) -> OracleVerdict {
    if expected == actual {
        OracleVerdict::passed(
            REPLAY_DETERMINISM_ORACLE,
            "replay reproduced the delivered boundary sequence and final abstract summary",
        )
    } else {
        OracleVerdict::failed(
            REPLAY_DETERMINISM_ORACLE,
            format!(
                "replay summary digest diverged: expected {}, actual {}",
                expected.digest, actual.digest
            ),
        )
    }
}

pub fn runtime_provider_turn(ok: bool, message: impl Into<String>) -> OracleVerdict {
    if ok {
        OracleVerdict::passed(RUNTIME_PROVIDER_TURN_ORACLE, message)
    } else {
        OracleVerdict::failed(RUNTIME_PROVIDER_TURN_ORACLE, message)
    }
}

pub fn pending_tool_completion(ok: bool, message: impl Into<String>) -> OracleVerdict {
    if ok {
        OracleVerdict::passed(PENDING_TOOL_COMPLETION_ORACLE, message)
    } else {
        OracleVerdict::failed(PENDING_TOOL_COMPLETION_ORACLE, message)
    }
}

pub fn runtime_final_value_semantic(ok: bool, message: impl Into<String>) -> OracleVerdict {
    if ok {
        OracleVerdict::passed(RUNTIME_FINAL_VALUE_SEMANTIC_ORACLE, message)
    } else {
        OracleVerdict::failed(RUNTIME_FINAL_VALUE_SEMANTIC_ORACLE, message)
    }
}

fn runtime_observed_fact<T>(event: &DeliveredBoundary, key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(
        event
            .observed
            .get("runtime_invariant_facts")?
            .get(key)?
            .clone(),
    )
    .ok()
}

pub const LIVE_PROVIDER_FAILURE_ORACLE: &str = "sim.oracle.live-provider-failure-terminalizes.v1";

/// Observed facts from driving a non-retryable provider failure through a LIVE
/// runtime turn (a real `session.turn().run()` parked on a scheduler-gated
/// provider event, not an isolated `provider.complete()`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct LiveProviderFailureFacts {
    /// The non-retryable fault injected mid-turn (e.g. `malformed_sse_chunk`).
    pub fault_kind: String,
    /// The turn was observed live and parked on the first scheduler-gated
    /// provider event before the failure was delivered (proves it is a live turn,
    /// not a synchronous isolated call).
    pub turn_was_live_parked: bool,
    /// The turn ended in a terminal failure (returned an error or a non-success
    /// outcome) rather than finishing successfully.
    pub terminalized_failure: bool,
    /// Count of assistant prose deltas committed during the failed turn.
    pub committed_assistant_prose_deltas: usize,
    /// Count of FinalValue events committed during the failed turn.
    pub committed_final_values: usize,
    /// Whether the committed assistant message was non-empty (leaked text).
    pub committed_assistant_message_nonempty: bool,
}

/// A live runtime turn that receives a non-retryable provider failure mid-flight
/// MUST terminalize with a terminal failure and commit NO provider output (no
/// leaked partial assistant prose, no FinalValue, no assistant message). This is
/// failing-capable: it fails loudly if the turn instead committed output or
/// finished successfully.
pub fn live_provider_failure_terminalizes(facts: &LiveProviderFailureFacts) -> OracleVerdict {
    let no_committed_output = facts.committed_assistant_prose_deltas == 0
        && facts.committed_final_values == 0
        && !facts.committed_assistant_message_nonempty;
    if facts.turn_was_live_parked && facts.terminalized_failure && no_committed_output {
        OracleVerdict::passed(
            LIVE_PROVIDER_FAILURE_ORACLE,
            format!(
                "a live runtime turn parked on a scheduler-gated provider event, received a non-retryable `{}` fault mid-turn, terminalized as a terminal failure, and committed no provider output (0 assistant prose deltas, 0 final values, empty assistant message)",
                facts.fault_kind
            ),
        )
    } else {
        OracleVerdict::failed(
            LIVE_PROVIDER_FAILURE_ORACLE,
            format!(
                "live provider failure turn did not terminalize cleanly for fault `{}`: live_parked={} terminalized_failure={} committed_assistant_prose_deltas={} committed_final_values={} committed_assistant_message_nonempty={}",
                facts.fault_kind,
                facts.turn_was_live_parked,
                facts.terminalized_failure,
                facts.committed_assistant_prose_deltas,
                facts.committed_final_values,
                facts.committed_assistant_message_nonempty
            ),
        )
    }
}

pub fn combine_oracles(oracles: &[OracleVerdict]) -> OracleVerdict {
    if let Some(failure) = oracles.iter().find(|oracle| !oracle.is_passed()) {
        return failure.clone();
    }
    OracleVerdict::passed(
        "sim.oracle.generated-workload.v1",
        format!("{} generated workload oracles passed", oracles.len()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::SchedulerDeliveryEvidence;
    use crate::trace::{
        DurableEffectAbstractSummary, SessionAbstractSummary, WorkerAbstractSummary,
    };
    use serde_json::json;

    #[test]
    fn unmapped_runtime_semantic_fails_loudly_and_protocol_suites_route_to_coverage_manifests() {
        let summary = AbstractWorldSummary::with_digest(0, 0, vec![], vec![], vec![]);

        // A brand-new RUNTIME contract with no per-contract adapter fails loudly
        // (the runtime suite is genuinely per-contract; there is no decorative
        // shared fallback).
        let runtime = runtime_contract_semantics("runtime.brand_new_contract", &[], &summary);
        assert!(
            !runtime.passed,
            "an unmapped runtime contract must fail loudly, not pass via a fallback"
        );
        assert!(
            runtime.reason.contains("no per-contract semantic adapter"),
            "runtime failure reason should explain the missing adapter, got: {}",
            runtime.reason
        );

        // The STANDARD/RLM/AGENT suites do NOT emit per-contract oracles: the
        // per-contract dispatcher routes them to the coverage-manifest path
        // instead of stamping a shared-bucket fallback verdict.
        for contracts in [
            STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
            RLM_PROTOCOL_SCENARIO_CONTRACTS,
            AGENT_SCENARIO_CONTRACTS,
        ] {
            let contract = contracts.first().expect("suite has a contract");
            let verdict = scenario_contract_semantics(contract, &[], &summary);
            assert!(
                !verdict.passed,
                "protocol suite `{}` must not produce a passing per-contract verdict",
                contract.suite
            );
            assert!(
                verdict.reason.contains("suite coverage manifest"),
                "protocol suite `{}` should route to the coverage manifest, got: {}",
                contract.suite,
                verdict.reason
            );
        }
    }

    #[test]
    fn scenario_contract_oracles_emit_one_named_verdict_per_contract() {
        let summary = AbstractWorldSummary::with_digest(
            2,
            24,
            vec![
                SessionAbstractSummary {
                    alias: "session-001".to_string(),
                    opened: true,
                    ingress_count: 1,
                    provider_outputs: vec![
                        "answer for session-001 turn 1".to_string(),
                        "answer for session-001 turn 2".to_string(),
                    ],
                    provider_exchange_counts: vec![1, 2],
                    graph_node_counts: vec![2, 4],
                    transcript_message_counts: vec![2, 4],
                    tool_outputs: vec!["tool result for session-001".to_string()],
                    exec_code_outputs: vec!["exec result for session-001".to_string()],
                    observer_turn_indices: vec![2],
                    observer_reconnects: 1,
                    queued_ingress_count: 1,
                    cancellation_count: 1,
                    trigger_count: 1,
                    backend_failure_count: 2,
                    provider_mutation_count: 2,
                    process_wake_count: 2,
                    durable_effect_keys: vec!["durable/session-001".to_string()],
                    lease_time_ticks: vec![1, 2],
                },
                SessionAbstractSummary {
                    alias: "session-002".to_string(),
                    opened: true,
                    ingress_count: 1,
                    provider_outputs: vec![
                        "answer for session-002 turn 1".to_string(),
                        "answer for session-002 turn 2".to_string(),
                    ],
                    provider_exchange_counts: vec![1, 2],
                    graph_node_counts: vec![2, 4],
                    transcript_message_counts: vec![2, 4],
                    tool_outputs: Vec::new(),
                    exec_code_outputs: Vec::new(),
                    observer_turn_indices: vec![2],
                    observer_reconnects: 0,
                    queued_ingress_count: 0,
                    cancellation_count: 0,
                    trigger_count: 0,
                    backend_failure_count: 0,
                    provider_mutation_count: 0,
                    process_wake_count: 0,
                    durable_effect_keys: Vec::new(),
                    lease_time_ticks: vec![1, 2],
                },
            ],
            vec![DurableEffectAbstractSummary {
                durable_key: "durable/session-001".to_string(),
                execution_count: 1,
                replay_count: 1,
                result_digest: "digest".to_string(),
            }],
            vec![WorkerAbstractSummary {
                worker_alias: "worker-001".to_string(),
                session_alias: "session-001".to_string(),
                active_incarnation_id: "worker-001:incarnation-002".to_string(),
                active_fencing_token: 2,
                lease_owner_changes: 1,
                stale_completion_rejections: 1,
            }],
        );

        let events = semantic_events();
        let verdicts = scenario_contract_oracles(&events, &summary);

        // RUNTIME is genuinely per-contract; STANDARD/RLM/AGENT collapse to one
        // failing-capable suite coverage manifest each (no shared-bucket
        // per-contract stamping that would overstate the oracle count).
        let expected_count = RUNTIME_SCENARIO_CONTRACTS.len() + 3;
        assert_eq!(verdicts.len(), expected_count);
        assert!(verdicts.iter().all(OracleVerdict::is_passed));

        let ids = verdicts
            .iter()
            .map(|verdict| verdict.oracle_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), verdicts.len());

        // Exactly the runtime contracts get distinct per-contract oracle ids.
        let runtime_per_contract = verdicts
            .iter()
            .filter(|verdict| {
                verdict.oracle_id.starts_with(SCENARIO_RUNTIME_CONTRACT_ORACLE)
                    && verdict.oracle_id.contains("runtime_scenario_")
            })
            .count();
        assert_eq!(runtime_per_contract, RUNTIME_SCENARIO_CONTRACTS.len());

        // Each protocol suite emits exactly ONE coverage manifest, and that
        // manifest enumerates every contract it covers (so the coverage is
        // explicit and auditable rather than hidden behind a shared bucket).
        for (base, contracts) in [
            (
                SCENARIO_STANDARD_CONTRACT_ORACLE,
                STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
            ),
            (SCENARIO_RLM_CONTRACT_ORACLE, RLM_PROTOCOL_SCENARIO_CONTRACTS),
            (SCENARIO_AGENT_CONTRACT_ORACLE, AGENT_SCENARIO_CONTRACTS),
        ] {
            let suite_verdicts = verdicts
                .iter()
                .filter(|verdict| verdict.oracle_id.starts_with(base))
                .collect::<Vec<_>>();
            assert_eq!(
                suite_verdicts.len(),
                1,
                "suite `{base}` must emit exactly one coverage manifest, not per-contract oracles"
            );
            let manifest = suite_verdicts[0];
            assert_eq!(manifest.oracle_id, format!("{base}:coverage-manifest"));
            for contract in contracts {
                assert!(
                    manifest.message.contains(contract.test_name),
                    "coverage manifest for `{base}` must enumerate covered contract `{}`",
                    contract.test_name
                );
            }
        }
    }

    #[test]
    fn state_machine_semantic_oracle_checks_contract_outcomes_not_presence() {
        let summary = semantic_summary();
        let events = semantic_events();
        let verdict = state_machine_semantic_invariants(&events, &summary);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Passed);
        assert_eq!(verdict.oracle_id, STATE_MACHINE_SEMANTIC_INVARIANTS_ORACLE);

        let mut cancelled_not_terminal = events.clone();
        let cancel = cancelled_not_terminal
            .iter_mut()
            .find(|event| event.kind == BoundaryKind::Cancellation)
            .expect("cancellation event");
        cancel
            .observed
            .as_object_mut()
            .expect("observed object")
            .insert("cancel_outcome".to_string(), json!("not_found"));
        let verdict = state_machine_semantic_invariants(&cancelled_not_terminal, &summary);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert!(
            verdict
                .message
                .contains("cancellation terminalizes a pending queued input")
        );

        let mut retry_not_terminal = events;
        retry_not_terminal.retain(|event| {
            !(event.kind == BoundaryKind::BackendFailure
                && event.observed.get("retryable").and_then(Value::as_bool) == Some(false))
        });
        let verdict = state_machine_semantic_invariants(&retry_not_terminal, &summary);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert!(verdict.message.contains("backend retry terminalization"));
    }

    #[test]
    fn coverage_oracles_are_failing_capable_not_presence_only() {
        let summary = semantic_summary();
        let events = semantic_events();

        // Positive: a workload that genuinely satisfies the per-boundary runtime
        // invariants passes every strengthened coverage oracle.
        for verdict in [
            queued_ingress_observed(&summary, &events),
            cancellation_observed(&summary, &events),
            trigger_delivery_observed(&summary, &events),
            observer_reconnect_observed(&summary, &events),
            backend_failure_observed(&summary, &events),
            provider_mutation_rejected(&summary, &events),
            process_wake_observed(&summary, &events),
            tool_boundary_observed(&summary, &events),
            exec_code_observed(&summary, &events),
        ] {
            assert!(
                verdict.is_passed(),
                "{} should pass on a valid workload: {}",
                verdict.oracle_id,
                verdict.message
            );
        }

        // Failing-capable: the boundary kinds remain PRESENT in the summary, but
        // with the runtime DTO/projection evidence stripped (empty events) every
        // oracle FAILS loudly instead of passing on presence alone.
        for verdict in [
            queued_ingress_observed(&summary, &[]),
            cancellation_observed(&summary, &[]),
            trigger_delivery_observed(&summary, &[]),
            observer_reconnect_observed(&summary, &[]),
            backend_failure_observed(&summary, &[]),
            provider_mutation_rejected(&summary, &[]),
            process_wake_observed(&summary, &[]),
            tool_boundary_observed(&summary, &[]),
            exec_code_observed(&summary, &[]),
        ] {
            assert!(
                !verdict.is_passed(),
                "{} must fail when its invariant evidence is missing (presence is not enough)",
                verdict.oracle_id
            );
            assert!(
                verdict.message.contains("observed but"),
                "{} should explain the violated invariant, got: {}",
                verdict.oracle_id,
                verdict.message
            );
        }
    }

    #[test]
    fn live_provider_failure_oracle_passes_on_clean_terminalization_and_fails_on_committed_output() {
        // Positive: a live, parked turn that terminalized as a failure and
        // committed nothing passes.
        let clean = LiveProviderFailureFacts {
            fault_kind: "malformed_sse_chunk".to_string(),
            turn_was_live_parked: true,
            terminalized_failure: true,
            committed_assistant_prose_deltas: 0,
            committed_final_values: 0,
            committed_assistant_message_nonempty: false,
        };
        assert!(live_provider_failure_terminalizes(&clean).is_passed());

        // Negative: the turn finished successfully instead of terminalizing.
        let succeeded = LiveProviderFailureFacts {
            terminalized_failure: false,
            ..clean.clone()
        };
        assert!(!live_provider_failure_terminalizes(&succeeded).is_passed());

        // Negative: the turn leaked a committed partial assistant prose delta.
        let leaked_prose = LiveProviderFailureFacts {
            committed_assistant_prose_deltas: 1,
            ..clean.clone()
        };
        assert!(!live_provider_failure_terminalizes(&leaked_prose).is_passed());

        // Negative: the turn committed a FinalValue despite failing.
        let leaked_final = LiveProviderFailureFacts {
            committed_final_values: 1,
            ..clean.clone()
        };
        assert!(!live_provider_failure_terminalizes(&leaked_final).is_passed());

        // Negative: a leaked non-empty assistant message.
        let leaked_message = LiveProviderFailureFacts {
            committed_assistant_message_nonempty: true,
            ..clean.clone()
        };
        assert!(!live_provider_failure_terminalizes(&leaked_message).is_passed());

        // Negative: the turn was never observed live/parked (could be a
        // synchronous isolated path rather than a live turn).
        let not_live = LiveProviderFailureFacts {
            turn_was_live_parked: false,
            ..clean
        };
        assert!(!live_provider_failure_terminalizes(&not_live).is_passed());
    }

    #[test]
    fn worker_stale_completion_oracle_requires_real_fencing() {
        // Negative: the real store did NOT fence (no incarnation change, no stale
        // rejection, fence not advanced) -> the oracle must FAIL. With the
        // fabrication deleted, this is the ONLY way the summary can look.
        let not_fenced = AbstractWorldSummary::with_digest(
            1,
            1,
            vec![],
            vec![],
            vec![WorkerAbstractSummary {
                worker_alias: "worker-001".to_string(),
                session_alias: "session-001".to_string(),
                active_incarnation_id: String::new(),
                active_fencing_token: 1,
                lease_owner_changes: 0,
                stale_completion_rejections: 0,
            }],
        );
        assert!(!worker_stale_completion_rejected(&not_fenced).is_passed());

        // Positive: the real store fenced (incarnation change, stale rejection,
        // monotonic fence advance) -> the oracle passes.
        let fenced = AbstractWorldSummary::with_digest(
            1,
            1,
            vec![],
            vec![],
            vec![WorkerAbstractSummary {
                worker_alias: "worker-001".to_string(),
                session_alias: "session-001".to_string(),
                active_incarnation_id: "worker-001:incarnation-002".to_string(),
                active_fencing_token: 2,
                lease_owner_changes: 1,
                stale_completion_rejections: 1,
            }],
        );
        assert!(worker_stale_completion_rejected(&fenced).is_passed());
    }

    #[test]
    fn worker_failover_continuation_oracle_requires_successor_commit() {
        let worker_event = |work: serde_json::Value| {
            delivered_with_payload(
                0,
                "worker-001:worker:001",
                "worker-001",
                BoundaryKind::Worker,
                json!({ "session": "session-001" }),
                json!({ "runtime_worker_store": { "worker_owned_work": work } }),
            )
        };
        let full = json!({
            "first_owner_claimed_work": true,
            "second_owner_resumed_work": true,
            "second_owner_outranks_first": true,
            "stale_work_completion_rejected": true,
        });

        // Positive: a successor reclaimed and continued the work.
        assert!(worker_failover_continues_work(&[worker_event(full.clone())]).is_passed());

        // Negative: no worker boundary at all.
        assert!(!worker_failover_continues_work(&[]).is_passed());

        // Negative: the successor did not resume the dead owner's work.
        let mut not_resumed = full.clone();
        not_resumed["second_owner_resumed_work"] = json!(false);
        assert!(!worker_failover_continues_work(&[worker_event(not_resumed)]).is_passed());

        // Negative: the dead owner's stale completion was NOT rejected.
        let mut stale_not_rejected = full.clone();
        stale_not_rejected["stale_work_completion_rejected"] = json!(false);
        assert!(!worker_failover_continues_work(&[worker_event(stale_not_rejected)]).is_passed());

        // Negative: the successor did not outrank the first owner's fence.
        let mut not_outranked = full;
        not_outranked["second_owner_outranks_first"] = json!(false);
        assert!(!worker_failover_continues_work(&[worker_event(not_outranked)]).is_passed());
    }

    #[test]
    fn suite_coverage_manifest_is_failing_capable() {
        // Missing suite-level evidence => the coverage manifest fails loudly
        // (it is a real, failing-capable oracle, not a decorative label).
        let empty = AbstractWorldSummary::with_digest(0, 0, vec![], vec![], vec![]);
        for contracts in [
            STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
            RLM_PROTOCOL_SCENARIO_CONTRACTS,
            AGENT_SCENARIO_CONTRACTS,
        ] {
            let verdict = scenario_suite_coverage_manifest(contracts, &[], &empty);
            assert!(!verdict.is_passed());
            assert!(verdict.message.contains("FAILED: suite-level evidence missing"));
        }

        // On a workload that produces the suite evidence it passes.
        let summary = semantic_summary();
        let events = semantic_events();
        for contracts in [
            STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
            RLM_PROTOCOL_SCENARIO_CONTRACTS,
            AGENT_SCENARIO_CONTRACTS,
        ] {
            assert!(scenario_suite_coverage_manifest(contracts, &events, &summary).is_passed());
        }
    }

    #[test]
    fn scheduler_owned_runtime_completion_oracle_rejects_missing_pending_evidence() {
        let verdict = scheduler_owned_runtime_completions(&[delivered_with_payload(
            0,
            "session-001:provider:001",
            "session-001",
            BoundaryKind::Provider,
            json!({}),
            json!({"provider_output": "answer"}),
        )]);

        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert_eq!(verdict.oracle_id, SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE);
        assert!(
            verdict
                .message
                .contains("delivered without pending runtime boundary evidence")
        );
    }

    #[test]
    fn scheduler_owned_runtime_completion_oracle_rejects_incomplete_pending_evidence() {
        for (name, completion) in [
            (
                "empty family",
                json!({
                    "completion_family": "",
                    "completion_units": [{"unit": "runtime:unit", "at": 0}],
                    "ready_at": 0,
                    "registered_after": "session-001:ingress"
                }),
            ),
            (
                "empty units",
                json!({
                    "completion_family": "provider_turn_completion",
                    "completion_units": [],
                    "ready_at": 0,
                    "registered_after": "session-001:ingress"
                }),
            ),
            (
                "wrong ready_at",
                json!({
                    "completion_family": "provider_turn_completion",
                    "completion_units": [{"unit": "runtime:unit", "at": 0}],
                    "ready_at": 99,
                    "registered_after": "session-001:ingress"
                }),
            ),
            (
                "empty registered_after",
                json!({
                    "completion_family": "provider_turn_completion",
                    "completion_units": [{"unit": "runtime:unit", "at": 0}],
                    "ready_at": 0,
                    "registered_after": ""
                }),
            ),
        ] {
            let verdict = scheduler_owned_runtime_completions(&[delivered_with_payload(
                0,
                "session-001:provider:001",
                "session-001",
                BoundaryKind::Provider,
                json!({"runtime_completion": completion}),
                json!({"provider_output": "answer"}),
            )]);

            assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
            assert_eq!(verdict.oracle_id, SCHEDULER_OWNED_RUNTIME_COMPLETION_ORACLE);
            assert!(
                verdict.message.contains("incomplete pending evidence"),
                "{name}: {}",
                verdict.message
            );
        }
    }

    #[test]
    fn standard_provider_error_oracle_requires_ordered_failure_and_parser_matrix() {
        let parser_matrix = delivered_with_payload(
            2,
            "session-001:provider-mutation:001",
            "session-001",
            BoundaryKind::ProviderMutation,
            json!({"runtime_completion": runtime_completion("provider_script_mutation", 2)}),
            provider_mutation_observed("malformed_sse_chunk"),
        );
        let provider = delivered_with_payload(
            1,
            "session-001:provider:001",
            "session-001",
            BoundaryKind::Provider,
            json!({"runtime_completion": runtime_completion("provider_turn_completion", 1)}),
            json!({"provider_output": "answer"}),
        );
        let failure_equal_sequence = delivered_with_payload(
            1,
            "session-001:backend-failure:001",
            "session-001",
            BoundaryKind::BackendFailure,
            json!({"runtime_completion": runtime_completion("backend_retry_or_failure", 1)}),
            json!({"backend_failure": true}),
        );
        let verdict = mini_standard_provider_error_without_checkpoint(&[
            provider.clone(),
            failure_equal_sequence,
            parser_matrix.clone(),
        ]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert_eq!(
            verdict.oracle_id,
            SCENARIO_MINI_STANDARD_PROVIDER_ERROR_ORACLE
        );

        let missing_runtime_completion = delivered_with_payload(
            0,
            "session-001:backend-failure:001",
            "session-001",
            BoundaryKind::BackendFailure,
            json!({}),
            json!({"backend_failure": true}),
        );
        let verdict = mini_standard_provider_error_without_checkpoint(&[
            missing_runtime_completion,
            provider.clone(),
            parser_matrix.clone(),
        ]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);

        let wrong_kind_before_provider = delivered_with_payload(
            0,
            "session-001:tool:001",
            "session-001",
            BoundaryKind::Tool,
            json!({"runtime_completion": runtime_completion("tool_return", 0)}),
            json!({"tool_output": "not a provider failure"}),
        );
        let verdict = mini_standard_provider_error_without_checkpoint(&[
            wrong_kind_before_provider,
            provider.clone(),
            parser_matrix.clone(),
        ]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);

        let ordered_failure = delivered_with_payload(
            0,
            "session-001:backend-failure:001",
            "session-001",
            BoundaryKind::BackendFailure,
            json!({"runtime_completion": runtime_completion("backend_retry_or_failure", 0)}),
            json!({"backend_failure": true}),
        );
        let verdict =
            mini_standard_provider_error_without_checkpoint(&[ordered_failure, provider.clone()]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);

        let provider_after_failure = delivered_with_payload(
            2,
            "session-001:provider:002",
            "session-001",
            BoundaryKind::Provider,
            json!({"runtime_completion": runtime_completion("provider_turn_completion", 2)}),
            json!({"provider_output": "answer"}),
        );
        let late_failure = delivered_with_payload(
            3,
            "session-001:backend-failure:002",
            "session-001",
            BoundaryKind::BackendFailure,
            json!({"runtime_completion": runtime_completion("backend_retry_or_failure", 3)}),
            json!({"backend_failure": true}),
        );
        let verdict = mini_standard_provider_error_without_checkpoint(&[
            provider_after_failure,
            late_failure,
            parser_matrix.clone(),
        ]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);

        let passing_failure = delivered_with_payload(
            0,
            "session-001:backend-failure:003",
            "session-001",
            BoundaryKind::BackendFailure,
            json!({"runtime_completion": runtime_completion("backend_retry_or_failure", 0)}),
            json!({"backend_failure": true}),
        );
        let verdict = mini_standard_provider_error_without_checkpoint(&[
            passing_failure,
            provider,
            parser_matrix,
        ]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Passed);
    }

    #[test]
    fn rlm_mini_oracle_rejects_exec_without_runtime_effect_outcome() {
        let events = vec![
            delivered_with_payload(
                0,
                "session-001:exec:001",
                "session-001",
                BoundaryKind::ExecCode,
                json!({"runtime_completion": runtime_completion("exec_result", 0)}),
                json!({
                    "exec_output": "cell ran",
                    "execution_count": 1
                }),
            ),
            delivered_with_payload(
                1,
                "session-001:provider:001",
                "session-001",
                BoundaryKind::Provider,
                json!({"runtime_completion": runtime_completion("provider_turn_completion", 1)}),
                json!({"provider_output": "continued"}),
            ),
        ];

        let verdict = mini_rlm_lashlang_cell_exec_continues(&events);

        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert_eq!(verdict.oracle_id, SCENARIO_MINI_RLM_CELL_EXEC_ORACLE);
        assert!(verdict.message.contains("did not continue after exec"));
    }

    #[test]
    fn rlm_mini_oracle_requires_provider_after_same_actor_exec() {
        let exec = delivered_with_payload(
            1,
            "session-001:exec:001",
            "session-001",
            BoundaryKind::ExecCode,
            json!({"runtime_completion": runtime_completion("exec_result", 1)}),
            json!({
                "exec_output": "cell ran",
                "runtime_effect_outcome": {"type": "exec_code"},
                "execution_count": 1
            }),
        );
        for (name, event) in [
            (
                "same sequence provider",
                delivered_with_payload(
                    1,
                    "session-001:provider:001",
                    "session-001",
                    BoundaryKind::Provider,
                    json!({"runtime_completion": runtime_completion("provider_turn_completion", 1)}),
                    json!({"provider_output": "continued"}),
                ),
            ),
            (
                "different actor provider",
                delivered_with_payload(
                    2,
                    "session-002:provider:001",
                    "session-002",
                    BoundaryKind::Provider,
                    json!({"runtime_completion": runtime_completion("provider_turn_completion", 2)}),
                    json!({"provider_output": "continued"}),
                ),
            ),
            (
                "non-provider event",
                delivered_with_payload(
                    2,
                    "session-001:tool:001",
                    "session-001",
                    BoundaryKind::Tool,
                    json!({"runtime_completion": runtime_completion("tool_return", 2)}),
                    json!({"tool_output": "continued"}),
                ),
            ),
        ] {
            let verdict = mini_rlm_lashlang_cell_exec_continues(&[exec.clone(), event]);
            assert_eq!(verdict.status, crate::trace::OracleStatus::Failed, "{name}");
            assert_eq!(verdict.oracle_id, SCENARIO_MINI_RLM_CELL_EXEC_ORACLE);
        }

        let continued = delivered_with_payload(
            2,
            "session-001:provider:001",
            "session-001",
            BoundaryKind::Provider,
            json!({"runtime_completion": runtime_completion("provider_turn_completion", 2)}),
            json!({"provider_output": "continued"}),
        );
        let verdict = mini_rlm_lashlang_cell_exec_continues(&[exec, continued]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Passed);
    }

    #[test]
    fn agent_mini_oracle_rejects_process_wake_without_join_session() {
        let summary = AbstractWorldSummary::with_digest(2, 2, Vec::new(), Vec::new(), Vec::new());
        let events = vec![
            delivered_with_payload(
                0,
                "session-001:process-wake:001",
                "session-001",
                BoundaryKind::ProcessWake,
                json!({"runtime_completion": runtime_completion("process_wake", 0)}),
                json!({
                    "process_wake": true,
                    "runtime_process_wake": {
                        "event_invocation": {
                            "subject": {
                                "process_id": "process-001"
                            }
                        }
                    }
                }),
            ),
            delivered_with_payload(
                1,
                "worker-001:stale-completion",
                "worker-001",
                BoundaryKind::Worker,
                json!({"runtime_completion": runtime_completion("worker_lease_completion", 1)}),
                json!({"session": "session-001"}),
            ),
        ];

        let verdict = mini_agent_parallel_spawn_join(&events, &summary);

        assert_eq!(verdict.status, crate::trace::OracleStatus::Failed);
        assert_eq!(verdict.oracle_id, SCENARIO_MINI_AGENT_PARALLEL_JOIN_ORACLE);
        assert!(
            verdict
                .message
                .contains("did not record deterministic process/worker ordering")
        );
    }

    #[test]
    fn agent_durable_input_mini_oracle_requires_all_resolution_evidence() {
        let summary = AbstractWorldSummary::with_digest(2, 3, Vec::new(), Vec::new(), Vec::new());
        let durable = delivered_with_payload(
            0,
            "session-001:durable:001:replay",
            "session-001",
            BoundaryKind::DurableEffect,
            json!({"runtime_completion": runtime_completion("durable_effect_completion", 0)}),
            json!({"replayed": true, "runtime_effect": {}}),
        );
        let process_wake = delivered_with_payload(
            1,
            "session-001:process-wake:001",
            "session-001",
            BoundaryKind::ProcessWake,
            json!({"runtime_completion": runtime_completion("process_wake", 1)}),
            json!({
                "session": "session-001",
                "runtime_process_wake": {
                    "event_invocation": {
                        "subject": {
                            "process_id": "process-001"
                        }
                    }
                }
            }),
        );
        let observer = delivered_with_payload(
            2,
            "session-001:observer:reconnect:001",
            "session-001",
            BoundaryKind::Observer,
            json!({}),
            json!({"reconnected": true}),
        );

        for (name, events) in [
            (
                "missing durable",
                vec![process_wake.clone(), observer.clone()],
            ),
            (
                "missing process wake",
                vec![durable.clone(), observer.clone()],
            ),
            (
                "missing observer",
                vec![durable.clone(), process_wake.clone()],
            ),
            (
                "wrong durable kind",
                vec![
                    delivered_with_payload(
                        0,
                        "session-001:tool:001",
                        "session-001",
                        BoundaryKind::Tool,
                        json!({"runtime_completion": runtime_completion("tool_return", 0)}),
                        json!({"replayed": true, "runtime_effect": {}}),
                    ),
                    process_wake.clone(),
                    observer.clone(),
                ],
            ),
            (
                "durable not replayed",
                vec![
                    delivered_with_payload(
                        0,
                        "session-001:durable:001:first",
                        "session-001",
                        BoundaryKind::DurableEffect,
                        json!({"runtime_completion": runtime_completion("durable_effect_completion", 0)}),
                        json!({"replayed": false, "runtime_effect": {}}),
                    ),
                    process_wake.clone(),
                    observer.clone(),
                ],
            ),
        ] {
            let verdict = mini_agent_durable_input_resolution(&events);
            assert_eq!(verdict.status, crate::trace::OracleStatus::Failed, "{name}");
            assert_eq!(verdict.oracle_id, SCENARIO_MINI_AGENT_DURABLE_INPUT_ORACLE);
        }
        let verdict = mini_agent_durable_input_resolution(&[durable, process_wake, observer]);
        assert_eq!(verdict.status, crate::trace::OracleStatus::Passed);

        let parallel = mini_agent_parallel_spawn_join(
            &[
                delivered_with_payload(
                    3,
                    "session-001:process-wake:002",
                    "session-001",
                    BoundaryKind::ProcessWake,
                    json!({"runtime_completion": runtime_completion("process_wake", 3)}),
                    json!({"session": "session-001"}),
                ),
                delivered_with_payload(
                    4,
                    "worker-001:lease:002",
                    "worker-001",
                    BoundaryKind::Worker,
                    json!({"runtime_completion": runtime_completion("worker_lease_completion", 4)}),
                    json!({"session": "session-001"}),
                ),
            ],
            &summary,
        );
        assert_eq!(
            parallel.status,
            crate::trace::OracleStatus::Passed,
            "non-empty process wake session should satisfy join evidence"
        );

        let reversed = mini_agent_parallel_spawn_join(
            &[
                delivered_with_payload(
                    6,
                    "session-001:process-wake:003",
                    "session-001",
                    BoundaryKind::ProcessWake,
                    json!({"runtime_completion": runtime_completion("process_wake", 6)}),
                    json!({"session": "session-001"}),
                ),
                delivered_with_payload(
                    5,
                    "worker-001:lease:003",
                    "worker-001",
                    BoundaryKind::Worker,
                    json!({"runtime_completion": runtime_completion("worker_lease_completion", 5)}),
                    json!({"session": "session-001"}),
                ),
            ],
            &summary,
        );
        assert_eq!(reversed.status, crate::trace::OracleStatus::Failed);

        let duplicate_sequence = mini_agent_parallel_spawn_join(
            &[
                delivered_with_payload(
                    7,
                    "session-001:process-wake:004",
                    "session-001",
                    BoundaryKind::ProcessWake,
                    json!({"runtime_completion": runtime_completion("process_wake", 7)}),
                    json!({"session": "session-001"}),
                ),
                delivered_with_payload(
                    7,
                    "worker-001:lease:004",
                    "worker-001",
                    BoundaryKind::Worker,
                    json!({"runtime_completion": runtime_completion("worker_lease_completion", 7)}),
                    json!({"session": "session-001"}),
                ),
            ],
            &summary,
        );
        assert_eq!(
            duplicate_sequence.status,
            crate::trace::OracleStatus::Failed
        );
    }

    fn semantic_summary() -> AbstractWorldSummary {
        AbstractWorldSummary::with_digest(
            2,
            24,
            vec![
                SessionAbstractSummary {
                    alias: "session-001".to_string(),
                    opened: true,
                    ingress_count: 1,
                    provider_outputs: vec![
                        "answer for session-001 turn 1".to_string(),
                        "answer for session-001 turn 2".to_string(),
                    ],
                    provider_exchange_counts: vec![1, 2],
                    graph_node_counts: vec![2, 4],
                    transcript_message_counts: vec![2, 4],
                    tool_outputs: vec!["tool result for session-001".to_string()],
                    exec_code_outputs: vec!["exec result for session-001".to_string()],
                    observer_turn_indices: vec![2],
                    observer_reconnects: 1,
                    queued_ingress_count: 1,
                    cancellation_count: 1,
                    trigger_count: 1,
                    backend_failure_count: 2,
                    provider_mutation_count: 3,
                    process_wake_count: 2,
                    durable_effect_keys: vec!["durable/session-001".to_string()],
                    lease_time_ticks: vec![1, 2],
                },
                SessionAbstractSummary {
                    alias: "session-002".to_string(),
                    opened: true,
                    ingress_count: 1,
                    provider_outputs: vec![
                        "answer for session-002 turn 1".to_string(),
                        "answer for session-002 turn 2".to_string(),
                    ],
                    provider_exchange_counts: vec![1, 2],
                    graph_node_counts: vec![2, 4],
                    transcript_message_counts: vec![2, 4],
                    tool_outputs: Vec::new(),
                    exec_code_outputs: Vec::new(),
                    observer_turn_indices: vec![2],
                    observer_reconnects: 0,
                    queued_ingress_count: 0,
                    cancellation_count: 0,
                    trigger_count: 0,
                    backend_failure_count: 0,
                    provider_mutation_count: 0,
                    process_wake_count: 0,
                    durable_effect_keys: Vec::new(),
                    lease_time_ticks: vec![1, 2],
                },
            ],
            vec![DurableEffectAbstractSummary {
                durable_key: "durable/session-001".to_string(),
                execution_count: 1,
                replay_count: 1,
                result_digest: "digest".to_string(),
            }],
            vec![WorkerAbstractSummary {
                worker_alias: "worker-001".to_string(),
                session_alias: "session-001".to_string(),
                active_incarnation_id: "worker-001:incarnation-002".to_string(),
                active_fencing_token: 2,
                lease_owner_changes: 1,
                stale_completion_rejections: 1,
            }],
        )
    }

    fn semantic_events() -> Vec<DeliveredBoundary> {
        [
            delivered_with_payload(
                0,
                "session-001:provider:001",
                "session-001",
                BoundaryKind::Provider,
                json!({
                    "text": "answer for session-001 turn 1",
                    "runtime_completion": runtime_completion("provider_turn_completion", 0),
                    "expected_provider_exchange_count": 1,
                }),
                json!({
                    "provider_kind": "openai-compatible",
                    "provider_output": "answer for session-001 turn 1",
                    "success": true,
                    "provider_exchange_count": 1,
                    "runtime_contract": {"status": "passed"},
                }),
            ),
            delivered_with_payload(
                1,
                "session-001:queue:001",
                "session-001",
                BoundaryKind::QueuedIngress,
                json!({
                    "active_turn_id": "session-001:provider:002",
                    "ingress_mode": "active_turn",
                    "source_key": "queue/session-001/001",
                    "text": "queued follow-up hidden from live turn",
                }),
                json!({
                    "ingress_mode": "active_turn",
                    "input_id": "input-001",
                    "input_state": "pending_active",
                    "queued_ingress": true,
                    "session": "session-001",
                    "source_key": "queue/session-001/001",
                    "active_turn_id": "session-001:provider:002",
                }),
            ),
            delivered_with_payload(
                2,
                "session-001:provider:002:provider-event:001:sse",
                "session-001",
                BoundaryKind::ProviderEvent,
                json!({
                    "turn_boundary_id": "session-001:provider:002",
                    "event_index": 1,
                    "event_name": "sse",
                }),
                json!({
                    "provider_event_release": true,
                    "released_while_turn_pending": true,
                    "turn_boundary_id": "session-001:provider:002",
                }),
            ),
            delivered_with_payload(
                2,
                "session-001:cancel:001",
                "session-001",
                BoundaryKind::Cancellation,
                json!({
                    "target": "session-001:queue:001",
                    "runtime_completion": {
                        "completion_family": "queued_input_cancellation",
                        "completion_units": [{"unit": "runtime:cancel_pending_turn_input", "at": 2}],
                        "ready_at": 2,
                        "registered_after": "session-001:queue:001"
                    }
                }),
                json!({
                    "cancel_outcome": "cancelled",
                    "cancelled": true,
                    "session": "session-001",
                    "target": "session-001:queue:001",
                }),
            ),
            delivered_with_payload(
                3,
                "session-001:observer:reconnect:001",
                "session-001",
                BoundaryKind::Observer,
                json!({}),
                json!({"reconnected": true, "turn_index": 2}),
            ),
            delivered_with_payload(
                4,
                "session-001:provider:002",
                "session-001",
                BoundaryKind::Provider,
                json!({
                    "text": "answer for session-001 turn 2",
                    "runtime_completion": runtime_completion("provider_turn_completion", 4),
                    "expected_provider_exchange_count": 2,
                }),
                json!({
                    "provider_kind": "openai-compatible",
                    "provider_output": "answer for session-001 turn 2",
                    "success": true,
                    "provider_exchange_count": 2,
                    "runtime_contract": {"status": "passed"},
                }),
            ),
            delivered_with_payload(
                5,
                "session-002:provider:001",
                "session-002",
                BoundaryKind::Provider,
                json!({
                    "text": "answer for session-002 turn 1",
                    "runtime_completion": runtime_completion("provider_turn_completion", 5),
                    "expected_provider_exchange_count": 1,
                }),
                json!({
                    "provider_kind": "anthropic",
                    "provider_output": "answer for session-002 turn 1",
                    "success": true,
                    "provider_exchange_count": 1,
                    "runtime_contract": {"status": "passed"},
                }),
            ),
            delivered_with_payload(
                6,
                "session-002:provider:002",
                "session-002",
                BoundaryKind::Provider,
                json!({
                    "text": "answer for session-002 turn 2",
                    "runtime_completion": runtime_completion("provider_turn_completion", 6),
                    "expected_provider_exchange_count": 2,
                }),
                json!({
                    "provider_kind": "anthropic",
                    "provider_output": "answer for session-002 turn 2",
                    "success": true,
                    "provider_exchange_count": 2,
                    "runtime_contract": {"status": "passed"},
                }),
            ),
            delivered_with_payload(
                7,
                "session-001:process-wake:001",
                "session-001",
                BoundaryKind::ProcessWake,
                json!({"dedupe_key": "process/wake/session-001/001"}),
                json!({
                    "claimed_once": true,
                    "dedupe_key": "process/wake/session-001/001",
                    "runtime_process_wake": {
                        "event_invocation": {
                            "subject": {
                                "process_id": "process-001"
                            }
                        }
                    },
                    "runtime_queued_work": {
                        "claimed": true,
                        "source_key": "process/wake/session-001/001"
                    },
                    "wake_id": "wake:duplicate"
                }),
            ),
            delivered_with_payload(
                8,
                "session-001:process-wake:002",
                "session-001",
                BoundaryKind::ProcessWake,
                json!({"dedupe_key": "process/wake/session-001/001"}),
                json!({
                    "claimed_once": false,
                    "dedupe_key": "process/wake/session-001/001",
                    "runtime_process_wake": {
                        "event_invocation": {
                            "subject": {
                                "process_id": "process-001"
                            }
                        }
                    },
                    "runtime_queued_work": {
                        "claimed": false,
                        "source_key": "process/wake/session-001/001"
                    },
                    "wake_id": "wake:duplicate"
                }),
            ),
            delivered_with_payload(
                9,
                "worker-001:worker:001",
                "worker-001",
                BoundaryKind::Worker,
                json!({"runtime_completion": runtime_completion("worker_lease_completion", 9)}),
                json!({
                    "stale_completion_rejected": true,
                    "runtime_active_lease": {},
                    "runtime_stale_completion": {},
                }),
            ),
            delivered_with_payload(
                10,
                "session-001:durable:001:first",
                "session-001",
                BoundaryKind::DurableEffect,
                json!({"runtime_completion": runtime_completion("durable_effect_completion", 10)}),
                json!({
                    "durable_key": "durable/session-001",
                    "replayed": false,
                    "runtime_effect": {"local_executor_called": true},
                    "result_digest": "digest",
                    "execution_count": 1,
                    "replay_count": 0,
                }),
            ),
            delivered_with_payload(
                11,
                "session-001:durable:001:replay",
                "session-001",
                BoundaryKind::DurableEffect,
                json!({"runtime_completion": runtime_completion("durable_effect_completion", 11)}),
                json!({
                    "durable_key": "durable/session-001",
                    "replayed": true,
                    "runtime_effect": {"local_executor_called": false},
                    "result_digest": "digest",
                    "execution_count": 1,
                    "replay_count": 1,
                }),
            ),
            delivered_with_payload(
                12,
                "session-001:tool:001",
                "session-001",
                BoundaryKind::Tool,
                json!({"runtime_completion": runtime_completion("tool_return", 12)}),
                json!({
                    "runtime_tool_output": {},
                    "runtime_tool_record": {},
                    "execution_count": 1,
                }),
            ),
            delivered_with_payload(
                13,
                "session-001:exec:001",
                "session-001",
                BoundaryKind::ExecCode,
                json!({"runtime_completion": runtime_completion("exec_result", 13)}),
                json!({
                    "runtime_effect_outcome": {
                        "result": {
                            "Ok": {
                                "tool_calls": []
                            }
                        },
                        "type": "exec_code"
                    },
                    "execution_count": 1,
                }),
            ),
            delivered_with_payload(
                14,
                "session-001:trigger:001",
                "session-001",
                BoundaryKind::Trigger,
                json!({
                    "session": "session-001",
                    "source_key": "trigger/button/session-001/001",
                    "started_process": true,
                }),
                json!({
                    "occurrence_id": "trigger:abc",
                    "reservation_count": 1,
                    "session": "session-001",
                    "source_key": "trigger/button/session-001/001",
                    "started_process": true,
                    "trigger_delivered": true,
                }),
            ),
            delivered_with_payload(
                15,
                "session-001:backend-failure:001",
                "session-001",
                BoundaryKind::BackendFailure,
                json!({
                    "operation": "commit_runtime_state:001",
                    "runtime_completion": runtime_completion("backend_retry_or_failure", 15),
                }),
                json!({
                    "attempt": 1,
                    "backend_failure": true,
                    "operation": "commit_runtime_state:001",
                    "production_store_error": {
                        "retryable_class": true,
                        "type": "lash_core::StoreError",
                        "variant": "HeadRevisionConflict"
                    },
                    "retryable": true,
                    "store_error_class": "retryable_conflict"
                }),
            ),
            delivered_with_payload(
                16,
                "session-001:backend-failure:002",
                "session-001",
                BoundaryKind::BackendFailure,
                json!({
                    "operation": "commit_runtime_state:001",
                    "runtime_completion": runtime_completion("backend_retry_or_failure", 16),
                }),
                json!({
                    "attempt": 2,
                    "backend_failure": true,
                    "operation": "commit_runtime_state:001",
                    "production_store_error": {
                        "retryable_class": false,
                        "type": "lash_core::StoreError",
                        "variant": "SessionExecutionLeaseExpired"
                    },
                    "retryable": false,
                    "store_error_class": "terminal_backend_error"
                }),
            ),
            delivered_with_payload(
                17,
                "session-001:provider-mutation:001",
                "session-001",
                BoundaryKind::ProviderMutation,
                json!({"runtime_completion": runtime_completion("provider_script_mutation", 17)}),
                json!({
                    "mutation": "malformed_sse_chunk",
                    "provider_parser_matrix": {
                        "matrix": {
                            "real_provider_parser_execution": true,
                            "provider_kinds": [
                                "anthropic",
                                "google_oauth",
                                "openai",
                                "openai-compatible"
                            ]
                        }
                    }
                }),
            ),
            delivered_with_payload(
                18,
                "session-001:provider-mutation:002",
                "session-001",
                BoundaryKind::ProviderMutation,
                json!({"runtime_completion": runtime_completion("provider_script_mutation", 18)}),
                json!({
                    "mutation": "rate_limit_error_envelope",
                    "provider_parser_matrix": {
                        "matrix": {
                            "real_provider_parser_execution": true,
                            "provider_kinds": [
                                "anthropic",
                                "google_oauth",
                                "openai",
                                "openai-compatible"
                            ],
                            "proofs": [
                                {"provider_kind": "openai-compatible", "terminal_reason": "provider_error", "status": 429, "classification": {"kind": "Http", "retryable": true, "status": 429}},
                                {"provider_kind": "openai", "terminal_reason": "provider_error", "status": 429, "classification": {"kind": "Http", "retryable": true, "status": 429}},
                                {"provider_kind": "anthropic", "terminal_reason": "provider_error", "status": 429, "classification": {"kind": "Http", "retryable": true, "status": 429}},
                                {"provider_kind": "google_oauth", "terminal_reason": "provider_error", "status": 429, "classification": {"kind": "Http", "retryable": true, "status": 429}}
                            ]
                        }
                    }
                }),
            ),
            delivered_with_payload(
                19,
                "session-001:provider-mutation:003",
                "session-001",
                BoundaryKind::ProviderMutation,
                json!({"runtime_completion": runtime_completion("provider_script_mutation", 19)}),
                json!({
                    "mutation": "dropped_terminal_event",
                    "provider_parser_matrix": {
                        "matrix": {
                            "real_provider_parser_execution": true,
                            "provider_kinds": [
                                "anthropic",
                                "google_oauth",
                                "openai",
                                "openai-compatible"
                            ],
                            "proofs": [
                                {"provider_kind": "openai-compatible", "terminal_reason": "provider_error", "classification": {"retryable": false}},
                                {"provider_kind": "openai", "terminal_reason": "provider_error", "classification": {"retryable": false}},
                                {"provider_kind": "anthropic", "terminal_reason": "provider_error", "classification": {"retryable": false}},
                                {"provider_kind": "google_oauth", "terminal_reason": "provider_error", "classification": {"retryable": false}}
                            ]
                        }
                    }
                }),
            ),
        ]
        .into()
    }

    fn delivered_with_payload(
        sequence: usize,
        boundary_id: &str,
        actor_alias: &str,
        kind: BoundaryKind,
        payload: serde_json::Value,
        observed: serde_json::Value,
    ) -> DeliveredBoundary {
        DeliveredBoundary {
            schema: crate::scheduler::BOUNDARY_EVENT_SCHEMA.to_string(),
            sequence,
            scheduler: SchedulerDeliveryEvidence {
                scheduler_controlled: true,
                delivered_at: sequence as u64,
                ..SchedulerDeliveryEvidence::default()
            },
            boundary_id: boundary_id.to_string(),
            actor_alias: actor_alias.to_string(),
            kind,
            at: sequence as u64,
            label: format!("{kind:?}"),
            payload,
            observed,
        }
    }

    fn runtime_completion(family: &str, ready_at: u64) -> serde_json::Value {
        json!({
            "completion_family": family,
            "completion_units": [
                {
                    "unit": format!("runtime:{family}"),
                    "at": ready_at
                }
            ],
            "ready_at": ready_at,
            "registered_after": "session-001:ingress"
        })
    }

    fn provider_mutation_observed(mutation: &str) -> serde_json::Value {
        json!({
            "mutation": mutation,
            "provider_parser_matrix": {
                "matrix": {
                    "real_provider_parser_execution": true,
                    "provider_kinds": [
                        "anthropic",
                        "google_oauth",
                        "openai",
                        "openai-compatible"
                    ]
                }
            }
        })
    }

    fn provider_event(sequence: usize, actor: &str, turn_boundary_id: &str) -> DeliveredBoundary {
        delivered_with_payload(
            sequence,
            &format!("{turn_boundary_id}:event:{sequence}"),
            actor,
            BoundaryKind::ProviderEvent,
            json!({ "turn_boundary_id": turn_boundary_id }),
            json!({}),
        )
    }

    fn provider_completion(sequence: usize, actor: &str, turn_boundary_id: &str) -> DeliveredBoundary {
        delivered_with_payload(
            sequence,
            turn_boundary_id,
            actor,
            BoundaryKind::Provider,
            json!({ "provider_kind": "openai" }),
            json!({ "provider_kind": "openai" }),
        )
    }

    #[test]
    fn interleaving_oracle_passes_when_two_sessions_overlap() {
        // Turn A and turn B are both live (each has released a provider event)
        // before either completes.
        let events = vec![
            provider_event(0, "session-a", "turn-a"),
            provider_event(1, "session-b", "turn-b"),
            provider_completion(2, "session-a", "turn-a"),
            provider_completion(3, "session-b", "turn-b"),
        ];
        assert_eq!(peak_concurrent_live_turns(&events), 2);
        assert!(provider_turn_interleaving_depth(&events).is_passed());
    }

    #[test]
    fn interleaving_oracle_fails_when_multi_session_turns_never_overlap() {
        // Two sessions each run a turn, but each turn completes before the next
        // one releases an event, so peak concurrency is 1.
        let events = vec![
            provider_event(0, "session-a", "turn-a"),
            provider_completion(1, "session-a", "turn-a"),
            provider_event(2, "session-b", "turn-b"),
            provider_completion(3, "session-b", "turn-b"),
        ];
        assert_eq!(peak_concurrent_live_turns(&events), 1);
        let verdict = provider_turn_interleaving_depth(&events);
        assert!(!verdict.is_passed());
        assert_eq!(verdict.oracle_id, PROVIDER_TURN_INTERLEAVING_ORACLE);
    }

    #[test]
    fn interleaving_oracle_passes_vacuously_for_single_session() {
        let events = vec![
            provider_event(0, "session-a", "turn-a"),
            provider_completion(1, "session-a", "turn-a"),
        ];
        assert_eq!(peak_concurrent_live_turns(&events), 1);
        assert!(provider_turn_interleaving_depth(&events).is_passed());
    }

    fn suspend_resume_event(suspended_before: bool, before: u64, after: u64) -> DeliveredBoundary {
        delivered_with_payload(
            0,
            "suspend-tool:suspend-resume:001",
            "suspend-tool",
            BoundaryKind::Tool,
            json!({ "suspend_resume": true, "tool": "await_tool", "output": {"ok": true} }),
            json!({
                "session": "suspend-tool",
                "tool_output": {"ok": true},
                "runtime_suspend": {
                    "suspend_kind": "tool",
                    "turn_suspended_before_completion": suspended_before,
                    "scheduler_delivered_completion": true,
                    "resolve_accepted": true,
                    "resumed_after_completion": after > before,
                    "completed_event_count_before_resolution": before,
                    "completed_event_count_after_resolution": after,
                    "final_assistant_message": "resumed",
                },
            }),
        )
    }

    #[test]
    fn suspend_resume_oracle_passes_when_turn_parked_then_resumed() {
        let events = vec![suspend_resume_event(true, 0, 1)];
        assert!(generated_suspend_resume(&events).is_passed());
    }

    #[test]
    fn suspend_resume_oracle_fails_when_turn_ran_synchronously() {
        // The tool completed before the scheduler delivered the completion
        // boundary: the turn never actually parked.
        let events = vec![suspend_resume_event(false, 1, 1)];
        let verdict = generated_suspend_resume(&events);
        assert!(!verdict.is_passed());
        assert_eq!(verdict.oracle_id, GENERATED_SUSPEND_RESUME_ORACLE);
    }

    #[test]
    fn suspend_resume_oracle_passes_vacuously_without_suspend_boundaries() {
        let events = vec![provider_completion(0, "session-a", "turn-a")];
        assert!(generated_suspend_resume(&events).is_passed());
    }

}
