use std::collections::BTreeSet;

use crate::trace::{AbstractWorldSummary, OracleVerdict};

pub const CROSS_SESSION_ISOLATION_ORACLE: &str = "sim.oracle.cross-session-isolation.v1";
pub const DURABLE_EFFECT_EXACTLY_ONCE_ORACLE: &str = "sim.oracle.durable-effect-exactly-once.v1";
pub const INGRESS_SESSION_OPENED_ORACLE: &str = "sim.oracle.ingress-session-opened.v1";
pub const LEASE_TIME_MONOTONIC_ORACLE: &str = "sim.oracle.lease-time-monotonic.v1";
pub const OBSERVER_CONVERGENCE_ORACLE: &str = "sim.oracle.observer-convergence.v1";
pub const REPLAY_DETERMINISM_ORACLE: &str = "sim.oracle.replay-determinism.v1";
pub const RUNTIME_PROVIDER_TURN_ORACLE: &str = "sim.oracle.runtime-provider-turn.v1";
pub const RUNTIME_SESSION_GRAPH_ORACLE: &str = "sim.oracle.runtime-session-graph.v1";
pub const WORKER_STALE_COMPLETION_ORACLE: &str = "sim.oracle.worker-stale-completion-rejected.v1";

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

pub fn lease_time_monotonic(summary: &AbstractWorldSummary) -> OracleVerdict {
    for session in &summary.sessions {
        if session
            .lease_time_ticks
            .windows(2)
            .any(|ticks| ticks[0] > ticks[1])
        {
            return OracleVerdict::failed(
                LEASE_TIME_MONOTONIC_ORACLE,
                format!(
                    "session `{}` lease/time ticks moved backwards",
                    session.alias
                ),
            );
        }
    }
    OracleVerdict::passed(
        LEASE_TIME_MONOTONIC_ORACLE,
        "lease/time boundary ticks were monotonic per session",
    )
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

pub fn combine_oracles(oracles: &[OracleVerdict]) -> OracleVerdict {
    if let Some(failure) = oracles.iter().find(|oracle| !oracle.is_passed()) {
        return failure.clone();
    }
    OracleVerdict::passed(
        "sim.oracle.generated-workload.v1",
        format!("{} generated workload oracles passed", oracles.len()),
    )
}
