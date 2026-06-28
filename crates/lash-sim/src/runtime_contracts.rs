use serde::{Deserialize, Serialize};

use crate::trace::{OracleStatus, OracleVerdict};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeTurnObservation {
    pub session_id: String,
    pub turn_index: usize,
    pub assistant_message: String,
    pub graph_node_count: usize,
    pub transcript_message_count: usize,
    pub activity_count: usize,
    pub provider_exchange_count: usize,
}

pub fn runtime_turn_contract(
    observation: &RuntimeTurnObservation,
    expected_session_id: &str,
    expected_turn_index: usize,
    expected_assistant_message: &str,
    expected_provider_exchange_count: usize,
) -> OracleVerdict {
    if observation.session_id != expected_session_id {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "session id diverged: expected `{expected_session_id}`, got `{}`",
                observation.session_id
            ),
        );
    }
    if observation.turn_index != expected_turn_index {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "turn index diverged: expected {expected_turn_index}, got {}",
                observation.turn_index
            ),
        );
    }
    if observation.assistant_message != expected_assistant_message {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "provider output diverged: expected `{expected_assistant_message}`, got `{}`",
                observation.assistant_message
            ),
        );
    }
    if observation.provider_exchange_count != expected_provider_exchange_count {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "provider exchange count diverged: expected {expected_provider_exchange_count}, got {}",
                observation.provider_exchange_count
            ),
        );
    }
    let expected_min_count = expected_turn_index * 2;
    if observation.graph_node_count < expected_min_count {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "session graph too small for turn {expected_turn_index}: {} nodes",
                observation.graph_node_count
            ),
        );
    }
    if observation.transcript_message_count < expected_min_count {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            format!(
                "transcript too small for turn {expected_turn_index}: {} messages",
                observation.transcript_message_count
            ),
        );
    }
    if observation.activity_count == 0 {
        return OracleVerdict::failed(
            "runtime.turn_contract",
            "turn emitted no runtime activities".to_string(),
        );
    }
    OracleVerdict::passed(
        "runtime.turn_contract",
        format!(
            "turn {expected_turn_index} matched session, transcript, graph, and provider output contracts"
        ),
    )
}

pub fn require_passed(verdict: &OracleVerdict) -> Result<(), String> {
    match verdict.status {
        OracleStatus::Passed => Ok(()),
        OracleStatus::Failed => Err(verdict.message.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> RuntimeTurnObservation {
        RuntimeTurnObservation {
            session_id: "session-001".to_string(),
            turn_index: 2,
            assistant_message: "answer".to_string(),
            graph_node_count: 4,
            transcript_message_count: 4,
            activity_count: 3,
            provider_exchange_count: 2,
        }
    }

    #[test]
    fn runtime_turn_contract_accepts_realistic_observation() {
        let verdict = runtime_turn_contract(&observation(), "session-001", 2, "answer", 2);
        assert_eq!(verdict.status, OracleStatus::Passed);
    }

    #[test]
    fn runtime_turn_contract_rejects_graph_regression() {
        let mut observed = observation();
        observed.graph_node_count = 1;
        let verdict = runtime_turn_contract(&observed, "session-001", 2, "answer", 2);
        assert_eq!(verdict.status, OracleStatus::Failed);
    }
}
