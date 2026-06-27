use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::scheduler::{BoundaryEvent, BoundaryKind, next_seed};
use crate::trace::StableAliases;

pub const GENERATOR_VERSION: &str = "lash-sim.generated-workload.v3";
pub const WORKLOAD_FAMILY: &str = "multi-session-runtime-semantic-boundaries";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GeneratedSession {
    pub alias: String,
    pub raw_session_id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GeneratedWorkload {
    pub seed: u64,
    pub profile: String,
    pub generator_version: String,
    pub workload_family: String,
    pub workload_id: String,
    pub sessions: Vec<GeneratedSession>,
    pub boundaries: Vec<BoundaryEvent>,
    pub aliases: StableAliases,
}

pub fn generate_workload(seed: u64, profile: &str, max_boundaries: usize) -> GeneratedWorkload {
    let mut aliases = StableAliases::default();
    let mut rng = seed;
    let session_count = 2 + (next_seed(&mut rng) as usize % 2);
    let sessions = (0..session_count)
        .map(|index| {
            let raw_session_id = format!("generated-session-{seed}-{index}");
            let alias = aliases.alias("session", raw_session_id.clone());
            GeneratedSession {
                alias,
                raw_session_id,
            }
        })
        .collect::<Vec<_>>();

    let mut boundaries = Vec::new();
    for (index, session) in sessions.iter().enumerate() {
        let base_at = (index as u64) * 8;
        let provider_texts = vec![
            format!("answer 1 for {}", session.alias),
            format!("answer 2 for {}", session.alias),
        ];
        boundaries.push(BoundaryEvent::new(
            format!("{}:ingress", session.alias),
            session.alias.clone(),
            BoundaryKind::Ingress,
            base_at,
            "session.open",
            json!({
                "raw_session_id": session.raw_session_id.clone(),
                "provider_script": "openai-compatible.chat-runtime-text-stream",
                "provider_texts": provider_texts.clone(),
            }),
        ));
        for (turn_offset, provider_text) in provider_texts.iter().enumerate() {
            let turn_index = turn_offset + 1;
            let provider_at = base_at + 1 + (turn_offset as u64 * 2);
            boundaries.push(BoundaryEvent::new(
                format!("{}:provider:{turn_index:03}", session.alias),
                session.alias.clone(),
                BoundaryKind::Provider,
                provider_at,
                "provider.chat.stream",
                json!({
                    "script": "openai-compatible.chat-runtime-text-stream",
                    "text": provider_text.clone(),
                    "turn_index": turn_index,
                    "expected_provider_exchange_count": turn_index,
                    "expected_graph_node_count": turn_index * 2,
                    "expected_transcript_message_count": turn_index * 2,
                }),
            ));
            boundaries.push(BoundaryEvent::new(
                format!("{}:observer:{turn_index:03}", session.alias),
                session.alias.clone(),
                BoundaryKind::Observer,
                provider_at + 1,
                "observer.snapshot",
                json!({
                    "turn_index": turn_index,
                    "expected_graph_node_count": turn_index * 2,
                    "expected_transcript_message_count": turn_index * 2,
                }),
            ));
        }
        boundaries.push(BoundaryEvent::new(
            format!("{}:lease-time", session.alias),
            session.alias.clone(),
            BoundaryKind::LeaseTime,
            base_at + 7,
            "lease.clock.advance",
            json!({
                "tick": base_at + 6,
            }),
        ));
    }
    if let Some(first_session) = sessions.first() {
        boundaries.push(BoundaryEvent::new(
            "durable-effect:sleep:first",
            first_session.alias.clone(),
            BoundaryKind::DurableEffect,
            (session_count as u64) * 8,
            "durable.sleep.complete",
            json!({
                "session": first_session.alias.clone(),
                "durable_key": format!("sleep/{}/1", first_session.alias),
                "result": { "completed": true, "wake_tick": (session_count as u64) * 8 },
                "runtime_effect": {
                    "kind": "sleep",
                    "effect_id": format!("effect/sleep/{}/1", first_session.alias),
                    "duration_ms": 1
                },
            }),
        ));
        boundaries.push(BoundaryEvent::new(
            "durable-effect:sleep:replay",
            first_session.alias.clone(),
            BoundaryKind::DurableEffect,
            (session_count as u64) * 8 + 1,
            "durable.sleep.replay",
            json!({
                "session": first_session.alias.clone(),
                "durable_key": format!("sleep/{}/1", first_session.alias),
                "result": { "completed": false, "wake_tick": 0 },
                "runtime_effect": {
                    "kind": "sleep",
                    "effect_id": format!("effect/sleep/{}/1", first_session.alias),
                    "duration_ms": 1
                },
            }),
        ));
        boundaries.push(BoundaryEvent::new(
            "worker:stale-completion",
            "worker-001",
            BoundaryKind::Worker,
            (session_count as u64) * 8 + 2,
            "worker.stale-completion-rejected",
            json!({
                "session": first_session.alias.clone(),
            }),
        ));
    }
    let required_boundaries = boundaries.len();
    boundaries.truncate(max_boundaries.max(required_boundaries));

    let workload_id = workload_id(seed, profile, &boundaries);
    GeneratedWorkload {
        seed,
        profile: profile.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        workload_family: WORKLOAD_FAMILY.to_string(),
        workload_id,
        sessions,
        boundaries,
        aliases,
    }
}

pub fn default_seed_count(profile: &str) -> usize {
    match profile {
        "fast" | "fast-random" => 2,
        "default" | "default-random" => 8,
        "full" | "full-random" => 32,
        _ => 2,
    }
}

pub fn default_max_boundaries(profile: &str) -> usize {
    match profile {
        "fast" | "fast-random" => 24,
        "default" | "default-random" => 96,
        "full" | "full-random" => 384,
        _ => 24,
    }
}

fn workload_id(seed: u64, profile: &str, boundaries: &[BoundaryEvent]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(profile.as_bytes());
    hasher.update(GENERATOR_VERSION.as_bytes());
    for boundary in boundaries {
        hasher.update(boundary.boundary_id.as_bytes());
        hasher.update(boundary.actor_alias.as_bytes());
        hasher.update(boundary.at.to_le_bytes());
    }
    let digest = hasher.finalize();
    hex_digest(&digest)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_generation_is_seed_and_version_deterministic() {
        let first = generate_workload(42, "fast-random", 24);
        let second = generate_workload(42, "fast-random", 24);

        assert_eq!(first, second);
        assert!(first.sessions.len() >= 2);
        assert_eq!(first.generator_version, GENERATOR_VERSION);
    }
}
