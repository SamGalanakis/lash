use std::collections::BTreeMap;

use lash::runtime::{
    RuntimeEffectCommand, RuntimeEffectEnvelope, RuntimeEffectKind, RuntimeInvocation, RuntimeScope,
};
use serde_json::{Value, json};

use crate::trace::value_digest;

#[derive(Clone, Debug, Default)]
pub struct DurableEffectJournal {
    entries: BTreeMap<String, DurableEffectEntry>,
}

impl DurableEffectJournal {
    pub fn complete(&mut self, durable_key: impl Into<String>, result: Value) -> Value {
        let durable_key = durable_key.into();
        match self.entries.get_mut(&durable_key) {
            Some(entry) => {
                entry.replay_count += 1;
                json!({
                    "durable_key": durable_key,
                    "result_digest": entry.result_digest,
                    "execution_count": entry.execution_count,
                    "replay_count": entry.replay_count,
                    "replayed": true,
                })
            }
            None => {
                let result_digest = value_digest(&result);
                let entry = DurableEffectEntry {
                    result_digest: result_digest.clone(),
                    execution_count: 1,
                    replay_count: 0,
                };
                self.entries.insert(durable_key.clone(), entry);
                json!({
                    "durable_key": durable_key,
                    "result_digest": result_digest,
                    "execution_count": 1,
                    "replay_count": 0,
                    "replayed": false,
                })
            }
        }
    }

    pub fn complete_runtime_effect(
        &mut self,
        envelope: RuntimeEffectEnvelope,
        result: Value,
    ) -> Value {
        let durable_key = envelope
            .invocation
            .replay_key()
            .unwrap_or("runtime-effect-without-replay-key")
            .to_string();
        let mut observed = self.complete(durable_key, result);
        observed["runtime_effect"] = json!({
            "kind": envelope.command.kind().as_str(),
            "effect_id": envelope.invocation.effect_id(),
            "replay_key": envelope.invocation.replay_key(),
            "envelope_hash": envelope.stable_hash().ok(),
        });
        observed
    }
}

pub fn sleep_effect_envelope(
    session_id: impl Into<String>,
    effect_id: impl Into<String>,
    replay_key: impl Into<String>,
    duration_ms: u64,
) -> RuntimeEffectEnvelope {
    RuntimeEffectEnvelope::new(
        RuntimeInvocation::effect(
            RuntimeScope::new(session_id),
            effect_id,
            RuntimeEffectKind::Sleep,
            replay_key,
        ),
        RuntimeEffectCommand::Sleep { duration_ms },
    )
}

#[derive(Clone, Debug)]
struct DurableEffectEntry {
    result_digest: String,
    execution_count: usize,
    replay_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durable_effect_journal_replays_by_key_without_reexecution() {
        let mut journal = DurableEffectJournal::default();
        let first = journal.complete("sleep:session-001:1", json!({"done": true}));
        let second = journal.complete("sleep:session-001:1", json!({"done": false}));

        assert_eq!(first["execution_count"], 1);
        assert_eq!(first["replayed"], false);
        assert_eq!(second["execution_count"], 1);
        assert_eq!(second["replay_count"], 1);
        assert_eq!(second["replayed"], true);
        assert_eq!(first["result_digest"], second["result_digest"]);
    }

    #[test]
    fn durable_effect_journal_uses_runtime_effect_replay_key() {
        let mut journal = DurableEffectJournal::default();
        let envelope =
            sleep_effect_envelope("session-001", "effect/sleep/1", "sleep/session-001/1", 5);
        let observed = journal.complete_runtime_effect(envelope, json!({"done": true}));

        assert_eq!(observed["durable_key"], "sleep/session-001/1");
        assert_eq!(observed["runtime_effect"]["kind"], "sleep");
        assert!(observed["runtime_effect"]["envelope_hash"].is_string());
    }
}
