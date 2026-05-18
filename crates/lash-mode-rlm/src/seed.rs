use lash_core::SessionAppendNode;
use serde_json::Value;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct RlmSeed {
    pub projected: lash_rlm_types::RlmProjectedSeedSnapshot,
    pub globals: serde_json::Map<String, Value>,
}

impl RlmSeed {
    pub fn from_tool_args(args: &Value) -> Result<Self, String> {
        let raw = match args.get("seed") {
            None | Some(Value::Null) => return Ok(Self::default()),
            Some(Value::Object(map)) => map,
            Some(_) => return Err("`seed` must be a record/dict".to_string()),
        };
        let mut out = Self::default();
        for (name, value) in raw.iter() {
            if let Some(inner) = lash_rlm_types::projection_inner(value) {
                out.projected.push(name.clone(), inner.clone());
            } else {
                out.globals.insert(name.clone(), value.clone());
            }
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.globals.is_empty() && self.projected.is_empty()
    }

    pub fn into_event_body(self) -> lash_rlm_types::RlmSeedPluginBody {
        lash_rlm_types::RlmSeedPluginBody {
            globals: self.globals,
            projected: self.projected,
        }
    }
}

pub fn rlm_seed_initial_nodes(seed: RlmSeed) -> Vec<SessionAppendNode> {
    if seed.is_empty() {
        return Vec::new();
    }
    vec![SessionAppendNode::mode_event(crate::rlm_mode_event(
        lash_rlm_types::RlmModeEvent::RlmSeed(seed.into_event_body()),
    ))]
}
