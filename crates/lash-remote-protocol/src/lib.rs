use std::collections::{BTreeMap, HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

include!("protocol/version.rs");
include!("protocol/llm.rs");
include!("protocol/turn_input.rs");
include!("protocol/observations.rs");
include!("protocol/processes.rs");
include!("protocol/triggers.rs");
include!("protocol/turn_result.rs");
include!("protocol/prompt.rs");
include!("protocol/tools.rs");
include!("protocol/usage_activity.rs");
include!("protocol/registry_errors.rs");

#[cfg(feature = "core-conversions")]
mod core_conversions;

#[cfg(feature = "core-conversions")]
pub use core_conversions::{RemoteTurnActivitySink, replay_collected_activities};

#[cfg(test)]
mod tests;
