#![allow(clippy::result_large_err)]

mod canonical_scripts;

pub mod backend_contention;
pub mod generator;
pub mod minimize;
pub mod oracles;
pub mod postgres_replay;
pub mod provider;
pub mod provider_mutations;
pub mod replay;
pub mod runner;
pub mod runtime_boundaries;
pub mod runtime_contracts;
pub mod runtime_providers;
pub mod scheduler;
pub mod sqlite_replay;
pub mod stack_policy;
pub mod store;
pub mod trace;

pub use provider::{
    ProviderWireEndpoint, ProviderWireEvent, ProviderWireRequestMatch, ProviderWireScript,
    ScriptedLlmHttpTransport, ScriptedTransportSchedule,
};
pub use runner::{
    FIXED_SCRIPT_PROFILE, FixedScriptManifest, FixedScriptProof, FixedScriptSummary,
    GeneratedPostgresReplayReport, GeneratedSimProfileReport, ScriptHashManifest,
    run_fixed_script_profile, run_generated_postgres_replay_for_seeds, run_generated_sim_profile,
    run_generated_sim_profile_for_seeds,
};
pub use stack_policy::{PRODUCT_STACK_BUDGET_BYTES, SIM_HARNESS_STACK_LIMIT_BYTES};
