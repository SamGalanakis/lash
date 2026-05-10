//! Library half of `lash-cli`. Exposes the pieces that other workspace
//! crates (notably the bench runners) reuse: the on-disk config schema and
//! the `lash_home` paths helpers.
//!
//! The binary half (`main.rs`) still owns the TUI / interactive command /
//! setup-wizard code. Those live in private modules under `bin`.

pub mod config;
pub mod paths;
