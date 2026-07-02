//! Test-support surface for embedders and the durable artifact stores.
//!
//! Gated behind the `testing` feature (and always available under `cfg(test)`)
//! so it never ships in a production build.

pub mod conformance;
