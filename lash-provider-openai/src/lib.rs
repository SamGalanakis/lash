#![allow(clippy::result_large_err)]

mod chat;
mod common;
mod config;
mod responses;
mod support;
#[cfg(test)]
mod tests;
mod transport;

pub use common::OPENROUTER_BASE_URL;
pub use config::{OpenAiCacheRetention, OpenAiGenericProvider, OpenAiWireApi};
pub use transport::OpenAiGenericProviderFactory;
