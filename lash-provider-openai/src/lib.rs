#![allow(clippy::result_large_err)]

mod chat;
mod common;
mod config;
mod responses;
mod support;
#[cfg(test)]
mod tests;
mod transport;

pub use common::{OPENAI_BASE_URL, OPENROUTER_BASE_URL};
pub use config::{OpenAiCacheRetention, OpenAiCompatibleProvider, OpenAiProvider};
pub use transport::{OpenAiCompatibleProviderFactory, OpenAiProviderFactory};
