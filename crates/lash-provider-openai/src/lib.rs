#![allow(clippy::result_large_err)]

mod chat;
pub mod codex;
mod common;
mod config;
mod driver;
mod factory;
mod provider;
mod responses;
pub mod responses_shared;
pub mod schema;
mod support;
#[cfg(feature = "testing")]
pub mod testing;
#[cfg(test)]
mod tests;

pub use codex::{CodexProvider, CodexProviderFactory};
pub use common::{OPENAI_BASE_URL, OPENROUTER_BASE_URL};
pub use config::{OpenAiCompatibleProvider, OpenAiProvider};
pub use factory::{OpenAiCompatibleProviderFactory, OpenAiProviderFactory};
