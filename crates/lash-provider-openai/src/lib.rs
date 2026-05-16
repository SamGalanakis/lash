#![allow(clippy::result_large_err)]

mod chat;
mod common;
mod config;
mod driver;
mod factory;
mod policy;
mod provider;
mod responses;
mod support;
#[cfg(test)]
mod tests;

pub use common::{OPENAI_BASE_URL, OPENROUTER_BASE_URL};
pub use config::{OpenAiCompatibleProvider, OpenAiProvider};
pub use factory::{OpenAiCompatibleProviderFactory, OpenAiProviderFactory};
