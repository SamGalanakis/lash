use super::support::*;

pub trait ProviderFactory: Send + Sync {
    fn kind(&self) -> &'static str;

    /// Instantiate a provider from its [`ProviderSpec::config`] blob.
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String>;
}
