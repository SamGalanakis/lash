use super::support::*;

pub trait ProviderFactory: Send + Sync {
    fn kind(&self) -> &'static str;

    /// Instantiate a provider from its [`ProviderSpec::config`] blob.
    fn deserialize(&self, config: serde_json::Value) -> Result<ProviderComponents, String>;
}

#[derive(Clone, Default)]
pub struct ProviderRegistry {
    factories: BTreeMap<&'static str, Arc<dyn ProviderFactory>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, factory: Arc<dyn ProviderFactory>) {
        self.factories.insert(factory.kind(), factory);
    }

    pub fn build_from_spec(&self, spec: &ProviderSpec) -> Result<ProviderComponents, String> {
        let factory = self.factories.get(spec.kind.as_str()).ok_or_else(|| {
            format!(
                "provider `{}` is not registered. Call `lash::register_provider_factory` at startup.",
                spec.kind
            )
        })?;
        factory.deserialize(spec.config.clone())
    }

    pub fn factory(&self, kind: &str) -> Option<&Arc<dyn ProviderFactory>> {
        self.factories.get(kind)
    }
}

static PROVIDER_REGISTRY: LazyLock<RwLock<ProviderRegistry>> =
    LazyLock::new(|| RwLock::new(ProviderRegistry::new()));

/// Register a provider factory in the global registry. Hosts call this
/// once per backend at process startup, before constructing any
/// `LashConfig` or session from disk.
pub fn register_provider_factory(factory: Arc<dyn ProviderFactory>) {
    PROVIDER_REGISTRY.write().unwrap().register(factory);
}

/// Materialize a provider from its serialized form using the global
/// registry. Returns `Err` if no factory is registered for `spec.kind`.
pub fn build_provider(spec: &ProviderSpec) -> Result<ProviderComponents, String> {
    PROVIDER_REGISTRY.read().unwrap().build_from_spec(spec)
}

/// Look up a registered provider factory by kind. Returns `None` if no
/// factory with that kind is registered.
pub fn provider_factory(kind: &str) -> Option<Arc<dyn ProviderFactory>> {
    PROVIDER_REGISTRY.read().unwrap().factory(kind).cloned()
}
