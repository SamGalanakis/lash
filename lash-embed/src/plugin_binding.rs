use crate::support::*;

/// Typed app-facing activation for an ordinary Lash plugin.
///
/// A binding is not a second plugin implementation model. It gives embed hosts
/// typed session configuration and typed per-turn context while still building
/// a normal [`PluginFactory`] whose session plugin registers capabilities
/// through `lash::PluginRegistrar`.
///
/// Plugin crates should expose a small domain extension trait over
/// [`TurnBuilder`] for their turn context instead of asking app route code to
/// call [`TurnBuilder::with_plugin_context`] directly:
///
/// ```ignore
/// trait ToneTurnExt {
///     fn with_tone(self, tone: Tone) -> Self;
/// }
///
/// impl ToneTurnExt for lash_embed::TurnBuilder {
///     fn with_tone(self, tone: Tone) -> Self {
///         self.with_plugin_context::<TonePlugin>(ToneTurnContext { tone })
///     }
/// }
/// ```
pub trait PluginBinding: Send + Sync + 'static {
    const ID: &'static str;
    type SessionConfig: Clone + Send + Sync + 'static;
    type TurnContext: Clone + Send + Sync + 'static;

    fn factory(config: &Self::SessionConfig) -> Arc<dyn PluginFactory>;

    fn requires_turn_context(_config: &Self::SessionConfig) -> bool {
        false
    }
}

pub(crate) type PluginConfigMap<P> =
    Arc<StdMutex<HashMap<String, <P as PluginBinding>::SessionConfig>>>;

pub(crate) struct PluginBindingFactory<P: PluginBinding> {
    configs: PluginConfigMap<P>,
}

impl<P: PluginBinding> PluginBindingFactory<P> {
    pub(crate) fn new(configs: PluginConfigMap<P>) -> Self {
        Self { configs }
    }
}

impl<P: PluginBinding> lash::PluginFactory for PluginBindingFactory<P> {
    fn id(&self) -> &'static str {
        P::ID
    }

    fn build(
        &self,
        ctx: &lash::PluginSessionContext,
    ) -> std::result::Result<Arc<dyn lash::SessionPlugin>, lash::PluginError> {
        let config = self
            .configs
            .lock()
            .map_err(|_| {
                lash::PluginError::Session(format!("plugin `{}` config lock poisoned", P::ID))
            })?
            .get(&ctx.session_id)
            .cloned();
        let Some(config) = config else {
            return Ok(Arc::new(InactivePluginBinding { id: P::ID }));
        };
        P::factory(&config).build(ctx)
    }
}

struct InactivePluginBinding {
    id: &'static str,
}

impl lash::SessionPlugin for InactivePluginBinding {
    fn id(&self) -> &'static str {
        self.id
    }

    fn register(
        &self,
        _reg: &mut lash::PluginRegistrar,
    ) -> std::result::Result<(), lash::PluginError> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ActivePluginBinding {
    pub(crate) id: &'static str,
    pub(crate) requires_turn_context: bool,
}
