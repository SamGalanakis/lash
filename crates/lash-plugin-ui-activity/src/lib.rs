use std::sync::Arc;

use lash_core::plugin::{PluginError, PluginFactory, PluginSessionContext, PluginSpecFactory};

pub struct UiActivityPluginFactory;

impl PluginFactory for UiActivityPluginFactory {
    fn id(&self) -> &'static str {
        "ui_activity"
    }

    fn build(
        &self,
        _ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash_core::plugin::SessionPlugin>, PluginError> {
        PluginSpecFactory::new(
            "ui_activity",
            Arc::new(|_ctx| {
                Ok(
                    lash_core::plugin::PluginSpec::new().with_after_turn(Arc::new(|ctx| {
                        Box::pin(async move {
                            let body = match &ctx.turn.outcome {
                                lash_core::TurnOutcome::Finished(_)
                                | lash_core::TurnOutcome::Handoff { .. } => {
                                    Some("Response complete".to_string())
                                }
                                lash_core::TurnOutcome::Stopped(_) => None,
                            };
                            let Some(body) = body else {
                                return Ok(Vec::new());
                            };
                            Ok(vec![lash_core::plugin::PluginDirective::emit_events(vec![
                                lash_core::PluginSurfaceEvent::Custom {
                                    name: "desktop_notification".to_string(),
                                    payload: serde_json::json!({
                                        "title": "lash",
                                        "body": body,
                                        "only_when_unfocused": true,
                                    }),
                                },
                            ])])
                        })
                    })),
                )
            }),
        )
        .build(_ctx)
    }
}
