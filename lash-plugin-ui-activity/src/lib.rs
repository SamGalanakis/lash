use std::sync::Arc;

use lash::plugin::{PluginError, PluginFactory, PluginSessionContext, PluginSpecFactory};

pub struct UiActivityPluginFactory;

impl PluginFactory for UiActivityPluginFactory {
    fn id(&self) -> &'static str {
        "ui_activity"
    }

    fn build(
        &self,
        _ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash::plugin::SessionPlugin>, PluginError> {
        PluginSpecFactory::new(
            "ui_activity",
            Arc::new(|_ctx| {
                Ok(lash::plugin::PluginSpec::new()
                    .with_after_turn(Arc::new(|ctx| {
                        Box::pin(async move {
                            let body = match ctx.turn.status {
                                lash::TurnStatus::Completed => {
                                    Some("Response complete".to_string())
                                }
                                lash::TurnStatus::Interrupted => None,
                                lash::TurnStatus::Failed => None,
                            };
                            let Some(body) = body else {
                                return Ok(Vec::new());
                            };
                            Ok(vec![lash::plugin::PluginDirective::emit_events(vec![
                                lash::PluginSurfaceEvent::Custom {
                                    name: "desktop_notification".to_string(),
                                    payload: serde_json::json!({
                                        "title": "lash",
                                        "body": body,
                                        "only_when_unfocused": true,
                                    }),
                                },
                            ])])
                        })
                    }))
                    .with_prompt_request(Arc::new(|ctx| {
                        Box::pin(async move {
                            Ok(vec![lash::PluginSurfaceEvent::Custom {
                                name: "desktop_notification".to_string(),
                                payload: serde_json::json!({
                                    "title": "lash",
                                    "body": ctx.request.question,
                                    "only_when_unfocused": true,
                                }),
                            }])
                        })
                    })))
            }),
        )
        .build(_ctx)
    }
}
