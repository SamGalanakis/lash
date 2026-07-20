//! Explicitly opt-in, development-only LLM Provider scenarios for failure UX.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use async_trait::async_trait;
use lash::direct::{LlmOutputPart, LlmStreamEvent};
use lash::provider::{
    LlmRequest, LlmResponse, LlmTransportError, Provider, ProviderComponents, ProviderHandle,
    ProviderOptions, ProviderReliability,
};

pub(crate) const DEV_PROVIDER_SCENARIO_ENV: &str = "AGENT_WORKBENCH_DEV_PROVIDER_SCENARIO";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DevProviderScenario {
    AuthFailureOnce,
    RateLimitOnce,
    FailedProcess,
    ExecBlocked,
}

impl DevProviderScenario {
    pub(crate) fn from_environment() -> Result<Option<Self>> {
        let Ok(value) = std::env::var(DEV_PROVIDER_SCENARIO_ENV) else {
            return Ok(None);
        };
        let value = value.trim();
        if value.is_empty() {
            return Ok(None);
        }
        let scenario = match value {
            "auth-failure-once" => Self::AuthFailureOnce,
            "rate-limit-once" => Self::RateLimitOnce,
            "failed-process" => Self::FailedProcess,
            "exec-blocked" => Self::ExecBlocked,
            other => bail!(
                "invalid {DEV_PROVIDER_SCENARIO_ENV} `{other}`; expected one of: \
                 auth-failure-once, rate-limit-once, failed-process, exec-blocked"
            ),
        };
        Ok(Some(scenario))
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::AuthFailureOnce => "auth-failure-once",
            Self::RateLimitOnce => "rate-limit-once",
            Self::FailedProcess => "failed-process",
            Self::ExecBlocked => "exec-blocked",
        }
    }

    pub(crate) fn provider(self) -> ProviderHandle {
        ProviderHandle::new(ProviderComponents::new(Box::new(DevFailureProvider {
            scenario: self,
            calls: Arc::new(AtomicUsize::new(0)),
            options: ProviderOptions {
                reliability: ProviderReliability::default()
                    .max_attempts(2)
                    .base_delay_ms(0)
                    .max_delay_ms(0),
                ..ProviderOptions::default()
            },
        })))
    }
}

#[derive(Clone, Debug)]
struct DevFailureProvider {
    scenario: DevProviderScenario,
    calls: Arc<AtomicUsize>,
    options: ProviderOptions,
}

#[async_trait]
impl Provider for DevFailureProvider {
    fn kind(&self) -> &'static str {
        "workbench-dev-failure"
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        serde_json::json!({ "scenario": self.scenario.as_str() })
    }

    fn requires_streaming(&self) -> bool {
        true
    }

    async fn complete(
        &mut self,
        request: LlmRequest,
    ) -> std::result::Result<LlmResponse, LlmTransportError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.scenario {
            DevProviderScenario::AuthFailureOnce if call == 0 => {
                send_delta(&request, "provider authentication check started");
                Err(
                    LlmTransportError::new("development provider rejected credentials mid-turn")
                        .with_status(401)
                        .with_code("dev_auth_rejected"),
                )
            }
            DevProviderScenario::AuthFailureOnce => Ok(streamed_response(
                &request,
                "<lashlang>\nfinish \"session recovered after provider auth failure\"\n</lashlang>",
            )),
            DevProviderScenario::RateLimitOnce if call == 0 => {
                // Cross the RLM prose boundary before failing so the retry
                // reset has visible first-attempt output to retract.
                send_delta(&request, "retry observer single-copy marker\n<lashlang>\n");
                Err(
                    LlmTransportError::new("development provider rate limit; retry is safe")
                        .with_status(429)
                        .with_code("dev_rate_limited"),
                )
            }
            DevProviderScenario::RateLimitOnce => Ok(streamed_response(
                &request,
                "retry observer single-copy marker\n<lashlang>\nfinish \"provider retry succeeded\"\n</lashlang>",
            )),
            DevProviderScenario::FailedProcess => Ok(streamed_response(
                &request,
                r#"<lashlang>
process FIG425_deterministic_failure() {
  fail "deterministic durable process failure"
}
start FIG425_deterministic_failure()
finish "started deterministic failing process"
</lashlang>"#,
            )),
            DevProviderScenario::ExecBlocked if call == 0 => Ok(streamed_response(
                &request,
                r#"<lashlang>
sleep for "10m"
finish "exec block unexpectedly returned"
</lashlang>"#,
            )),
            DevProviderScenario::ExecBlocked => Ok(streamed_response(
                &request,
                "<lashlang>\nfinish \"session recovered after break glass\"\n</lashlang>",
            )),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Provider> {
        Box::new(self.clone())
    }
}

fn streamed_response(request: &LlmRequest, text: &str) -> LlmResponse {
    send_delta(request, text);
    LlmResponse {
        full_text: text.to_string(),
        parts: vec![LlmOutputPart::Text {
            text: text.to_string(),
            response_meta: None,
        }],
        response_metadata: Default::default(),
        ..LlmResponse::default()
    }
}

fn send_delta(request: &LlmRequest, text: &str) {
    if let Some(events) = request.stream_events.as_ref() {
        events.send(LlmStreamEvent::Delta(text.to_string()));
    }
}
