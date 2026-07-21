//! Provider-routing (`OpenAiCompat::provider_routing`) wire-shape tests.

use super::*;

#[test]
fn openrouter_chat_body_requires_supported_parameters_with_json_schema_output() {
    let mut req = request(vec![LlmMessage::text(LlmRole::User, "extract")]);
    req.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
        name: "extraction".to_string(),
        schema: json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"]
        })
        .into(),
        strict: true,
    }));

    let body = openrouter_provider()
        .build_chat_request_body(&req, false)
        .unwrap();

    assert_eq!(body["provider"], json!({ "require_parameters": true }));
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "extraction");
}

#[test]
fn openrouter_chat_body_requires_supported_parameters_without_output_spec() {
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

    let body = openrouter_provider()
        .build_chat_request_body(&req, false)
        .unwrap();

    assert_eq!(body["provider"], json!({ "require_parameters": true }));
    assert!(body.get("response_format").is_none());
}

// An explicitly declared but empty `provider_routing` stays on the wire as
// `require_parameters: false` rather than vanishing: the body mirrors the
// declared config exactly. Absence of the whole object means "this endpoint
// does not do provider routing", which is a different statement.
#[test]
fn declared_empty_provider_routing_emits_require_parameters_false() {
    let compat: OpenAiCompat = serde_json::from_value(json!({ "provider_routing": {} })).unwrap();
    assert_eq!(
        compat.provider_routing,
        Some(ProviderRoutingPrefs {
            require_parameters: false
        })
    );

    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);
    let body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .with_compat(compat)
        .build_chat_request_body(&req, false)
        .unwrap();

    assert_eq!(body["provider"], json!({ "require_parameters": false }));
}

#[test]
fn default_compat_and_direct_openai_bodies_omit_provider_routing() {
    let req = request(vec![LlmMessage::text(LlmRole::User, "hello")]);

    let default_compat_body = OpenAiCompatibleProvider::new("key", OPENROUTER_BASE_URL)
        .build_chat_request_body(&req, false)
        .unwrap();
    let direct_openai_body = OpenAiProvider::new("key")
        .build_responses_request_body(&req, false)
        .unwrap();

    assert!(default_compat_body.get("provider").is_none());
    assert!(direct_openai_body.get("provider").is_none());
}
