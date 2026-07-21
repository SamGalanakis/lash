pub(crate) const CODEX_RESPONSES_TEXT: &str =
    include_str!("../provider-scripts/canonical/codex.responses-text-stream.json");
pub(crate) const CODEX_RESPONSES_RATE_LIMIT: &str =
    include_str!("../provider-scripts/canonical/codex.responses-rate-limit-429.json");
pub(crate) const CODEX_RESPONSES_DISCONNECT: &str =
    include_str!("../provider-scripts/canonical/codex.responses-mid-stream-disconnect.json");
pub(crate) const CODEX_RESPONSES_TOOL_CALL: &str =
    include_str!("../provider-scripts/canonical/codex.responses-tool-call-stream.json");
pub(crate) const OPENAI_COMPAT_TOOL_CALL: &str = include_str!(
    "../provider-scripts/canonical/openai-compatible.chat-tool-call-split-stream.json"
);
pub(crate) const OPENAI_COMPAT_RATE_LIMIT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-rate-limit-429.json");
pub(crate) const OPENAI_COMPAT_VALIDATION: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-validation-error.json");
pub(crate) const OPENAI_COMPAT_DISCONNECT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-mid-stream-disconnect.json");
pub(crate) const OPENAI_COMPAT_RESPONSE_START_TIMEOUT: &str = include_str!(
    "../provider-scripts/canonical/openai-compatible.chat-response-start-timeout.json"
);
pub(crate) const OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-stream-chunk-timeout.json");
pub(crate) const OPENAI_RESPONSES_TEXT: &str =
    include_str!("../provider-scripts/canonical/openai.responses-text-stream.json");
pub(crate) const ANTHROPIC_MESSAGES_TEXT: &str =
    include_str!("../provider-scripts/canonical/anthropic.messages-text-stream.json");
pub(crate) const GOOGLE_STREAM_GENERATE_TEXT: &str =
    include_str!("../provider-scripts/canonical/google.stream-generate-content-text-stream.json");
pub(crate) const GOOGLE_GENERATE_TEXT: &str =
    include_str!("../provider-scripts/canonical/google.generate-content-text.json");

#[derive(Clone, Copy)]
pub(crate) struct CanonicalScript {
    pub(crate) path: &'static str,
    pub(crate) content: &'static str,
}

pub(crate) const CANONICAL_SCRIPTS: &[CanonicalScript] = &[
    CanonicalScript {
        path: "provider-scripts/canonical/anthropic.messages-text-stream.json",
        content: ANTHROPIC_MESSAGES_TEXT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/codex.responses-mid-stream-disconnect.json",
        content: CODEX_RESPONSES_DISCONNECT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/codex.responses-rate-limit-429.json",
        content: CODEX_RESPONSES_RATE_LIMIT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/codex.responses-text-stream.json",
        content: CODEX_RESPONSES_TEXT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/codex.responses-tool-call-stream.json",
        content: CODEX_RESPONSES_TOOL_CALL,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/google.generate-content-text.json",
        content: GOOGLE_GENERATE_TEXT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/google.stream-generate-content-text-stream.json",
        content: GOOGLE_STREAM_GENERATE_TEXT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-mid-stream-disconnect.json",
        content: OPENAI_COMPAT_DISCONNECT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-rate-limit-429.json",
        content: OPENAI_COMPAT_RATE_LIMIT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-response-start-timeout.json",
        content: OPENAI_COMPAT_RESPONSE_START_TIMEOUT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-stream-chunk-timeout.json",
        content: OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-tool-call-split-stream.json",
        content: OPENAI_COMPAT_TOOL_CALL,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-validation-error.json",
        content: OPENAI_COMPAT_VALIDATION,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai.responses-text-stream.json",
        content: OPENAI_RESPONSES_TEXT,
    },
];
