# Recording provider traffic

Provider traffic is recorded at the existing `LlmHttpTransport` seam. The
`RecordingLlmHttpTransport` decorator wraps the same transport passed to a
provider's `with_transport` method and writes Provider Wire Script v1 files,
which `ScriptedLlmHttpTransport` replays without network access.

The recorder does not persist request bodies. FIG-523 traces the serialized,
pre-authentication body when an exact request snapshot is needed. A recording
contains only caller-selected, stable request matchers and the provider
response.

## Capture a response

1. Create a directory outside the repository for the capture. Never record
   directly into `provider-scripts`.
2. Use a dedicated provider project and a prompt containing unique marker
   strings. Pass every marker to `with_user_content_markers`.
3. Wrap the real transport and install the wrapper through `with_transport`:

   ```rust
   use std::sync::Arc;

   use lash_llm_transport::ReqwestLlmHttpTransport;
   use lash_sim::{ProviderRecordingConfig, RecordingLlmHttpTransport};

   let transport = Arc::new(RecordingLlmHttpTransport::new(
       Arc::new(ReqwestLlmHttpTransport::default()),
       ProviderRecordingConfig::new(
           "/tmp/lash-provider-captures",
           "openai_rate_limit",
           "openai",
       )
       .with_user_content_markers(["FIG530_CAPTURE_MARKER"])
       .with_notes("purpose and provider account tier"),
   ));

   let provider = provider.with_transport(transport.clone());
   ```

4. Make one targeted call. Retries are numbered separately as
   `<name>.001.json`, `<name>.002.json`, and so on.
5. Inspect every emitted file before moving it into the repository. Search for
   the unique prompt markers, credential prefixes, account identifiers, email
   addresses, and project names. Check the response body and all headers by
   hand. A capture that cannot be confidently reviewed must be discarded.
6. Replace the default empty request match with stable structural matchers such
   as model, streaming mode, endpoint, schema presence, and required header
   presence. Never match credential values or user text.
7. Replay the file through the real provider parser and add an assertion that
   distinguishes the observed behavior: retryable throttle versus terminal
   quota, exact backoff versus no backoff, structured auth classification, or
   a specific terminal reason.
8. Move the reviewed file to
   `crates/lash-sim/provider-scripts/recorded-reality/` and keep its generated
   `captured_live` provenance. Confirm that `source` is the endpoint path and
   `captured_at` is the capture timestamp, then append the exact phrase
   `redaction reviewed` to `provenance.notes`. CI must only replay it.

The decorator redacts authentication, cookie, and token header values in place;
scrubs request credential values if a provider echoes them; redacts sensitive
JSON fields; and replaces configured user markers. Streaming bodies are held
until the exchange ends so a secret split across chunks cannot cross the
redaction boundary. Redaction happens before the first file write. Non-UTF-8
responses are rejected instead of being persisted without inspection.

## Update an assembled-request snapshot

Run the focused snapshot test:

```console
cargo nextest run -p lash-sim --locked -E 'test(/request_snapshot/)'
```

When an intentional request-shape change fails the snapshot, review and update
the expected JSON first. Run the test again and copy the reported `body_len`,
then run it once more and copy the reported `body_sha256`. This assertion order
keeps the structural JSON diff visible before the compact integrity checks.

## When live reproduction is impractical

Use a verbatim body from provider documentation or a real incident report.
Record the source URL and whether any envelope or replay-only request metadata
was added in the fixture's `provenance`. Do not synthesize plausible error
wording, status codes, retry headers, or delays. A documented body without a
documented retry instruction must assert that no backoff was inferred.

The initial recorded-reality corpus follows this sourced path because no
provider credentials were available in the capture environment. Its bodies
come from provider documentation, provider-owned issue trackers, provider
forums, or cited real client reports; every fixture carries its own source.
