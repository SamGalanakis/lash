# Mock Chat Service

This example is a backend-only chat service built around the `lash-embed`
facade.
It intentionally avoids HTTP, database schemas, frontend streaming protocols, and
provider credentials.

The host application owns:

- conversation IDs
- request/response DTOs
- runtime storage in memory
- event collection
- the mock provider

`lash-embed` owns:

- turn execution
- model request construction
- event emission
- session read state
- standard-mode conversation semantics
- the `LashCore` / session / turn wrapper over the lower-level runtime

Run it:

```bash
cargo run -p mock-chat-service
```

Run the contract-style test:

```bash
cargo test -p mock-chat-service
```

The example is also an API pressure harness. App embedders should not need to
assemble `SessionPolicy`, `ExecutionMode`, mode plugin factories, or
`LashRuntime::builder()` directly.
