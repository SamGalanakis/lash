# Scenario: Time Query

Purpose: exercise a simple tool-backed answer where the model may use Lashlang.
This previously exposed model text after code being parsed as Lashlang.

Setup:

- Start the harness in RLM mode with a fresh output directory.

Steps:

```text
send What time is it?
idle
screen
screenshot time-query
```

Inspect:

- The final visible answer should contain a time or a clear inability to get it.
- If a red parser error appears, inspect `ui-trace.json` to see whether the model
  emitted malformed text inside a paired `<lashlang>` block.
- The screen should not show duplicated retries that bury the latest response.
