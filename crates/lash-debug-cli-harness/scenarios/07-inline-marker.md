# Scenario: Inline Tag In Prose

Purpose: verify that inline text mentioning `<lashlang>` does not start an
execution cell.

Setup:

- Start the harness in RLM mode.

Steps:

```text
send Explain in one sentence that the tag <lashlang> only starts a cell when it is alone on its own line.
idle
screen
screenshot inline-marker
```

Inspect:

- The answer may mention `<lashlang>` inline.
- There should be no Lashlang execution caused by the inline mention.
- There should be no parser error.
