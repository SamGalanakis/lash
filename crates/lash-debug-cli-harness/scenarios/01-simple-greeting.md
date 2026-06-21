# Scenario: Simple Greeting

Purpose: verify a normal RLM greeting does not leak `<lashlang>` tags, does not show a
parser error, and leaves the prompt usable.

Setup:

- Start the harness in RLM mode with a fresh output directory.

Steps:

```text
send hi
idle
screen
screenshot simple-greeting
```

Inspect:

- The visible answer should be a short greeting like `Hi! How can I help?`.
- There should be no red error panel.
- There should be no visible `<lashlang>` tag in the final assistant answer.
- The bottom input prompt should be usable and the status should be `Idle`.
