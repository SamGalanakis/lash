# Scenario: Large Output

Purpose: check that large command output can be captured into variables and
summarized without flooding the UI.

Setup:

- Start the harness in RLM mode.
- Use a fresh temporary working directory.

Steps:

```text
send Generate 250 numbered lines with a shell command, keep the output in a variable, and tell me only how many lines there were.
idle
screen
screenshot large-output
```

Inspect:

- The visible answer should summarize the line count.
- The UI should not print all 250 lines unless the user explicitly asked for it.
- There should be no truncation error caused by merely saving command output to a
  variable.
