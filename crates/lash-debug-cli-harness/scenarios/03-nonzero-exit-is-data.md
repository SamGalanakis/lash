# Scenario: Nonzero Exit Is Data

Purpose: verify the model can inspect a failing shell command without Lash
treating the nonzero exit as an unwrapped module failure.

Setup:

- Start the harness in RLM mode.
- Use a fresh temporary working directory.

Steps:

```text
send Run a shell command that exits with code 7, capture the result, and tell me the exit code. Do not treat the nonzero exit as a tool failure.
idle
screen
screenshot nonzero-exit
```

Inspect:

- The answer should mention exit code `7`.
- The UI should not show `unwrapped failed module operation`.
- If the model reports the nonzero exit as an exception, check whether that came
  from model code or from Lash runtime behavior.
