# Scenario: File Summary

Purpose: verify a realistic read/summarize workflow in a temporary workspace.

Setup:

- Create a fresh temporary working directory.
- Before starting or while in another shell, create `inventory.txt` in that
  directory with:

```text
apples 3
oranges 5
pears 2
```

- Start the harness in RLM mode with `--working-dir` set to that directory.

Steps:

```text
send Read inventory.txt and summarize the total item count.
idle
screen
screenshot file-summary
```

Inspect:

- The answer should mention a total of `10`.
- The UI should make it clear that the command ran in the requested workspace.
