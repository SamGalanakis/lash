# Scenario: Multiturn Workspace

Purpose: verify that a session can do work across multiple user turns in the
same temporary working directory.

Setup:

- Create a fresh temporary working directory.
- Start the harness in RLM mode with `--working-dir` set to that directory.

Steps:

```text
send Create notes.md with a heading "QC Notes" and a line "first turn complete".
idle
screenshot multiturn-step-1
send Append a line "second turn complete" to notes.md.
idle
screen
screenshot multiturn-step-2
```

Inspect:

- The final `notes.md` should contain both lines.
- The UI should not require exact magic response text; judge the actual work.
- If the model claims it updated the file but did not, record that as model
  behavior unless the trace shows Lash ignored a valid write.
