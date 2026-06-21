# Lash Debug CLI Manual Scenarios

These scenarios are instructions for an agent or human operator driving the real
`lash` TUI through `lash-debug-cli-harness`.

They are not machine-judged fixtures. Start the harness, perform the steps, then
inspect the visible screen, `terminal.ansi`, `ui-trace.json`, and PNG screenshots.

## Purpose

Use these scenarios to observe the CLI as a user would. The harness should only
facilitate keyboard input, terminal capture, logs, and screenshots. The scenario
runner is the agent/operator, not code in the harness.

## Setup

Use a fresh temporary working directory and a fresh output directory for each
scenario. Put scenario setup files in the temporary working directory, not in the
repo.

Typical start command:

```sh
workspace=$(mktemp -d /tmp/lash-scenario-workspace-XXXXXX)
output=$(mktemp -d /tmp/lash-scenario-output-XXXXXX)

cargo run -p lash-debug-cli-harness -- \
  --execution-mode rlm \
  --lash-home /home/sam/.lash \
  --working-dir "$workspace" \
  --output-dir "$output" \
  --rows 45 \
  --cols 140 \
  --no-build
```

If the scenario needs files, create them in `$workspace` before sending the first
message.

## Driving The CLI

Type harness commands into the harness process. Use `send` for normal user
messages and `idle` after each submitted user turn.

Useful harness commands:

```text
send hi
idle
screen
screenshot after-hi
log
quit
```

Use `screenshot NAME` after the important state is visible. It writes:

- `screens/NAME.txt`
- `screens/NAME.svg`
- `screens/NAME.png`

Inspect the PNG visually. Use `screen` or `screens/NAME.txt` for exact text.
Use `ui-trace.json` and `terminal.ansi` when the visible UI is unclear.

## When To Stop

Stop the scenario when one of these is true:

- The scenario's expected behavior is visible and a screenshot has been saved.
- A clear failure is visible, such as a red parser error, leaked `<lashlang>` tags,
  stuck non-idle state, wrong working directory, or unreadable UI.
- The model completes the turn but does the wrong work. Capture the screen and
  classify it as model behavior unless the trace shows Lash ignored valid code.
- The turn exceeds the configured timeout or appears stuck. Capture the latest
  screen and logs before killing the harness.

Use `quit` for a clean exit. Use `kill` only if the TUI is stuck.

## Reporting Results

Report back in this format:

```text
Scenario:
Result: pass | fail | inconclusive
Workspace:
Output dir:
Screenshots:
Observed:
Issues:
Likely cause: CLI | Lash runtime | model | harness | unclear
Repo changes made: none
```

When running these as QC subagents, do not edit repo files. Temporary workspace
files and harness artifacts are fine.

## Checklist

- Use a fresh temporary working directory unless the scenario says otherwise.
- Capture at least one PNG screenshot with `screenshot NAME`.
- Check the PNG visually, not only `screen.txt`.
- Record trace and screenshot paths in the report.
- Do not make repo changes during scenario execution unless explicitly asked.
