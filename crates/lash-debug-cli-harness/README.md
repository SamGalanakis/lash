# lash-debug-cli-harness

`lash-debug-cli-harness` is a thin PTY controller for the real interactive
`lash` TUI. It is for debugging the UI the way a user sees it: send keys, wait,
print the screen, and save screenshots/logs.

It does not run or judge scenarios. Scenario files under `scenarios/` are
markdown instructions for an agent or human operator.

## Start

Use your normal `LASH_HOME` when testing a real provider:

```sh
cargo run -p lash-debug-cli-harness -- \
  --execution-mode rlm \
  --lash-home /home/sam/.lash \
  --output-dir /tmp/lash-debug-run \
  --rows 45 \
  --cols 140 \
  --no-build
```

Use a throwaway working directory when checking file edits:

```sh
mkdir -p /tmp/lash-debug-workspace
cargo run -p lash-debug-cli-harness -- \
  --execution-mode rlm \
  --lash-home /home/sam/.lash \
  --working-dir /tmp/lash-debug-workspace \
  --output-dir /tmp/lash-debug-run \
  --no-build
```

## Commands

After `HARNESS ready`, drive the TUI with newline-delimited commands:

```text
send hi
idle
screen
screenshot after-hi
quit
```

Supported commands:

- `send TEXT`: type text and press Enter
- `type TEXT` or `paste TEXT`: write text bytes without pressing Enter
- `key NAME`: press `Enter`, `Tab`, `Esc`, `Backspace`, arrows, `Home`, `End`,
  `PageUp`, `PageDown`, `Ctrl-C`, `Alt-Up`, etc.
- `idle`: wait until the submitted turn settles back to `Idle`
- `wait TEXT`: wait until visible screen text contains `TEXT`
- `screen`: print the current visible screen as text
- `screenshot NAME`: save `screens/NAME.txt`, `screens/NAME.svg`, and
  `screens/NAME.png`
- `artifacts`: print artifact paths
- `log`: print `lash.log` when `--lash-home` was set
- `quit`: send `/exit`, save artifacts, and exit
- `kill`: kill the child process, save artifacts, and exit

## Artifacts

The harness writes:

- `terminal.ansi`: complete raw PTY byte stream
- `screen.txt`, `screen.svg`, `screen.png`: final visible screen
- `screens/latest.txt/svg/png`: periodically refreshed live screen
- `screens/NAME.txt/svg/png`: explicit screenshots
- `ui-trace.json`: Lash debug UI trace
- `metadata.json`: command, mode, paths, status, and elapsed time

Use the PNG screenshots for visual inspection. Use `screen.txt` and
`ui-trace.json` to understand exact text and trace state.

## Manual Scenarios

Read `scenarios/*.md`, run the described steps, and inspect the artifacts. The
scenario markdown is intentionally not executable; judgement belongs to the
agent/operator looking at the real UI.
