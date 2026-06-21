# Theme QA

Purpose: verify that both bundled CLI themes are readable in a real terminal
capture, and that the System theme respects the user's terminal colors instead
of painting large gray panels over the screen.

Recommended harness size: `--rows 40 --cols 120`.

This scenario changes the persisted `theme` setting. Use a disposable copy of
`LASH_HOME` if you do not want to alter your normal CLI config.

## Start

```sh
workspace=$(mktemp -d /tmp/lash-theme-workspace-XXXXXX)
output=$(mktemp -d /tmp/lash-theme-output-XXXXXX)

cargo run -p lash-debug-cli-harness -- \
  --execution-mode rlm \
  --lash-home /home/sam/.lash \
  --working-dir "$workspace" \
  --output-dir "$output" \
  --rows 40 \
  --cols 120 \
  --no-build
```

## Lash Theme

```text
key Ctrl-P
wait Commands
type theme lash
key Enter
wait Theme set to Lash
screenshot theme-lash-idle
send /info
wait Runtime
screenshot theme-lash-info
key Ctrl-C
```

## System Theme

```text
key Ctrl-P
wait Commands
type theme system
key Enter
wait Theme set to System
screenshot theme-system-idle
send /info
wait Runtime
screenshot theme-system-info
key Ctrl-C
```

## Selection And Toast

```text
paste selectable theme text
wait selectable theme text
screenshot theme-system-draft
```

Drag across part of the draft input. At the recommended size the input text row
is usually row 37; run `screen` first if your terminal layout differs.

```text
mouse select 3 37 14 37
wait Copied to clipboard
screenshot theme-system-selection-toast
key Ctrl-C
```

## Optional Provider Turn

Run this section only when the configured provider is expected to answer
quickly.

```text
send Reply with exactly: theme smoke ok
idle
wait theme smoke ok
screenshot theme-system-response
```

## Expected Behavior

- `theme-lash-*` screenshots keep Lash's custom dark surfaces and orange brand
  accents.
- `theme-system-*` screenshots use the terminal's default background for the
  main canvas, input area, status area, toast body, and modal surfaces. No broad
  gray bars should appear unless they come from the terminal theme itself.
- Text remains readable in both themes: primary text, muted labels, borders,
  selection highlight, and error/status colors should all be distinguishable.
- The selection screenshot shows a visible selected range and a readable
  `Copied to clipboard` toast.
- `/info` can be dismissed with Ctrl-C after each capture.

Report failures with the output directory and the exact screenshot names.
