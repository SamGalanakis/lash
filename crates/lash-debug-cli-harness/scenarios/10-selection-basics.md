# Selection Basics

Purpose: verify that Lash's full TUI treats selected text as the active
interaction, matching the basic opencode-style contract: drag selection wins
over click activation, Escape clears selection, and screenshots capture the
selected and cleared states.

Recommended harness size: `--rows 24 --cols 80`.

```text
paste abc123 selectable text
wait abc123
screenshot before-selection
```

Drag across part of the input text using zero-based coordinates. At the
recommended size the input row is usually row 21; use `screen` first if the
prompt layout differs in your environment.

```text
mouse select 3 21 16 21
screenshot input-selection
key Esc
screenshot selection-cleared
quit
```

Expected behavior:

- The live terminal shows a contiguous highlighted input selection after
  `mouse select`.
- `screens/input-selection.*` and `screens/selection-cleared.*` are written for
  the selected and cleared states. The harness raster screenshots preserve text
  layout; style-level highlight assertions live in the render tests.
- The Escape key clears the selection without submitting or dismissing anything
  else visible on the screen.
