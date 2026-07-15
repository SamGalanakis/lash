// Visual identity for each WorkflowDocument node kind. One signature hue per
// kind so the canvas reads as a legend at a glance. `glyph` is a short mark
// shown in the node's kind badge.
export const NODE_KINDS = {
  process: { label: 'process', hue: 188, glyph: '⬡', accent: '#38e0d0' },
  call: { label: 'call', hue: 44, glyph: '⚡', accent: '#f5c451' },
  effect: { label: 'effect', hue: 268, glyph: '◆', accent: '#b18cff' },
  data: { label: 'data', hue: 152, glyph: '≋', accent: '#5fd08a' },
  container: { label: 'branch', hue: 210, glyph: '⋔', accent: '#6ab0ff' },
  opaque: { label: 'opaque', hue: 8, glyph: '{}', accent: '#ff8f6b' },
  terminal: { label: 'terminal', hue: 340, glyph: '◉', accent: '#ff6b9d' },
};

export function kindMeta(kind) {
  return NODE_KINDS[kind] ?? { label: kind, hue: 220, glyph: '•', accent: '#8aa0c0' };
}

// Effect verbs that render as a "waiting"-capable node (sleeps / signal waits).
export const WAITING_EFFECTS = new Set(['sleep', 'wait_signal', 'await_join']);

// Human labels for the toy display tool operations.
export const OP_LABELS = {
  show_message: 'show message',
  set_status: 'set status',
  add_item: 'add item',
  set_light: 'set light',
  set_progress: 'set progress',
  highlight: 'highlight',
};
