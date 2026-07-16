// Visual identity for each WorkflowDocument node kind. One signature hue per
// kind so the canvas reads as a legend at a glance. `glyph` is a short mark
// shown in the node's kind badge.
export const NODE_KINDS = {
  process: { label: 'process', hue: 188, glyph: '⬡', accent: '#38e0d0' },
  call: { label: 'call', hue: 44, glyph: '⚡', accent: '#f5c451' },
  effect: { label: 'effect', hue: 268, glyph: '◆', accent: '#b18cff' },
  data: { label: 'data', hue: 152, glyph: '≋', accent: '#5fd08a' },
  computation: { label: 'computation', hue: 72, glyph: 'ƒ', accent: '#cdd94a' },
  state_update: { label: 'state', hue: 312, glyph: '≔', accent: '#e77ec8' },
  container: { label: 'branch / loop', hue: 210, glyph: '⋔', accent: '#6ab0ff' },
  opaque: { label: 'opaque', hue: 8, glyph: '{}', accent: '#ff8f6b' },
  terminal: { label: 'terminal', hue: 340, glyph: '◉', accent: '#ff6b9d' },
};

export function kindMeta(kind) {
  return NODE_KINDS[kind] ?? { label: kind, hue: 220, glyph: '•', accent: '#8aa0c0' };
}

// A container node (data.kind === 'container') carries an explicit `subkind`
// once the backend projects/round-trips it (and the host sets it on freshly
// added containers). When present it is authoritative; otherwise we recover it
// from the child slots + derived title (older projections, defensive path).
export function containerSubkind(node) {
  const explicit = node?.data?.subkind;
  if (explicit) return explicit;
  const slots = (node?.data?.children ?? []).map((g) => g.slot);
  if (slots.includes('then') || slots.includes('else')) return 'if';
  if (slots.includes('element')) return 'comprehension';
  const title = (node?.data?.title ?? '').trim();
  if (/^for\b/.test(title)) return 'for';
  if (/^while\b/.test(title)) return 'while';
  return 'loop';
}

// Badge glyph + label per container sub-kind (all share the container accent so
// while renders exactly like the existing for / if containers).
export const CONTAINER_SUBKINDS = {
  if: { label: 'if', glyph: '⋔' },
  for: { label: 'for', glyph: '↻' },
  while: { label: 'while', glyph: '⟳' },
  comprehension: { label: 'comprehension', glyph: '⟦⟧' },
  loop: { label: 'loop', glyph: '↻' },
};

// Effect verbs that render as a "waiting"-capable node (sleeps / signal waits).
export const WAITING_EFFECTS = new Set(['sleep', 'wait_signal', 'await_join']);

// Kinds the "+ Add node" palette can insert into a scope, in menu order. Leaf
// kinds map to a NODE_KINDS entry; the four container kinds (if/while/for/
// comprehension) all render as `container` nodes but carry a distinct subkind.
export const ADDABLE_KINDS = [
  'call',
  'effect',
  'data',
  'computation',
  'state_update',
  'terminal',
  'if',
  'while',
  'for',
  'comprehension',
];

// Menu label + glyph + accent for an addable kind (containers borrow the
// container accent + their sub-kind glyph so the palette reads like the legend).
export function addableMeta(kind) {
  const sub = CONTAINER_SUBKINDS[kind];
  if (sub) {
    return { label: sub.label, glyph: sub.glyph, accent: NODE_KINDS.container.accent };
  }
  const meta = NODE_KINDS[kind] ?? kindMeta(kind);
  return { label: meta.label, glyph: meta.glyph, accent: meta.accent };
}

// Human labels for the toy display tool operations.
export const OP_LABELS = {
  show_message: 'show message',
  set_status: 'set status',
  add_item: 'add item',
  set_light: 'set light',
  set_progress: 'set progress',
  highlight: 'highlight',
};
