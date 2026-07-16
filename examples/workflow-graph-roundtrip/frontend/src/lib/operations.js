// The operation catalog is the single data home for the "+ Add node" palette:
// what a non-expert can add, its friendly label, and the typed default fields
// that seed a fully-formed node. At runtime the catalog comes from the backend
// (`GET /operations`); FALLBACK_OPERATIONS below is the same-shape catalog used
// only when an older backend does not serve that endpoint, so the palette
// always shows real labels and never falls back to raw kind strings.

import { NODE_KINDS, CONTAINER_SUBKINDS, kindMeta } from './nodeKinds.js';

// Same shape as a `GET /operations` entry:
//   { id, label, nodeKind, subkind?, operation?, effect?, terminalKind?,
//     fields: [{ name, type, default }] }
// `type` is string | number | boolean | expression | identifier |
// assignment_target. Defaults mirror the backend's own catalog so an add
// round-trips identically whether the catalog was fetched or fell back.
export const FALLBACK_OPERATIONS = [
  {
    id: 'display.show_message',
    label: 'Show message',
    nodeKind: 'call',
    operation: 'show_message',
    fields: [{ name: 'text', type: 'string', default: '' }],
  },
  {
    id: 'display.set_status',
    label: 'Set status',
    nodeKind: 'call',
    operation: 'set_status',
    fields: [
      { name: 'key', type: 'string', default: '' },
      { name: 'value', type: 'string', default: '' },
    ],
  },
  {
    id: 'display.add_item',
    label: 'Add item',
    nodeKind: 'call',
    operation: 'add_item',
    fields: [
      { name: 'list', type: 'string', default: '' },
      { name: 'item', type: 'string', default: '' },
    ],
  },
  {
    id: 'display.set_light',
    label: 'Set light',
    nodeKind: 'call',
    operation: 'set_light',
    fields: [
      { name: 'name', type: 'string', default: '' },
      { name: 'state', type: 'string', default: '' },
    ],
  },
  {
    id: 'display.set_progress',
    label: 'Set progress',
    nodeKind: 'call',
    operation: 'set_progress',
    fields: [{ name: 'pct', type: 'number', default: 0 }],
  },
  {
    id: 'display.highlight',
    label: 'Highlight',
    nodeKind: 'call',
    operation: 'highlight',
    fields: [{ name: 'target', type: 'string', default: '' }],
  },
  {
    id: 'effect.sleep',
    label: 'Sleep',
    nodeKind: 'effect',
    effect: 'sleep',
    fields: [{ name: 'duration', type: 'expression', default: '"1s"' }],
  },
  {
    id: 'effect.wait_signal',
    label: 'Wait for signal',
    nodeKind: 'effect',
    effect: 'wait_signal',
    fields: [{ name: 'signal', type: 'string', default: 'continue' }],
  },
  {
    id: 'control.if',
    label: 'If / branch',
    nodeKind: 'container',
    subkind: 'if',
    fields: [{ name: 'condition', type: 'expression', default: 'true' }],
  },
  {
    id: 'control.while',
    label: 'While loop',
    nodeKind: 'container',
    subkind: 'while',
    fields: [{ name: 'condition', type: 'expression', default: 'false' }],
  },
  {
    id: 'control.for',
    label: 'For each',
    nodeKind: 'container',
    subkind: 'for',
    fields: [
      { name: 'binding', type: 'identifier', default: 'item' },
      { name: 'iterable', type: 'expression', default: '[1, 2, 3]' },
    ],
  },
  {
    id: 'control.comprehension',
    label: 'Comprehension',
    nodeKind: 'container',
    subkind: 'comprehension',
    fields: [{ name: 'binding', type: 'identifier', default: 'items' }],
  },
  {
    id: 'stmt.assign',
    label: 'Set variable',
    nodeKind: 'state_update',
    fields: [
      { name: 'target', type: 'assignment_target', default: 'state.count' },
      { name: 'expression', type: 'expression', default: '0' },
    ],
  },
  {
    id: 'stmt.let',
    label: 'Define value',
    nodeKind: 'data',
    fields: [
      { name: 'binding', type: 'identifier', default: 'value' },
      { name: 'expression', type: 'expression', default: '0' },
    ],
  },
  {
    id: 'stmt.compute',
    label: 'Compute',
    nodeKind: 'computation',
    fields: [{ name: 'expression', type: 'expression', default: '1 + 1' }],
  },
  {
    id: 'stmt.finish',
    label: 'Finish',
    nodeKind: 'terminal',
    terminalKind: 'finish',
    fields: [{ name: 'expression', type: 'expression', default: '0' }],
  },
  // Power-only: an escape hatch that inserts a verbatim Lashlang statement.
  {
    id: 'stmt.opaque',
    label: 'Raw statement',
    nodeKind: 'opaque',
    powerOnly: true,
    fields: [{ name: 'source', type: 'string', default: 'display.show_message({ text: "raw" })' }],
  },
];

// Palette groups, in menu order. Each catalog entry lands in the first group
// whose `kinds` include its nodeKind; unmatched kinds fall into "Other".
const GROUPS = [
  { id: 'actions', label: 'Actions', kinds: ['call', 'effect'] },
  { id: 'control', label: 'Control flow', kinds: ['container', 'terminal'] },
  { id: 'data', label: 'Data', kinds: ['data', 'computation', 'state_update'] },
  { id: 'advanced', label: 'Advanced', kinds: ['opaque'] },
];

// Group a catalog into `[{ id, label, items }]`, dropping empty groups.
// Simplified mode hides power-only operations (raw statements / opaque).
export function groupOperations(catalog, { includePower = true } = {}) {
  const entries = (catalog && catalog.length ? catalog : FALLBACK_OPERATIONS).filter(
    (op) => includePower || !op.powerOnly,
  );
  const groups = GROUPS.map((g) => ({
    id: g.id,
    label: g.label,
    items: entries.filter((op) => g.kinds.includes(op.nodeKind)),
  }));
  const claimed = new Set(GROUPS.flatMap((g) => g.kinds));
  const other = entries.filter((op) => !claimed.has(op.nodeKind));
  if (other.length) groups.push({ id: 'other', label: 'Other', items: other });
  return groups.filter((g) => g.items.length);
}

// Menu glyph + accent for a catalog entry (containers borrow the container
// accent + their sub-kind glyph, so the palette reads like the canvas legend).
export function operationMeta(op) {
  if (op.nodeKind === 'container' && op.subkind) {
    const sub = CONTAINER_SUBKINDS[op.subkind] ?? CONTAINER_SUBKINDS.loop;
    return { glyph: sub.glyph, accent: NODE_KINDS.container.accent };
  }
  const meta = NODE_KINDS[op.nodeKind] ?? kindMeta(op.nodeKind);
  return { glyph: meta.glyph, accent: meta.accent };
}
