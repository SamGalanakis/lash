// Helpers shared by the typed field forms.

// The `/validate` kind for an editable text slot. Expression-valued slots
// (conditions, iterables, computed values, `$expr` record args) validate as
// `expression`; assignment targets as `assignment_target`; simple `let`/loop
// bindings as `identifier`.
export function validationKind(slot) {
  switch (slot) {
    case 'target':
      return 'assignment_target';
    case 'binding':
    case 'clauseBinding':
      return 'identifier';
    default:
      return 'expression';
  }
}

// A value is "too complex" for the visual builders (multi-line or long); the
// field then stays on the raw-text editor rather than a builder widget.
export function isComplexExpression(text) {
  if (typeof text !== 'string') return false;
  return text.includes('\n') || text.length > 96;
}

// Comparison operators for the if/while condition builder, in display order.
export const COMPARISON_OPS = [
  { value: '<', label: '<' },
  { value: '<=', label: '≤' },
  { value: '==', label: '=' },
  { value: '!=', label: '≠' },
  { value: '>=', label: '≥' },
  { value: '>', label: '>' },
];

const OP_ALT = new Set(COMPARISON_OPS.map((o) => o.value));

// Parse a single top-level `lhs <op> rhs` comparison for the builder. Returns
// null when the text is not a simple comparison (the field then stays raw).
export function parseComparison(text) {
  const m = /^\s*([^<>=!]+?)\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$/.exec(text ?? '');
  if (!m) return null;
  // Reject a right side that itself contains a comparison operator (chained /
  // compound expressions belong in the raw editor).
  if (/[<>]|[=!]=/.test(m[3])) return null;
  if (!OP_ALT.has(m[2])) return null;
  return { lhs: m[1].trim(), op: m[2], rhs: m[3].trim() };
}

// Classify a value literal for the value builder. `expression` means "not a
// simple literal" — such values stay on the raw editor.
export function parseLiteral(text) {
  const t = (text ?? '').trim();
  if (t === '') return { type: 'string', value: '' };
  if (/^-?\d+(\.\d+)?$/.test(t)) return { type: 'number', value: t };
  if (t === 'true' || t === 'false') return { type: 'boolean', value: t };
  const s = /^"((?:[^"\\]|\\.)*)"$/.exec(t);
  if (s) return { type: 'string', value: s[1] };
  return { type: 'expression', value: t };
}

// Encode a builder value back into canonical Lashlang literal text.
export function encodeLiteral(type, value) {
  switch (type) {
    case 'number': {
      const n = String(value ?? '').trim();
      return n === '' ? '0' : n;
    }
    case 'boolean':
      return value === true || value === 'true' ? 'true' : 'false';
    case 'string':
      return JSON.stringify(String(value ?? ''));
    default:
      return String(value ?? '');
  }
}
