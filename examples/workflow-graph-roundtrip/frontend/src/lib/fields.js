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

// A value is "too complex" to edit through the Simplified single-line builder
// when it spans multiple lines or is long — Simplified then shows it read-only
// with a "switch to Power" hint instead of silently hiding it.
export function isComplexExpression(text) {
  if (typeof text !== 'string') return false;
  return text.includes('\n') || text.length > 96;
}
