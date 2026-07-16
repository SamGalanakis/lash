import { describe, it, expect } from 'vitest';
import {
  parseLiteral,
  encodeLiteral,
  parseList,
  encodeList,
  parseComparison,
  stripOuterParens,
  isBoolLiteral,
  isSimpleReference,
  isComplexExpression,
  clauseAdded,
  clauseRemoved,
  makeClause,
} from './fields.js';

// Mirror of ExpressionField's builder-selection predicate. Kept in the test so
// the selection logic that decides raw-vs-builder is itself under coverage —
// this is the logic whose gap let the paren bug ship green.
function autoRaw(builder, value, vars = []) {
  if (isComplexExpression(value)) return true;
  const empty = (value ?? '').trim() === '';
  if (builder === 'comparison') return !(parseComparison(value) || empty || isBoolLiteral(value));
  if (builder === 'value') return parseLiteral(value).type === 'expression';
  if (builder === 'list') {
    return !(
      parseList(value) ||
      empty ||
      (isSimpleReference(value) && vars.includes((value ?? '').trim()))
    );
  }
  if (builder === 'target') return !(empty || isSimpleReference(value));
  return true;
}

describe('literal parse/encode', () => {
  it('classifies scalar literals', () => {
    expect(parseLiteral('42')).toEqual({ type: 'number', value: '42' });
    expect(parseLiteral('-3.5')).toEqual({ type: 'number', value: '-3.5' });
    expect(parseLiteral('true')).toEqual({ type: 'boolean', value: 'true' });
    expect(parseLiteral('"hi"')).toEqual({ type: 'string', value: 'hi' });
    expect(parseLiteral('')).toEqual({ type: 'string', value: '' });
  });

  it('treats non-literals as expression (stay raw)', () => {
    expect(parseLiteral('a + b').type).toBe('expression');
    expect(parseLiteral('foo(1)').type).toBe('expression');
  });

  it('encodes canonical Lashlang literals', () => {
    expect(encodeLiteral('number', '7')).toBe('7');
    expect(encodeLiteral('number', '')).toBe('0');
    expect(encodeLiteral('boolean', 'true')).toBe('true');
    expect(encodeLiteral('boolean', 'false')).toBe('false');
    expect(encodeLiteral('string', 'done')).toBe('"done"');
    expect(encodeLiteral('string', 'a"b')).toBe('"a\\"b"');
  });

  it('round-trips plain scalar literals through parse -> encode', () => {
    for (const raw of ['"hello"', '"done"', '42', '-3.5', 'true', 'false']) {
      const lit = parseLiteral(raw);
      expect(encodeLiteral(lit.type, lit.value)).toBe(raw);
    }
  });
});

describe('list parse/encode', () => {
  it('parses a flat scalar list', () => {
    expect(parseList('[1, 2, 3]')).toEqual({
      items: [
        { type: 'number', value: '1' },
        { type: 'number', value: '2' },
        { type: 'number', value: '3' },
      ],
    });
  });

  it('parses an empty list', () => {
    expect(parseList('[]')).toEqual({ items: [] });
  });

  it('parses mixed scalar types with quoted commas', () => {
    const parsed = parseList('["a, b", true, 5]');
    expect(parsed).toEqual({
      items: [
        { type: 'string', value: 'a, b' },
        { type: 'boolean', value: 'true' },
        { type: 'number', value: '5' },
      ],
    });
  });

  it('rejects non-flat / nested / expression lists', () => {
    expect(parseList('[a + b]')).toBeNull();
    expect(parseList('[[1], 2]')).toBeNull();
    expect(parseList('not a list')).toBeNull();
    expect(parseList('[1, 2')).toBeNull();
  });

  it('encodes scalar items to canonical list text', () => {
    expect(
      encodeList([
        { type: 'number', value: '1' },
        { type: 'string', value: 'x' },
      ]),
    ).toBe('[1, "x"]');
    expect(encodeList([])).toBe('[]');
  });

  it('round-trips list text through parse -> encode', () => {
    for (const raw of ['[1, 2, 3]', '["a", "b"]', '[true, false]', '[]']) {
      expect(encodeList(parseList(raw).items)).toBe(raw);
    }
  });
});

describe('comparison parse', () => {
  it('parses a simple comparison', () => {
    expect(parseComparison('count < 10')).toEqual({ lhs: 'count', op: '<', rhs: '10' });
    expect(parseComparison('name == "bob"')).toEqual({ lhs: 'name', op: '==', rhs: '"bob"' });
    expect(parseComparison('a >= b')).toEqual({ lhs: 'a', op: '>=', rhs: 'b' });
  });

  it('round-trips a comparison through parse -> rebuild', () => {
    const src = 'total != 0';
    const { lhs, op, rhs } = parseComparison(src);
    expect(`${lhs} ${op} ${rhs}`).toBe(src);
  });

  it('rejects chained / compound comparisons (stay raw)', () => {
    expect(parseComparison('a < b < c')).toBeNull();
    expect(parseComparison('plain text')).toBeNull();
  });
});

describe('parenthesized conditions (the lens emits these)', () => {
  it('strips a balanced outer paren pair', () => {
    expect(stripOuterParens('(x < 2)')).toBe('x < 2');
    expect(stripOuterParens('((a))')).toBe('a');
    expect(stripOuterParens('  (state.count < 3)  ')).toBe('state.count < 3');
  });

  it('leaves non-wrapping parens intact', () => {
    expect(stripOuterParens('(a) < (b)')).toBe('(a) < (b)');
    expect(stripOuterParens('x < 2')).toBe('x < 2');
    expect(stripOuterParens('(a) && (b)')).toBe('(a) && (b)');
  });

  it('decodes a lens-emitted `(x < 2)` to clean operands', () => {
    expect(parseComparison('(x < 2)')).toEqual({ lhs: 'x', op: '<', rhs: '2' });
  });

  it('decodes `(state.count < 3)` with a dotted lhs', () => {
    expect(parseComparison('(state.count < 3)')).toEqual({
      lhs: 'state.count',
      op: '<',
      rhs: '3',
    });
  });

  it('re-encodes bare (lens re-wraps) so a round-trip stays balanced', () => {
    const parsed = parseComparison('(x < 2)');
    const emitted = `${parsed.lhs} ${parsed.op} ${parsed.rhs}`;
    expect(emitted).toBe('x < 2'); // bare — no stray parens, balanced
    // and re-parsing the wrapped form the lens would produce is stable
    expect(parseComparison(`(${emitted})`)).toEqual(parsed);
  });

  it('does NOT mis-parse a compound expression as a comparison', () => {
    // Naive splitting would yield lhs="(x" — must reject and fall back to raw.
    expect(parseComparison('(x < 2) && ok')).toBeNull();
  });
});

describe('builder-selection (autoRaw) — locks the raw-vs-builder decision', () => {
  it('keeps a parenthesized comparison on the builder, not raw', () => {
    expect(autoRaw('comparison', '(x < 2)')).toBe(false);
    expect(autoRaw('comparison', '(state.count < 3)')).toBe(false);
  });

  it('starts a fresh comparison for empty / bare-boolean conditions', () => {
    expect(autoRaw('comparison', '')).toBe(false);
    expect(autoRaw('comparison', 'true')).toBe(false);
  });

  it('falls back to raw for compound / non-comparison conditions', () => {
    expect(autoRaw('comparison', '(x < 2) && ok')).toBe(true);
    expect(autoRaw('comparison', 'compute(x)')).toBe(true);
  });

  it('list: literal list and in-scope var use the builder, else raw', () => {
    expect(autoRaw('list', '[1, 2, 3]')).toBe(false);
    expect(autoRaw('list', 'items', ['items'])).toBe(false);
    expect(autoRaw('list', 'items', [])).toBe(true); // not in scope
    expect(autoRaw('list', 'range(0, n)')).toBe(true);
  });

  it('value: literals use the builder, expressions stay raw', () => {
    expect(autoRaw('value', '42')).toBe(false);
    expect(autoRaw('value', '"hi"')).toBe(false);
    expect(autoRaw('value', 'a + b')).toBe(true);
  });

  it('target: dotted references use the builder, index/calls stay raw', () => {
    expect(autoRaw('target', 'state.count')).toBe(false);
    expect(autoRaw('target', 'total')).toBe(false);
    expect(autoRaw('target', 'arr[0]')).toBe(true);
  });
});

describe('predicates', () => {
  it('detects bare boolean literals', () => {
    expect(isBoolLiteral('true')).toBe(true);
    expect(isBoolLiteral(' false ')).toBe(true);
    expect(isBoolLiteral('x')).toBe(false);
  });

  it('detects simple variable references', () => {
    expect(isSimpleReference('items')).toBe(true);
    expect(isSimpleReference('state.list')).toBe(true);
    expect(isSimpleReference('a + b')).toBe(false);
    expect(isSimpleReference('[1, 2]')).toBe(false);
  });

  it('flags multi-line / long values as complex', () => {
    expect(isComplexExpression('a\nb')).toBe(true);
    expect(isComplexExpression('x'.repeat(200))).toBe(true);
    expect(isComplexExpression('a + b')).toBe(false);
  });
});

describe('comprehension clause edits', () => {
  it('seeds a fresh clause by kind', () => {
    expect(makeClause('for')).toEqual({ kind: 'for', binding: 'x', iterable: '[1, 2, 3]' });
    expect(makeClause('if')).toEqual({ kind: 'if', condition: 'true' });
  });

  it('adds a clause immutably', () => {
    const base = [{ kind: 'for', binding: 'a', iterable: 'xs' }];
    const next = clauseAdded(base, 'if');
    expect(next).toHaveLength(2);
    expect(next[1]).toEqual({ kind: 'if', condition: 'true' });
    expect(base).toHaveLength(1); // original untouched
  });

  it('adds a clause to an empty/undefined list', () => {
    expect(clauseAdded(undefined, 'for')).toHaveLength(1);
  });

  it('removes a clause by index immutably', () => {
    const base = [
      { kind: 'for', binding: 'a', iterable: 'xs' },
      { kind: 'if', condition: 'a > 0' },
    ];
    const next = clauseRemoved(base, 0);
    expect(next).toEqual([{ kind: 'if', condition: 'a > 0' }]);
    expect(base).toHaveLength(2);
  });
});
