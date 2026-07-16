import { describe, it, expect } from 'vitest';
import {
  parseLiteral,
  encodeLiteral,
  parseList,
  encodeList,
  parseComparison,
  isBoolLiteral,
  isSimpleReference,
  isComplexExpression,
  clauseAdded,
  clauseRemoved,
  makeClause,
} from './fields.js';

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
