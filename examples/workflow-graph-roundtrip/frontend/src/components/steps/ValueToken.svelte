<script>
  import { parseType } from '../../lib/facets.js';

  // A read-mode rendering of one value on a step row (item 3). A value produced
  // by an EARLIER step is an in-scope variable → a colored TOKEN chip carrying
  // its name + a plain-language kind. A bare literal is plain STATIC text; a
  // compound value is an EXPRESSION pill. The descriptor comes from
  // steps.js#describeValue — this component only paints it.
  let { desc, placeholder = 'not set' } = $props();

  // A soft, non-technical word for a facet type, used as the token's kind tag.
  function kindWord(type) {
    const t = parseType(type);
    switch (t.kind) {
      case 'int':
      case 'float':
        return 'number';
      case 'str':
        return 'text';
      case 'bool':
        return 'yes/no';
      case 'list':
        return 'list';
      case 'enum':
        return 'choice';
      case 'dict':
      case 'object':
        return 'record';
      case 'any':
        return 'value';
      default:
        return t.name ? 'value' : 'value';
    }
  }

  // Hue family per type so tokens read as a legend at a glance.
  function tone(type) {
    const t = parseType(type);
    if (t.kind === 'int' || t.kind === 'float') return 'num';
    if (t.kind === 'str' || t.kind === 'enum') return 'text';
    if (t.kind === 'bool') return 'bool';
    if (t.kind === 'list') return 'list';
    return 'any';
  }
</script>

{#if desc.kind === 'token'}
  <span class="tok tok--{tone(desc.varType)}" title="a value from an earlier step ({desc.varType})">
    <span class="tok-dot"></span>
    <span class="tok-name">{desc.name}</span>
    <span class="tok-kind">{kindWord(desc.varType)}</span>
  </span>
{:else if desc.kind === 'expression'}
  <code class="expr" title="a calculated value">{desc.text}</code>
{:else if desc.display === ''}
  <span class="empty">{placeholder}</span>
{:else}
  <span class="lit lit--{desc.literalType}">{desc.display}</span>
{/if}

<style>
  .tok {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-ui);
    font-size: 12.5px;
    font-weight: 600;
    padding: 1px 9px 1px 7px;
    border-radius: 999px;
    border: 1px solid;
    line-height: 1.5;
    vertical-align: baseline;
  }
  .tok-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
  }
  .tok-name {
    letter-spacing: 0.01em;
  }
  .tok-kind {
    font-family: var(--font-mono);
    font-size: 8.5px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
  }
  .tok--num {
    color: var(--cyan);
    border-color: color-mix(in srgb, var(--cyan) 45%, transparent);
    background: color-mix(in srgb, var(--cyan) 12%, transparent);
  }
  .tok--text {
    color: #5fd08a;
    border-color: color-mix(in srgb, #5fd08a 45%, transparent);
    background: color-mix(in srgb, #5fd08a 12%, transparent);
  }
  .tok--bool {
    color: var(--violet);
    border-color: color-mix(in srgb, var(--violet) 45%, transparent);
    background: color-mix(in srgb, var(--violet) 12%, transparent);
  }
  .tok--list {
    color: #6ab0ff;
    border-color: color-mix(in srgb, #6ab0ff 45%, transparent);
    background: color-mix(in srgb, #6ab0ff 12%, transparent);
  }
  .tok--any {
    color: var(--text-dim);
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--text-dim) 10%, transparent);
  }

  .lit {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }
  .lit--string::before,
  .lit--string::after {
    content: '\201C';
    color: var(--text-faint);
    font-weight: 400;
  }
  .lit--string::after {
    content: '\201D';
  }
  .lit--number {
    font-variant-numeric: tabular-nums;
    color: var(--cyan);
  }
  .lit--boolean {
    color: var(--violet);
  }
  .expr {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--amber);
    background: color-mix(in srgb, var(--amber) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--amber) 28%, transparent);
    border-radius: 6px;
    padding: 1px 6px;
  }
  .empty {
    font-style: italic;
    font-size: 12.5px;
    color: var(--text-faint);
  }
</style>
