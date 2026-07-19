<script>
  import ExpressionField from '../ExpressionField.svelte';
  import ValueToken from './ValueToken.svelte';
  import { describeValue } from '../../lib/steps.js';

  // One tweakable value on a step row (item 5). It reads as a calm token/chip
  // (ValueToken) until tapped; tapping opens the shared, facet-driven
  // ExpressionField (item 3) so the guardrails — type-compatible variable
  // picker, enum dropdown, scalar builder, raw escape hatch — are reused rather
  // than reinvented. Composite values keep the expression field as the
  // pill-in-text fallback. Blur commits and returns to the calm view.
  let {
    value = '',
    availableVars = [],
    expectedType = null,
    builder = 'value',
    placeholder = 'not set',
    onInput,
    onCommit,
  } = $props();

  let editing = $state(false);
  const desc = $derived(describeValue(value, availableVars));

  function open() {
    editing = true;
  }
  function done() {
    editing = false;
    onCommit?.();
  }

  // Focus the first control inside the editor as soon as it appears, so a tap
  // lands straight in the field rather than needing a second click.
  function autofocus(node) {
    const el = node.querySelector('input, textarea, select');
    if (el) queueMicrotask(() => el.focus());
  }
</script>

{#if editing}
  <span class="ev-edit" use:autofocus>
    <ExpressionField
      {value}
      kind="expression"
      {builder}
      {availableVars}
      {expectedType}
      {placeholder}
      onInput={(t) => onInput?.(t)}
      onCommit={done}
    />
  </span>
{:else}
  <button
    class="ev-read"
    type="button"
    title="Tap to change"
    onclick={open}
  >
    <ValueToken {desc} {placeholder} />
    <span class="ev-pen" aria-hidden="true">✎</span>
  </button>
{/if}

<style>
  .ev-read {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 2px 4px;
    cursor: pointer;
    transition:
      background 0.12s ease,
      border-color 0.12s ease;
  }
  .ev-read:hover {
    background: color-mix(in srgb, var(--cyan) 8%, transparent);
    border-color: color-mix(in srgb, var(--cyan) 28%, var(--line));
  }
  .ev-pen {
    font-size: 10px;
    color: var(--text-faint);
    opacity: 0;
    transition: opacity 0.12s ease;
  }
  .ev-read:hover .ev-pen {
    opacity: 1;
  }
  .ev-edit {
    display: inline-flex;
    min-width: 180px;
    max-width: 420px;
    flex: 1;
  }
</style>
