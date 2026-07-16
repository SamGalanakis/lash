<script>
  import { getContext, tick } from 'svelte';
  import { validateFragment } from '../lib/api.js';
  import { isComplexExpression } from '../lib/fields.js';

  // A single editable Lashlang-text slot, rendered by mode:
  //  - Simplified: a friendly single-line builder input + a variable picker fed
  //    by the node's in-scope vars, with inline validation. If the value is too
  //    complex to represent simply it is shown read-only with a Power hint.
  //  - Power: a raw multi-line code input (plus the same picker + validation).
  let {
    value = '',
    kind = 'expression',
    availableVars = [],
    placeholder = 'expression…',
    onInput,
    onCommit,
    monospace = true,
  } = $props();

  const mode = getContext('mode');

  let el = $state(null);
  let pickerOpen = $state(false);
  let error = $state(null);
  let checking = $state(false);
  let timer = null;
  let seq = 0;

  const complex = $derived(isComplexExpression(value));
  const readOnly = $derived(mode.simplified && complex);
  const vars = $derived(Array.isArray(availableVars) ? availableVars : []);

  async function runValidate(text) {
    const trimmed = (text ?? '').trim();
    if (!trimmed) {
      error = null;
      checking = false;
      return;
    }
    const mine = (seq += 1);
    checking = true;
    const res = await validateFragment(kind, trimmed);
    if (mine !== seq) return; // superseded by a newer keystroke
    checking = false;
    error = res.ok ? null : (res.error?.message ?? 'invalid');
  }

  function scheduleValidate(text) {
    clearTimeout(timer);
    timer = setTimeout(() => runValidate(text), 400);
  }

  function handleInput(event) {
    const text = event.currentTarget.value;
    onInput?.(text);
    scheduleValidate(text);
  }

  function handleBlur() {
    clearTimeout(timer);
    runValidate(value);
    onCommit?.();
  }

  async function insertVar(name) {
    pickerOpen = false;
    const node = el;
    const start = node?.selectionStart ?? value.length;
    const end = node?.selectionEnd ?? value.length;
    const next = value.slice(0, start) + name + value.slice(end);
    onInput?.(next);
    await tick();
    if (node) {
      const caret = start + name.length;
      node.focus();
      try {
        node.setSelectionRange(caret, caret);
      } catch {
        /* setSelectionRange unsupported — caret position is best-effort */
      }
    }
    scheduleValidate(next);
  }
</script>

<div class="xf" class:has-error={!!error} class:is-ro={readOnly}>
  {#if readOnly}
    <code class="xf-ro">{value}</code>
    <span class="xf-ro-hint" title="This value is too detailed for the simple editor"
      >switch to Power to edit</span
    >
  {:else}
    <div class="xf-row">
      {#if mode.power}
        <textarea
          bind:this={el}
          class="xf-input nodrag"
          class:mono={monospace}
          rows="1"
          {value}
          {placeholder}
          spellcheck="false"
          oninput={handleInput}
          onblur={handleBlur}
        ></textarea>
      {:else}
        <input
          bind:this={el}
          class="xf-input nodrag"
          class:mono={monospace}
          type="text"
          {value}
          {placeholder}
          spellcheck="false"
          oninput={handleInput}
          onblur={handleBlur}
        />
      {/if}
      {#if vars.length}
        <div class="xf-picker">
          <button
            class="xf-pick-btn"
            class:is-open={pickerOpen}
            title="Insert an in-scope variable"
            aria-label="Insert variable"
            onpointerdown={(e) => e.stopPropagation()}
            onclick={(e) => {
              e.stopPropagation();
              pickerOpen = !pickerOpen;
            }}>{'{x}'}</button
          >
          {#if pickerOpen}
            <div class="xf-pick-menu">
              <div class="xf-pick-label">in scope</div>
              {#each vars as v (v)}
                <button
                  class="xf-pick-item"
                  onpointerdown={(e) => e.stopPropagation()}
                  onclick={(e) => {
                    e.stopPropagation();
                    insertVar(v);
                  }}>{v}</button
                >
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
    {#if error}
      <div class="xf-err">{error}</div>
    {/if}
  {/if}
</div>

<style>
  .xf {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
    flex: 1;
  }
  .xf-row {
    display: flex;
    align-items: stretch;
    gap: 5px;
    min-width: 0;
  }
  .xf-input {
    flex: 1;
    min-width: 0;
    width: 100%;
    background: #0a0d13;
    border: 1px solid var(--line);
    border-radius: 7px;
    color: #eaf2ff;
    font-family: var(--font-ui);
    font-size: 11.5px;
    line-height: 1.5;
    padding: 5px 8px;
    transition:
      border-color 0.15s ease,
      box-shadow 0.15s ease;
  }
  .xf-input.mono {
    font-family: var(--font-mono);
  }
  textarea.xf-input {
    resize: vertical;
    tab-size: 2;
  }
  .xf-input:focus {
    outline: none;
    border-color: var(--accent, var(--cyan));
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent, var(--cyan)) 14%, transparent);
  }
  .xf.has-error .xf-input {
    border-color: var(--rose);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--rose) 16%, transparent);
  }
  .xf-err {
    font-family: var(--font-mono);
    font-size: 9px;
    line-height: 1.35;
    color: var(--rose);
    letter-spacing: 0.01em;
  }

  .xf-picker {
    position: relative;
    flex-shrink: 0;
  }
  .xf-pick-btn {
    height: 100%;
    background: color-mix(in srgb, var(--cyan) 10%, var(--ink-2));
    border: 1px solid color-mix(in srgb, var(--cyan) 30%, var(--line));
    border-radius: 7px;
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 0 7px;
    cursor: pointer;
    transition:
      background 0.15s ease,
      border-color 0.15s ease;
  }
  .xf-pick-btn:hover,
  .xf-pick-btn.is-open {
    background: color-mix(in srgb, var(--cyan) 20%, var(--ink-2));
    border-color: var(--cyan);
  }
  .xf-pick-menu {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    z-index: 30;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 120px;
    max-height: 190px;
    overflow-y: auto;
    padding: 5px;
    background: var(--ink-2);
    border: 1px solid var(--line-strong);
    border-radius: 9px;
    box-shadow: 0 14px 30px -12px rgba(0, 0, 0, 0.7);
  }
  .xf-pick-label {
    font-family: var(--font-mono);
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    padding: 2px 6px 4px;
  }
  .xf-pick-item {
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 4px 7px;
    cursor: pointer;
    transition:
      background 0.12s ease,
      border-color 0.12s ease;
  }
  .xf-pick-item:hover {
    background: color-mix(in srgb, var(--cyan) 14%, transparent);
    border-color: color-mix(in srgb, var(--cyan) 40%, transparent);
  }

  .xf-ro {
    display: block;
    background: #0a0d13;
    border: 1px dashed var(--line-strong);
    border-radius: 7px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.5;
    padding: 5px 8px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 96px;
    overflow: auto;
  }
  .xf-ro-hint {
    font-family: var(--font-mono);
    font-size: 8.5px;
    color: var(--amber);
    letter-spacing: 0.03em;
  }
</style>
