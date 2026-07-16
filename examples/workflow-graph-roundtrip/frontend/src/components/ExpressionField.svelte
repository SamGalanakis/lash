<script>
  import { getContext, tick } from 'svelte';
  import { validateFragment } from '../lib/api.js';
  import {
    isComplexExpression,
    COMPARISON_OPS,
    parseComparison,
    parseLiteral,
    encodeLiteral,
  } from '../lib/fields.js';

  // A single editable Lashlang-text slot, rendered by mode:
  //  - Power: a raw multi-line code input (plus variable picker + validation).
  //  - Simplified: a lightweight visual builder so a non-expert can express the
  //    common cases without typing syntax — `builder="value"` gives a
  //    number/text/true·false literal picker, `builder="comparison"` gives a
  //    [variable ▾][op ▾][value] comparison builder. Every field keeps a "raw"
  //    escape toggle for anything the builder cannot express.
  let {
    value = '',
    kind = 'expression',
    builder = null,
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
  let timer = null;
  let seq = 0;

  // Per-field override for which editor shows: null = follow the automatic pick.
  let userRaw = $state(null);

  const hasBuilder = $derived(!!builder && !mode.power);
  const cmp = $derived(builder === 'comparison' ? parseComparison(value) : null);
  const lit = $derived(builder === 'value' ? parseLiteral(value) : null);

  // Whether the raw editor (rather than a builder) should show by default.
  const autoRaw = $derived.by(() => {
    if (!hasBuilder) return true;
    if (isComplexExpression(value)) return true;
    if (builder === 'comparison') return !(cmp || (value ?? '').trim() === '');
    if (builder === 'value') return lit?.type === 'expression';
    return true;
  });
  const showRaw = $derived(mode.power || (userRaw ?? autoRaw));
  const vars = $derived(Array.isArray(availableVars) ? availableVars : []);

  // --- inline validation -----------------------------------------------------
  async function runValidate(text) {
    const trimmed = (text ?? '').trim();
    if (!trimmed) {
      error = null;
      return;
    }
    const mine = (seq += 1);
    const res = await validateFragment(kind, trimmed);
    if (mine !== seq) return;
    error = res.ok ? null : (res.error?.message ?? 'invalid');
  }
  function scheduleValidate(text) {
    clearTimeout(timer);
    timer = setTimeout(() => runValidate(text), 400);
  }
  function emit(text) {
    onInput?.(text);
    scheduleValidate(text);
  }

  // --- raw editor ------------------------------------------------------------
  function handleRawInput(event) {
    emit(event.currentTarget.value);
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
    emit(next);
    await tick();
    if (node) {
      const caret = start + name.length;
      node.focus();
      try {
        node.setSelectionRange(caret, caret);
      } catch {
        /* best-effort caret restore */
      }
    }
  }

  // --- comparison builder ----------------------------------------------------
  let lhs = $state('');
  let op = $state('<');
  let rhs = $state('');
  let cmpEditing = $state(false);
  $effect(() => {
    // Resync parts from the canonical value unless the user is mid-edit.
    if (cmpEditing || builder !== 'comparison') return;
    const parsed = parseComparison(value);
    if (parsed) {
      lhs = parsed.lhs;
      op = parsed.op;
      rhs = parsed.rhs;
    } else if ((value ?? '').trim() === '') {
      lhs = '';
      rhs = '';
    }
  });
  function emitCmp() {
    emit(lhs === '' && rhs === '' ? '' : `${lhs} ${op} ${rhs}`.trim());
  }

  // --- value builder ---------------------------------------------------------
  let litType = $state('string');
  let litVal = $state('');
  let litEditing = $state(false);
  $effect(() => {
    if (litEditing || builder !== 'value') return;
    const parsed = parseLiteral(value);
    if (parsed.type !== 'expression') {
      litType = parsed.type;
      litVal = parsed.value;
    }
  });
  function emitLit() {
    emit(encodeLiteral(litType, litVal));
  }
  function onLitType(next) {
    litType = next;
    if (next === 'boolean' && litVal !== 'true' && litVal !== 'false') litVal = 'true';
    if (next === 'number' && !/^-?\d*\.?\d*$/.test(litVal)) litVal = '';
    emitLit();
  }
</script>

<div class="xf" class:has-error={!!error}>
  {#if showRaw}
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
          oninput={handleRawInput}
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
          oninput={handleRawInput}
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
      {#if hasBuilder}
        <button
          class="xf-toggle"
          title="Switch to the visual builder"
          aria-label="Use visual builder"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            userRaw = false;
          }}>◧</button
        >
      {/if}
    </div>
  {:else if builder === 'comparison'}
    <div class="xf-row xf-cmp">
      {#if vars.length}
        <select
          class="xf-cmp-lhs"
          value={lhs}
          onpointerdown={(e) => e.stopPropagation()}
          onfocus={() => (cmpEditing = true)}
          onchange={(e) => {
            lhs = e.currentTarget.value;
            emitCmp();
          }}
          onblur={() => {
            cmpEditing = false;
            onCommit?.();
          }}
        >
          {#if !vars.includes(lhs)}<option value={lhs}>{lhs || '— variable —'}</option>{/if}
          {#each vars as v (v)}<option value={v}>{v}</option>{/each}
        </select>
      {:else}
        <input
          class="xf-cmp-lhs xf-cmp-input"
          value={lhs}
          placeholder="variable"
          spellcheck="false"
          onpointerdown={(e) => e.stopPropagation()}
          onfocus={() => (cmpEditing = true)}
          oninput={(e) => {
            lhs = e.currentTarget.value;
            emitCmp();
          }}
          onblur={() => {
            cmpEditing = false;
            onCommit?.();
          }}
        />
      {/if}
      <select
        class="xf-cmp-op"
        value={op}
        onpointerdown={(e) => e.stopPropagation()}
        onfocus={() => (cmpEditing = true)}
        onchange={(e) => {
          op = e.currentTarget.value;
          emitCmp();
        }}
        onblur={() => {
          cmpEditing = false;
          onCommit?.();
        }}
      >
        {#each COMPARISON_OPS as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
      </select>
      <input
        class="xf-cmp-rhs xf-cmp-input"
        value={rhs}
        placeholder="value"
        spellcheck="false"
        onpointerdown={(e) => e.stopPropagation()}
        onfocus={() => (cmpEditing = true)}
        oninput={(e) => {
          rhs = e.currentTarget.value;
          emitCmp();
        }}
        onblur={() => {
          cmpEditing = false;
          onCommit?.();
        }}
      />
      <button
        class="xf-toggle"
        title="Edit as raw expression"
        aria-label="Edit as raw expression"
        onpointerdown={(e) => e.stopPropagation()}
        onclick={(e) => {
          e.stopPropagation();
          userRaw = true;
        }}>{'</>'}</button
      >
    </div>
  {:else}
    <div class="xf-row xf-val">
      <select
        class="xf-val-type"
        value={litType}
        onpointerdown={(e) => e.stopPropagation()}
        onchange={(e) => onLitType(e.currentTarget.value)}
      >
        <option value="number">number</option>
        <option value="string">text</option>
        <option value="boolean">true/false</option>
      </select>
      {#if litType === 'boolean'}
        <select
          class="xf-val-input"
          value={litVal}
          onpointerdown={(e) => e.stopPropagation()}
          onchange={(e) => {
            litVal = e.currentTarget.value;
            emitLit();
          }}
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      {:else}
        <input
          class="xf-val-input xf-cmp-input"
          type={litType === 'number' ? 'number' : 'text'}
          value={litVal}
          placeholder={litType === 'number' ? '0' : 'text'}
          spellcheck="false"
          onpointerdown={(e) => e.stopPropagation()}
          onfocus={() => (litEditing = true)}
          oninput={(e) => {
            litVal = e.currentTarget.value;
            emitLit();
          }}
          onblur={() => {
            litEditing = false;
            onCommit?.();
          }}
        />
      {/if}
      <button
        class="xf-toggle"
        title="Edit as raw expression"
        aria-label="Edit as raw expression"
        onpointerdown={(e) => e.stopPropagation()}
        onclick={(e) => {
          e.stopPropagation();
          userRaw = true;
        }}>{'</>'}</button
      >
    </div>
  {/if}
  {#if error}
    <div class="xf-err">{error}</div>
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
  .xf.has-error .xf-input,
  .xf.has-error .xf-cmp-input {
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

  /* builders */
  .xf-cmp,
  .xf-val {
    align-items: center;
  }
  .xf-cmp-input,
  .xf-cmp-lhs,
  .xf-cmp-op,
  .xf-val-type,
  .xf-val-input {
    background: #0a0d13;
    border: 1px solid color-mix(in srgb, var(--accent, var(--cyan)) 26%, var(--line));
    border-radius: 6px;
    color: #eaf2ff;
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 4px 6px;
    min-width: 0;
  }
  .xf-cmp-input:focus,
  .xf-cmp-lhs:focus,
  .xf-cmp-op:focus,
  .xf-val-type:focus,
  .xf-val-input:focus {
    outline: none;
    border-color: var(--accent, var(--cyan));
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent, var(--cyan)) 14%, transparent);
  }
  .xf-cmp-lhs {
    flex: 1 1 44%;
    color: var(--accent, var(--cyan));
    font-weight: 600;
    cursor: pointer;
  }
  .xf-cmp-op {
    flex: 0 0 auto;
    text-align: center;
    cursor: pointer;
  }
  .xf-cmp-rhs {
    flex: 1 1 30%;
  }
  .xf-val-type {
    flex: 0 0 auto;
    color: var(--text-dim);
    cursor: pointer;
  }
  .xf-val-input {
    flex: 1;
  }

  .xf-toggle {
    flex-shrink: 0;
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: 6px;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 9px;
    padding: 0 6px;
    cursor: pointer;
    transition:
      color 0.15s ease,
      border-color 0.15s ease;
  }
  .xf-toggle:hover {
    color: var(--accent, var(--cyan));
    border-color: color-mix(in srgb, var(--accent, var(--cyan)) 45%, var(--line));
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
</style>
