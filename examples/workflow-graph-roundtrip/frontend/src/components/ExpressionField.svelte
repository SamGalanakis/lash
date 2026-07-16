<script>
  import { getContext, tick } from 'svelte';
  import { validateFragment } from '../lib/api.js';
  import {
    isComplexExpression,
    isBoolLiteral,
    isSimpleReference,
    COMPARISON_OPS,
    parseComparison,
    parseLiteral,
    encodeLiteral,
    parseList,
  } from '../lib/fields.js';
  import ScalarBuilder from './ScalarBuilder.svelte';

  // A single editable Lashlang-text slot, rendered by mode:
  //  - Power: a raw multi-line code input (plus variable picker + validation).
  //  - Simplified: a lightweight visual builder so a non-expert can express the
  //    common cases without typing syntax — `builder="value"` a typed scalar,
  //    `builder="comparison"` a [variable ▾][op ▾][scalar] builder, `builder="list"`
  //    a list-of-scalars chip editor or an in-scope list variable. Every field
  //    keeps a raw escape toggle for anything the builder cannot express.
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
  const vars = $derived(Array.isArray(availableVars) ? availableVars : []);

  const autoRaw = $derived.by(() => {
    if (!hasBuilder) return true;
    if (isComplexExpression(value)) return true;
    const empty = (value ?? '').trim() === '';
    if (builder === 'comparison') return !(cmp || empty || isBoolLiteral(value));
    if (builder === 'value') return lit?.type === 'expression';
    if (builder === 'list') {
      return !(
        parseList(value) ||
        empty ||
        (isSimpleReference(value) && vars.includes((value ?? '').trim()))
      );
    }
    return true;
  });
  const showRaw = $derived(mode.power || (userRaw ?? autoRaw));

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
    if (cmpEditing || builder !== 'comparison') return;
    const parsed = parseComparison(value);
    if (parsed) {
      lhs = parsed.lhs;
      op = parsed.op;
      rhs = parsed.rhs;
    } else if ((value ?? '').trim() === '' || isBoolLiteral(value)) {
      lhs = '';
      rhs = '';
    }
  });
  function emitCmp() {
    emit(lhs === '' && rhs === '' ? '' : `${lhs} ${op} ${rhs}`.trim());
  }

  // --- list builder ----------------------------------------------------------
  let items = $state([]);
  let listMode = $state('items');
  $effect(() => {
    if (builder !== 'list') return;
    const trimmed = (value ?? '').trim();
    if (isSimpleReference(value) && vars.includes(trimmed)) {
      listMode = 'var';
      return;
    }
    const parsed = parseList(value);
    if (parsed) {
      listMode = 'items';
      items = parsed.items.map((it) => encodeLiteral(it.type, it.value));
    }
  });
  const listVar = $derived(
    isSimpleReference(value) && vars.includes((value ?? '').trim())
      ? (value ?? '').trim()
      : (vars[0] ?? ''),
  );
  function emitItems() {
    emit(`[${items.join(', ')}]`);
  }
  function setListMode(next) {
    listMode = next;
    if (next === 'var') emit(vars[0] ?? '');
    else emitItems();
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
          class="xf-cmp-lhs xf-cmp-plain"
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
      <ScalarBuilder
        value={rhs}
        compact
        onChange={(enc) => {
          rhs = enc;
          emitCmp();
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
  {:else if builder === 'list'}
    <div class="xf-list">
      <div class="xf-list-head">
        {#if vars.length}
          <span class="xf-listmode">
            <button
              class="xf-listmode-btn"
              class:is-active={listMode === 'items'}
              onpointerdown={(e) => e.stopPropagation()}
              onclick={(e) => {
                e.stopPropagation();
                setListMode('items');
              }}>list</button
            >
            <button
              class="xf-listmode-btn"
              class:is-active={listMode === 'var'}
              onpointerdown={(e) => e.stopPropagation()}
              onclick={(e) => {
                e.stopPropagation();
                setListMode('var');
              }}>variable</button
            >
          </span>
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
      {#if listMode === 'var' && vars.length}
        <select
          class="xf-list-var"
          value={listVar}
          onpointerdown={(e) => e.stopPropagation()}
          onchange={(e) => emit(e.currentTarget.value)}
          onblur={() => onCommit?.()}
        >
          {#each vars as v (v)}<option value={v}>{v}</option>{/each}
        </select>
      {:else}
        <div class="xf-list-items">
          {#each items as item, i (i)}
            <div class="xf-list-item">
              <ScalarBuilder
                value={item}
                compact
                onChange={(enc) => {
                  items[i] = enc;
                  emitItems();
                }}
              />
              <button
                class="xf-list-del"
                title="Remove item"
                aria-label="Remove item"
                onpointerdown={(e) => e.stopPropagation()}
                onclick={(e) => {
                  e.stopPropagation();
                  items = items.filter((_, j) => j !== i);
                  emitItems();
                }}>×</button
              >
            </div>
          {/each}
          <button
            class="xf-list-add"
            onpointerdown={(e) => e.stopPropagation()}
            onclick={(e) => {
              e.stopPropagation();
              items = [...items, '0'];
              emitItems();
            }}>+ item</button
          >
        </div>
      {/if}
    </div>
  {:else}
    <div class="xf-row xf-val">
      <ScalarBuilder value={value} onChange={(enc) => emit(enc)} />
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
  .xf.has-error .xf-cmp-plain {
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

  /* comparison builder */
  .xf-cmp,
  .xf-val {
    align-items: center;
  }
  .xf-cmp-lhs,
  .xf-cmp-op {
    background: #0a0d13;
    border: 1px solid color-mix(in srgb, var(--accent, var(--cyan)) 26%, var(--line));
    border-radius: 6px;
    color: #eaf2ff;
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 4px 6px;
    min-width: 0;
  }
  .xf-cmp-lhs:focus,
  .xf-cmp-op:focus {
    outline: none;
    border-color: var(--accent, var(--cyan));
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent, var(--cyan)) 14%, transparent);
  }
  .xf-cmp-lhs {
    flex: 1 1 40%;
    color: var(--accent, var(--cyan));
    font-weight: 600;
    cursor: pointer;
  }
  .xf-cmp-op {
    flex: 0 0 auto;
    text-align: center;
    cursor: pointer;
  }

  /* list builder */
  .xf-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    min-width: 0;
  }
  .xf-list-head {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .xf-listmode {
    display: inline-flex;
    gap: 1px;
    padding: 1px;
    background: var(--ink-2);
    border: 1px solid var(--line-strong);
    border-radius: 6px;
  }
  .xf-listmode-btn {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 7px;
    border-radius: 5px;
    cursor: pointer;
  }
  .xf-listmode-btn.is-active {
    color: var(--ink);
    background: var(--cyan);
  }
  .xf-list-items {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .xf-list-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .xf-list-del {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-size: 13px;
    line-height: 1;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    padding: 0;
    cursor: pointer;
    flex-shrink: 0;
  }
  .xf-list-del:hover {
    color: var(--rose);
    background: color-mix(in srgb, var(--rose) 16%, transparent);
  }
  .xf-list-add {
    align-self: flex-start;
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.03em;
    color: color-mix(in srgb, var(--cyan) 80%, var(--text-dim));
    background: transparent;
    border: 1px dashed color-mix(in srgb, var(--cyan) 40%, var(--line));
    border-radius: 6px;
    padding: 2px 8px;
    cursor: pointer;
  }
  .xf-list-add:hover {
    color: var(--cyan);
    border-color: var(--cyan);
  }
  .xf-list-var {
    background: #0a0d13;
    border: 1px solid color-mix(in srgb, var(--cyan) 30%, var(--line));
    border-radius: 6px;
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 6px;
    cursor: pointer;
  }
  .xf-list-var:focus {
    outline: none;
    border-color: var(--cyan);
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
