<script>
  import { Handle, Position } from '@xyflow/svelte';
  import { getContext } from 'svelte';
  import {
    kindMeta,
    containerSubkind,
    CONTAINER_SUBKINDS,
    ADDABLE_KINDS,
    addableMeta,
  } from '../../lib/nodeKinds.js';

  let { id, data } = $props();

  // Which group's "+ Add node" menu is open (by slot), if any.
  let menuSlot = $state(null);
  function toggleMenu(slot) {
    menuSlot = menuSlot === slot ? null : slot;
  }
  function pickKind(slot, kind) {
    menuSlot = null;
    data.onAddNode?.(id, slot, kind);
  }

  const run = getContext('run');
  const node = $derived(data.node);
  const isProcess = $derived(node.data.kind === 'process');
  const meta = $derived(kindMeta(node.data.kind));
  const status = $derived(run.overlay[id] ?? null);
  const groups = $derived(data.groups ?? []);

  // Recover the container sub-kind (if / for / while / comprehension) — it is not
  // an explicit DTO field. Each sub-kind surfaces the editable slice of the
  // backend lens in place of its (redundant) derived title: `if`/`while` edit a
  // condition, `for` edits its loop binding + iterable, and a comprehension
  // edits its binding + for/if clauses.
  const subkind = $derived(isProcess ? null : containerSubkind(node));
  const subMeta = $derived(subkind ? CONTAINER_SUBKINDS[subkind] ?? CONTAINER_SUBKINDS.loop : null);
  const isWhile = $derived(subkind === 'while');
  const isIf = $derived(subkind === 'if');
  const isFor = $derived(subkind === 'for');
  const isComprehension = $derived(subkind === 'comprehension');
  // if / while both surface a single editable condition input.
  const isConditional = $derived(isWhile || isIf);
  const clauses = $derived(node.data.clauses ?? []);

  function onTitleInput(e) {
    node.data.title = e.currentTarget.value;
    node.data.nameSource = 'label';
  }
  function onCondition(e) {
    node.data.condition = e.currentTarget.value;
  }
  // `for` binding + iterable are required by the backend lens — keep the raw
  // value (never coerce empty to undefined the way an optional binding would).
  function onForBinding(e) {
    node.data.binding = e.currentTarget.value;
  }
  function onIterable(e) {
    node.data.iterable = e.currentTarget.value;
  }
  // Comprehension binding is optional — empty means "no `let` name".
  function onBinding(e) {
    const v = e.currentTarget.value;
    node.data.binding = v === '' ? undefined : v;
  }
  function onClauseBinding(i, e) {
    node.data.clauses[i].binding = e.currentTarget.value;
  }
  function onClauseIterable(i, e) {
    node.data.clauses[i].iterable = e.currentTarget.value;
  }
  function onClauseCondition(i, e) {
    node.data.clauses[i].condition = e.currentTarget.value;
  }
</script>

<div
  class="wf-container"
  class:is-process={isProcess}
  class:is-running={status === 'running'}
  class:is-done={status === 'succeeded'}
  style="--accent:{meta.accent}; width:{data.width}px; height:{data.height}px;"
>
  <Handle type="target" position={Position.Top} />

  <header class="ct-head">
    <span class="ct-badge"
      ><span class="ct-glyph">{isProcess ? meta.glyph : subMeta.glyph}</span>{isProcess
        ? 'process'
        : subMeta.label}</span
    >
    {#if isProcess}
      <input class="ct-title" value={node.data.title} oninput={onTitleInput} spellcheck="false" />
    {:else if isConditional}
      <span class="ct-cond">
        <span class="ct-paren">(</span>
        <input
          class="ct-cond-input nodrag"
          value={node.data.condition ?? ''}
          oninput={onCondition}
          spellcheck="false"
          placeholder="condition"
        />
        <span class="ct-paren">)</span>
      </span>
    {:else if isFor}
      <span class="ct-for">
        <input
          class="ct-cond-input ct-for-bind nodrag"
          value={node.data.binding ?? ''}
          oninput={onForBinding}
          spellcheck="false"
          placeholder="binding"
        />
        <span class="ct-kw">in</span>
        <input
          class="ct-cond-input nodrag"
          value={node.data.iterable ?? ''}
          oninput={onIterable}
          spellcheck="false"
          placeholder="iterable"
        />
      </span>
    {:else if isComprehension}
      <span class="ct-for">
        <span class="ct-kw">let</span>
        <input
          class="ct-cond-input ct-for-bind nodrag"
          value={node.data.binding ?? ''}
          oninput={onBinding}
          spellcheck="false"
          placeholder="binding (optional)"
        />
      </span>
    {:else}
      <span class="ct-title-static">{node.data.title}</span>
    {/if}
    {#if status}<span class="ct-status ct-status--{status}">{status}</span>{/if}
    {#if !isProcess}
      <button
        class="ct-del"
        title="Delete branch"
        aria-label="Delete branch"
        onclick={(e) => {
          e.stopPropagation();
          data.onDelete(id);
        }}>×</button
      >
    {/if}
  </header>

  {#if isComprehension && clauses.length}
    <div class="ct-clauses nodrag">
      {#each clauses as clause, i (i)}
        <div class="ct-clause">
          {#if clause.kind === 'for'}
            <span class="ct-kw">for</span>
            <input
              class="ct-cond-input ct-for-bind"
              value={clause.binding ?? ''}
              oninput={(e) => onClauseBinding(i, e)}
              spellcheck="false"
              placeholder="binding"
            />
            <span class="ct-kw">in</span>
            <input
              class="ct-cond-input"
              value={clause.iterable ?? ''}
              oninput={(e) => onClauseIterable(i, e)}
              spellcheck="false"
              placeholder="iterable"
            />
          {:else}
            <span class="ct-kw">if</span>
            <input
              class="ct-cond-input"
              value={clause.condition ?? ''}
              oninput={(e) => onClauseCondition(i, e)}
              spellcheck="false"
              placeholder="condition"
            />
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  {#each groups as g (g.slot)}
    <span class="ct-slot" style="left:{g.x}px; top:{g.y - 4}px; width:{g.w}px;">{g.slot}</span>
  {/each}

  {#each groups as g (g.slot)}
    {#if g.addable && data.onAddNode}
      <div class="ct-add nodrag" style="left:{g.x}px; top:{g.y + g.h - 32}px; width:{g.w}px;">
        <button
          class="ct-add-btn"
          class:is-open={menuSlot === g.slot}
          title="Add a node to the end of this {g.slot}"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            toggleMenu(g.slot);
          }}
        >
          + Add node
        </button>
        {#if menuSlot === g.slot}
          <div class="ct-add-menu">
            {#each ADDABLE_KINDS as k (k)}
              {@const m = addableMeta(k)}
              <button
                class="ct-add-item"
                style="--c:{m.accent}"
                onpointerdown={(e) => e.stopPropagation()}
                onclick={(e) => {
                  e.stopPropagation();
                  pickKind(g.slot, k);
                }}
              >
                <span class="ct-add-glyph">{m.glyph}</span>{m.label}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {/each}

  <Handle type="source" position={Position.Bottom} />
</div>

<style>
  .wf-container {
    position: relative;
    border: 1px solid color-mix(in srgb, var(--accent) 26%, var(--line));
    border-radius: 16px;
    background: color-mix(in srgb, var(--accent) 5%, rgba(12, 16, 24, 0.55));
    box-shadow: inset 0 0 60px -30px color-mix(in srgb, var(--accent) 60%, transparent);
    backdrop-filter: blur(1px);
  }
  .wf-container.is-process {
    border-style: solid;
    border-color: color-mix(in srgb, var(--accent) 22%, var(--line));
    background: rgba(10, 14, 21, 0.45);
  }
  .ct-head {
    position: absolute;
    top: 10px;
    left: 14px;
    right: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .ct-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 34%, transparent);
    padding: 2px 7px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .ct-glyph {
    font-size: 11px;
  }
  .ct-title {
    background: transparent;
    border: none;
    border-bottom: 1px dashed transparent;
    color: var(--text);
    font-family: var(--font-ui);
    font-weight: 600;
    font-size: 14px;
    padding: 1px 0;
    flex: 1;
    min-width: 0;
  }
  .ct-title:focus {
    outline: none;
    border-bottom-color: var(--accent);
  }
  .ct-title-static {
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 12px;
    flex: 1;
  }
  .ct-cond {
    display: flex;
    align-items: center;
    gap: 3px;
    flex: 1;
    min-width: 0;
  }
  .ct-paren {
    font-family: var(--font-mono);
    font-size: 14px;
    color: color-mix(in srgb, var(--accent) 75%, var(--text-dim));
  }
  .ct-cond-input {
    flex: 1;
    min-width: 0;
    background: var(--ink-2);
    border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--line));
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11.5px;
    padding: 3px 7px;
  }
  .ct-cond-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .ct-for {
    display: flex;
    align-items: center;
    gap: 5px;
    flex: 1;
    min-width: 0;
  }
  .ct-for-bind {
    flex: 0 1 40%;
    color: var(--accent);
    font-weight: 600;
  }
  .ct-kw {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    color: color-mix(in srgb, var(--accent) 70%, var(--text-dim));
    flex-shrink: 0;
  }
  .ct-clauses {
    position: absolute;
    top: 38px;
    left: 14px;
    right: 12px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .ct-clause {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
  }
  .ct-status {
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 5px;
    border-radius: 5px;
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 18%, transparent);
  }
  .ct-status--succeeded {
    color: #5fd08a;
    background: rgba(95, 208, 138, 0.16);
  }
  .ct-del {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-size: 17px;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    padding: 0;
  }
  .ct-del:hover {
    color: var(--rose);
    background: color-mix(in srgb, var(--rose) 16%, transparent);
  }
  .ct-slot {
    position: absolute;
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    border-top: 1px dashed color-mix(in srgb, var(--accent) 30%, transparent);
    padding-top: 3px;
    text-align: center;
    pointer-events: none;
  }
  .wf-container.is-running {
    border-color: var(--cyan);
    box-shadow:
      inset 0 0 70px -34px color-mix(in srgb, var(--cyan) 70%, transparent),
      0 0 0 1px color-mix(in srgb, var(--cyan) 40%, transparent);
  }

  .ct-add {
    position: absolute;
    display: flex;
    justify-content: center;
    z-index: 3;
  }
  .ct-add-btn {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.04em;
    color: color-mix(in srgb, var(--accent) 80%, var(--text-dim));
    background: color-mix(in srgb, var(--accent) 8%, rgba(10, 14, 21, 0.85));
    border: 1px dashed color-mix(in srgb, var(--accent) 45%, var(--line));
    border-radius: 8px;
    padding: 5px 12px;
    cursor: pointer;
    transition:
      color 0.15s ease,
      background 0.15s ease,
      border-color 0.15s ease;
  }
  .ct-add-btn:hover,
  .ct-add-btn.is-open {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, rgba(10, 14, 21, 0.92));
    border-style: solid;
    border-color: var(--accent);
  }
  .ct-add-menu {
    position: absolute;
    top: calc(100% + 5px);
    left: 50%;
    transform: translateX(-50%);
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 3px;
    padding: 6px;
    min-width: 190px;
    background: var(--ink-2, #0d1119);
    border: 1px solid var(--line-strong, #2a3346);
    border-radius: 10px;
    box-shadow: 0 14px 34px -12px rgba(0, 0, 0, 0.7);
    z-index: 20;
  }
  .ct-add-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-dim);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 7px;
    cursor: pointer;
    text-align: left;
    transition:
      color 0.12s ease,
      background 0.12s ease,
      border-color 0.12s ease;
  }
  .ct-add-item:hover {
    color: var(--text);
    background: color-mix(in srgb, var(--c) 14%, transparent);
    border-color: color-mix(in srgb, var(--c) 45%, transparent);
  }
  .ct-add-glyph {
    color: var(--c);
    font-size: 11px;
    width: 13px;
    text-align: center;
    flex-shrink: 0;
  }
</style>
