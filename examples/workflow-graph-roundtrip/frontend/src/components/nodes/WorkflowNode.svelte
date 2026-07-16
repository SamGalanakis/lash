<script>
  import { Handle, Position } from '@xyflow/svelte';
  import { getContext } from 'svelte';
  import { kindMeta, OP_LABELS, WAITING_EFFECTS } from '../../lib/nodeKinds.js';
  import ExpressionField from '../ExpressionField.svelte';

  let { id, data } = $props();

  const run = getContext('run');
  const mode = getContext('mode');
  const node = $derived(data.node);
  const meta = $derived(kindMeta(node.data.kind));
  const status = $derived(run.overlay[id] ?? null);

  const kind = $derived(node.data.kind);
  const isAssign = $derived(kind === 'state_update');
  const isComputation = $derived(kind === 'computation');
  const isData = $derived(kind === 'data');
  const isTerminal = $derived(kind === 'terminal');
  // call / effect are configured through their typed `fields` (record args /
  // duration / signal) + an optional `let <binding> =` name; their raw call
  // expression is rebuilt by the lens, so it is not shown as an input here.
  const isInvoke = $derived(kind === 'call' || kind === 'effect');
  const hasBinding = $derived(isInvoke || isData);
  // state_update / computation always carry an editable expression; data /
  // terminal expose one only when the backend surfaces `expression` on them.
  const hasExpr = $derived(
    isAssign || isComputation || ((isData || isTerminal) && node.data.expression !== undefined),
  );

  const availableVars = $derived(node.data.availableVars ?? []);
  const reads = $derived(data.reads ?? []);
  const moveTargets = $derived(data.getMoveTargets ? data.getMoveTargets(id) : []);
  let moveOpen = $state(false);

  const subtitle = $derived(
    node.data.operation
      ? (OP_LABELS[node.data.operation] ?? node.data.operation)
      : (node.data.effect ??
          (isAssign ? 'assignment' : isComputation ? 'sequenced compute' : kind)),
  );
  const isWaitEffect = $derived(node.data.effect && WAITING_EFFECTS.has(node.data.effect));
  const terminalKind = $derived(node.data.terminalKind ?? null);

  const fieldKeys = $derived(Object.keys(node.data.fields ?? {}));

  function commit() {
    data.onCommit?.();
  }

  function defaultTarget() {
    const m = (node.data.title ?? '').trim().match(/^update\s+(.+)$/i);
    return m ? m[1] : '';
  }
  function onBinding(e) {
    const v = e.currentTarget.value;
    node.data.binding = v === '' ? undefined : v;
  }
  function onTitleInput(e) {
    node.data.title = e.currentTarget.value;
    node.data.nameSource = 'label';
  }
  function onDescInput(e) {
    node.data.description = e.currentTarget.value || undefined;
    node.data.nameSource = 'label';
  }
  function onNumber(key, e) {
    const n = Number(e.currentTarget.value);
    if (!Number.isNaN(n)) node.data.fields[key] = n;
  }

  function fieldType(v) {
    if (typeof v === 'number') return 'number';
    if (typeof v === 'boolean') return 'boolean';
    if (typeof v === 'string') return 'string';
    if (v !== null && typeof v === 'object' && typeof v.$expr === 'string') return 'expr';
    return 'other';
  }
</script>

<div
  class="wf-node"
  class:is-running={status === 'running'}
  class:is-waiting={status === 'waiting'}
  class:is-done={status === 'succeeded'}
  class:is-failed={status === 'failed'}
  style="--accent:{meta.accent}; width:{data.width}px; min-height:{data.height}px;"
>
  <Handle type="target" position={Position.Top} />

  <header class="wf-head">
    <span class="wf-badge" title={kind}>
      <span class="wf-glyph">{meta.glyph}</span>{meta.label}
    </span>
    <span class="wf-sub">{subtitle}</span>
    {#if isTerminal && terminalKind}
      <span class="wf-term wf-term--{terminalKind}">{terminalKind}</span>
    {/if}
    {#if status}
      <span class="wf-status wf-status--{status}">{status}</span>
    {/if}
    {#if moveTargets.length}
      <span class="wf-move-wrap nodrag">
        <button
          class="wf-icon"
          class:is-open={moveOpen}
          title="Move to another scope"
          aria-label="Move node to another scope"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            moveOpen = !moveOpen;
          }}>⤴</button
        >
        {#if moveOpen}
          <div class="wf-move-menu">
            <div class="wf-move-label">move into</div>
            {#each moveTargets as t (t.key)}
              <button
                class="wf-move-item"
                onpointerdown={(e) => e.stopPropagation()}
                onclick={(e) => {
                  e.stopPropagation();
                  moveOpen = false;
                  data.onMoveTo?.(id, t.dest);
                }}>{t.label}</button
              >
            {/each}
          </div>
        {/if}
      </span>
    {/if}
    {#if data.onReorder}
      <span class="wf-reorder nodrag" title="Reorder within scope">
        <button
          class="wf-move"
          title="Move earlier in execution order"
          aria-label="Move node earlier"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            data.onReorder(id, 'up');
          }}>▲</button
        >
        <button
          class="wf-move"
          title="Move later in execution order"
          aria-label="Move node later"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            data.onReorder(id, 'down');
          }}>▼</button
        >
      </span>
    {/if}
    <button
      class="wf-del"
      title="Delete node"
      aria-label="Delete node"
      onclick={(e) => {
        e.stopPropagation();
        data.onDelete(id);
      }}>×</button
    >
  </header>

  <input
    class="wf-title"
    value={node.data.title}
    oninput={onTitleInput}
    onchange={commit}
    spellcheck="false"
    placeholder="title"
  />
  {#if node.data.nameSource === 'label' || node.data.description}
    <input
      class="wf-desc"
      value={node.data.description ?? ''}
      oninput={onDescInput}
      onchange={commit}
      spellcheck="false"
      placeholder="description…"
    />
  {/if}

  {#if reads.length}
    <div class="wf-reads" title={`reads variables from an enclosing scope: ${reads.join(', ')}`}>
      <span class="wf-reads-k">reads</span>
      {#each reads as v (v)}<span class="wf-reads-v">{v}</span>{/each}
    </div>
  {/if}

  {#if hasBinding}
    <div class="wf-bind">
      <span class="wf-bind-kw">let</span>
      <input
        class="wf-bind-input nodrag"
        value={node.data.binding ?? ''}
        oninput={onBinding}
        onchange={commit}
        spellcheck="false"
        placeholder="name (optional)"
      />
      <span class="wf-bind-eq">=</span>
    </div>
  {/if}

  {#if fieldKeys.length}
    <div class="wf-fields">
      {#each fieldKeys as key (key)}
        {@const v = node.data.fields[key]}
        <label class="wf-field">
          <span class="wf-key">{key}</span>
          {#if fieldType(v) === 'number'}
            <input
              class="wf-input wf-input--num"
              type="number"
              value={v}
              oninput={(e) => onNumber(key, e)}
              onchange={commit}
            />
          {:else if fieldType(v) === 'boolean'}
            <input
              class="wf-check"
              type="checkbox"
              checked={v}
              onchange={(e) => {
                node.data.fields[key] = e.currentTarget.checked;
                commit();
              }}
            />
          {:else if fieldType(v) === 'string'}
            <input
              class="wf-input"
              type="text"
              value={v}
              spellcheck="false"
              oninput={(e) => (node.data.fields[key] = e.currentTarget.value)}
              onchange={commit}
            />
          {:else if fieldType(v) === 'expr'}
            <ExpressionField
              value={v.$expr}
              kind="expression"
              {availableVars}
              placeholder="expression…"
              onInput={(text) => (node.data.fields[key] = { $expr: text })}
              onCommit={commit}
            />
          {:else}
            <code class="wf-ro">{JSON.stringify(v)}</code>
          {/if}
        </label>
      {/each}
    </div>
  {/if}

  {#if hasExpr}
    <div class="wf-expr">
      {#if isAssign}
        <div class="wf-expr-row">
          <div class="wf-expr-lhs-wrap">
            <ExpressionField
              value={node.data.target ?? defaultTarget()}
              kind="assignment_target"
              {availableVars}
              placeholder="target"
              onInput={(text) => (node.data.target = text)}
              onCommit={commit}
            />
          </div>
          <span class="wf-expr-op" title="assignment">≔</span>
        </div>
      {/if}
      <ExpressionField
        value={node.data.expression ?? ''}
        kind="expression"
        {availableVars}
        placeholder="expression…"
        onInput={(text) => (node.data.expression = text)}
        onCommit={commit}
      />
      <div class="wf-expr-note">
        {isAssign
          ? 'typed assignment · canonical form rebuilt on save'
          : isComputation
            ? 'typed computation · canonical form rebuilt on save'
            : 'value expression · canonical form rebuilt on save'}
      </div>
    </div>
  {/if}

  {#if isWaitEffect}
    <div class="wf-wait-note">
      {node.data.effect === 'sleep' ? 'pauses the run' : 'waits for a signal'}
    </div>
  {/if}

  <Handle type="source" position={Position.Bottom} />
</div>

<style>
  .wf-node {
    position: relative;
    background: linear-gradient(180deg, var(--node-hi), var(--node));
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 9px 11px 11px;
    box-shadow: var(--shadow);
    transition:
      box-shadow 0.25s ease,
      border-color 0.25s ease,
      transform 0.2s ease;
  }

  .wf-head {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 7px;
  }
  .wf-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 32%, transparent);
    padding: 2px 6px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .wf-glyph {
    font-size: 11px;
    line-height: 1;
  }
  .wf-sub {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }
  .wf-term {
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 5px;
    border-radius: 5px;
  }
  .wf-term--finish {
    color: #5fd08a;
    background: rgba(95, 208, 138, 0.16);
  }
  .wf-term--fail {
    color: var(--rose);
    background: color-mix(in srgb, var(--rose) 18%, transparent);
  }
  .wf-status {
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 5px;
    border-radius: 5px;
  }
  .wf-status--running {
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 18%, transparent);
  }
  .wf-status--waiting {
    color: var(--amber);
    background: color-mix(in srgb, var(--amber) 18%, transparent);
  }
  .wf-status--succeeded {
    color: #5fd08a;
    background: rgba(95, 208, 138, 0.16);
  }
  .wf-status--failed {
    color: var(--rose);
    background: color-mix(in srgb, var(--rose) 20%, transparent);
  }
  .wf-del {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-size: 17px;
    line-height: 1;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    padding: 0;
    transition:
      color 0.15s ease,
      background 0.15s ease;
  }
  .wf-del:hover {
    color: var(--rose);
    background: color-mix(in srgb, var(--rose) 16%, transparent);
  }

  .wf-move-wrap {
    position: relative;
    flex-shrink: 0;
  }
  .wf-icon {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-size: 12px;
    line-height: 1;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    padding: 0;
    transition:
      color 0.15s ease,
      background 0.15s ease;
  }
  .wf-icon:hover,
  .wf-icon.is-open {
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 16%, transparent);
  }
  .wf-move-menu {
    position: absolute;
    top: calc(100% + 5px);
    right: 0;
    z-index: 30;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 150px;
    max-height: 200px;
    overflow-y: auto;
    padding: 5px;
    background: var(--ink-2);
    border: 1px solid var(--line-strong);
    border-radius: 9px;
    box-shadow: 0 14px 30px -12px rgba(0, 0, 0, 0.7);
  }
  .wf-move-label {
    font-family: var(--font-mono);
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    padding: 2px 6px 4px;
  }
  .wf-move-item {
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 10.5px;
    padding: 4px 7px;
    cursor: pointer;
    transition:
      color 0.12s ease,
      background 0.12s ease,
      border-color 0.12s ease;
  }
  .wf-move-item:hover {
    color: var(--text);
    background: color-mix(in srgb, var(--cyan) 12%, transparent);
    border-color: color-mix(in srgb, var(--cyan) 40%, transparent);
  }

  .wf-reorder {
    display: inline-flex;
    flex-direction: column;
    gap: 1px;
    flex-shrink: 0;
  }
  .wf-move {
    border: none;
    background: transparent;
    color: var(--text-faint);
    font-size: 8px;
    line-height: 1;
    width: 17px;
    height: 10px;
    padding: 0;
    border-radius: 3px;
    cursor: pointer;
    transition:
      color 0.15s ease,
      background 0.15s ease;
  }
  .wf-move:hover {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }

  .wf-title {
    width: 100%;
    background: transparent;
    border: none;
    border-bottom: 1px dashed transparent;
    color: var(--text);
    font-family: var(--font-ui);
    font-weight: 600;
    font-size: 14px;
    padding: 1px 0 3px;
    margin-bottom: 2px;
  }
  .wf-title:focus,
  .wf-desc:focus {
    outline: none;
    border-bottom-color: var(--accent);
  }
  .wf-desc {
    width: 100%;
    background: transparent;
    border: none;
    border-bottom: 1px dashed transparent;
    color: var(--text-dim);
    font-family: var(--font-ui);
    font-size: 11.5px;
    padding: 1px 0 4px;
    margin-bottom: 4px;
  }

  .wf-reads {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    margin: 2px 0 6px;
    padding: 4px 7px;
    background: color-mix(in srgb, var(--cyan) 7%, transparent);
    border: 1px dashed color-mix(in srgb, var(--cyan) 30%, var(--line));
    border-radius: 7px;
  }
  .wf-reads-k {
    font-family: var(--font-mono);
    font-size: 8.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: color-mix(in srgb, var(--cyan) 80%, var(--text-dim));
  }
  .wf-reads-v {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 12%, transparent);
    border-radius: 5px;
    padding: 1px 6px;
  }

  .wf-bind {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 2px;
    margin-bottom: 4px;
  }
  .wf-bind-kw {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    color: var(--text-faint);
    flex-shrink: 0;
  }
  .wf-bind-eq {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--accent);
    flex-shrink: 0;
  }
  .wf-bind-input {
    flex: 1;
    min-width: 0;
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 500;
    padding: 5px 8px;
    transition: border-color 0.15s ease;
  }
  .wf-bind-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
  }

  .wf-fields {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 4px;
  }
  .wf-field {
    display: grid;
    grid-template-columns: 62px 1fr;
    align-items: center;
    gap: 8px;
  }
  .wf-key {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .wf-input {
    width: 100%;
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11.5px;
    padding: 5px 8px;
    transition: border-color 0.15s ease;
  }
  .wf-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .wf-input--num {
    font-variant-numeric: tabular-nums;
  }
  .wf-check {
    justify-self: start;
    accent-color: var(--accent);
    width: 15px;
    height: 15px;
  }
  .wf-ro {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-dim);
  }
  .wf-wait-note {
    margin-top: 8px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--amber);
    opacity: 0.8;
    letter-spacing: 0.04em;
  }

  .wf-expr {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .wf-expr-row {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .wf-expr-lhs-wrap {
    flex: 1;
    min-width: 0;
    display: flex;
  }
  .wf-expr-op {
    font-family: var(--font-mono);
    font-size: 14px;
    color: var(--accent);
    flex-shrink: 0;
  }
  .wf-expr-note {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-faint);
    letter-spacing: 0.03em;
  }

  /* run overlay states */
  .wf-node.is-running {
    border-color: var(--cyan);
    box-shadow:
      0 0 0 1px var(--cyan),
      0 0 26px -4px color-mix(in srgb, var(--cyan) 65%, transparent),
      var(--shadow);
  }
  .wf-node.is-waiting {
    border-color: var(--amber);
    animation: wf-pulse 1.1s ease-in-out infinite;
  }
  .wf-node.is-done {
    border-left-color: #5fd08a;
  }
  .wf-node.is-failed {
    border-color: var(--rose);
    box-shadow:
      0 0 0 1px var(--rose),
      0 0 26px -6px color-mix(in srgb, var(--rose) 60%, transparent),
      var(--shadow);
  }
  @keyframes wf-pulse {
    0%,
    100% {
      box-shadow:
        0 0 0 1px color-mix(in srgb, var(--amber) 55%, transparent),
        0 0 8px -2px color-mix(in srgb, var(--amber) 30%, transparent),
        var(--shadow);
    }
    50% {
      box-shadow:
        0 0 0 1px var(--amber),
        0 0 28px -2px color-mix(in srgb, var(--amber) 75%, transparent),
        var(--shadow);
    }
  }
</style>
