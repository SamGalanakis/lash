<script>
  import { Handle, Position } from '@xyflow/svelte';
  import { getContext } from 'svelte';
  import { kindMeta } from '../../lib/nodeKinds.js';

  let { id, data } = $props();

  const run = getContext('run');
  const node = $derived(data.node);
  const isProcess = $derived(node.data.kind === 'process');
  const meta = $derived(kindMeta(node.data.kind));
  const status = $derived(run.overlay[id] ?? null);
  const groups = $derived(data.groups ?? []);

  function onTitleInput(e) {
    node.data.title = e.currentTarget.value;
    node.data.nameSource = 'label';
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
    <span class="ct-badge"><span class="ct-glyph">{meta.glyph}</span>{isProcess ? 'process' : 'branch'}</span
    >
    {#if isProcess}
      <input class="ct-title" value={node.data.title} oninput={onTitleInput} spellcheck="false" />
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

  {#each groups as g (g.slot)}
    <span class="ct-slot" style="left:{g.x}px; top:{g.y - 4}px; width:{g.w}px;">{g.slot}</span>
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
</style>
