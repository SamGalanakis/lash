<script>
  import { setContext } from 'svelte';
  import { SvelteFlow, Background, Controls, MiniMap, BackgroundVariant } from '@xyflow/svelte';
  import WorkflowNode from './components/nodes/WorkflowNode.svelte';
  import ContainerNode from './components/nodes/ContainerNode.svelte';
  import OpaqueNode from './components/nodes/OpaqueNode.svelte';
  import DisplayPanel from './components/DisplayPanel.svelte';
  import SourceView from './components/SourceView.svelte';
  import { fetchWorkflow, fetchWorkflows, selectWorkflow, saveWorkflow } from './lib/api.js';
  import { buildFlow, deleteNodeFromDoc, addNodeToDoc } from './lib/graph.js';
  import { loadPositions, savePosition, clearPositions } from './lib/positions.js';
  import { RunController } from './lib/runStore.svelte.js';
  import { NODE_KINDS, ADDABLE_KINDS, addableMeta } from './lib/nodeKinds.js';

  const run = new RunController();
  setContext('run', run);

  const nodeTypes = { workflow: WorkflowNode, container: ContainerNode, opaque: OpaqueNode };

  let draftDoc = $state(null);
  let canonicalSource = $state('');
  let savedVersion = $state(0);
  let flowNodes = $state([]);
  let flowEdges = $state([]);
  let flowKey = $state(0);

  let positions = loadPositions();
  let dirty = $state(false);
  let loading = $state(true);
  let saving = $state(false);
  let loadError = $state(null);
  let saveError = $state(null);
  let saveOk = $state(null);

  let catalog = $state([]);
  let selectedId = $state('onboarding');
  let switching = $state(false);

  const legend = Object.entries(NODE_KINDS);

  function onDelete(id) {
    deleteNodeFromDoc(draftDoc, id);
    dirty = true;
    saveOk = null;
    rebuild();
  }

  // Insert a new default node of `kind` at the end of a container/process group
  // (ownerId + slot). The new node lives only in the draft until Save round-trips
  // it through the backend, which mints a canonical id.
  function onAddNode(ownerId, slot, kind) {
    addNodeToDoc(draftDoc, { ownerId, slot }, kind);
    dirty = true;
    saveOk = null;
    rebuild();
  }

  // Insert a new default node at the end of the top-level (`main`) scope.
  let mainMenuOpen = $state(false);
  function onAddMain(kind) {
    mainMenuOpen = false;
    addNodeToDoc(draftDoc, { main: true }, kind);
    dirty = true;
    saveOk = null;
    rebuild();
  }

  function rebuild() {
    const { flowNodes: fn, flowEdges: fe } = buildFlow(draftDoc, positions, onDelete, onAddNode);
    flowNodes = fn;
    flowEdges = fe;
    flowKey += 1; // remount SvelteFlow so layout + fitView re-run cleanly
  }

  async function loadInitial() {
    loading = true;
    loadError = null;
    try {
      const [doc, list] = await Promise.all([fetchWorkflow(), fetchWorkflows().catch(() => [])]);
      catalog = list;
      adoptDocument(doc);
    } catch (err) {
      loadError = err?.message ?? String(err);
    } finally {
      loading = false;
    }
  }

  // Switching examples resets the backend's current workflow to that built-in
  // and loads it just like the initial GET: fresh draft, re-layout, clear
  // overlay. The current draft is intentionally discarded (backend contract).
  async function onSelect(id) {
    if (switching || id === selectedId) return;
    switching = true;
    saveError = null;
    saveOk = null;
    run.stop();
    run.reset();
    try {
      const doc = await selectWorkflow(id);
      selectedId = id;
      adoptDocument(doc);
    } catch (err) {
      loadError = err?.message ?? String(err);
    } finally {
      switching = false;
    }
  }

  function adoptDocument(doc) {
    // Deep clone into a fresh reactive draft the host owns and mutates.
    draftDoc = structuredClone(doc);
    canonicalSource = doc.source;
    savedVersion = doc.version;
    dirty = false;
    rebuild();
  }

  async function onSave() {
    if (!draftDoc) return;
    saving = true;
    saveError = null;
    saveOk = null;
    // Serialize the draft (a plain snapshot, without any layout/UI cruft).
    const payload = JSON.parse(JSON.stringify(draftDoc));
    const result = await saveWorkflow(payload);
    saving = false;
    if (result.ok) {
      adoptDocument(result.document);
      saveOk = `saved as v${result.document.version}`;
      run.reset();
    } else {
      saveError = result.error ?? {
        code: `http_${result.status}`,
        message: 'save failed',
        details: {},
      };
    }
  }

  function onPlay() {
    saveOk = null;
    run.start();
  }

  function onResetLayout() {
    clearPositions();
    positions = {};
    rebuild();
  }

  function onReload() {
    run.stop();
    loadInitial();
  }

  function handleDragStop({ targetNode, nodes }) {
    const moved = nodes && nodes.length ? nodes : targetNode ? [targetNode] : [];
    for (const n of moved) {
      if (n?.position) savePosition(n.id, n.position);
    }
    positions = loadPositions();
  }

  loadInitial();
</script>

<div class="studio">
  <header class="topbar">
    <div class="brand">
      <span class="mark">◧</span>
      <div class="brand-text">
        <div class="brand-title">Workflow Graph Studio</div>
        <div class="brand-sub">lashlang · code ⇄ graph ⇄ live run</div>
      </div>
    </div>
    <nav class="legend">
      {#each legend as [kind, meta] (kind)}
        <span class="legend-item" style="--c:{meta.accent}">
          <span class="legend-dot"></span>{meta.label}
        </span>
      {/each}
      <span class="legend-sep"></span>
      <span class="legend-item legend-edge"><span class="legend-line seq"></span>sequence</span>
      <span class="legend-item legend-edge"><span class="legend-line data"></span>data</span>
    </nav>
  </header>

  <main class="body">
    <section class="canvas">
      {#if loading}
        <div class="overlay-msg">projecting graph…</div>
      {:else if loadError}
        <div class="overlay-msg error">
          backend unreachable<br /><span class="mono">{loadError}</span>
          <button class="btn" onclick={onReload}>retry</button>
        </div>
      {:else}
        {#key flowKey}
          <SvelteFlow
            bind:nodes={flowNodes}
            bind:edges={flowEdges}
            {nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            minZoom={0.2}
            maxZoom={2}
            proOptions={{ hideAttribution: false }}
            onnodedragstop={handleDragStop}
          >
            <Background variant={BackgroundVariant.Dots} gap={26} size={1.4} />
            <Controls showLock={false} />
            <MiniMap
              pannable
              zoomable
              nodeColor={(n) =>
                NODE_KINDS[n.data?.node?.data?.kind]?.accent ?? '#5c6a80'}
              maskColor="rgba(8,11,17,0.72)"
            />
          </SvelteFlow>
        {/key}
        <div class="palette">
          <button
            class="palette-btn"
            class:is-open={mainMenuOpen}
            onclick={() => (mainMenuOpen = !mainMenuOpen)}
            title="Add a node to the top-level (main) scope"
          >
            + Add node <span class="palette-scope">· main</span>
          </button>
          {#if mainMenuOpen}
            <div class="palette-menu">
              {#each ADDABLE_KINDS as k (k)}
                {@const m = addableMeta(k)}
                <button class="palette-item" style="--c:{m.accent}" onclick={() => onAddMain(k)}>
                  <span class="palette-glyph">{m.glyph}</span>{m.label}
                </button>
              {/each}
            </div>
          {/if}
        </div>
        <div class="canvas-hint">drag to arrange · positions saved locally, never in source</div>
      {/if}
    </section>

    <aside class="side">
      {#if catalog.length}
        <div class="wf-select">
          <label class="wf-select-label" for="wf-picker">workflow</label>
          <div class="wf-select-wrap">
            <select
              id="wf-picker"
              class="wf-select-input"
              value={selectedId}
              disabled={switching || loading}
              onchange={(e) => onSelect(e.currentTarget.value)}
              title={catalog.find((w) => w.id === selectedId)?.description ?? ''}
            >
              {#each catalog as w (w.id)}
                <option value={w.id} title={w.description}>{w.name}</option>
              {/each}
            </select>
            <span class="wf-select-caret">{switching ? '…' : '▾'}</span>
          </div>
          <div class="wf-select-desc">
            {catalog.find((w) => w.id === selectedId)?.description ?? ''}
          </div>
        </div>
      {/if}

      <div class="controls">
        <button
          class="btn btn-play"
          onclick={onPlay}
          disabled={loading || !!loadError}
          title="Create a new run of the saved workflow"
        >
          <span class="btn-glyph">{run.running ? '❚❚' : '▶'}</span>
          {run.running ? 'running…' : 'Play'}
        </button>
        <button
          class="btn btn-save"
          onclick={onSave}
          disabled={saving || loading || !!loadError}
          title="Send the edited graph → graph→code → new version"
        >
          {saving ? 'saving…' : 'Save'}
        </button>
        <div class="ctrl-minor">
          <button class="btn btn-ghost" onclick={onResetLayout} title="Clear saved positions"
            >reset layout</button
          >
          <button class="btn btn-ghost" onclick={onReload} title="Reload from backend, discard draft"
            >reload</button
          >
        </div>
      </div>

      {#if saveError}
        <div class="banner banner-err">
          <div class="banner-title">invalid edit · {saveError.code}</div>
          <div class="banner-msg">{saveError.message}</div>
          {#if saveError.details?.nodeId}
            <div class="banner-detail">node {saveError.details.nodeId}</div>
          {/if}
          <div class="banner-foot">draft kept — fix it and Save again</div>
        </div>
      {/if}
      {#if saveOk && !saveError}
        <div class="banner banner-ok">{saveOk}</div>
      {/if}
      {#if run.error}
        <div class="banner banner-err"><div class="banner-title">run error</div><div class="banner-msg">{run.error}</div></div>
      {/if}

      <DisplayPanel {run} />

      {#if draftDoc}
        <SourceView source={canonicalSource} version={savedVersion} {dirty} />
      {/if}
    </aside>
  </main>
</div>

<style>
  .studio {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(56, 224, 208, 0.06), transparent 60%),
      radial-gradient(1000px 500px at 100% 0%, rgba(177, 140, 255, 0.06), transparent 55%),
      var(--ink);
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 12px 22px;
    border-bottom: 1px solid var(--line);
    background: rgba(10, 13, 19, 0.7);
    backdrop-filter: blur(8px);
    z-index: 5;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .mark {
    font-size: 26px;
    color: var(--cyan);
    text-shadow: 0 0 18px color-mix(in srgb, var(--cyan) 60%, transparent);
  }
  .brand-title {
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.01em;
  }
  .brand-sub {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
    letter-spacing: 0.05em;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.03em;
  }
  .legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 3px;
    background: var(--c);
    box-shadow: 0 0 8px -1px var(--c);
  }
  .legend-sep {
    width: 1px;
    height: 16px;
    background: var(--line-strong);
  }
  .legend-line {
    width: 18px;
    height: 0;
    border-top: 2px solid;
  }
  .legend-line.seq {
    border-color: rgba(150, 170, 205, 0.6);
  }
  .legend-line.data {
    border-top-style: dashed;
    border-color: var(--cyan);
  }

  .body {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  .canvas {
    position: relative;
    flex: 1;
    min-width: 0;
  }
  .canvas-hint {
    position: absolute;
    left: 14px;
    bottom: 12px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--text-faint);
    background: rgba(10, 13, 19, 0.6);
    padding: 4px 9px;
    border-radius: 6px;
    border: 1px solid var(--line);
    pointer-events: none;
  }
  .palette {
    position: absolute;
    top: 14px;
    left: 14px;
    z-index: 6;
  }
  .palette-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.03em;
    color: var(--text);
    background: rgba(10, 13, 19, 0.82);
    border: 1px dashed color-mix(in srgb, var(--cyan) 50%, var(--line-strong));
    border-radius: 9px;
    padding: 7px 12px;
    cursor: pointer;
    backdrop-filter: blur(6px);
    transition:
      border-color 0.15s ease,
      background 0.15s ease;
  }
  .palette-btn:hover,
  .palette-btn.is-open {
    border-style: solid;
    border-color: var(--cyan);
    background: rgba(14, 18, 26, 0.92);
  }
  .palette-scope {
    color: var(--text-faint);
  }
  .palette-menu {
    margin-top: 6px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 3px;
    padding: 6px;
    min-width: 210px;
    background: var(--ink-2);
    border: 1px solid var(--line-strong);
    border-radius: 11px;
    box-shadow: 0 16px 38px -14px rgba(0, 0, 0, 0.72);
  }
  .palette-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 8px;
    cursor: pointer;
    text-align: left;
    transition:
      color 0.12s ease,
      background 0.12s ease,
      border-color 0.12s ease;
  }
  .palette-item:hover {
    color: var(--text);
    background: color-mix(in srgb, var(--c) 14%, transparent);
    border-color: color-mix(in srgb, var(--c) 45%, transparent);
  }
  .palette-glyph {
    color: var(--c);
    font-size: 12px;
    width: 14px;
    text-align: center;
    flex-shrink: 0;
  }

  .overlay-msg {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    gap: 14px;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-dim);
    text-align: center;
  }
  .overlay-msg.error {
    color: var(--rose);
  }
  .overlay-msg .mono {
    font-size: 10px;
    color: var(--text-faint);
  }

  .side {
    width: 418px;
    flex-shrink: 0;
    border-left: 1px solid var(--line);
    background: rgba(10, 13, 19, 0.5);
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .wf-select {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .wf-select-label {
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
  }
  .wf-select-wrap {
    position: relative;
  }
  .wf-select-input {
    appearance: none;
    width: 100%;
    background: linear-gradient(180deg, var(--node-hi), var(--node));
    border: 1px solid var(--line-strong);
    border-radius: 10px;
    color: var(--text);
    font-family: var(--font-ui);
    font-weight: 600;
    font-size: 14px;
    padding: 11px 34px 11px 13px;
    cursor: pointer;
    transition:
      border-color 0.2s ease,
      box-shadow 0.2s ease;
  }
  .wf-select-input:hover:not(:disabled) {
    border-color: color-mix(in srgb, var(--cyan) 45%, var(--line-strong));
  }
  .wf-select-input:focus {
    outline: none;
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--cyan) 14%, transparent);
  }
  .wf-select-input:disabled {
    opacity: 0.55;
    cursor: wait;
  }
  .wf-select-input option {
    background: var(--ink-2);
    color: var(--text);
  }
  .wf-select-caret {
    position: absolute;
    right: 13px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--cyan);
    font-size: 11px;
    pointer-events: none;
  }
  .wf-select-desc {
    font-family: var(--font-mono);
    font-size: 10px;
    line-height: 1.45;
    color: var(--text-faint);
    min-height: 14px;
  }

  .controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .ctrl-minor {
    grid-column: 1 / -1;
    display: flex;
    gap: 10px;
  }
  .btn {
    border: 1px solid var(--line-strong);
    background: var(--ink-3);
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    padding: 11px 14px;
    border-radius: 10px;
    transition:
      transform 0.12s ease,
      background 0.2s ease,
      box-shadow 0.2s ease,
      border-color 0.2s ease;
  }
  .btn:hover:not(:disabled) {
    transform: translateY(-1px);
  }
  .btn:active:not(:disabled) {
    transform: translateY(0);
  }
  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .btn-glyph {
    font-size: 11px;
    margin-right: 4px;
  }
  .btn-play {
    background: linear-gradient(180deg, #1c8f83, #147268);
    border-color: color-mix(in srgb, var(--cyan) 50%, transparent);
    color: #eafffb;
  }
  .btn-play:hover:not(:disabled) {
    box-shadow: 0 8px 24px -10px var(--cyan);
  }
  .btn-save {
    background: linear-gradient(180deg, #2b3247, #1d2333);
    border-color: var(--line-strong);
  }
  .btn-save:hover:not(:disabled) {
    box-shadow: 0 8px 22px -12px rgba(120, 150, 200, 0.6);
  }
  .btn-ghost {
    flex: 1;
    font-size: 11px;
    font-weight: 500;
    padding: 8px 10px;
    background: transparent;
    color: var(--text-dim);
    border-style: dashed;
  }
  .btn-ghost:hover {
    color: var(--text);
    background: var(--ink-3);
  }

  .banner {
    border-radius: 11px;
    padding: 11px 13px;
    font-size: 12px;
  }
  .banner-err {
    background: color-mix(in srgb, var(--rose) 12%, var(--ink-2));
    border: 1px solid color-mix(in srgb, var(--rose) 40%, transparent);
  }
  .banner-title {
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--rose);
    margin-bottom: 4px;
  }
  .banner-msg {
    color: var(--text);
    line-height: 1.4;
  }
  .banner-detail,
  .banner-foot {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--text-faint);
    margin-top: 5px;
  }
  .banner-ok {
    background: rgba(95, 208, 138, 0.12);
    border: 1px solid rgba(95, 208, 138, 0.4);
    color: #9fe6bb;
    font-family: var(--font-mono);
    font-size: 11px;
    text-align: center;
  }

  @media (max-width: 900px) {
    .body {
      flex-direction: column;
    }
    .side {
      width: 100%;
      border-left: none;
      border-top: 1px solid var(--line);
      max-height: 48vh;
    }
  }
</style>
