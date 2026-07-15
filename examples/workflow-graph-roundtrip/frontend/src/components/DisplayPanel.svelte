<script>
  import Lights from './Lights.svelte';
  import ProgressBar from './ProgressBar.svelte';

  let { run } = $props();

  const d = $derived(run.display);
  const statuses = $derived(Object.entries(d.statuses ?? {}));
  const lists = $derived(Object.entries(d.lists ?? {}));
  const hasRun = $derived(run.runId !== null);
</script>

<section class="display" class:live={run.running}>
  <header class="disp-head">
    <div class="disp-title">
      <span class="dot" class:running={run.running} class:done={run.finished}></span>
      Display
    </div>
    <div class="disp-meta">
      {#if hasRun}
        <span class="run-id">run {run.runId.slice(0, 8)}</span>
        <span class="run-ver">v{run.workflowVersion}</span>
        <span class="run-ev">{run.eventCount} evt</span>
      {:else}
        <span class="run-id idle">no run yet</span>
      {/if}
    </div>
  </header>

  <ProgressBar value={d.progress} />

  <div class="block">
    <div class="block-label">lights</div>
    <Lights lights={d.lights} />
  </div>

  {#if statuses.length}
    <div class="block">
      <div class="block-label">status</div>
      <div class="statuses">
        {#each statuses as [k, v] (k)}
          <div class="stat">
            <span class="stat-k">{k}</span><span class="stat-v">{v}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if lists.length}
    <div class="block">
      <div class="block-label">lists</div>
      {#each lists as [name, items] (name)}
        <div class="list">
          <div class="list-name">
            {name}
            <span class="list-count">{items.length}</span>
            {#if d.highlighted === name}<span class="hl-tag">focus</span>{/if}
          </div>
          <ul>
            {#each items as item, i (i)}
              <li>{item}</li>
            {/each}
          </ul>
        </div>
      {/each}
    </div>
  {/if}

  <div class="block messages-block">
    <div class="block-label">messages <span class="mcount">{d.messages.length}</span></div>
    <div class="messages">
      {#if d.messages.length === 0}
        <div class="msg-empty">— the running workflow prints here —</div>
      {:else}
        {#each d.messages as m, i (i)}
          <div class="msg"><span class="msg-idx">{String(i + 1).padStart(2, '0')}</span>{m}</div>
        {/each}
      {/if}
    </div>
  </div>

  {#if d.highlighted}
    <div class="hl-banner">focus · {d.highlighted}</div>
  {/if}
</section>

<style>
  .display {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 15px 15px 18px;
    background: linear-gradient(180deg, rgba(20, 26, 38, 0.7), rgba(14, 18, 27, 0.7));
    border: 1px solid var(--line);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
  }
  .display.live::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    animation: sweep 1.8s linear infinite;
  }
  @keyframes sweep {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(100%);
    }
  }
  .disp-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .disp-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.01em;
  }
  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--text-faint);
  }
  .dot.running {
    background: var(--cyan);
    box-shadow: 0 0 10px 1px var(--cyan);
    animation: blink 1s ease-in-out infinite;
  }
  .dot.done {
    background: #5fd08a;
    box-shadow: 0 0 8px 1px #5fd08a;
  }
  @keyframes blink {
    50% {
      opacity: 0.35;
    }
  }
  .disp-meta {
    display: flex;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--text-faint);
  }
  .disp-meta span {
    background: var(--ink);
    border: 1px solid var(--line);
    padding: 2px 6px;
    border-radius: 5px;
  }
  .run-id.idle {
    color: var(--text-faint);
  }

  .block-label {
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .mcount,
  .list-count {
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 14%, transparent);
    border-radius: 5px;
    padding: 0 5px;
    font-size: 9px;
  }

  .statuses {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .stat {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--ink);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 5px 9px;
  }
  .stat-k {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-faint);
  }
  .stat-v {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--amber);
    font-weight: 500;
  }

  .list {
    margin-bottom: 8px;
  }
  .list-name {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 5px;
  }
  .hl-tag {
    color: var(--amber);
    background: color-mix(in srgb, var(--amber) 16%, transparent);
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 1px 5px;
    border-radius: 4px;
  }
  .list ul {
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .list li {
    font-size: 12px;
    color: var(--text);
    background: var(--ink);
    border: 1px solid var(--line);
    border-left: 2px solid #5fd08a;
    border-radius: 6px;
    padding: 5px 9px;
    animation: item-in 0.35s ease;
  }
  @keyframes item-in {
    from {
      opacity: 0;
      transform: translateX(-6px);
    }
  }

  .messages-block {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .messages {
    display: flex;
    flex-direction: column;
    gap: 5px;
    overflow-y: auto;
    max-height: 220px;
    padding-right: 4px;
  }
  .msg-empty {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-faint);
    font-style: italic;
  }
  .msg {
    display: flex;
    gap: 9px;
    align-items: baseline;
    font-size: 12.5px;
    color: var(--text);
    background: var(--ink);
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 6px 10px;
    animation: item-in 0.3s ease;
  }
  .msg-idx {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--cyan);
    opacity: 0.7;
  }
  .hl-banner {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--amber);
    text-align: center;
  }
</style>
