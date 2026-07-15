<script>
  let { source, version, dirty } = $props();
</script>

<section class="src">
  <header class="src-head">
    <div class="src-title">
      canonical source
      <span class="src-ver">v{version}</span>
    </div>
    {#if dirty}
      <span class="src-dirty">unsaved edits</span>
    {:else}
      <span class="src-clean">saved · lash-rendered</span>
    {/if}
  </header>
  <pre class="src-body"><code>{source}</code></pre>
  <div class="src-foot">graph→code output · positions never appear here</div>
</section>

<style>
  .src {
    display: flex;
    flex-direction: column;
    background: linear-gradient(180deg, rgba(18, 23, 34, 0.7), rgba(12, 16, 24, 0.7));
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    min-height: 0;
  }
  .src-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 15px 10px;
    border-bottom: 1px solid var(--line);
  }
  .src-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 13px;
  }
  .src-ver {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--cyan);
    background: color-mix(in srgb, var(--cyan) 14%, transparent);
    padding: 1px 6px;
    border-radius: 5px;
  }
  .src-dirty {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--amber);
    letter-spacing: 0.06em;
  }
  .src-clean {
    font-family: var(--font-mono);
    font-size: 9.5px;
    color: var(--text-faint);
    letter-spacing: 0.04em;
  }
  .src-body {
    margin: 0;
    padding: 13px 15px;
    overflow: auto;
    max-height: 260px;
    background: #0a0d13;
  }
  .src-body code {
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.6;
    color: #cfe3ff;
    white-space: pre;
  }
  .src-foot {
    padding: 8px 15px;
    border-top: 1px solid var(--line);
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-faint);
    letter-spacing: 0.03em;
  }
</style>
