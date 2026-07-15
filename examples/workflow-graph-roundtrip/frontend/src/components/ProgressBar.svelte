<script>
  let { value = 0 } = $props();
  const pct = $derived(Math.max(0, Math.min(100, value)));
</script>

<div class="prog">
  <div class="prog-head">
    <span class="prog-label">progress</span>
    <span class="prog-val">{Math.round(pct)}%</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill" style="width:{pct}%"></div>
    <div class="prog-ticks">
      {#each Array(20) as _, i (i)}
        <span class="tick"></span>
      {/each}
    </div>
  </div>
</div>

<style>
  .prog-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
  }
  .prog-label {
    font-family: var(--font-mono);
    font-size: 9.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
  }
  .prog-val {
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 600;
    color: var(--cyan);
    font-variant-numeric: tabular-nums;
  }
  .prog-track {
    position: relative;
    height: 14px;
    background: var(--ink);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }
  .prog-fill {
    position: absolute;
    inset: 0 auto 0 0;
    background: linear-gradient(90deg, #2fb6a8, var(--cyan) 70%, #7ff0e4);
    box-shadow: 0 0 16px -2px color-mix(in srgb, var(--cyan) 70%, transparent);
    transition: width 0.5s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .prog-ticks {
    position: absolute;
    inset: 0;
    display: flex;
    justify-content: space-between;
    padding: 0 2px;
    pointer-events: none;
  }
  .tick {
    width: 1px;
    background: rgba(0, 0, 0, 0.35);
  }
</style>
