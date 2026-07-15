<script>
  let { lights } = $props();

  const NAMED = {
    green: '#5fd08a',
    blue: '#5aa9ff',
    cyan: '#38e0d0',
    red: '#ff6b6b',
    rose: '#ff6b9d',
    amber: '#f5c451',
    yellow: '#f5d451',
    orange: '#ff9d5c',
    violet: '#b18cff',
    purple: '#b18cff',
    white: '#eef3fb',
    on: '#38e0d0',
  };
  const OFF = new Set(['off', 'false', '0', 'none', '']);

  function color(state) {
    const s = String(state ?? '').toLowerCase();
    if (OFF.has(s)) return null;
    return NAMED[s] ?? state;
  }

  const entries = $derived(Object.entries(lights ?? {}));
</script>

<div class="lights">
  {#if entries.length === 0}
    <div class="lights-empty">no lights yet</div>
  {:else}
    {#each entries as [name, state] (name)}
      {@const c = color(state)}
      <div class="light" class:on={c} style={c ? `--c:${c}` : ''}>
        <span class="bulb"></span>
        <span class="light-name">{name}</span>
        <span class="light-state">{state}</span>
      </div>
    {/each}
  {/if}
</div>

<style>
  .lights {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 4px 2px;
  }
  .lights-empty {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-faint);
    padding: 6px 2px;
  }
  .light {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    min-width: 58px;
  }
  .bulb {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #2a3444, #12161f);
    border: 1px solid var(--line-strong);
    box-shadow: inset 0 -3px 8px rgba(0, 0, 0, 0.6);
    transition:
      background 0.35s ease,
      box-shadow 0.35s ease;
  }
  .light.on .bulb {
    background: radial-gradient(circle at 35% 30%, color-mix(in srgb, var(--c) 92%, white), var(--c));
    border-color: color-mix(in srgb, var(--c) 60%, transparent);
    box-shadow:
      0 0 18px 1px color-mix(in srgb, var(--c) 70%, transparent),
      0 0 40px 2px color-mix(in srgb, var(--c) 35%, transparent),
      inset 0 -2px 6px rgba(0, 0, 0, 0.3);
    animation: bulb-in 0.4s ease;
  }
  @keyframes bulb-in {
    0% {
      transform: scale(0.7);
    }
    60% {
      transform: scale(1.12);
    }
    100% {
      transform: scale(1);
    }
  }
  .light-name {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.03em;
  }
  .light-state {
    font-family: var(--font-mono);
    font-size: 8.5px;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .light.on .light-name {
    color: var(--text);
  }
</style>
