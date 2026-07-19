<script>
  // Small, dependency-free tool marks. They deliberately share one stroke
  // language so integrations read as a family while remaining recognizable.
  let { kind = 'action' } = $props();
</script>

{#if kind === 'slack'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M9.3 3.2a2 2 0 0 1 2 2v4.1H9.3V5.2a2 2 0 0 1 0-2Z" />
    <path d="M3.2 9.3a2 2 0 0 1 2-2h4.1v2H5.2a2 2 0 0 1-2 0Z" />
    <path d="M14.7 20.8a2 2 0 0 1-2-2v-4.1h2v4.1a2 2 0 0 1 0 2Z" />
    <path d="M20.8 14.7a2 2 0 0 1-2 2h-4.1v-2h4.1a2 2 0 0 1 2 0Z" />
    <path d="M12.7 5.2a2 2 0 1 1 4 0v2.1h2.1a2 2 0 1 1 0 4h-6.1V5.2Z" />
    <path d="M11.3 18.8a2 2 0 1 1-4 0v-2.1H5.2a2 2 0 1 1 0-4h6.1v6.1Z" />
  </svg>
{:else if kind === 'github'}
  <svg class="filled" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 2.2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.9c-2.8.6-3.4-1.2-3.4-1.2-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 0 1.6 1 1.6 1 .9 1.6 2.4 1.1 3 .9.1-.7.4-1.1.7-1.4-2.3-.3-4.7-1.1-4.7-5a3.9 3.9 0 0 1 1-2.7c-.1-.3-.5-1.3.1-2.7 0 0 .9-.3 2.8 1a9.7 9.7 0 0 1 5.1 0c2-1.3 2.8-1 2.8-1 .6 1.4.2 2.4.1 2.7a3.9 3.9 0 0 1 1 2.7c0 3.9-2.4 4.7-4.7 5 .4.3.7.9.7 1.8v2.9c0 .3.2.6.7.5A10 10 0 0 0 12 2.2Z" />
  </svg>
{:else if kind === 'email'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M3.5 6.5h17v11h-17z" />
    <path d="m4 7 8 6 8-6" />
    <path d="m4 17 5.7-5M20 17l-5.7-5" />
  </svg>
{:else if kind === 'web'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <circle cx="10.5" cy="10.5" r="6.5" />
    <path d="M4.4 10.5h12.2M10.5 4c1.8 1.8 2.7 4 2.7 6.5s-.9 4.7-2.7 6.5c-1.8-1.8-2.7-4-2.7-6.5S8.7 5.8 10.5 4ZM15.3 15.3 21 21" />
  </svg>
{:else if kind === 'llm'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 2.8c.6 4.2 3 6.6 7.2 7.2-4.2.6-6.6 3-7.2 7.2-.6-4.2-3-6.6-7.2-7.2 4.2-.6 6.6-3 7.2-7.2Z" />
    <path d="M18.4 15.8c.3 1.9 1.4 3 3.3 3.3-1.9.3-3 1.4-3.3 3.3-.3-1.9-1.4-3-3.3-3.3 1.9-.3 3-1.4 3.3-3.3Z" />
  </svg>
{:else if kind === 'agent'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 3v3M9.5 3h5" />
    <rect x="4" y="6" width="16" height="13" rx="3" />
    <path d="M4 11H2.5v4H4M20 11h1.5v4H20M8 16h8" />
    <circle cx="9" cy="11" r="1" />
    <circle cx="15" cy="11" r="1" />
  </svg>
{:else if kind === 'message'}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 4.5h16v12H9l-5 4v-16Z" />
    <path d="M8 9h8M8 12.5h5" />
  </svg>
{:else}
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="m13.5 2.8-8 11.1h6l-1 7.3 8-11.1h-6l1-7.3Z" />
  </svg>
{/if}

<style>
  svg {
    display: block;
    width: 20px;
    height: 20px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.7;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .filled {
    fill: currentColor;
    stroke: none;
  }
</style>
