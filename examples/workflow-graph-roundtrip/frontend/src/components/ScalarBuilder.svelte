<script>
  import { parseLiteral, encodeLiteral } from '../lib/fields.js';

  // A typed scalar literal editor: [number | text | true·false] + a value input,
  // emitting canonical Lashlang (`3`, `"done"`, `true`). Reused by the value
  // builder, the comparison RHS, and each list item so string values pick up
  // their quotes automatically. Non-scalar values never reach it (the parent
  // routes those to the raw editor).
  let { value = '', onChange, compact = false } = $props();

  let type = $state('string');
  let text = $state('');
  let editing = $state(false);

  $effect(() => {
    if (editing) return;
    const parsed = parseLiteral(value);
    if (parsed.type !== 'expression') {
      type = parsed.type;
      text = parsed.value;
    }
  });

  function emit() {
    onChange?.(encodeLiteral(type, text));
  }
  function onType(next) {
    type = next;
    if (next === 'boolean' && text !== 'true' && text !== 'false') text = 'true';
    if (next === 'number' && !/^-?\d*\.?\d*$/.test(text)) text = '';
    emit();
  }
</script>

<span class="sb" class:compact>
  <select
    class="sb-type"
    value={type}
    onpointerdown={(e) => e.stopPropagation()}
    onchange={(e) => onType(e.currentTarget.value)}
  >
    <option value="number">num</option>
    <option value="string">text</option>
    <option value="boolean">bool</option>
  </select>
  {#if type === 'boolean'}
    <select
      class="sb-val"
      value={text}
      onpointerdown={(e) => e.stopPropagation()}
      onchange={(e) => {
        text = e.currentTarget.value;
        emit();
      }}
    >
      <option value="true">true</option>
      <option value="false">false</option>
    </select>
  {:else}
    <input
      class="sb-val"
      type={type === 'number' ? 'number' : 'text'}
      value={text}
      placeholder={type === 'number' ? '0' : 'text'}
      spellcheck="false"
      onpointerdown={(e) => e.stopPropagation()}
      onfocus={() => (editing = true)}
      oninput={(e) => {
        text = e.currentTarget.value;
        emit();
      }}
      onblur={() => (editing = false)}
    />
  {/if}
</span>

<style>
  .sb {
    display: inline-flex;
    align-items: stretch;
    gap: 3px;
    min-width: 0;
    flex: 1;
  }
  .sb-type,
  .sb-val {
    background: #0a0d13;
    border: 1px solid color-mix(in srgb, var(--accent, var(--cyan)) 26%, var(--line));
    border-radius: 6px;
    color: #eaf2ff;
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 4px 6px;
    min-width: 0;
  }
  .sb-type {
    flex: 0 0 auto;
    color: var(--text-dim);
    cursor: pointer;
  }
  .sb-val {
    flex: 1;
    width: 100%;
  }
  .sb-type:focus,
  .sb-val:focus {
    outline: none;
    border-color: var(--accent, var(--cyan));
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent, var(--cyan)) 14%, transparent);
  }
  .compact .sb-val {
    flex: 1 1 50px;
  }
</style>
