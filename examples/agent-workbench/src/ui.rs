pub const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="data:," />
  <title>agent workbench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@500;600;700;800&family=Chivo+Mono:wght@400;500&family=Spectral:wght@400;500;600;ital,400&display=swap">
  <style>
    :root {
      /* warm-iron neutral ramp · hue ~75 · OKLCH, chroma reduced at extremes */
      --form-deep:     oklch(0.13 0.003 75);
      --form:          oklch(0.16 0.004 75);
      --form-raised:   oklch(0.20 0.005 75);
      --surface-hover: oklch(0.23 0.005 75);
      --ash:           oklch(0.27 0.005 75);
      --ash-mid:       oklch(0.36 0.006 75);
      --ash-text:      oklch(0.64 0.012 80);
      --chalk-dim:     oklch(0.74 0.012 80);
      --chalk-mid:     oklch(0.88 0.014 85);
      --chalk:         oklch(0.94 0.018 90);

      /* signal accents — rare by design */
      --sodium:        oklch(0.74 0.155 65);
      --sodium-soft:   oklch(0.80 0.110 65);
      --lichen:        oklch(0.66 0.060 130);
      --info:          oklch(0.68 0.040 245);
      --error:         oklch(0.58 0.180 25);

      --line:          oklch(0.91 0.018 90 / 0.12);
      --line-strong:   oklch(0.74 0.155 65 / 0.32);
      --line-danger:   oklch(0.58 0.180 25 / 0.45);
      --sodium-tint:   oklch(0.74 0.155 65 / 0.10);
      --error-tint:    oklch(0.58 0.180 25 / 0.12);
      --focus-ring:    0 0 0 2px var(--form), 0 0 0 4px var(--sodium);

      /* 4pt spacing scale */
      --space-3xs: 0.125rem;
      --space-2xs: 0.25rem;
      --space-xs:  0.5rem;
      --space-sm:  0.75rem;
      --space-md:  1rem;
      --space-lg:  1.5rem;
      --space-xl:  2rem;

      --font-display: "Big Shoulders Display", "Helvetica Neue", sans-serif;
      --font-body:    "Spectral", "Iowan Old Style", Georgia, serif;
      --font-ui:      "Chivo Mono", ui-monospace, monospace;

      color-scheme: dark;
    }

    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      margin: 0;
      color: var(--chalk-mid);
      font-family: var(--font-body);
      font-size: clamp(0.95rem, 0.85rem + 0.4vw, 1.2rem);
      line-height: 1.55;
      background:
        repeating-linear-gradient(135deg, oklch(0.94 0.018 90 / 0.012) 0 1px, transparent 1px 8px),
        var(--form-deep);
    }

    button, input, select, textarea { font: inherit; color: inherit; }

    ::selection { background: var(--sodium); color: var(--form-deep); }

    :focus-visible { outline: none; box-shadow: var(--focus-ring); border-radius: 3px; }

    .slash { color: var(--sodium); }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 288px minmax(380px, 1fr) 360px;
    }

    /* ─── left rail ─── */

    .left {
      min-width: 0;
      border-right: 1px solid var(--line);
      background: var(--form);
      display: grid;
      grid-template-rows: auto auto auto 1fr auto;
      gap: var(--space-xl);
      padding: var(--space-lg);
    }

    .brand {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.4rem;
      line-height: 1.05;
      letter-spacing: 0;
      min-width: 0;
    }

    .brand-name { display: block; }

    .brand small {
      display: block;
      margin-top: var(--space-2xs);
      font-family: var(--font-ui);
      font-weight: 400;
      font-size: 0.75rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--ash-text);
    }

    .trigger-bay {
      display: grid;
      justify-items: center;
      gap: var(--space-sm);
    }

    .trigger-buttons {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-xs);
      width: 100%;
    }

    .host-trigger {
      --trigger-color: var(--error);
      --trigger-line: var(--line-danger);
      position: relative;
      width: 100%;
      max-width: 132px;
      aspect-ratio: 1;
      border: 0;
      padding: 0;
      background: none;
      cursor: pointer;
      display: grid;
      place-items: center;
      touch-action: manipulation;
      -webkit-user-select: none;
      user-select: none;
    }

    .host-trigger[data-button="Blue"] {
      --trigger-color: var(--info);
      --trigger-line: oklch(0.68 0.040 245 / 0.58);
    }

    .trigger-face {
      position: absolute;
      inset: 13px;
      border-radius: 50%;
      background: var(--form-raised);
      border: 1px solid var(--trigger-line);
      display: grid;
      place-items: center;
      gap: var(--space-3xs);
      font-family: var(--font-ui);
      transition: transform 120ms cubic-bezier(0.22, 1, 0.36, 1), border-color 160ms ease-out;
    }

    .trigger-face strong {
      font-size: 1.05rem;
      font-weight: 500;
      letter-spacing: 0.16em;
      color: var(--trigger-color);
    }

    .trigger-face span {
      font-size: 0.75rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--ash-text);
    }

    .host-trigger:active .trigger-face { transform: scale(0.97); border-color: var(--sodium); }
    .host-trigger:focus-visible .trigger-face { border-color: var(--sodium); }
    .host-trigger[aria-disabled="true"] { cursor: not-allowed; filter: saturate(0.4) brightness(0.7); }

    .trigger-caption {
      margin: 0;
      max-width: 22ch;
      text-align: center;
      font-family: var(--font-ui);
      font-size: 0.75rem;
      line-height: 1.5;
      color: var(--ash-text);
    }

    .model-config { display: grid; gap: var(--space-md); }

    .field { display: grid; gap: var(--space-2xs); }

    .field span {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--ash-text);
    }

    .field input,
    .field select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--form-deep);
      color: var(--chalk);
      font-family: var(--font-ui);
      font-size: 0.82rem;
      padding: 0 var(--space-sm);
      outline: none;
    }

    .field input:focus-visible,
    .field select:focus-visible { box-shadow: var(--focus-ring); border-color: var(--line-strong); }

    .field input.unknown { border-color: var(--line-danger); }

    .field-hint {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      line-height: 1.4;
      color: var(--error);
    }
    .field-hint[hidden] { display: none; }

    .status-stack {
      align-self: end;
      display: grid;
      gap: var(--space-xs);
      font-family: var(--font-ui);
      font-size: 0.75rem;
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      gap: var(--space-xs);
      padding-top: var(--space-xs);
      border-top: 1px solid var(--line);
    }

    .status-row span:first-child { color: var(--ash-text); letter-spacing: 0.1em; text-transform: uppercase; }
    .status-row span:last-child { color: var(--chalk-dim); }

    /* ─── center ─── */

    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      background: var(--form-deep);
    }

    .topbar {
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-md);
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--line);
    }

    .title-block { min-width: 0; }

    .title {
      margin: 0;
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.6rem;
      line-height: 1;
    }

    .subtitle {
      margin-top: var(--space-2xs);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      color: var(--ash-text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .topbar-right { display: flex; align-items: center; gap: var(--space-sm); }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: var(--space-xs);
      min-height: 30px;
      padding: 0 var(--space-sm);
      border: 1px solid var(--line);
      border-radius: 999px;
      font-family: var(--font-ui);
      font-size: 0.75rem;
      color: var(--ash-text);
      white-space: nowrap;
    }

    .topbar-right { position: relative; }

    .help {
      display: grid;
      place-items: center;
      width: 26px;
      height: 26px;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: transparent;
      font-family: var(--font-ui);
      font-size: 0.75rem;
      color: var(--ash-text);
      cursor: pointer;
    }
    .help:hover,
    .help[aria-expanded="true"] { color: var(--chalk); border-color: var(--line-strong); }

    .help-panel {
      position: absolute;
      top: calc(100% + var(--space-xs));
      right: 0;
      z-index: 20;
      width: 320px;
      max-width: 80vw;
      padding: var(--space-md);
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--form-raised);
      box-shadow: 0 12px 30px oklch(0.06 0.002 75 / 0.55);
    }
    .help-panel[hidden] { display: none; }

    .help-panel h2 {
      margin: 0 0 var(--space-xs);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--sodium);
    }
    .help-panel h2:not(:first-child) { margin-top: var(--space-md); }

    .help-panel dl {
      margin: 0;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: var(--space-2xs) var(--space-sm);
    }
    .help-panel dt {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      color: var(--chalk);
      white-space: nowrap;
    }
    .help-panel dd {
      margin: 0;
      font-size: 0.78rem;
      color: var(--chalk-dim);
    }

    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ash-mid); }
    .pill.run .dot { background: var(--sodium); animation: pulse 1.6s ease-in-out infinite; }

    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    @media (prefers-reduced-motion: reduce) {
      .pill.run .dot,
      .work-card.running .work-dot { animation: none; }
    }

    .stop {
      min-height: 30px;
      padding: 0 var(--space-sm);
      border: 1px solid var(--line-danger);
      border-radius: 4px;
      background: transparent;
      color: var(--error);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.06em;
      cursor: pointer;
    }
    .stop:hover { background: var(--error-tint); }
    .stop[hidden] { display: none; }

    .ghost-btn {
      min-height: 30px;
      padding: 0 var(--space-sm);
      border: 1px solid var(--line);
      border-radius: 4px;
      background: transparent;
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.06em;
      cursor: pointer;
    }
    .ghost-btn:hover { color: var(--chalk); border-color: var(--line-strong); }
    .ghost-btn.armed { color: var(--error); border-color: var(--line-danger); }
    .ghost-btn:disabled { opacity: 0.45; cursor: not-allowed; }

    .timeline {
      min-height: 0;
      overflow: auto;
      padding: var(--space-md) var(--space-lg);
      scroll-behavior: smooth;
    }

    /* editorial transcript — hanging role labels, hairline dividers, no chat bubbles */

    .message {
      display: grid;
      grid-template-columns: 84px minmax(0, 1fr);
      gap: var(--space-md);
      padding: var(--space-md) 0;
      border-bottom: 1px solid var(--line);
    }

    .msg-role {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--ash-text);
      padding-top: var(--space-3xs);
    }

    .message.user .msg-role { color: var(--sodium); }
    .message.trigger .msg-role,
    .message.error .msg-role { color: var(--error); }

    .message.event {
      display: block;
      padding: var(--space-xs) 0;
    }
    .message.event .msg-role { display: none; }
    .message.event .msg-body {
      max-width: none;
      text-align: center;
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      color: var(--ash-text);
    }

    .msg-body {
      max-width: 68ch;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--chalk);
    }

    .message.assistant .msg-body { color: var(--chalk-mid); }

    .retry {
      display: inline-block;
      margin-top: var(--space-xs);
      border: 1px solid var(--line-strong);
      border-radius: 4px;
      background: transparent;
      color: var(--sodium);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      padding: var(--space-2xs) var(--space-sm);
      cursor: pointer;
    }
    .retry:hover { background: var(--sodium-tint); }

    .note {
      padding: var(--space-sm) 0;
      border-bottom: 1px solid var(--line);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      color: var(--ash-text);
      text-align: center;
    }

    /* machine-activity panels — restrained, mono, collapsible */

    .reasoning,
    .code-block,
    .tool {
      margin: var(--space-sm) 0;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--form);
      font-family: var(--font-ui);
      font-size: 0.78rem;
      color: var(--chalk-dim);
    }

    .reasoning,
    .code-block { padding: var(--space-sm) var(--space-md); }

    .reasoning summary,
    .code-block summary,
    .tool summary {
      cursor: pointer;
      color: var(--info);
      letter-spacing: 0.04em;
    }
    .reasoning summary { color: var(--sodium-soft); }
    .reasoning summary:hover,
    .code-block summary:hover,
    .tool summary:hover { color: var(--chalk); }

    .reasoning pre,
    .code-block pre,
    .tool pre {
      margin: var(--space-xs) 0 0;
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--form-deep);
      color: var(--chalk-mid);
      padding: var(--space-sm);
      font-size: 0.75rem;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .code-block.fail { border-color: var(--line-danger); }
    .code-block.fail summary,
    .tool.fail strong { color: var(--error); }

    .tool {
      display: grid;
      gap: var(--space-xs);
      padding: var(--space-sm) var(--space-md);
    }

    .tool-head { display: flex; align-items: center; gap: var(--space-xs); flex-wrap: wrap; }
    .tool strong { color: var(--info); font-weight: 500; }

    .tool .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: var(--space-3xs) var(--space-xs);
      color: var(--ash-text);
      background: var(--form-deep);
      font-size: 0.75rem;
      letter-spacing: 0.06em;
    }

    .tool-head span:last-child { color: var(--ash-text); font-size: 0.75rem; }
    .tool-summary { color: var(--chalk); overflow-wrap: anywhere; }
    .tool details { border-top: 1px solid var(--line); padding-top: var(--space-xs); }

    .composer {
      border-top: 1px solid var(--line);
      padding: var(--space-md) var(--space-lg) var(--space-lg);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: var(--space-sm);
    }

    .composer textarea {
      width: 100%;
      min-height: 56px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--form);
      color: var(--chalk);
      font-family: var(--font-body);
      padding: var(--space-sm) var(--space-md);
      outline: none;
      line-height: 1.45;
    }

    .composer textarea:focus-visible { box-shadow: var(--focus-ring); border-color: var(--line-strong); }

    .send {
      min-width: 96px;
      align-self: end;
      min-height: 56px;
      border: 0;
      border-radius: 4px;
      background: var(--sodium);
      color: var(--form-deep);
      font-family: var(--font-ui);
      font-weight: 500;
      letter-spacing: 0.06em;
      cursor: pointer;
    }
    .send:hover { background: var(--sodium-soft); }
    .send:disabled { background: var(--ash); color: var(--ash-text); cursor: not-allowed; }

    /* ─── right rail · work ledger ─── */

    .right {
      min-width: 0;
      border-left: 1px solid var(--line);
      background: var(--form);
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }

    .rail-head {
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-sm);
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--line);
    }

    .rail-title { font-family: var(--font-display); font-weight: 700; font-size: 1.25rem; }
    .rail-count { font-family: var(--font-ui); font-size: 0.75rem; color: var(--ash-text); }
    .rail-count.stale { color: var(--error); cursor: pointer; }

    .work-list { min-height: 0; overflow: auto; padding: 0 var(--space-lg); }

    .work-card {
      display: grid;
      gap: var(--space-2xs);
      padding: var(--space-md) 0;
      border-bottom: 1px solid var(--line);
    }

    .work-top { display: flex; align-items: center; gap: var(--space-xs); min-width: 0; }

    .work-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ash-mid); flex: none; }
    .work-card.running .work-dot { background: var(--sodium); animation: pulse 1.6s ease-in-out infinite; }
    .work-card.succeeded .work-dot,
    .work-card.completed .work-dot,
    .work-card.done .work-dot { background: var(--lichen); }
    .work-card.failed .work-dot,
    .work-card.cancelled .work-dot { background: var(--error); }

    .work-label {
      min-width: 0;
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--chalk);
      font-size: 0.92rem;
    }

    .work-state {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ash-text);
      white-space: nowrap;
    }

    .work-meta,
    .work-events {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      line-height: 1.5;
      color: var(--ash-text);
      overflow-wrap: anywhere;
    }

    .empty {
      margin: var(--space-md) 0;
      border: 1px dashed var(--line);
      border-radius: 4px;
      color: var(--ash-text);
      padding: var(--space-lg);
      text-align: center;
      font-family: var(--font-ui);
      font-size: 0.78rem;
      line-height: 1.6;
    }

    @media (max-width: 1060px) {
      .shell { grid-template-columns: 248px minmax(320px, 1fr); }
      .brand { font-size: 1.2rem; }
      .right {
        grid-column: 1 / -1;
        min-height: 320px;
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }

    @media (max-width: 720px) {
      .shell { display: flex; flex-direction: column; }
      .left { grid-template-rows: auto auto auto auto auto; }
      .composer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="left">
      <div class="brand">
        <span class="brand-name">agent<span class="slash">/</span>workbench</span>
        <small>rlm runtime console</small>
      </div>

      <div class="trigger-bay">
        <div class="trigger-buttons">
          <button class="host-trigger" type="button" data-trigger-button data-button="Red"
                  aria-label="Fire the red button event"
                  title="Emit the host-owned button event.">
            <span class="trigger-face">
              <strong>RED</strong>
              <span>fire</span>
            </span>
          </button>
          <button class="host-trigger" type="button" data-trigger-button data-button="Blue"
                  aria-label="Fire the blue button event"
                  title="Emit the host-owned button event.">
            <span class="trigger-face">
              <strong>BLUE</strong>
              <span>fire</span>
            </span>
          </button>
        </div>
        <p class="trigger-caption">click a dial to emit its button event</p>
      </div>

      <form class="model-config" id="modelConfig">
        <label class="field">
          <span>model</span>
          <input id="modelInput" list="modelList" autocomplete="off" spellcheck="false"
                 title="Provider model slug, e.g. openai/gpt-5.5. Applies to the next turn." />
          <datalist id="modelList"></datalist>
          <small id="modelHint" class="field-hint" hidden></small>
        </label>
        <label class="field">
          <span>thinking</span>
          <select id="variantSelect" title="Reasoning effort for the next turn."></select>
        </label>
      </form>

      <div></div>

      <div class="status-stack">
        <div class="status-row"><span>model</span><span id="modelName">loading</span></div>
        <div class="status-row"><span>web</span><span id="webState">loading</span></div>
        <div class="status-row"><span>session</span><span id="sessionId">—</span></div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div class="title-block">
          <h1 class="title">chat</h1>
          <div class="subtitle" id="streamState">ready</div>
        </div>
        <div class="topbar-right">
          <button id="reset" class="ghost-btn" type="button" title="Clear the conversation and reset the session">reset</button>
          <button id="stop" class="stop" type="button" hidden title="Cancel the running turn (Esc).">stop turn</button>
          <span id="busyPill" class="pill"><span id="busyDot" class="dot"></span><span id="busyText" aria-live="polite">idle</span></span>
          <button id="helpBtn" class="help" type="button" aria-expanded="false" aria-controls="helpPanel"
                  aria-label="Help and keyboard shortcuts">?</button>
          <div id="helpPanel" class="help-panel" role="region" aria-label="Workbench help" tabindex="-1" hidden>
            <h2>keyboard</h2>
            <dl>
              <dt>enter</dt><dd>send the turn</dd>
              <dt>shift + enter</dt><dd>insert a newline</dd>
              <dt>esc</dt><dd>stop the running turn</dd>
              <dt>↑ (empty composer)</dt><dd>recall your last prompt</dd>
              <dt>click a dial</dt><dd>emit the host event</dd>
            </dl>
            <h2>what's what</h2>
            <dl>
              <dt>trigger</dt><dd>fires a host-owned turn — a scheduled or external event, not your prompt</dd>
              <dt>work</dt><dd>background processes (sub-agents and tools) the turn spawned</dd>
              <dt>model · thinking</dt><dd>apply to the next turn you send</dd>
            </dl>
          </div>
        </div>
      </header>

      <section id="timeline" class="timeline" aria-busy="false">
        <div id="timelineEmpty" class="empty">
          no turns yet. ask the agent something below, or click a dial to fire the host event.
        </div>
      </section>

      <form id="composer" class="composer">
        <textarea id="prompt" rows="2" placeholder="ask the agent — enter to send, shift + enter for a newline"></textarea>
        <button id="send" class="send" type="submit">send</button>
      </form>
    </main>

    <aside class="right">
      <div class="rail-head">
        <div class="rail-title">work</div>
        <div id="workCount" class="rail-count">0 processes</div>
      </div>
      <div id="workList" class="work-list"></div>
    </aside>
  </div>

  <script>
    const timeline = document.getElementById("timeline");
    const timelineEmpty = document.getElementById("timelineEmpty");
    const workList = document.getElementById("workList");
    const workCount = document.getElementById("workCount");
    const composer = document.getElementById("composer");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const triggerButtons = Array.from(document.querySelectorAll("[data-trigger-button]"));
    const modelInput = document.getElementById("modelInput");
    const variantSelect = document.getElementById("variantSelect");
    const streamState = document.getElementById("streamState");
    const busyPill = document.getElementById("busyPill");
    const busyText = document.getElementById("busyText");
    const modelName = document.getElementById("modelName");
    const webState = document.getElementById("webState");
    const sessionId = document.getElementById("sessionId");

    const renderedMessages = new Set();
    let assistantDraft = null;
    let reasoning = null;
    let pendingCodeBlock = null;
    let pendingTools = [];
    let busy = false;
    let controller = null;
    let lastRequest = null;
    let workStale = false;
    let turnTimer = 0;
    let turnStart = 0;

    function clearEmpty() {
      const e = document.getElementById("timelineEmpty");
      if (e) e.remove();
    }

    function modelEmpty() { return !modelInput.value.trim(); }

    function setBusy(next, label) {
      busy = next;
      sendButton.disabled = next || modelEmpty();
      stopButton.hidden = true;
      for (const button of triggerButtons) button.setAttribute("aria-disabled", String(next));
      busyPill.classList.toggle("run", next);
      busyText.textContent = next ? "running" : "idle";
      timeline.setAttribute("aria-busy", String(next));
      clearInterval(turnTimer);
      turnTimer = 0;
      if (next) {
        turnStart = performance.now();
        streamState.textContent = label || "turn running";
        turnTimer = setInterval(() => {
          const secs = Math.floor((performance.now() - turnStart) / 1000);
          streamState.textContent = "running · " + secs + "s";
        }, 1000);
      } else {
        streamState.textContent = label || "ready";
      }
    }

    function scrollToEnd() {
      timeline.scrollTop = timeline.scrollHeight;
    }

    function roleLabel(role) {
      if (role === "user") return "you";
      if (role === "assistant") return "agent";
      if (role === "trigger") return "trigger";
      if (role === "event") return "event";
      return role;
    }

    function renderMessage(message) {
      if (renderedMessages.has(message.id)) return;
      renderedMessages.add(message.id);
      clearEmpty();
      if (message.role === "assistant" && assistantDraft) {
        const parent = assistantDraft.closest(".message");
        if (parent) parent.remove();
        assistantDraft = null;
      }
      const node = document.createElement("div");
      node.className = "message " + message.role;
      const role = document.createElement("div");
      role.className = "msg-role";
      role.textContent = roleLabel(message.role);
      const body = document.createElement("div");
      body.className = "msg-body";
      body.textContent = message.text;
      node.append(role, body);
      timeline.appendChild(node);
      scrollToEnd();
    }

    function cleanErrorText(message) {
      let text = String(message || "request failed").trim();
      if (/<!doctype|<html/i.test(text)) text = "the server returned an error page (not a normal response)";
      return text.length > 280 ? text.slice(0, 280) + "…" : text;
    }

    function renderError(message, opts) {
      clearEmpty();
      const node = document.createElement("div");
      node.className = "message error";
      const role = document.createElement("div");
      role.className = "msg-role";
      role.textContent = "error";
      const body = document.createElement("div");
      body.className = "msg-body";
      body.textContent = cleanErrorText(message);
      if (opts && opts.retry && lastRequest) {
        const retry = document.createElement("button");
        retry.className = "retry";
        retry.type = "button";
        retry.textContent = "retry turn";
        retry.addEventListener("click", () => postCommand(lastRequest.url, lastRequest.payload));
        body.append(document.createElement("br"), retry);
      }
      node.append(role, body);
      timeline.appendChild(node);
      scrollToEnd();
    }

    function renderNote(text) {
      clearEmpty();
      const node = document.createElement("div");
      node.className = "note";
      node.textContent = text;
      timeline.appendChild(node);
      scrollToEnd();
    }

    function ensureAssistantDraft() {
      if (assistantDraft) return assistantDraft;
      clearEmpty();
      const node = document.createElement("div");
      node.className = "message assistant";
      const role = document.createElement("div");
      role.className = "msg-role";
      role.textContent = "agent";
      assistantDraft = document.createElement("div");
      assistantDraft.className = "msg-body";
      node.append(role, assistantDraft);
      timeline.appendChild(node);
      return assistantDraft;
    }

    function selectedModelPayload() {
      return {
        model: modelInput.value.trim(),
        model_variant: variantSelect.value
      };
    }

    const STATUS_LABELS = {
      running: "running",
      pending: "queued",
      queued: "queued",
      succeeded: "done",
      completed: "done",
      done: "done",
      failed: "failed",
      cancelled: "stopped",
      canceled: "stopped"
    };

    const EVENT_LABELS = {
      reasoning_delta: "thinking",
      assistant_prose_delta: "writing reply",
      queued_work_started: "queued work started",
      tool_call_started: "tool started",
      tool_call_completed: "tool finished",
      code_block_started: "code started",
      code_block_completed: "code finished",
      submitted_value: "result submitted",
      tool_value: "tool result",
      child_usage: "sub-agent usage",
      retry_status: "retrying",
      error: "error",
      started: "started",
      completed: "finished"
    };

    function humanize(token) {
      return String(token || "").replaceAll("_", " ").trim();
    }

    function eventLabel(type) {
      return EVENT_LABELS[type] || humanize(type) || "update";
    }

    function statusLabel(terminal) {
      return STATUS_LABELS[terminal] || humanize(terminal) || "unknown";
    }

    function kindLabel(kind) {
      if (kind === "subagent") return "sub-agent";
      return humanize(kind);
    }

    function renderTerminalValue(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function appendAssistantText(delta) {
      if (!delta) return;
      ensureAssistantDraft().textContent += delta;
      scrollToEnd();
    }

    function toolOutcome(event) {
      return event?.output?.outcome || {};
    }

    function toolResult(event) {
      const outcome = toolOutcome(event);
      return outcome.payload ?? event?.output?.value ?? event?.result ?? null;
    }

    function toolSucceeded(event) {
      return toolOutcome(event).status === "success" || event?.success === true;
    }

    function cleanArgs(args) {
      const out = { ...(args || {}) };
      delete out.__session_id__;
      return out;
    }

    function compactToolPayload(event) {
      return {
        args: cleanArgs(event.args),
        status: toolOutcome(event).status || (toolSucceeded(event) ? "success" : "unknown"),
        result: toolResult(event)
      };
    }

    function displayToolName(name) {
      if (name === "search_web") return "web.search";
      if (name === "fetch_url") return "web.fetch";
      if (name === "spawn_agent") return "agents.spawn";
      return name || "tool";
    }

    function appendTool(event, parent = timeline) {
      if (parent === timeline) clearEmpty();
      const ok = toolSucceeded(event);
      const el = document.createElement("div");
      el.className = "tool" + (ok ? "" : " fail");
      el.innerHTML = `<div class="tool-head"><strong></strong><span class="badge"></span><span></span></div><div class="tool-summary"></div><details><summary>JSON payload</summary><pre></pre></details>`;
      el.querySelector("strong").textContent = displayToolName(event.name);
      el.querySelector(".badge").textContent = ok ? "completed" : "failed";
      el.querySelector(".tool-head span:last-child").textContent = `${ok ? "ok" : "failed"} in ${event.duration_ms || 0}ms`;
      const result = toolResult(event);
      el.querySelector(".tool-summary").textContent = ok
        ? summarizeToolResult(event.name, result)
        : summarizeToolFailure(result);
      el.querySelector("pre").textContent = JSON.stringify(compactToolPayload(event), null, 2);
      parent.appendChild(el);
      scrollToEnd();
    }

    function summarizeToolResult(name, result) {
      if (name === "search_web") {
        const count = Array.isArray(result?.results) ? result.results.length : 0;
        return `Search completed · ${count} result${count === 1 ? "" : "s"}`;
      }
      if (name === "fetch_url") {
        const chars = typeof result?.content === "string" ? result.content.length : 0;
        return `Fetched ${result?.url || "URL"} · ${chars} chars`;
      }
      if (name === "spawn_agent") {
        return result?.summary || result?.session_id || "Subagent completed";
      }
      return "Tool completed";
    }

    function summarizeToolFailure(result) {
      if (typeof result === "string") return result;
      return result?.message || result?.error || "Tool failed";
    }

    function appendCodeBlock(event, linkedTools = []) {
      clearEmpty();
      const el = document.createElement("details");
      el.className = "code-block" + (event.success === false ? " fail" : "");
      el.open = false;
      el.innerHTML = "<summary></summary><pre></pre>";
      const toolCount = linkedTools.length || (event.tool_call_ids || []).length;
      const toolLabel = toolCount ? ` · ${toolCount} tool${toolCount === 1 ? "" : "s"}` : "";
      el.querySelector("summary").textContent = `${event.language || "code"} ${event.success ? "completed" : "failed"} in ${event.duration_ms || 0}ms${toolLabel}`;
      el.querySelector("pre").textContent = event.code || "";
      for (const tool of linkedTools) appendTool(tool, el);
      timeline.appendChild(el);
      scrollToEnd();
    }

    function appendCompletedTool(event) {
      if (pendingCodeBlock) {
        pendingTools.push(event);
      } else {
        appendTool(event);
      }
    }

    function completeCodeBlock(event) {
      const linkedIds = new Set(event.tool_call_ids || []);
      const linkedTools = pendingTools.filter(tool => tool.call_id && linkedIds.has(tool.call_id));
      const unlinkedTools = pendingTools.filter(tool => !tool.call_id || !linkedIds.has(tool.call_id));
      appendCodeBlock(
        { ...event, code: event.code || pendingCodeBlock?.code || "" },
        linkedTools
      );
      pendingCodeBlock = null;
      for (const tool of unlinkedTools) appendTool(tool);
      pendingTools = [];
    }

    function thinkingPanel(label) {
      const el = document.createElement("details");
      el.className = "reasoning";
      el.open = true;
      el.innerHTML = "<summary></summary><pre></pre>";
      el.querySelector("summary").textContent = label;
      return el;
    }

    function appendReasoning(delta) {
      if (!reasoning) {
        clearEmpty();
        reasoning = thinkingPanel("thinking");
        timeline.appendChild(reasoning);
      }
      reasoning.querySelector("pre").textContent += delta;
      scrollToEnd();
    }

    function renderEventRow(label, detail) {
      clearEmpty();
      const node = document.createElement("div");
      node.className = "message event";
      const role = document.createElement("div");
      role.className = "msg-role";
      role.textContent = "event";
      const body = document.createElement("div");
      body.className = "msg-body";
      body.textContent = detail ? `${label} · ${detail}` : label;
      node.append(role, body);
      timeline.appendChild(node);
      scrollToEnd();
    }

    function renderQueuedWorkStarted(event) {
      const causes = event.causes || [];
      const wake = causes.find(cause => cause.event_type === "process.wake");
      if (wake) {
        renderEventRow("agent woken", wake.text || "process.wake");
      } else {
        const count = (event.batch_ids || []).length;
        renderEventRow("queued turn started", count ? `${count} batch${count === 1 ? "" : "es"}` : "");
      }
      setBusy(true, "agent running");
    }

    function finishTransientRows() {
      for (const tool of pendingTools) appendTool(tool);
      pendingCodeBlock = null;
      pendingTools = [];
      reasoning = null;
    }

    function handleTurnEvent(event) {
      if (event.type === "queued_work_started") renderQueuedWorkStarted(event);
      if (event.type === "assistant_prose_delta") appendAssistantText(event.text);
      if (event.type === "reasoning_delta") appendReasoning(event.text);
      if (event.type === "code_block_started") pendingCodeBlock = event;
      if (event.type === "code_block_completed") completeCodeBlock(event);
      if (event.type === "tool_call_completed") appendCompletedTool(event);
      if (event.type === "submitted_value") appendAssistantText(renderTerminalValue(event.value));
      if (event.type === "tool_value") appendAssistantText(renderTerminalValue(event.value));
      if (event.type === "error") renderError(event.message, { retry: true });
    }

    async function postCommand(url, payload) {
      if (busy) return;
      lastRequest = { url, payload };
      controller = new AbortController();
      assistantDraft = null;
      reasoning = null;
      pendingCodeBlock = null;
      pendingTools = [];
      setBusy(true, "starting turn");
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload ? JSON.stringify(payload) : "{}",
          signal: controller.signal
        });
        if (!response.ok) {
          const text = await response.text();
          renderError(text || "request failed", { retry: true });
          setBusy(false, "ready");
          return;
        }
      } catch (error) {
        if (error.name === "AbortError") {
          renderNote("turn stopped");
        } else {
          renderError(error.message || String(error), { retry: true });
        }
        setBusy(false, "ready");
      } finally {
        controller = null;
        refreshWork();
      }
    }

    async function connectEvents() {
      while (true) {
        try {
          const response = await fetch("/api/events", { cache: "no-store" });
          if (!response.ok || !response.body) throw new Error("event stream failed (" + response.status + ")");
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) handleStreamLine(line);
          }
          handleStreamLine(buffer);
        } catch (error) {
          renderNote("event stream reconnecting");
        }
        await new Promise(resolve => setTimeout(resolve, 900));
      }
    }

    function handleStreamLine(line) {
      if (!line.trim()) return;
      let item;
      try {
        item = JSON.parse(line);
      } catch (_) {
        renderNote("skipped a malformed update from the agent");
        return;
      }
      handleStreamItem(item);
    }

    function handleStreamItem(item) {
      if (item.type === "message") renderMessage(item.message);
      if (item.type === "event") handleTurnEvent(item.event);
      if (item.type === "error") renderError(item.message, { retry: true });
      if (item.type === "done") {
        finishTransientRows();
        setBusy(false, "ready");
      }
    }

    function stopTurn() { if (controller) controller.abort(); }
    stopButton.addEventListener("click", stopTurn);

    /* reset — replaces the server-side session and clears the transcript, with a two-step confirm */
    const resetButton = document.getElementById("reset");
    let resetArmed = false;
    let resetArmTimer = 0;

    function disarmReset() {
      clearTimeout(resetArmTimer);
      resetArmTimer = 0;
      resetArmed = false;
      resetButton.classList.remove("armed");
      resetButton.textContent = "reset";
    }

    function clearTranscript() {
      timeline.innerHTML = "";
      renderedMessages.clear();
      assistantDraft = null;
      reasoning = null;
      pendingCodeBlock = null;
      pendingTools = [];
      lastUserText = "";
      const empty = document.createElement("div");
      empty.id = "timelineEmpty";
      empty.className = "empty";
      empty.textContent = "no turns yet. ask the agent something below, or click a dial to fire the host event.";
      timeline.appendChild(empty);
    }

    async function doReset() {
      resetButton.disabled = true;
      try {
        const response = await fetch("/api/reset", { method: "POST" });
        if (!response.ok) throw new Error("reset failed (" + response.status + ")");
        const state = await response.json();
        clearTranscript();
        lastRequest = null;
        sessionId.textContent = state.settings.session_id || "—";
        renderWork([]);
        setBusy(false, "ready");
      } catch (error) {
        renderError(cleanErrorText(error.message || error));
      } finally {
        resetButton.disabled = false;
      }
    }

    resetButton.addEventListener("click", () => {
      if (busy) return;
      if (!resetArmed) {
        resetArmed = true;
        resetButton.classList.add("armed");
        resetButton.textContent = "confirm reset?";
        resetArmTimer = setTimeout(disarmReset, 3000);
        return;
      }
      disarmReset();
      doReset();
    });

    const helpBtn = document.getElementById("helpBtn");
    const helpPanel = document.getElementById("helpPanel");
    function setHelp(open) {
      helpPanel.hidden = !open;
      helpBtn.setAttribute("aria-expanded", String(open));
      if (open) helpPanel.focus();
    }
    helpBtn.addEventListener("click", () => setHelp(helpPanel.hidden));
    document.addEventListener("click", event => {
      if (!helpPanel.hidden && !helpPanel.contains(event.target) && event.target !== helpBtn) setHelp(false);
    });

    window.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      if (!helpPanel.hidden) { setHelp(false); helpBtn.focus(); return; }
      if (busy) { event.preventDefault(); stopTurn(); }
    });

    let lastUserText = "";

    composer.addEventListener("submit", event => {
      event.preventDefault();
      const text = promptInput.value.trim();
      if (!text) return;
      if (modelEmpty()) { validateModel(); modelInput.focus(); return; }
      lastUserText = text;
      promptInput.value = "";
      postCommand("/api/turn", { text, ...selectedModelPayload() });
    });

    promptInput.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        composer.requestSubmit();
        return;
      }
      if (event.key === "ArrowUp" && !promptInput.value && lastUserText) {
        event.preventDefault();
        promptInput.value = lastUserText;
      }
    });

    function fireTrigger(event) {
      const button = event.currentTarget;
      if (busy || button.getAttribute("aria-disabled") === "true") return;
      postCommand("/api/button-trigger", { button: button.dataset.button });
    }

    for (const button of triggerButtons) {
      button.addEventListener("click", fireTrigger);
    }

    const knownModels = new Set();
    const modelHint = document.getElementById("modelHint");

    function validateModel() {
      const value = modelInput.value.trim();
      const empty = !value;
      const unknown = value && knownModels.size && !knownModels.has(value);
      modelInput.classList.toggle("unknown", empty || unknown);
      modelHint.hidden = !(empty || unknown);
      if (empty) modelHint.textContent = "a model is required before sending a turn";
      else if (unknown) modelHint.textContent = "unrecognized model — it will still be sent as typed";
      sendButton.disabled = busy || empty;
      modelName.textContent = [value, variantSelect.value].filter(Boolean).join(" / ") || "—";
    }

    async function loadState() {
      let state;
      try {
        const response = await fetch("/api/state");
        if (!response.ok) throw new Error("state request failed (" + response.status + ")");
        state = await response.json();
      } catch (error) {
        modelName.textContent = "unavailable";
        webState.textContent = "unavailable";
        sessionId.textContent = "—";
        streamState.textContent = "could not load session";
        renderError("couldn't load the workbench session: " + (error.message || error) +
          ". check the server is running, then reload.");
        throw error;
      }
      modelInput.value = state.settings.default_model || "";
      const modelList = document.getElementById("modelList");
      modelList.innerHTML = "";
      knownModels.clear();
      const known = state.settings.models || (state.settings.default_model ? [state.settings.default_model] : []);
      for (const slug of known) {
        knownModels.add(slug);
        const option = document.createElement("option");
        option.value = slug;
        modelList.appendChild(option);
      }
      variantSelect.innerHTML = "";
      for (const variant of state.settings.model_variants || ["low", "medium", "high"]) {
        const option = document.createElement("option");
        option.value = variant;
        option.textContent = variant;
        if (variant === state.settings.default_model_variant) option.selected = true;
        variantSelect.appendChild(option);
      }
      validateModel();
      modelInput.addEventListener("input", validateModel);
      variantSelect.addEventListener("change", validateModel);
      webState.textContent = state.settings.web_configured ? "ready" : "not configured";
      sessionId.textContent = state.settings.session_id;
      for (const message of state.messages) renderMessage(message);
    }

    function formatTime(ms) {
      if (!ms) return "";
      return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function renderWork(items) {
      workList.innerHTML = "";
      workStale = false;
      workCount.classList.remove("stale");
      workCount.title = "";
      workCount.textContent = items.length + (items.length === 1 ? " process" : " processes");
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "no visible processes";
        workList.appendChild(empty);
        return;
      }
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "work-card " + item.terminal;
        card.title = "process " + item.process_id;

        const top = document.createElement("div");
        top.className = "work-top";
        const dot = document.createElement("span");
        dot.className = "work-dot";
        const label = document.createElement("div");
        label.className = "work-label";
        label.title = item.label;
        label.textContent = item.label;
        const state = document.createElement("div");
        state.className = "work-state";
        state.textContent = statusLabel(item.terminal);
        top.append(dot, label, state);

        const meta = document.createElement("div");
        meta.className = "work-meta";
        meta.textContent = [kindLabel(item.kind), formatTime(item.updated_at_ms)]
          .filter(Boolean).join(" · ");

        const events = document.createElement("div");
        events.className = "work-events";
        const recent = item.events.slice(-3).map(event => eventLabel(event.event_type));
        events.textContent = recent.length ? "latest: " + recent.join(" → ") : "no events yet";

        card.append(top, meta, events);
        workList.appendChild(card);
      }
    }

    async function refreshWork() {
      try {
        const response = await fetch("/api/work");
        if (!response.ok) throw new Error("work request failed");
        renderWork(await response.json());
      } catch (_) {
        if (!workStale) {
          workStale = true;
          workCount.textContent = "updates paused — retry";
          workCount.title = "couldn't reach the process registry. polling continues; click to retry now.";
          workCount.classList.add("stale");
        }
      }
    }

    workCount.addEventListener("click", () => { if (workStale) refreshWork(); });

    loadState().catch(() => {}).finally(() => {
      refreshWork();
      connectEvents();
    });
    setInterval(refreshWork, 1400);
  </script>
</body>
</html>
"##;
