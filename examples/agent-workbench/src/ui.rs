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
      font-size: 16px;
      line-height: 1.55;
      background:
        linear-gradient(180deg, oklch(0.20 0.005 75 / 0.36), transparent 38vh),
        var(--form-deep);
    }

    button, input, select, textarea { font: inherit; color: inherit; }

    ::selection { background: var(--sodium); color: var(--form-deep); }

    :focus-visible { outline: none; box-shadow: var(--focus-ring); border-radius: 3px; }

    .slash { color: var(--sodium); }

    .shell {
      height: 100dvh;
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-areas: "left main right";
      grid-template-columns: minmax(280px, 320px) minmax(360px, 1fr) minmax(280px, 360px);
    }

    /* ─── left rail ─── */

    .left {
      grid-area: left;
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--line);
      background: var(--form);
      display: grid;
      grid-template-rows: auto auto auto 1fr auto;
      gap: var(--space-xl);
      padding: var(--space-lg);
      overflow: auto;
    }

    .left > * { min-width: 0; }

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
      max-width: 13rem;
    }

    .host-trigger {
      --trigger-color: var(--error);
      --trigger-line: var(--line-danger);
      position: relative;
      width: min(100%, 7rem);
      aspect-ratio: 1;
      border: 0;
      padding: 0;
      background: none;
      cursor: pointer;
      display: grid;
      place-items: center;
      justify-self: center;
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
      inset: 0.625rem;
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
      font-size: 0.98rem;
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

    .schedule-card {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(135deg, var(--sodium-tint), transparent 58%),
        var(--form-raised);
      padding: var(--space-sm);
      display: grid;
      gap: var(--space-2xs);
    }

    .schedule-card-label,
    .schedule-card-code {
      font-family: var(--font-ui);
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .schedule-card-label {
      color: var(--sodium-soft);
      font-size: 0.68rem;
    }

    .schedule-card-code {
      color: var(--chalk);
      font-size: 0.82rem;
      letter-spacing: 0.04em;
      text-transform: none;
      overflow-wrap: anywhere;
    }

    .schedule-card p {
      margin: 0;
      color: var(--chalk-dim);
      font-size: 0.88rem;
      line-height: 1.45;
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
      align-items: baseline;
      gap: var(--space-xs);
      padding-top: var(--space-xs);
      border-top: 1px solid var(--line);
    }

    .status-row span:first-child {
      color: var(--ash-text);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      flex: none;
    }
    .status-row span:last-child {
      min-width: 0;
      color: var(--chalk-dim);
      text-align: right;
      overflow-wrap: anywhere;
    }

    /* ─── center ─── */

    .main {
      grid-area: main;
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      background: var(--form-deep);
    }

    .main.explorer-open {
      grid-template-rows: auto minmax(120px, 0.78fr) minmax(360px, 1.22fr) auto;
    }

    .topbar {
      min-width: 0;
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

    .topbar-right {
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: var(--space-sm);
      flex-wrap: wrap;
    }

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
      box-shadow: 0 6px 8px oklch(0.06 0.002 75 / 0.45);
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

    .composer textarea::placeholder {
      color: var(--ash-text);
      opacity: 1;
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
      grid-area: right;
      min-width: 0;
      border-left: 1px solid var(--line);
      background: var(--form);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-height: 0;
    }

    .right > * { min-width: 0; }

    .rail-head {
      min-width: 0;
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-sm);
      flex-wrap: wrap;
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--line);
    }

    .rail-title { font-family: var(--font-display); font-weight: 700; font-size: 1.25rem; }
    .rail-count { font-family: var(--font-ui); font-size: 0.75rem; color: var(--ash-text); white-space: nowrap; }
    .rail-count.stale { color: var(--error); cursor: pointer; }

    .work-list {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      padding: 0 var(--space-lg);
    }

    .work-card {
      display: grid;
      gap: var(--space-2xs);
      padding: var(--space-md) 0;
      border-bottom: 1px solid var(--line);
    }

    .work-card.selected { background: rgba(209, 169, 74, 0.08); }

    .work-top {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto auto;
      align-items: center;
      gap: var(--space-xs);
      min-width: 0;
    }

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

    .work-diagram-button {
      width: 2rem;
      height: 2rem;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.018);
      display: grid;
      place-items: center;
      cursor: pointer;
      padding: 0;
      flex: none;
      transition: border-color 140ms ease-out, background 140ms ease-out, transform 120ms ease-out;
    }

    .work-diagram-button:hover {
      border-color: var(--line-strong);
      background: var(--sodium-tint);
      transform: translateY(-1px);
    }

    .work-diagram-button:active { transform: translateY(0); }

    .diagram-button-icon {
      position: relative;
      width: 1.125rem;
      height: 0.875rem;
      display: block;
    }

    .diagram-button-icon::before,
    .diagram-button-icon::after {
      content: "";
      position: absolute;
      left: 0.25rem;
      right: 0.25rem;
      border-top: 1px solid var(--ash-text);
    }

    .diagram-button-icon::before { top: 0.25rem; }
    .diagram-button-icon::after { bottom: 0.25rem; }

    .diagram-button-icon span {
      position: absolute;
      width: 0.375rem;
      height: 0.375rem;
      border: 1px solid var(--sodium);
      border-radius: 2px;
      background: var(--form);
    }

    .diagram-button-icon span:nth-child(1) { left: 0; top: 0; }
    .diagram-button-icon span:nth-child(2) { right: 0; top: 0.4375rem; }
    .diagram-button-icon span:nth-child(3) { left: 0; bottom: 0; }

    .work-meta,
    .work-events {
      font-family: var(--font-ui);
      font-size: 0.75rem;
      line-height: 1.5;
      color: var(--ash-text);
      overflow-wrap: anywhere;
    }

    .execution-explorer[hidden] { display: none; }

    .execution-explorer {
      min-width: 0;
      min-height: 0;
      border-top: 1px solid var(--line-strong);
      background: var(--form);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }

    .shell.execution-fullscreen .execution-explorer {
      position: fixed;
      inset: var(--space-md);
      z-index: 35;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      box-shadow: 0 1.5rem 4rem rgb(0 0 0 / 0.46);
    }

    .execution-head {
      min-width: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-md);
      align-items: start;
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--line);
      background: var(--form-raised);
    }

    .execution-heading {
      min-width: 0;
      display: grid;
      gap: var(--space-xs);
    }

    .execution-title-row {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: var(--space-sm);
    }

    .execution-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--chalk);
      font-family: var(--font-display);
      font-size: 1.45rem;
      font-weight: 700;
      line-height: 1;
    }

    .execution-status {
      font-family: var(--font-ui);
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ash-text);
      white-space: nowrap;
    }

    .execution-meta {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.72rem;
    }

    .execution-lineage {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: var(--space-xs);
      overflow-x: auto;
      padding-bottom: 1px;
      scrollbar-width: thin;
    }

    .lineage-item,
    .lineage-bridge {
      flex: none;
      min-height: 1.7rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.018);
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.68rem;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      gap: var(--space-2xs);
      padding: 0 var(--space-xs);
      white-space: nowrap;
    }

    .lineage-item {
      cursor: pointer;
    }

    .lineage-item.current {
      color: var(--chalk);
      border-color: var(--line-strong);
      background: var(--sodium-tint);
    }

    .lineage-bridge {
      border-style: dashed;
    }

    .lineage-separator {
      flex: none;
      color: var(--ash-mid);
      font-family: var(--font-ui);
      font-size: 0.7rem;
    }

    .execution-actions {
      display: flex;
      align-items: center;
      gap: var(--space-xs);
      flex: none;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .execution-action {
      height: 2rem;
      min-width: 2rem;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.018);
      color: var(--chalk);
      cursor: pointer;
      display: grid;
      place-items: center;
      padding: 0 var(--space-xs);
      font-family: var(--font-ui);
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      line-height: 1;
      text-transform: uppercase;
    }

    .execution-action:hover {
      border-color: var(--line-strong);
      background: var(--sodium-tint);
    }

    .execution-action:disabled {
      cursor: not-allowed;
      opacity: 0.42;
    }

    .execution-body {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(240px, 300px);
      overflow: hidden;
    }

    .diagram-canvas {
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      position: relative;
      background-color: var(--form-deep);
      background-image:
        linear-gradient(oklch(0.94 0.018 90 / 0.055) 1px, transparent 1px),
        linear-gradient(90deg, oklch(0.94 0.018 90 / 0.055) 1px, transparent 1px);
      background-size: 1.5rem 1.5rem;
    }

    .diagram-stage {
      position: absolute;
      left: 0;
      top: 0;
      transform-origin: 0 0;
      will-change: transform;
    }

    .diagram-svg {
      position: absolute;
      inset: 0;
      overflow: visible;
      pointer-events: none;
    }

    .diagram-edge {
      fill: none;
      stroke: var(--ash-mid);
      stroke-width: 2;
    }

    .diagram-edge.selected {
      stroke: var(--sodium);
      stroke-width: 3;
    }

    .diagram-edge.dimmed {
      opacity: 0.28;
      stroke-dasharray: 7 7;
    }

    .diagram-edge-label {
      position: absolute;
      transform: translate(-50%, -50%);
      max-width: 9rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--form);
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.64rem;
      line-height: 1;
      padding: 0.25rem 0.45rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: none;
    }

    .diagram-edge-label.selected {
      border-color: var(--line-strong);
      color: var(--sodium-soft);
    }

    .diagram-edge-label.dimmed { opacity: 0.42; }

    .diagram-node {
      position: absolute;
      width: 17.5rem;
      height: 7rem;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--form-raised);
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      gap: var(--space-2xs);
      padding: var(--space-sm);
      box-shadow: 0 0.75rem 1.8rem rgb(0 0 0 / 0.28);
      overflow: hidden;
      cursor: pointer;
    }

    .diagram-node.selected {
      border-color: var(--sodium);
      box-shadow: 0 0 0 1px var(--sodium), 0 0.75rem 1.8rem rgb(0 0 0 / 0.28);
    }

    .diagram-node.child-execution { border-style: dashed; }
    .diagram-node.subagent-bridge { background: var(--form); }

    .diagram-node.unobserved {
      opacity: 0.68;
      background: var(--form);
    }

    .diagram-node.running { border-color: var(--sodium); }
    .diagram-node.completed,
    .diagram-node.succeeded,
    .diagram-node.done,
    .diagram-node.observed { border-color: rgba(139, 179, 120, 0.76); }
    .diagram-node.failed { border-color: var(--error); }
    .diagram-node.cancelled,
    .diagram-node.canceled { border-color: var(--line-danger); }

    .diagram-port {
      position: absolute;
      top: calc(50% - 0.25rem);
      width: 0.5rem;
      height: 0.5rem;
      border: 1px solid var(--ash-text);
      border-radius: 50%;
      background: var(--form-deep);
    }

    .diagram-port.in { left: -0.25rem; }
    .diagram-port.out { right: -0.25rem; }

    .diagram-node.completed .diagram-port,
    .diagram-node.succeeded .diagram-port,
    .diagram-node.done .diagram-port,
    .diagram-node.observed .diagram-port { border-color: var(--lichen); }
    .diagram-node.running .diagram-port { border-color: var(--sodium); }
    .diagram-node.failed .diagram-port { border-color: var(--error); }

    .diagram-node-kind {
      font-family: var(--font-ui);
      font-size: 0.64rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ash-text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .diagram-node-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--chalk);
      font-size: 1.08rem;
      font-weight: 600;
      line-height: 1.15;
    }

    .diagram-node-description {
      min-width: 0;
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
      white-space: normal;
      color: var(--chalk-dim);
      font-size: 0.74rem;
      line-height: 1.25;
    }

    .diagram-node-meta {
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.68rem;
      line-height: 1.35;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .diagram-empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.78rem;
      text-align: center;
      padding: var(--space-lg);
    }

    .empty {
      min-width: 0;
      margin: var(--space-md) 0;
      border: 1px dashed var(--line);
      border-radius: 4px;
      color: var(--ash-text);
      padding: var(--space-lg);
      text-align: center;
      font-family: var(--font-ui);
      font-size: 0.78rem;
      line-height: 1.6;
      overflow-wrap: anywhere;
    }

    .node-inspector {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: var(--form);
      padding: var(--space-md);
      display: grid;
      align-content: start;
      gap: var(--space-sm);
    }

    .inspector-empty {
      color: var(--ash-text);
      font-family: var(--font-ui);
      font-size: 0.75rem;
      line-height: 1.5;
    }

    .inspector-title {
      margin: 0;
      color: var(--chalk);
      font-family: var(--font-display);
      font-size: 1.15rem;
      line-height: 1.05;
      overflow-wrap: anywhere;
    }

    .inspector-description {
      margin: 0;
      color: var(--chalk-dim);
      font-size: 0.82rem;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    .inspector-grid {
      display: grid;
      gap: var(--space-xs);
      font-family: var(--font-ui);
      font-size: 0.72rem;
      line-height: 1.45;
    }

    .inspector-row {
      display: grid;
      grid-template-columns: 5rem minmax(0, 1fr);
      gap: var(--space-xs);
      padding-top: var(--space-xs);
      border-top: 1px solid var(--line);
    }

    .inspector-row span:first-child {
      color: var(--ash-text);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .inspector-row span:last-child {
      color: var(--chalk-dim);
      overflow-wrap: anywhere;
    }

    .inspector-links {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-xs);
    }

    .inspector-link {
      min-height: 1.8rem;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.018);
      color: var(--sodium-soft);
      font-family: var(--font-ui);
      font-size: 0.7rem;
      padding: 0 var(--space-xs);
      cursor: pointer;
    }

    .inspector-link:hover {
      border-color: var(--line-strong);
      background: var(--sodium-tint);
    }

    @media (max-width: 1180px) {
      .shell {
        min-height: 100dvh;
        height: auto;
        overflow: visible;
        grid-template-areas:
          "left main"
          "right right";
        grid-template-columns: minmax(260px, 300px) minmax(0, 1fr);
        grid-template-rows: auto auto;
      }
      .brand { font-size: 1.2rem; }
      .left {
        overflow: visible;
      }
      .main {
        min-height: 78dvh;
      }
      .right {
        min-height: 360px;
        border-left: 0;
        border-top: 1px solid var(--line);
      }
      .work-list {
        overflow: visible;
      }
      .main.explorer-open {
        grid-template-rows: auto minmax(180px, auto) minmax(420px, 62dvh) auto;
      }
      .execution-body {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: minmax(0, 1fr) minmax(180px, auto);
      }
      .node-inspector {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }

    @media (max-width: 720px) {
      .shell {
        grid-template-areas:
          "left"
          "main"
          "right";
        grid-template-columns: minmax(0, 1fr);
      }
      .left {
        grid-template-rows: auto auto auto;
        gap: var(--space-lg);
        overflow: visible;
      }
      .trigger-bay {
        justify-items: stretch;
      }
      .trigger-caption { justify-self: center; }
      .status-stack { align-self: stretch; }
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }
      .topbar-right {
        width: 100%;
        flex-wrap: wrap;
        justify-content: flex-start;
      }
      .main { min-height: 70dvh; }
      .message {
        grid-template-columns: 1fr;
        gap: var(--space-xs);
      }
      .composer { grid-template-columns: 1fr; }
      .send {
        width: 100%;
        min-height: 48px;
      }
      .shell.execution-fullscreen .execution-explorer {
        inset: var(--space-xs);
      }
      .execution-head {
        padding: var(--space-sm);
        grid-template-columns: minmax(0, 1fr);
      }
      .execution-title-row {
        align-items: flex-start;
        flex-direction: column;
        gap: var(--space-2xs);
      }
      .execution-actions {
        justify-content: flex-start;
      }
      .inspector-row {
        grid-template-columns: 1fr;
      }
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
        <p class="trigger-caption">red and blue emit host events</p>
        <section class="schedule-card" aria-label="Cron schedule example">
          <div class="schedule-card-label">try scheduling</div>
          <div class="schedule-card-code">cron.Schedule(...)</div>
          <p>Ask the agent to schedule something; Restate runs the registered cron trigger.</p>
        </section>
      </div>

      <form class="model-config" id="modelConfig">
        <label class="field">
          <span>model</span>
          <input id="modelInput" list="modelList" autocomplete="off" spellcheck="false"
                 title="Provider model slug, e.g. anthropic/claude-sonnet-4.6. Applies to chat sends and button events." />
          <datalist id="modelList"></datalist>
          <small id="modelHint" class="field-hint" hidden></small>
        </label>
        <label class="field">
          <span>thinking</span>
          <select id="variantSelect" title="Optional reasoning effort. Use provider default for models without configurable thinking."></select>
        </label>
      </form>

      <div></div>

      <div class="status-stack">
        <div class="status-row"><span>web</span><span id="webState">loading</span></div>
        <div class="status-row"><span>session</span><span id="sessionId">—</span></div>
      </div>
    </aside>

    <main id="mainPanel" class="main">
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
              <dt>click red or blue</dt><dd>emit a host event</dd>
            </dl>
            <h2>what's what</h2>
            <dl>
              <dt>trigger</dt><dd>starts background work from a registered source</dd>
              <dt>work</dt><dd>background processes (sub-agents and tools) the turn spawned</dd>
              <dt>model · thinking</dt><dd>apply to the next turn you send</dd>
            </dl>
          </div>
        </div>
      </header>

      <section id="timeline" class="timeline" aria-busy="false">
        <div id="timelineEmpty" class="empty">
          no turns yet. ask the agent something below, or click red or blue to fire a host event.
        </div>
      </section>

      <section id="executionExplorer" class="execution-explorer" aria-label="Lashlang execution explorer" hidden>
        <header class="execution-head">
          <div class="execution-heading">
            <div class="execution-title-row">
              <div id="executionTitle" class="execution-title">Execution Explorer</div>
              <div id="executionStatus" class="execution-status">idle</div>
            </div>
            <div id="executionMeta" class="execution-meta"></div>
            <nav id="executionLineage" class="execution-lineage" aria-label="Execution lineage"></nav>
          </div>
          <div class="execution-actions" role="group" aria-label="Execution graph actions">
            <button id="executionZoomOut" class="execution-action" type="button" title="Zoom out" aria-label="Zoom out">−</button>
            <button id="executionZoomIn" class="execution-action" type="button" title="Zoom in" aria-label="Zoom in">+</button>
            <button id="executionFit" class="execution-action" type="button" title="Fit graph to panel" aria-label="Fit graph to panel">fit</button>
            <button id="executionResetView" class="execution-action" type="button" title="Reset pan and zoom" aria-label="Reset pan and zoom">1:1</button>
            <button id="executionExportPng" class="execution-action" type="button" title="Export full graph as a high resolution PNG" aria-label="Export execution graph as PNG" disabled>png</button>
            <button id="executionExportSvg" class="execution-action" type="button" title="Export full graph as SVG" aria-label="Export execution graph as SVG" disabled>svg</button>
            <button id="executionFullscreen" class="execution-action" type="button" title="Toggle fullscreen explorer" aria-label="Toggle fullscreen explorer">full</button>
            <button id="executionClose" class="execution-action" type="button" title="Close explorer" aria-label="Close execution explorer">×</button>
          </div>
        </header>
        <div class="execution-body">
          <div id="executionCanvas" class="diagram-canvas">
            <div class="diagram-empty">select an execution</div>
          </div>
          <aside id="nodeInspector" class="node-inspector" aria-label="Node inspector">
            <div class="inspector-empty">select a graph node</div>
          </aside>
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
        <div id="workCount" class="rail-count">0 executions</div>
      </div>
      <div id="workList" class="work-list"></div>
    </aside>
  </div>

  <script>
    const shell = document.querySelector(".shell");
    const mainPanel = document.getElementById("mainPanel");
    const timeline = document.getElementById("timeline");
    const timelineEmpty = document.getElementById("timelineEmpty");
    const workList = document.getElementById("workList");
    const workCount = document.getElementById("workCount");
    const executionExplorer = document.getElementById("executionExplorer");
    const executionTitle = document.getElementById("executionTitle");
    const executionStatus = document.getElementById("executionStatus");
    const executionMeta = document.getElementById("executionMeta");
    const executionLineage = document.getElementById("executionLineage");
    const executionCanvas = document.getElementById("executionCanvas");
    const nodeInspector = document.getElementById("nodeInspector");
    const executionClose = document.getElementById("executionClose");
    const executionZoomOut = document.getElementById("executionZoomOut");
    const executionZoomIn = document.getElementById("executionZoomIn");
    const executionFit = document.getElementById("executionFit");
    const executionResetView = document.getElementById("executionResetView");
    const executionFullscreen = document.getElementById("executionFullscreen");
    const executionExportPng = document.getElementById("executionExportPng");
    const executionExportSvg = document.getElementById("executionExportSvg");
    const composer = document.getElementById("composer");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const buttonTriggerButtons = Array.from(document.querySelectorAll("[data-trigger-button]"));
    const triggerButtons = buttonTriggerButtons;
    const modelInput = document.getElementById("modelInput");
    const variantSelect = document.getElementById("variantSelect");
    const streamState = document.getElementById("streamState");
    const busyPill = document.getElementById("busyPill");
    const busyText = document.getElementById("busyText");
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
    let selectedWorkKey = null;
    let activeExecutionGraphKey = null;
    let currentExecutionExport = null;
    let graphIndex = { graphs: [], lineage_edges: [] };
    let graphIndexByKey = new Map();
    let executionView = {
      graph: null,
      nodes: [],
      edges: [],
      layout: null,
      stage: null,
      selectedNodeId: null,
      scale: 1,
      panX: 0,
      panY: 0,
      dragging: false,
      dragX: 0,
      dragY: 0
    };
    let turnTimer = 0;
    let turnStart = 0;

    function clearEmpty() {
      const e = document.getElementById("timelineEmpty");
      if (e) e.remove();
    }

    function modelEmpty() { return !modelInput.value.trim(); }

    function setBusy(next, label) {
      busy = next;
      syncCommandAvailability();
      stopButton.hidden = true;
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

    function syncCommandAvailability() {
      const unavailable = busy || modelEmpty();
      sendButton.disabled = unavailable;
      for (const button of triggerButtons) {
        button.setAttribute("aria-disabled", String(unavailable));
      }
    }

    const STATUS_LABELS = {
      running: "running",
      pending: "queued",
      queued: "queued",
      observed: "seen",
      unobserved: "not reached",
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
      const key = String(terminal || "").toLowerCase();
      return STATUS_LABELS[key] || humanize(key) || "unknown";
    }

    function kindLabel(kind) {
      if (kind === "subagent") return "sub-agent";
      if (kind === "process") return "process";
      if (kind === "process_event") return "step";
      if (kind === "terminal") return "result";
      if (kind === "branch") return "decision";
      if (kind === "branch_arm") return "path";
      if (kind === "child_process") return "child process";
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
      const summary = el.querySelector("summary");
      summary.textContent = `${event.language || "code"} ${event.success ? "completed" : "failed"} in ${event.duration_ms || 0}ms${toolLabel}`;
      if (event.graph_key) {
        const graphButton = document.createElement("button");
        graphButton.className = "work-diagram-button";
        graphButton.type = "button";
        graphButton.title = "open execution graph";
        graphButton.setAttribute("aria-label", "Open execution graph for this code block");
        graphButton.innerHTML = `<span class="diagram-button-icon" aria-hidden="true"><span></span><span></span><span></span></span>`;
        graphButton.addEventListener("click", click => {
          click.preventDefault();
          click.stopPropagation();
          openExecutionGraph(event.graph_key);
        });
        summary.append(" ", graphButton);
      }
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

    function replaySnapshotTimeline(items) {
      pendingCodeBlock = null;
      pendingTools = [];
      for (const item of items || []) {
        handleStreamItem(item);
      }
      finishTransientRows();
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
      empty.textContent = "no turns yet. ask the agent something below, or click red or blue to fire a host event.";
      timeline.appendChild(empty);
    }

    async function doReset() {
      resetButton.disabled = true;
      try {
        const response = await fetch("/api/reset", { method: "POST" });
        if (!response.ok) throw new Error("reset failed (" + response.status + ")");
        const state = await response.json();
        clearTranscript();
        selectedWorkKey = null;
        closeExecutionExplorer();
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
    executionClose.addEventListener("click", closeExecutionExplorer);
    executionExportPng.addEventListener("click", () => exportExecutionGraph("png"));
    executionExportSvg.addEventListener("click", () => exportExecutionGraph("svg"));
    executionZoomOut.addEventListener("click", () => zoomExecution(0.82));
    executionZoomIn.addEventListener("click", () => zoomExecution(1.18));
    executionFit.addEventListener("click", fitExecutionGraph);
    executionResetView.addEventListener("click", resetExecutionView);
    executionFullscreen.addEventListener("click", toggleExecutionFullscreen);

    window.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      if (!executionExplorer.hidden) { event.preventDefault(); closeExecutionExplorer(); return; }
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

    function fireButtonTrigger(event) {
      const button = event.currentTarget;
      if (busy || button.getAttribute("aria-disabled") === "true") return;
      if (modelEmpty()) { validateModel(); modelInput.focus(); return; }
      postCommand("/api/button-trigger", { button: button.dataset.button, ...selectedModelPayload() });
    }

    for (const button of buttonTriggerButtons) {
      button.addEventListener("click", fireButtonTrigger);
    }

    const knownModels = new Set();
    const modelHint = document.getElementById("modelHint");

    function validateModel() {
      const value = modelInput.value.trim();
      const empty = !value;
      modelInput.classList.toggle("unknown", empty);
      modelHint.hidden = !empty;
      if (empty) modelHint.textContent = "a model is required before sending a turn";
      syncCommandAvailability();
    }

    async function loadState() {
      let state;
      try {
        const response = await fetch("/api/state");
        if (!response.ok) throw new Error("state request failed (" + response.status + ")");
        state = await response.json();
      } catch (error) {
        webState.textContent = "unavailable";
        sessionId.textContent = "—";
        streamState.textContent = "could not load session";
        renderError("couldn't load the workbench session: " + (error.message || error) +
          ". check the server is running, then reload.");
        throw error;
      }
      modelInput.value = state.settings.model || "";
      const modelList = document.getElementById("modelList");
      modelList.innerHTML = "";
      knownModels.clear();
      const known = state.settings.models || (state.settings.model ? [state.settings.model] : []);
      for (const slug of known) {
        knownModels.add(slug);
        const option = document.createElement("option");
        option.value = slug;
        modelList.appendChild(option);
      }
      variantSelect.innerHTML = "";
      const selectedVariant = state.settings.model_variant || "";
      for (const variant of state.settings.model_variants || ["", "low", "medium", "high"]) {
        const option = document.createElement("option");
        option.value = variant;
        option.textContent = variant || "provider default";
        if (variant === selectedVariant) option.selected = true;
        variantSelect.appendChild(option);
      }
      validateModel();
      modelInput.addEventListener("input", validateModel);
      variantSelect.addEventListener("change", validateModel);
      webState.textContent = state.settings.web_configured ? "ready" : "not configured";
      sessionId.textContent = state.settings.session_id;
      if (Array.isArray(state.timeline) && state.timeline.length) {
        replaySnapshotTimeline(state.timeline);
      } else {
        for (const message of state.messages) renderMessage(message);
      }
    }

    function formatTime(ms) {
      if (!ms) return "";
      return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function shortId(value) {
      const text = String(value || "").replace(/^process:/, "");
      return text.length > 12 ? text.slice(0, 8) + "..." : text;
    }

    function renderWork(items) {
      const rows = executionRows(items, graphIndex.graphs || []);
      workList.innerHTML = "";
      workStale = false;
      workCount.classList.remove("stale");
      workCount.title = "";
      workCount.textContent = rows.length + (rows.length === 1 ? " execution" : " executions");
      if (!rows.length) {
        selectedWorkKey = null;
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "no visible executions";
        workList.appendChild(empty);
        return;
      }
      for (const row of rows) {
        const card = document.createElement("article");
        card.className = "work-card " + statusClass(row.status) + (row.key === selectedWorkKey ? " selected" : "");
        card.title = row.graph_key ? "execution " + row.graph_key : row.title;
        card.addEventListener("click", () => {
          selectedWorkKey = row.key;
          if (row.graph_key) openExecutionGraph(row.graph_key);
          renderWork(items);
        });

        const top = document.createElement("div");
        top.className = "work-top";
        const dot = document.createElement("span");
        dot.className = "work-dot";
        const label = document.createElement("div");
        label.className = "work-label";
        label.title = row.title;
        label.textContent = row.title;
        const state = document.createElement("div");
        state.className = "work-state";
        state.textContent = statusLabel(row.status);
        const graphButton = document.createElement("button");
        graphButton.className = "work-diagram-button";
        graphButton.type = "button";
        graphButton.disabled = !row.graph_key;
        graphButton.title = row.graph_key ? "open execution graph" : "graph not observed yet";
        graphButton.setAttribute("aria-label", "Open execution graph for " + row.title);
        graphButton.innerHTML = `<span class="diagram-button-icon" aria-hidden="true"><span></span><span></span><span></span></span>`;
        graphButton.addEventListener("click", event => {
          event.stopPropagation();
          selectedWorkKey = row.key;
          renderWork(items);
          if (row.graph_key) openExecutionGraph(row.graph_key);
        });
        top.append(dot, label, state, graphButton);

        const meta = document.createElement("div");
        meta.className = "work-meta";
        meta.textContent = row.meta;

        const events = document.createElement("div");
        events.className = "work-events";
        events.textContent = row.detail;

        card.append(top, meta, events);
        workList.appendChild(card);
      }
      if (!selectedWorkKey && rows[0]) selectedWorkKey = rows[0].key;
    }

    function executionRows(workItems, graphSummaries) {
      const rows = [];
      const processKeys = new Set((workItems || []).map(item => item.graph_key || ("process:" + item.process_id)));
      for (const graph of graphSummaries || []) {
        if (graph.kind === "process" && processKeys.has(graph.graph_key)) continue;
        rows.push({
          key: graph.graph_key,
          graph_key: graph.graph_key,
          title: graph.title || graph.entry_name || shortId(graph.graph_key),
          status: graph.status || "observed",
          meta: [
            kindLabel(graph.kind),
            graph.node_count + (graph.node_count === 1 ? " node" : " nodes"),
            graph.child_count ? graph.child_count + " children" : "",
            graph.scope?.session_id ? "session " + shortId(graph.scope.session_id) : ""
          ].filter(Boolean).join(" · "),
          detail: "graph " + shortId(graph.graph_key)
        });
      }
      for (const item of workItems || []) {
        const graphKey = item.graph_key || ("process:" + item.process_id);
        const graph = graphIndexByKey.get(graphKey);
        const recent = item.events.slice(-3).map(event => eventLabel(event.event_type));
        rows.push({
          key: graphKey,
          graph_key: graphKey,
          title: item.label || shortId(item.process_id),
          status: item.terminal,
          meta: [
            kindLabel(item.kind),
            graph ? graph.node_count + (graph.node_count === 1 ? " node" : " nodes") : "pending graph",
            formatTime(item.updated_at_ms)
          ].filter(Boolean).join(" · "),
          detail: recent.length ? "latest: " + recent.join(" -> ") : "no events yet"
        });
      }
      rows.sort((left, right) => {
        const leftForeground = left.graph_key && left.graph_key.startsWith("effect:") ? 0 : 1;
        const rightForeground = right.graph_key && right.graph_key.startsWith("effect:") ? 0 : 1;
        return leftForeground - rightForeground || left.title.localeCompare(right.title);
      });
      return rows;
    }

    function openExecutionGraph(graphKey, opts = {}) {
      if (!graphKey) return;
      activeExecutionGraphKey = graphKey;
      selectedWorkKey = graphKey;
      executionExplorer.hidden = false;
      mainPanel.classList.add("explorer-open");
      executionTitle.textContent = "Execution Explorer";
      executionStatus.textContent = "loading";
      executionMeta.textContent = "id " + shortId(graphKey);
      renderLineage(graphKey);
      renderDiagramEmpty("loading");
      if (!opts.keepFocus) executionClose.focus();
      refreshExecutionGraph(graphKey);
    }

    function closeExecutionExplorer() {
      executionExplorer.hidden = true;
      mainPanel.classList.remove("explorer-open");
      shell.classList.remove("execution-fullscreen");
      activeExecutionGraphKey = null;
      executionView.graph = null;
      executionView.stage = null;
    }

    function renderDiagramEmpty(text) {
      currentExecutionExport = null;
      setGraphExportAvailable(false);
      executionCanvas.innerHTML = "";
      const empty = document.createElement("div");
      empty.className = "diagram-empty";
      empty.textContent = text;
      executionCanvas.appendChild(empty);
      renderNodeInspector(null);
    }

    function renderExecutionGraph(graph) {
      const nodes = normalizedDiagramNodes(graph);
      const edges = normalizedDiagramEdges(graph, nodes);
      executionTitle.textContent = executionTitleForGraph(graph);
      executionStatus.textContent = statusLabel(graph.status || "observed");
      executionMeta.textContent = [
        kindLabel(graphKind(graph)),
        nodes.length + (nodes.length === 1 ? " node" : " nodes"),
        edges.length + (edges.length === 1 ? " edge" : " edges"),
        (graph.children || []).length ? (graph.children || []).length + " child links" : "",
        graph.graph_key ? shortId(graph.graph_key) : ""
      ].filter(Boolean).join(" · ");
      renderLineage(graph.graph_key);

      if (!nodes.length) {
        renderDiagramEmpty("no diagram data yet");
        return;
      }

      const layout = layoutDiagram(nodes, edges);
      currentExecutionExport = {
        graph,
        nodes,
        edges,
        layout,
        title: executionTitle.textContent,
        status: executionStatus.textContent,
        meta: executionMeta.textContent,
      };
      setGraphExportAvailable(true);
      executionCanvas.innerHTML = "";
      const stage = document.createElement("div");
      stage.className = "diagram-stage";
      stage.style.width = layout.width + "px";
      stage.style.height = layout.height + "px";

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.classList.add("diagram-svg");
      svg.setAttribute("width", String(layout.width));
      svg.setAttribute("height", String(layout.height));
      svg.setAttribute("viewBox", "0 0 " + layout.width + " " + layout.height);
      stage.appendChild(svg);

      for (const edge of edges) renderDiagramEdge(stage, svg, edge, layout.positions);
      for (const node of nodes) stage.appendChild(renderDiagramNode(node, layout.positions.get(node.id)));
      executionCanvas.appendChild(stage);

      executionView = {
        ...executionView,
        graph,
        nodes,
        edges,
        layout,
        stage,
        selectedNodeId: nodes[0]?.id || null,
        scale: 1,
        panX: 0,
        panY: 0
      };
      selectDiagramNode(executionView.selectedNodeId);
      requestAnimationFrame(fitExecutionGraph);
    }

    function graphKind(graph) {
      const subject = graph.subject || {};
      if (subject.type === "effect" && subject.kind === "exec_code") return "foreground";
      if (subject.type === "process") return "process";
      return subject.kind || subject.type || graph.entry_kind || "execution";
    }

    function executionTitleForGraph(graph) {
      const summary = graphIndexByKey.get(graph.graph_key);
      if (summary?.title) return summary.title;
      if (graph.entry_name && graph.entry_name !== "main") return graph.entry_name;
      return graph.graph_key ? "Lashlang " + shortId(graph.graph_key) : "Execution Explorer";
    }

    function normalizedDiagramNodes(graph) {
      const graphStatus = graph.status || "observed";
      const nodes = (graph.nodes || []).map(node => {
        const normalized = { ...node };
        if (normalized.kind === "process" && isUnobservedStatus(normalized.status)) {
          normalized.status = graphStatus;
        }
        return normalized;
      });
      for (const edge of lineageEdgesFromGraph(graph.graph_key)) {
        const childSummary = edge.child_graph_key ? graphIndexByKey.get(edge.child_graph_key) : null;
        nodes.push({
          id: "lineage:" + edge.bridge_graph_key + ":" + (edge.child_graph_key || "pending"),
          kind: edge.child_graph_key ? "child_execution" : "subagent_bridge",
          label: childSummary?.title || edge.bridge_title || edge.bridge_graph_key,
          status: edge.pending ? "pending" : (childSummary?.status || edge.bridge_status || "observed"),
          occurrence: null,
          latest_error: edge.error || null,
          target_graph_key: edge.child_graph_key,
          bridge_graph_key: edge.bridge_graph_key,
          child_session_id: edge.child_session_id,
        });
      }
      return nodes;
    }

    function isUnobservedStatus(status) {
      return !status || String(status).toLowerCase() === "unobserved";
    }

    function normalizedDiagramEdges(graph, nodes) {
      const ids = new Set(nodes.map(node => node.id));
      const edges = (graph.edges || [])
        .filter(edge => ids.has(edge.from) && ids.has(edge.to))
        .map(edge => ({ ...edge }));
      for (const edge of lineageEdgesFromGraph(graph.graph_key)) {
        const childId = "lineage:" + edge.bridge_graph_key + ":" + (edge.child_graph_key || "pending");
        if (ids.has(edge.parent_node_id) && ids.has(childId)) {
          edges.push({
            id: "child-edge:" + edge.bridge_graph_key + ":" + (edge.child_graph_key || "pending"),
            from: edge.parent_node_id,
            to: childId,
            label: edge.child_graph_key ? "opens" : "pending",
            selected: !edge.pending,
          });
        }
      }
      return edges;
    }

    function lineageEdgesFromGraph(graphKey) {
      return (graphIndex.lineage_edges || []).filter(edge => edge.parent_graph_key === graphKey);
    }

    function layoutDiagram(nodes, edges) {
      const nodeWidth = 280;
      const nodeHeight = 112;
      const columnGap = 128;
      const rowGap = 40;
      const padX = 56;
      const padY = 56;
      const byId = new Map(nodes.map(node => [node.id, node]));
      const incoming = new Map(nodes.map(node => [node.id, []]));
      const outgoing = new Map(nodes.map(node => [node.id, []]));
      for (const edge of edges) {
        if (!byId.has(edge.from) || !byId.has(edge.to)) continue;
        outgoing.get(edge.from).push(edge);
        incoming.get(edge.to).push(edge);
      }

      let roots = nodes.filter(node => node.kind === "process");
      if (!roots.length) roots = nodes.filter(node => incoming.get(node.id).length === 0);
      if (!roots.length && nodes[0]) roots = [nodes[0]];

      const depth = new Map();
      const queue = [];
      for (const root of roots) {
        depth.set(root.id, 0);
        queue.push(root.id);
      }
      let traversalBudget = Math.max(1, nodes.length * Math.max(1, edges.length) + nodes.length);
      while (queue.length && traversalBudget-- > 0) {
        const id = queue.shift();
        const nextDepth = (depth.get(id) || 0) + 1;
        for (const edge of outgoing.get(id) || []) {
          if ((depth.get(edge.to) ?? -1) < nextDepth) {
            depth.set(edge.to, nextDepth);
            queue.push(edge.to);
          }
        }
      }
      for (const node of nodes) if (!depth.has(node.id)) depth.set(node.id, 0);

      const columns = new Map();
      for (const node of nodes) {
        const column = depth.get(node.id) || 0;
        if (!columns.has(column)) columns.set(column, []);
        columns.get(column).push(node);
      }
      for (const columnNodes of columns.values()) columnNodes.sort(compareDiagramNodes);

      const orderedColumns = Array.from(columns.keys()).sort((a, b) => a - b);
      const maxRows = Math.max(1, ...orderedColumns.map(column => columns.get(column).length));
      const maxColumnHeight = maxRows * nodeHeight + (maxRows - 1) * rowGap;
      const positions = new Map();

      for (const column of orderedColumns) {
        const columnNodes = columns.get(column);
        const columnHeight = columnNodes.length * nodeHeight + Math.max(0, columnNodes.length - 1) * rowGap;
        const yStart = padY + Math.max(0, (maxColumnHeight - columnHeight) / 2);
        columnNodes.forEach((node, index) => {
          positions.set(node.id, {
            x: padX + column * (nodeWidth + columnGap),
            y: yStart + index * (nodeHeight + rowGap),
            width: nodeWidth,
            height: nodeHeight,
          });
        });
      }

      const columnCount = Math.max(1, orderedColumns.length);
      return {
        positions,
        width: Math.max(720, padX * 2 + columnCount * nodeWidth + (columnCount - 1) * columnGap),
        height: Math.max(420, padY * 2 + maxColumnHeight),
      };
    }

    function compareDiagramNodes(a, b) {
      const order = { process: 0, branch: 1, branch_arm: 2, process_event: 3, terminal: 4, child_process: 5 };
      return (order[a.kind] ?? 20) - (order[b.kind] ?? 20)
        || String(a.label || a.id).localeCompare(String(b.label || b.id));
    }

    function renderDiagramEdge(stage, svg, edge, positions) {
      const from = positions.get(edge.from);
      const to = positions.get(edge.to);
      if (!from || !to) return;
      const x1 = from.x + from.width;
      const y1 = from.y + from.height / 2;
      const x2 = to.x;
      const y2 = to.y + to.height / 2;
      const curve = Math.max(48, Math.abs(x2 - x1) * 0.45);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.classList.add("diagram-edge");
      const edgeClass = diagramEdgeClass(edge);
      if (edgeClass) path.classList.add(edgeClass);
      path.setAttribute("d", `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`);
      svg.appendChild(path);

      if (edge.label) {
        const label = document.createElement("div");
        label.className = "diagram-edge-label" + (edgeClass ? " " + edgeClass : "");
        label.textContent = edge.label;
        label.style.left = ((x1 + x2) / 2) + "px";
        label.style.top = ((y1 + y2) / 2) + "px";
        stage.appendChild(label);
      }
    }

    function diagramEdgeClass(edge) {
      if (edge.selected === true || edge.selection === "selected") return "selected";
      if (edge.selected === false || edge.selection === "rejected") return "dimmed";
      return "";
    }

    function renderDiagramNode(node, position) {
      const el = document.createElement("div");
      const status = statusClass(node.status || "unobserved");
      el.className = "diagram-node " + status;
      if (node.kind === "child_execution") el.classList.add("child-execution");
      if (node.kind === "subagent_bridge") el.classList.add("subagent-bridge");
      if (node.id === executionView.selectedNodeId) el.classList.add("selected");
      el.dataset.nodeId = node.id;
      el.style.left = position.x + "px";
      el.style.top = position.y + "px";
      el.style.width = position.width + "px";
      el.style.height = position.height + "px";
      el.addEventListener("click", event => {
        event.stopPropagation();
        selectDiagramNode(node.id);
        if (node.target_graph_key) openExecutionGraph(node.target_graph_key, { keepFocus: true });
      });

      const inPort = document.createElement("span");
      inPort.className = "diagram-port in";
      const outPort = document.createElement("span");
      outPort.className = "diagram-port out";
      const kind = document.createElement("div");
      kind.className = "diagram-node-kind";
      kind.textContent = kindLabel(node.kind);
      const title = document.createElement("div");
      title.className = "diagram-node-title";
      const labelTitle = node.label_metadata && node.label_metadata.title ? node.label_metadata.title : "";
      const labelDescription = node.label_metadata && node.label_metadata.description ? node.label_metadata.description : "";
      const nodeTitle = labelTitle || node.label || node.id;
      title.title = labelDescription ? nodeTitle + "\n" + labelDescription : nodeTitle;
      title.textContent = nodeTitle;
      const description = document.createElement("div");
      description.className = "diagram-node-description";
      description.title = labelDescription;
      description.textContent = labelDescription;
      const meta = document.createElement("div");
      meta.className = "diagram-node-meta";
      meta.textContent = diagramNodeMeta(node);
      el.append(inPort, outPort, kind, title, description, meta);
      return el;
    }

    function selectDiagramNode(nodeId) {
      executionView.selectedNodeId = nodeId;
      if (executionView.stage) {
        for (const node of executionView.stage.querySelectorAll(".diagram-node")) {
          node.classList.toggle("selected", node.dataset.nodeId === nodeId);
        }
      }
      const node = executionView.nodes.find(candidate => candidate.id === nodeId) || null;
      renderNodeInspector(node);
    }

    function renderNodeInspector(node) {
      nodeInspector.innerHTML = "";
      if (!node) {
        const empty = document.createElement("div");
        empty.className = "inspector-empty";
        empty.textContent = "select a graph node";
        nodeInspector.appendChild(empty);
        return;
      }
      const title = document.createElement("h3");
      title.className = "inspector-title";
      title.textContent = (node.label_metadata && node.label_metadata.title) || node.label || node.id;
      const descriptionText = (node.label_metadata && node.label_metadata.description) || "";
      const description = document.createElement("p");
      description.className = "inspector-description";
      description.textContent = descriptionText || kindLabel(node.kind);
      const grid = document.createElement("div");
      grid.className = "inspector-grid";
      const rows = [
        ["status", statusLabel(node.status)],
        ["kind", kindLabel(node.kind)],
        ["duration", node.duration_ms !== null && node.duration_ms !== undefined ? node.duration_ms + "ms" : "—"],
        ["occurrence", node.occurrence || "—"],
        ["node id", node.id],
        ["bridge", node.bridge_graph_key || "—"],
        ["session", node.child_session_id || executionView.graph?.scope?.session_id || "—"],
        ["error", node.latest_error || "—"],
      ];
      for (const [key, value] of rows) {
        const row = document.createElement("div");
        row.className = "inspector-row";
        const k = document.createElement("span");
        k.textContent = key;
        const v = document.createElement("span");
        v.textContent = String(value);
        row.append(k, v);
        grid.appendChild(row);
      }
      nodeInspector.append(title, description, grid);
      const links = inspectorGraphLinks(node);
      if (links.length) {
        const linkWrap = document.createElement("div");
        linkWrap.className = "inspector-links";
        for (const link of links) {
          const button = document.createElement("button");
          button.className = "inspector-link";
          button.type = "button";
          button.textContent = link.label;
          button.addEventListener("click", () => openExecutionGraph(link.graphKey, { keepFocus: true }));
          linkWrap.appendChild(button);
        }
        nodeInspector.appendChild(linkWrap);
      }
    }

    function inspectorGraphLinks(node) {
      const links = [];
      if (node.target_graph_key) {
        links.push({ label: "open child", graphKey: node.target_graph_key });
      }
      for (const edge of lineageEdgesFromGraph(executionView.graph?.graph_key || "")) {
        if (edge.parent_node_id === node.id && edge.child_graph_key) {
          links.push({ label: graphIndexByKey.get(edge.child_graph_key)?.title || "open child", graphKey: edge.child_graph_key });
        }
      }
      return links;
    }

    function applyExecutionTransform() {
      if (!executionView.stage) return;
      executionView.stage.style.transform = `translate(${executionView.panX}px, ${executionView.panY}px) scale(${executionView.scale})`;
    }

    function zoomExecution(factor, origin) {
      if (!executionView.stage) return;
      const rect = executionCanvas.getBoundingClientRect();
      const point = origin || { x: rect.width / 2, y: rect.height / 2 };
      const oldScale = executionView.scale;
      const nextScale = Math.max(0.25, Math.min(2.5, oldScale * factor));
      const worldX = (point.x - executionView.panX) / oldScale;
      const worldY = (point.y - executionView.panY) / oldScale;
      executionView.scale = nextScale;
      executionView.panX = point.x - worldX * nextScale;
      executionView.panY = point.y - worldY * nextScale;
      applyExecutionTransform();
    }

    function fitExecutionGraph() {
      if (!executionView.layout || !executionView.stage) return;
      const rect = executionCanvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const scale = Math.max(0.25, Math.min(1.15, Math.min(
        (rect.width - 48) / executionView.layout.width,
        (rect.height - 48) / executionView.layout.height
      )));
      executionView.scale = scale;
      executionView.panX = Math.max(24, (rect.width - executionView.layout.width * scale) / 2);
      executionView.panY = Math.max(24, (rect.height - executionView.layout.height * scale) / 2);
      applyExecutionTransform();
    }

    function resetExecutionView() {
      executionView.scale = 1;
      executionView.panX = 0;
      executionView.panY = 0;
      applyExecutionTransform();
    }

    function toggleExecutionFullscreen() {
      shell.classList.toggle("execution-fullscreen");
      requestAnimationFrame(fitExecutionGraph);
    }

    function setGraphExportAvailable(available) {
      executionExportPng.disabled = !available;
      executionExportSvg.disabled = !available;
    }

    function exportExecutionGraph(format) {
      if (!currentExecutionExport) return;
      const svg = buildExecutionExportSvg(currentExecutionExport);
      const base = executionExportFilename(currentExecutionExport);
      if (format === "svg") {
        downloadBlob(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }), base + ".svg");
      } else {
        exportExecutionGraphPng(svg, currentExecutionExport, base + ".png");
      }
    }

    function executionExportFilename(state) {
      const parts = [
        state.title || "execution-graph",
        state.graph.graph_key ? graphExportIdPart(state.graph.graph_key) : "",
      ].filter(Boolean);
      return sanitizeFilename(parts.join("-")) || "execution-graph";
    }

    function graphExportIdPart(processId) {
      return String(processId || "").replace(/^process:/, "").slice(0, 8);
    }

    function sanitizeFilename(value) {
      return String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9._-]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 96);
    }

    function downloadBlob(blob, filename) {
      const link = document.createElement("a");
      const url = URL.createObjectURL(blob);
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 500);
    }

    function exportExecutionGraphPng(svg, state, filename) {
      const dimensions = graphExportDimensions(state);
      const imageBlob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(imageBlob);
      const image = new Image();
      image.decoding = "async";
      image.onload = () => {
        const maxEdge = Math.max(dimensions.width, dimensions.height);
        const scale = Math.max(1, Math.min(3, Math.floor(8192 / maxEdge) || 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(dimensions.width * scale);
        canvas.height = Math.round(dimensions.height * scale);
        const ctx = canvas.getContext("2d");
        ctx.scale(scale, scale);
        ctx.drawImage(image, 0, 0);
        canvas.toBlob(blob => {
          URL.revokeObjectURL(url);
          if (blob) downloadBlob(blob, filename);
        }, "image/png");
      };
      image.onerror = () => URL.revokeObjectURL(url);
      image.src = url;
    }

    function graphExportDimensions(state) {
      const pad = 32;
      const headerHeight = 88;
      const width = Math.max(960, state.layout.width + pad * 2);
      const height = Math.max(560, state.layout.height + headerHeight + pad);
      return { width, height, pad, headerHeight };
    }

    function buildExecutionExportSvg(state) {
      const dimensions = graphExportDimensions(state);
      const diagramY = dimensions.headerHeight;
      const colors = {
        deep: "#0f0f0d",
        form: "#1b1a17",
        raised: "#25231f",
        line: "rgba(239, 226, 194, 0.18)",
        lineStrong: "rgba(213, 169, 74, 0.45)",
        ash: "#696258",
        ashText: "#9c968b",
        chalk: "#f2eee0",
        chalkDim: "#bdb7a8",
        sodium: "#d5a949",
        sodiumSoft: "#e3bf68",
        lichen: "#8bb378",
        error: "#c74b38",
      };
      const out = [];
      out.push(`<?xml version="1.0" encoding="UTF-8"?>`);
      out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${dimensions.width}" height="${dimensions.height}" viewBox="0 0 ${dimensions.width} ${dimensions.height}" role="img" aria-label="${escapeXml(state.title || "Lashlang graph")}">`);
      out.push(`<defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(239, 226, 194, 0.06)" stroke-width="1"/></pattern></defs>`);
      out.push(`<rect width="${dimensions.width}" height="${dimensions.height}" fill="${colors.deep}"/>`);
      out.push(`<rect y="${diagramY}" width="${dimensions.width}" height="${dimensions.height - diagramY}" fill="url(#grid)"/>`);
      out.push(`<rect width="${dimensions.width}" height="${diagramY}" fill="${colors.form}"/>`);
      out.push(`<line x1="0" y1="${diagramY}" x2="${dimensions.width}" y2="${diagramY}" stroke="${colors.line}" stroke-width="1"/>`);
      out.push(`<text x="${dimensions.pad}" y="38" fill="${colors.chalk}" font-family="Big Shoulders Display, Helvetica Neue, Arial, sans-serif" font-size="28" font-weight="700">${escapeXml(state.title || "Lashlang graph")}</text>`);
      out.push(`<text x="${dimensions.pad}" y="64" fill="${colors.ashText}" font-family="Chivo Mono, ui-monospace, monospace" font-size="12">${escapeXml(state.meta || state.status || "")}</text>`);
      out.push(`<text x="${dimensions.width - dimensions.pad}" y="42" text-anchor="end" fill="${colors.ashText}" font-family="Chivo Mono, ui-monospace, monospace" font-size="12" letter-spacing="1">${escapeXml(state.status || "")}</text>`);
      out.push(`<g transform="translate(${dimensions.pad}, ${diagramY})">`);
      for (const edge of state.edges) out.push(exportEdgeSvg(edge, state.layout.positions, colors));
      for (const node of state.nodes) out.push(exportNodeSvg(node, state.layout.positions.get(node.id), colors));
      out.push(`</g></svg>`);
      return out.join("");
    }

    function exportEdgeSvg(edge, positions, colors) {
      const from = positions.get(edge.from);
      const to = positions.get(edge.to);
      if (!from || !to) return "";
      const x1 = from.x + from.width;
      const y1 = from.y + from.height / 2;
      const x2 = to.x;
      const y2 = to.y + to.height / 2;
      const curve = Math.max(48, Math.abs(x2 - x1) * 0.45);
      const edgeClass = diagramEdgeClass(edge);
      const selected = edgeClass === "selected";
      const dimmed = edgeClass === "dimmed";
      const stroke = selected ? colors.sodium : colors.ash;
      const strokeWidth = selected ? 3 : 2;
      const dash = dimmed ? ` stroke-dasharray="7 7"` : "";
      const opacity = dimmed ? ` opacity="0.35"` : "";
      const parts = [`<path d="M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="${strokeWidth}"${dash}${opacity}/>`];
      if (edge.label) {
        const label = String(edge.label);
        const width = Math.min(132, Math.max(42, label.length * 7 + 22));
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        parts.push(`<rect x="${cx - width / 2}" y="${cy - 13}" width="${width}" height="26" rx="13" fill="${colors.form}" stroke="${selected ? colors.lineStrong : colors.line}"${opacity}/>`);
        parts.push(`<text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="${selected ? colors.sodiumSoft : colors.ashText}" font-family="Chivo Mono, ui-monospace, monospace" font-size="11"${opacity}>${escapeXml(label)}</text>`);
      }
      return parts.join("");
    }

    function exportNodeSvg(node, position, colors) {
      if (!position) return "";
      const status = statusClass(node.status || "unobserved");
      const unobserved = isUnobservedStatus(node.status);
      const border = nodeBorderColor(status, colors);
      const title = (node.label_metadata && node.label_metadata.title) || node.label || node.id;
      const description = (node.label_metadata && node.label_metadata.description) || "";
      const titleLines = wrapExportText(title, 26, 1);
      const descriptionLines = wrapExportText(description, 44, 2);
      const meta = truncateExportText(diagramNodeMeta(node), 42);
      const opacity = unobserved ? ` opacity="0.68"` : "";
      const out = [];
      out.push(`<g${opacity}>`);
      out.push(`<rect x="${position.x}" y="${position.y}" width="${position.width}" height="${position.height}" rx="6" fill="${unobserved ? colors.form : colors.raised}" stroke="${border}" stroke-width="1"/>`);
      out.push(`<circle cx="${position.x}" cy="${position.y + position.height / 2}" r="4" fill="${colors.deep}" stroke="${border}" stroke-width="1"/>`);
      out.push(`<circle cx="${position.x + position.width}" cy="${position.y + position.height / 2}" r="4" fill="${colors.deep}" stroke="${border}" stroke-width="1"/>`);
      out.push(`<text x="${position.x + 16}" y="${position.y + 28}" fill="${colors.ashText}" font-family="Chivo Mono, ui-monospace, monospace" font-size="11" letter-spacing="2">${escapeXml(kindLabel(node.kind).toUpperCase())}</text>`);
      out.push(exportTextLines(titleLines, position.x + 16, position.y + 52, colors.chalk, "Spectral, Georgia, serif", 19, 20, "600"));
      if (descriptionLines.length) {
        out.push(exportTextLines(descriptionLines, position.x + 16, position.y + 73, colors.chalkDim, "Spectral, Georgia, serif", 12, 15, "400"));
      }
      out.push(`<text x="${position.x + 16}" y="${position.y + position.height - 14}" fill="${colors.ashText}" font-family="Chivo Mono, ui-monospace, monospace" font-size="11">${escapeXml(meta)}</text>`);
      out.push(`</g>`);
      return out.join("");
    }

    function nodeBorderColor(status, colors) {
      if (status === "running") return colors.sodium;
      if (status === "completed" || status === "succeeded" || status === "done" || status === "observed") return colors.lichen;
      if (status === "failed") return colors.error;
      if (status === "cancelled" || status === "canceled") return colors.error;
      return colors.line;
    }

    function exportTextLines(lines, x, y, fill, family, size, lineHeight, weight) {
      if (!lines.length) return "";
      const tspans = lines.map((line, index) =>
        `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`
      ).join("");
      return `<text x="${x}" y="${y}" fill="${fill}" font-family="${family}" font-size="${size}" font-weight="${weight}">${tspans}</text>`;
    }

    function wrapExportText(value, maxChars, maxLines) {
      const words = String(value || "").trim().split(/\s+/).filter(Boolean);
      const lines = [];
      let line = "";
      for (const word of words) {
        const next = line ? line + " " + word : word;
        if (next.length <= maxChars) {
          line = next;
          continue;
        }
        if (line) lines.push(line);
        line = word;
        if (lines.length >= maxLines) break;
      }
      if (line && lines.length < maxLines) lines.push(line);
      if (lines.length === maxLines && words.join(" ").length > lines.join(" ").length) {
        lines[lines.length - 1] = truncateExportText(lines[lines.length - 1], maxChars);
      }
      return lines;
    }

    function truncateExportText(value, maxChars) {
      const text = String(value || "");
      if (text.length <= maxChars) return text;
      return text.slice(0, Math.max(0, maxChars - 1)) + "...";
    }

    function escapeXml(value) {
      return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function statusClass(status) {
      return String(status || "unobserved").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    }

    function diagramNodeMeta(node) {
      return [
        statusLabel(node.status),
        node.occurrence ? "x" + node.occurrence : "",
        node.duration_ms !== null && node.duration_ms !== undefined ? node.duration_ms + "ms" : "",
        node.latest_error || "",
      ].filter(Boolean).join(" · ");
    }

    function renderLineage(graphKey) {
      executionLineage.innerHTML = "";
      const path = lineagePath(graphKey);
      if (!path.length) {
        const item = document.createElement("button");
        item.className = "lineage-item current";
        item.type = "button";
        item.textContent = shortId(graphKey);
        executionLineage.appendChild(item);
        return;
      }
      path.forEach((entry, index) => {
        if (index) {
          const sep = document.createElement("span");
          sep.className = "lineage-separator";
          sep.textContent = ">";
          executionLineage.appendChild(sep);
        }
        if (entry.type === "bridge") {
          const bridge = document.createElement("span");
          bridge.className = "lineage-bridge";
          bridge.textContent = entry.label;
          executionLineage.appendChild(bridge);
        } else {
          const button = document.createElement("button");
          button.className = "lineage-item" + (entry.graphKey === graphKey ? " current" : "");
          button.type = "button";
          button.textContent = entry.label;
          button.addEventListener("click", () => openExecutionGraph(entry.graphKey, { keepFocus: true }));
          executionLineage.appendChild(button);
        }
      });
    }

    function lineagePath(graphKey) {
      const parentEdge = (graphIndex.lineage_edges || []).find(edge => edge.child_graph_key === graphKey);
      const summary = graphIndexByKey.get(graphKey);
      if (!parentEdge) {
        return [{ type: "graph", graphKey, label: summary?.title || shortId(graphKey) }];
      }
      return [
        ...lineagePath(parentEdge.parent_graph_key),
        { type: "bridge", label: parentEdge.bridge_title || shortId(parentEdge.bridge_graph_key) },
        { type: "graph", graphKey, label: summary?.title || shortId(graphKey) }
      ];
    }

    async function refreshExecutionGraph(graphKey) {
      if (!graphKey || executionExplorer.hidden) return;
      try {
        const response = await fetch("/api/lashlang-graph/" + encodeURIComponent(graphKey));
        if (!response.ok) throw new Error("graph unavailable");
        renderExecutionGraph(await response.json());
      } catch (_) {
        executionStatus.textContent = "unavailable";
        executionMeta.textContent = "id " + shortId(graphKey);
        renderDiagramEmpty("no diagram data yet");
      }
    }

    async function refreshWork() {
      try {
        const [workResponse, graphResponse] = await Promise.all([
          fetch("/api/work"),
          fetch("/api/lashlang-graphs")
        ]);
        if (!workResponse.ok) throw new Error("work request failed");
        if (!graphResponse.ok) throw new Error("graph index request failed");
        const items = await workResponse.json();
        graphIndex = await graphResponse.json();
        graphIndexByKey = new Map((graphIndex.graphs || []).map(graph => [graph.graph_key, graph]));
        renderWork(items);
        if (activeExecutionGraphKey) {
          renderLineage(activeExecutionGraphKey);
          refreshExecutionGraph(activeExecutionGraphKey);
        }
      } catch (_) {
        if (!workStale) {
          workStale = true;
          workCount.textContent = "updates paused — retry";
          workCount.title = "couldn't reach the execution index. polling continues; click to retry now.";
          workCount.classList.add("stale");
        }
      }
    }

    executionCanvas.addEventListener("pointerdown", event => {
      if (event.target.closest(".diagram-node")) return;
      executionView.dragging = true;
      executionView.dragX = event.clientX - executionView.panX;
      executionView.dragY = event.clientY - executionView.panY;
      executionCanvas.setPointerCapture(event.pointerId);
    });
    executionCanvas.addEventListener("pointermove", event => {
      if (!executionView.dragging) return;
      executionView.panX = event.clientX - executionView.dragX;
      executionView.panY = event.clientY - executionView.dragY;
      applyExecutionTransform();
    });
    executionCanvas.addEventListener("pointerup", event => {
      executionView.dragging = false;
      if (executionCanvas.hasPointerCapture(event.pointerId)) {
        executionCanvas.releasePointerCapture(event.pointerId);
      }
    });
    executionCanvas.addEventListener("wheel", event => {
      if (!executionView.stage) return;
      event.preventDefault();
      const rect = executionCanvas.getBoundingClientRect();
      zoomExecution(event.deltaY > 0 ? 0.9 : 1.1, {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      });
    }, { passive: false });

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
