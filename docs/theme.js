/* lash · docs · theme bootstrap + toggle
   Runs BEFORE paint so there's no light/dark flash, then wires the
   #theme-toggle button (if present) to flip the theme and re-run
   the landscape generator (if loaded). */
(function () {
  "use strict";

  // ── pre-paint init ──────────────────────────────────────
  const KEY = "lash-theme";
  const stored = (function () {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  })();
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", stored || preferred);

  // ── visit marker ───────────────────────────────────────
  // Tag the document so entrance animations (spine draw, TOC stagger)
  // only play on first visit per session. After that, navigating
  // between pages feels snappy instead of replaying every animation.
  try {
    const visited = sessionStorage.getItem("lash-visited");
    if (visited) {
      document.documentElement.classList.add("lash-revisit");
    } else {
      sessionStorage.setItem("lash-visited", "1");
    }
  } catch (e) { /* private mode — let animations play */ }

  // ── FOUC guard ──────────────────────────────────────────
  // On full navs the page paints raw HTML (serif fallback, no
  // docs-layout grid) before fonts download and docs.js mounts
  // the shell. Hide body until fonts are ready, with a failsafe
  // timeout so a flaky font CDN can't strand the page.
  const markReady = () => document.documentElement.classList.add("lash-ready");
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(markReady);
  } else {
    markReady();
  }
  // Failsafe — paint at most ~600ms after page load even if a font
  // never resolves. Better to flash than to stall.
  setTimeout(markReady, 600);

  // ── toggle wiring (runs after DOM is ready) ───────────────
  function wire() {
    const html       = document.documentElement;
    const themeBtn   = document.getElementById("theme-toggle");
    const themeGlyph = document.getElementById("theme-glyph");
    const themeLabel = document.getElementById("theme-label");
    if (!themeBtn) return;

    // sun and moon icons rendered inline as SVG — unicode glyphs
    // (☀ ☾) don't carry across our font stack so they show as
    // tofu boxes on some systems. Inline SVG always renders.
    const SUN = '<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="3"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4"/></svg>';
    const MOON = '<svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M11.8 10.4a5 5 0 1 1-6.2-7 5.3 5.3 0 0 0-.3 1.7 4.6 4.6 0 0 0 4.6 4.6c.7 0 1.3-.1 1.9-.4z"/></svg>';

    function syncButton() {
      const current = html.getAttribute("data-theme") || "dark";
      if (current === "dark") {
        if (themeGlyph) themeGlyph.innerHTML = SUN;
        if (themeLabel) themeLabel.textContent = "light";
        themeBtn.setAttribute("aria-label", "switch to light theme");
      } else {
        if (themeGlyph) themeGlyph.innerHTML = MOON;
        if (themeLabel) themeLabel.textContent = "dark";
        themeBtn.setAttribute("aria-label", "switch to dark theme");
      }
    }

    themeBtn.addEventListener("click", () => {
      const current = html.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (e) {}
      syncButton();
      if (typeof window.__LASH_SCENE_RUN === "function") {
        window.__LASH_SCENE_RUN();
      }
    });
    syncButton();

    // honour OS theme change when user has no explicit override
    try {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (ev) => {
        const userPref = localStorage.getItem(KEY);
        if (userPref === "dark" || userPref === "light") return;
        html.setAttribute("data-theme", ev.matches ? "dark" : "light");
        syncButton();
        if (typeof window.__LASH_SCENE_RUN === "function") {
          window.__LASH_SCENE_RUN();
        }
      });
    } catch (e) { /* older browsers — silently ignore */ }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
