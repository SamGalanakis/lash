// Conditional Mermaid loader — only fetches the library when the page
// actually has .mermaid blocks. Saves ~2 MB on diagram-free pages.
// Theme variables are sampled from live CSS custom properties so a
// runtime theme toggle (data-theme on <html>) re-renders the diagrams
// against the active palette. Click-to-fullscreen with pan/zoom is
// wired up on every .diagram after the first render.
(function () {
  let _mermaid = null;
  // first-time-load promise; init() awaits it before each render
  let _loadPromise = null;

  function themeVarsFromCSS() {
    const cs = getComputedStyle(document.documentElement);
    const v = (k) => cs.getPropertyValue(k).trim();
    const bg     = v("--bg")        || "#1b1a17";
    const elev   = v("--bg-elev")   || "#1b1a17";
    const faint  = v("--bg-faint")  || "#201d18";
    const tint   = v("--bg-tint")   || "#2b2722";
    const ink    = v("--ink")       || "#e8e4d0";
    const inkMid = v("--ink-mid")   || "#c0bca8";
    const inkDim = v("--ink-dim")   || "#a09e91";
    const rule   = v("--rule")      || "#2b2925";
    const ruleS  = v("--rule-strong") || "#5a5a50";
    const sodium = v("--sodium")    || "#e8a33c";
    return {
      background:           bg,
      primaryColor:         elev,
      primaryTextColor:     ink,
      primaryBorderColor:   sodium,
      secondaryColor:       faint,
      tertiaryColor:        tint,
      lineColor:            inkDim,
      textColor:            ink,
      mainBkg:              elev,
      nodeBorder:           sodium,
      clusterBkg:           tint,
      clusterBorder:        ruleS,
      edgeLabelBackground:  bg,
      titleColor:           ink,
      // Chivo Mono is the docs' label face — use it inside diagrams
      // so they read as the same family of artefacts. NEVER Inter.
      fontFamily:           '"Chivo Mono", ui-monospace, monospace',
      // sequence-diagram extras
      actorBorder:          sodium,
      actorBkg:             elev,
      actorTextColor:       ink,
      signalColor:          inkMid,
      signalTextColor:      ink,
      labelBoxBkgColor:     elev,
      labelBoxBorderColor:  ruleS,
      labelTextColor:       ink,
      noteBkgColor:         tint,
      noteBorderColor:      ruleS,
      noteTextColor:        ink,
    };
  }

  // Snapshot a diagram's source the first time we see it; we use this
  // when re-rendering after a theme change.
  function snapshotSources() {
    document.querySelectorAll(".mermaid:not([data-mermaid-source])").forEach((el) => {
      el.dataset.mermaidSource = el.textContent;
    });
  }

  // Reset already-rendered diagrams back to their source text so the
  // next mermaid.run() actually re-draws them against the new palette.
  function resetForRerender() {
    document.querySelectorAll(".mermaid[data-processed='true']").forEach((el) => {
      const src = el.dataset.mermaidSource;
      if (typeof src === "string") {
        el.textContent = src;
        el.removeAttribute("data-processed");
        // mermaid stamps a fresh id each run; clear it so the next run
        // doesn't collide with its own previous output
        el.removeAttribute("id");
      }
    });
  }

  function ensureLoaded() {
    if (_mermaid) return Promise.resolve(_mermaid);
    if (_loadPromise) return _loadPromise;
    const url = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    _loadPromise = import(url).then(({ default: m }) => {
      _mermaid = m;
      // expose so docs.js can also trigger re-renders after SPA swaps
      window.__lashMermaid = m;
      return m;
    });
    return _loadPromise;
  }

  function init() {
    if (!document.querySelector(".mermaid")) return;
    snapshotSources();
    ensureLoaded()
      .then((m) => {
        m.initialize({
          startOnLoad: false,
          theme: "base",
          securityLevel: "loose",
          themeVariables: themeVarsFromCSS(),
        });
        resetForRerender();
        return m.run({ querySelector: ".mermaid:not([data-processed='true'])" });
      })
      .then(setupZoom)
      .catch((err) => {
        // surface to console, but don't break the page
        console.error("Mermaid load failed:", err);
      });
  }

  // Public re-render entry: idempotent, safe to call repeatedly. Used
  // by the theme toggle and by docs.js after SPA content swaps.
  function rerender() {
    if (!document.querySelector(".mermaid")) return;
    init();
  }
  window.__lashMermaidRerender = rerender;

  // Watch <html data-theme> for changes — re-render whenever it flips.
  function watchTheme() {
    const obs = new MutationObserver(() => rerender());
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  function setupZoom() {
    document.querySelectorAll(".diagram").forEach((d) => {
      if (d.querySelector(".diagram-zoom")) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "diagram-zoom";
      btn.setAttribute("aria-label", "Expand diagram");
      btn.title = "Expand";
      btn.textContent = "⤢";
      btn.addEventListener("click", () => openDialog(d));
      d.appendChild(btn);
    });
  }

  // Pan/zoom state for the currently-open dialog.
  const view = { scale: 1, tx: 0, ty: 0 };
  const MIN_SCALE = 0.3;
  const MAX_SCALE = 8;

  function ensureDialog() {
    let dlg = document.getElementById("diagram-modal");
    if (dlg) return dlg;
    dlg = document.createElement("dialog");
    dlg.id = "diagram-modal";
    dlg.className = "diagram-modal";
    dlg.innerHTML = [
      '<div class="diagram-modal__header">',
      '  <span class="diagram-modal__title"></span>',
      '  <div class="diagram-modal__controls">',
      '    <button class="diagram-modal__btn" type="button" data-action="zoom-out" aria-label="Zoom out">−</button>',
      '    <span class="diagram-modal__zoom">100%</span>',
      '    <button class="diagram-modal__btn" type="button" data-action="zoom-in" aria-label="Zoom in">+</button>',
      '    <button class="diagram-modal__btn" type="button" data-action="reset" aria-label="Reset view">Reset</button>',
      '    <button class="diagram-modal__btn" type="button" data-action="close" aria-label="Close">Close ⤬</button>',
      "  </div>",
      "</div>",
      '<div class="diagram-modal__body">',
      '  <div class="diagram-modal__stage"></div>',
      "</div>",
    ].join("");

    const body = dlg.querySelector(".diagram-modal__body");
    const stage = dlg.querySelector(".diagram-modal__stage");
    const zoomLabel = dlg.querySelector(".diagram-modal__zoom");

    const apply = () => {
      stage.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`;
      zoomLabel.textContent = Math.round(view.scale * 100) + "%";
    };

    const reset = () => {
      view.scale = 1;
      view.tx = 0;
      view.ty = 0;
      apply();
    };

    const zoomAt = (factor, cx, cy) => {
      const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.scale * factor));
      const ratio = newScale / view.scale;
      view.tx = cx - (cx - view.tx) * ratio;
      view.ty = cy - (cy - view.ty) * ratio;
      view.scale = newScale;
      apply();
    };

    const zoomFromCenter = (factor) => {
      const rect = body.getBoundingClientRect();
      zoomAt(factor, rect.width / 2, rect.height / 2);
    };

    dlg.querySelector("[data-action='zoom-in']").addEventListener("click", () => zoomFromCenter(1.25));
    dlg.querySelector("[data-action='zoom-out']").addEventListener("click", () => zoomFromCenter(1 / 1.25));
    dlg.querySelector("[data-action='reset']").addEventListener("click", reset);
    dlg.querySelector("[data-action='close']").addEventListener("click", () => dlg.close());

    body.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = body.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        zoomAt(factor, cx, cy);
      },
      { passive: false }
    );

    let drag = null;
    body.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      drag = { x: e.clientX - view.tx, y: e.clientY - view.ty };
      body.classList.add("is-panning");
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      view.tx = e.clientX - drag.x;
      view.ty = e.clientY - drag.y;
      apply();
    });
    window.addEventListener("mouseup", () => {
      drag = null;
      body.classList.remove("is-panning");
    });

    dlg.addEventListener("click", (e) => {
      if (e.target === dlg && !drag) dlg.close();
    });

    dlg.addEventListener("keydown", (e) => {
      if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomFromCenter(1.25);
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomFromCenter(1 / 1.25);
      } else if (e.key === "0") {
        e.preventDefault();
        reset();
      }
    });

    dlg._diagramReset = reset;

    document.body.appendChild(dlg);
    return dlg;
  }

  function openDialog(diagramEl) {
    const dlg = ensureDialog();
    const titleText = diagramEl.querySelector(".diagram-title")?.textContent.trim() || "Diagram";
    dlg.querySelector(".diagram-modal__title").textContent = titleText;
    const stage = dlg.querySelector(".diagram-modal__stage");

    dlg._diagramReset();

    stage.innerHTML = "";
    const svg = diagramEl.querySelector(".mermaid svg");
    if (svg) {
      const placeholder = document.createComment("diagram-zoom-placeholder");
      const savedAttrs = {
        width: svg.getAttribute("width"),
        height: svg.getAttribute("height"),
        style: svg.getAttribute("style"),
      };
      svg.parentNode.insertBefore(placeholder, svg);
      stage.appendChild(svg);
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.removeAttribute("style");

      dlg.addEventListener(
        "close",
        () => {
          if (savedAttrs.width === null) svg.removeAttribute("width");
          else svg.setAttribute("width", savedAttrs.width);
          if (savedAttrs.height === null) svg.removeAttribute("height");
          else svg.setAttribute("height", savedAttrs.height);
          if (savedAttrs.style === null) svg.removeAttribute("style");
          else svg.setAttribute("style", savedAttrs.style);
          if (placeholder.parentNode) {
            placeholder.parentNode.replaceChild(svg, placeholder);
          }
          stage.innerHTML = "";
        },
        { once: true }
      );
    } else {
      const clone = diagramEl.querySelector(".mermaid")?.cloneNode(true);
      if (clone) stage.appendChild(clone);
    }

    dlg.showModal();
  }

  function boot() {
    init();
    watchTheme();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
