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
    // Pin to a specific minor — `mermaid@11` floats and v11.15 has been
    // observed throwing "Syntax error in text" on diagrams that parsed
    // fine on earlier minors. Bump deliberately when upgrading.
    const url = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
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
      btn.innerHTML =
        '<span class="diagram-zoom__glyph" aria-hidden="true">⤢</span><span>Expand</span>';
      btn.addEventListener("click", () => openDialog(d));
      d.appendChild(btn);
    });
  }

  // Pan/zoom state for the currently-open dialog. Zoom changes the
  // active SVG's viewBox instead of CSS-scaling a composited layer, so
  // text and strokes stay vector-rendered while zooming.
  const view = {
    svg: null,
    baseBox: null,
    box: null,
    scale: 1,
    body: null,
    zoomLabel: null,
  };
  const MIN_SCALE = 0.3;
  const MAX_SCALE = 8;

  function formatNumber(n) {
    return Number(n.toFixed(3)).toString();
  }

  function cloneBox(box) {
    return { x: box.x, y: box.y, width: box.width, height: box.height };
  }

  function boxToViewBox(box) {
    return [box.x, box.y, box.width, box.height].map(formatNumber).join(" ");
  }

  function parseViewBox(value) {
    if (!value) return null;
    const parts = value.trim().split(/[\s,]+/).map(Number);
    if (parts.length !== 4 || parts.some((part) => !Number.isFinite(part))) return null;
    const [x, y, width, height] = parts;
    if (width <= 0 || height <= 0) return null;
    return { x, y, width, height };
  }

  function readSvgViewBox(svg) {
    const attrBox = parseViewBox(svg.getAttribute("viewBox"));
    if (attrBox) return attrBox;

    const width = Number.parseFloat(svg.getAttribute("width")) || svg.getBoundingClientRect().width;
    const height = Number.parseFloat(svg.getAttribute("height")) || svg.getBoundingClientRect().height;
    if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
      return { x: 0, y: 0, width, height };
    }

    return { x: 0, y: 0, width: 1000, height: 700 };
  }

  function updateZoomLabel() {
    if (view.zoomLabel) {
      view.zoomLabel.textContent = Math.round(view.scale * 100) + "%";
    }
  }

  function setViewBox(box) {
    if (!view.svg || !view.baseBox) return;
    view.box = cloneBox(box);
    view.scale = view.baseBox.width / view.box.width;
    view.svg.setAttribute("viewBox", boxToViewBox(view.box));
    updateZoomLabel();
  }

  function resetView() {
    if (!view.baseBox) {
      view.scale = 1;
      updateZoomLabel();
      return;
    }
    setViewBox(view.baseBox);
  }

  function clearView() {
    view.svg = null;
    view.baseBox = null;
    view.box = null;
    view.scale = 1;
    updateZoomLabel();
  }

  function pointFromClient(clientX, clientY) {
    if (!view.svg || !view.box) return null;
    if (typeof view.svg.createSVGPoint === "function") {
      const matrix = view.svg.getScreenCTM();
      if (matrix) {
        const point = view.svg.createSVGPoint();
        point.x = clientX;
        point.y = clientY;
        return point.matrixTransform(matrix.inverse());
      }
    }
    return {
      x: view.box.x + view.box.width / 2,
      y: view.box.y + view.box.height / 2,
    };
  }

  function unitsPerScreenPixel() {
    if (!view.svg || !view.box) return { x: 1, y: 1 };
    const matrix = view.svg.getScreenCTM?.();
    if (matrix && matrix.a && matrix.d) {
      return { x: 1 / matrix.a, y: 1 / matrix.d };
    }
    const rect = view.svg.getBoundingClientRect();
    return {
      x: view.box.width / Math.max(rect.width, 1),
      y: view.box.height / Math.max(rect.height, 1),
    };
  }

  function zoomAt(factor, clientX, clientY) {
    if (!view.box || !view.baseBox) return;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, view.scale * factor));
    if (newScale === view.scale) return;

    const point = pointFromClient(clientX, clientY);
    const ratio = newScale / view.scale;
    const nextWidth = view.box.width / ratio;
    const nextHeight = view.box.height / ratio;
    const px = (point.x - view.box.x) / view.box.width;
    const py = (point.y - view.box.y) / view.box.height;

    setViewBox({
      x: point.x - px * nextWidth,
      y: point.y - py * nextHeight,
      width: nextWidth,
      height: nextHeight,
    });
  }

  function zoomFromCenter(factor) {
    if (!view.body) return;
    const rect = view.body.getBoundingClientRect();
    zoomAt(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
  }

  function closeDialog(dlg) {
    if (dlg.open) dlg.close();
  }

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
    view.body = body;
    view.zoomLabel = zoomLabel;

    dlg.querySelector("[data-action='zoom-in']").addEventListener("click", () => zoomFromCenter(1.25));
    dlg.querySelector("[data-action='zoom-out']").addEventListener("click", () => zoomFromCenter(1 / 1.25));
    dlg.querySelector("[data-action='reset']").addEventListener("click", resetView);
    dlg.querySelector("[data-action='close']").addEventListener("click", () => closeDialog(dlg));

    body.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        zoomAt(factor, e.clientX, e.clientY);
      },
      { passive: false }
    );

    let drag = null;
    body.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      drag = {
        x: e.clientX,
        y: e.clientY,
        box: view.box ? cloneBox(view.box) : null,
        units: unitsPerScreenPixel(),
      };
      body.classList.add("is-panning");
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag || !drag.box) return;
      setViewBox({
        x: drag.box.x - (e.clientX - drag.x) * drag.units.x,
        y: drag.box.y - (e.clientY - drag.y) * drag.units.y,
        width: drag.box.width,
        height: drag.box.height,
      });
    });
    window.addEventListener("mouseup", () => {
      drag = null;
      body.classList.remove("is-panning");
    });

    dlg.addEventListener("click", (e) => {
      if (e.target === dlg && !drag) closeDialog(dlg);
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
        resetView();
      }
    });

    document.body.appendChild(dlg);
    return dlg;
  }

  function openDialog(diagramEl) {
    const dlg = ensureDialog();
    const titleText = diagramEl.querySelector(".diagram-title")?.textContent.trim() || "Diagram";
    dlg.querySelector(".diagram-modal__title").textContent = titleText;
    const stage = dlg.querySelector(".diagram-modal__stage");

    stage.innerHTML = "";
    const svg = diagramEl.querySelector(".mermaid svg");
    if (svg) {
      const placeholder = document.createComment("diagram-zoom-placeholder");
      const savedAttrs = {
        width: svg.getAttribute("width"),
        height: svg.getAttribute("height"),
        style: svg.getAttribute("style"),
        viewBox: svg.getAttribute("viewBox"),
        preserveAspectRatio: svg.getAttribute("preserveAspectRatio"),
      };
      svg.parentNode.insertBefore(placeholder, svg);
      stage.appendChild(svg);
      svg.removeAttribute("style");
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", "100%");
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
      view.svg = svg;
      view.baseBox = readSvgViewBox(svg);
      setViewBox(view.baseBox);

      dlg.addEventListener(
        "close",
        () => {
          if (savedAttrs.width === null) svg.removeAttribute("width");
          else svg.setAttribute("width", savedAttrs.width);
          if (savedAttrs.height === null) svg.removeAttribute("height");
          else svg.setAttribute("height", savedAttrs.height);
          if (savedAttrs.style === null) svg.removeAttribute("style");
          else svg.setAttribute("style", savedAttrs.style);
          if (savedAttrs.viewBox === null) svg.removeAttribute("viewBox");
          else svg.setAttribute("viewBox", savedAttrs.viewBox);
          if (savedAttrs.preserveAspectRatio === null) svg.removeAttribute("preserveAspectRatio");
          else svg.setAttribute("preserveAspectRatio", savedAttrs.preserveAspectRatio);
          if (placeholder.parentNode) {
            placeholder.parentNode.replaceChild(svg, placeholder);
          }
          stage.innerHTML = "";
          clearView();
        },
        { once: true }
      );
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
