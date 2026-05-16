// Conditional Mermaid loader — only fetches the library when the page
// actually has .mermaid blocks. Saves ~2 MB on diagram-free pages.
// Also wires up click-to-fullscreen with pan/zoom on every .diagram once
// mermaid renders.
(function () {
  function init() {
    if (!document.querySelector(".mermaid")) return;

    const LIGHT = {
      background: "#ffffff",
      primaryColor: "#ffffff",
      primaryTextColor: "#1d1c19",
      primaryBorderColor: "#cfcabd",
      lineColor: "#7a7872",
      secondaryColor: "#fafaf7",
      tertiaryColor: "#f3f1ec",
      clusterBkg: "#fafaf7",
      clusterBorder: "#cfcabd",
      edgeLabelBackground: "#ffffff",
      textColor: "#1d1c19",
      fontFamily: "Inter, system-ui, sans-serif",
      actorBorder: "#c97a1f",
      actorBkg: "#ffffff",
      actorTextColor: "#1d1c19",
      signalColor: "#4c4a45",
      signalTextColor: "#1d1c19",
      labelBoxBkgColor: "#fafaf7",
      labelBoxBorderColor: "#cfcabd",
      labelTextColor: "#1d1c19",
      noteBkgColor: "#f6ebd8",
      noteBorderColor: "#cfcabd",
      noteTextColor: "#1d1c19",
    };
    const DARK = {
      background: "#1b1a17",
      primaryColor: "#1b1a17",
      primaryTextColor: "#e8e4d0",
      primaryBorderColor: "#5a5a50",
      lineColor: "#8a8678",
      secondaryColor: "#141412",
      tertiaryColor: "#201d18",
      clusterBkg: "#141412",
      clusterBorder: "#5a5a50",
      edgeLabelBackground: "#1b1a17",
      textColor: "#e8e4d0",
      fontFamily: "Inter, system-ui, sans-serif",
      actorBorder: "#e8a33c",
      actorBkg: "#1b1a17",
      actorTextColor: "#e8e4d0",
      signalColor: "#c0bca8",
      signalTextColor: "#e8e4d0",
      labelBoxBkgColor: "#141412",
      labelBoxBorderColor: "#5a5a50",
      labelTextColor: "#e8e4d0",
      noteBkgColor: "#2b2218",
      noteBorderColor: "#5a5a50",
      noteTextColor: "#e8e4d0",
    };
    function activeTheme() {
      const explicit = document.documentElement.getAttribute("data-theme");
      if (explicit === "dark") return DARK;
      if (explicit === "light") return LIGHT;
      return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? DARK
        : LIGHT;
    }

    const url = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    import(url)
      .then(({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          securityLevel: "loose",
          themeVariables: activeTheme(),
        });
        return mermaid.run({ querySelector: ".mermaid" });
      })
      .then(setupZoom)
      .catch((err) => {
        console.error("Mermaid load failed:", err);
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

    // Header buttons
    dlg.querySelector("[data-action='zoom-in']").addEventListener("click", () => zoomFromCenter(1.25));
    dlg.querySelector("[data-action='zoom-out']").addEventListener("click", () => zoomFromCenter(1 / 1.25));
    dlg.querySelector("[data-action='reset']").addEventListener("click", reset);
    dlg.querySelector("[data-action='close']").addEventListener("click", () => dlg.close());

    // Wheel zoom around cursor
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

    // Drag to pan
    let drag = null;
    body.addEventListener("mousedown", (e) => {
      // ignore right-click
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

    // Click backdrop closes (but not when dragging)
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg && !drag) dlg.close();
    });

    // Keyboard: + / − / 0 to reset
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

    // Expose helpers on the dialog for openDialog
    dlg._diagramReset = reset;

    document.body.appendChild(dlg);
    return dlg;
  }

  function openDialog(diagramEl) {
    const dlg = ensureDialog();
    const titleText = diagramEl.querySelector(".diagram-title")?.textContent.trim() || "Diagram";
    dlg.querySelector(".diagram-modal__title").textContent = titleText;
    const stage = dlg.querySelector(".diagram-modal__stage");

    // Reset pan/zoom for the new diagram
    dlg._diagramReset();

    // Move the original SVG into the stage (avoids cross-document ID collisions on
    // mermaid's internal markers/gradients), restore on close.
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
