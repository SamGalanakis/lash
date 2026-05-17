/* ─────────────────────────────────────────────────────────────
   lash · docs · procedural landscape footer
   The nun catches sodium light from one or more suns; the
   brazier on the left peak re-emits it into the spine that
   runs the length of the page. Strand count = sun count.
   Landing uses a taller frame + larger figures; subpages use
   the compact default. Configure via window.__LASH_SCENE = {
     aspect: 7/16,     // 5/16 default for subpages, 7/16 for landing
     scaleMin: 1.2,    // figure scale minimum (default 1.0)
     scaleMax: 1.5,    // figure scale maximum (default 1.2)
     spineSelector: ".pager"  // element whose left rail the spine descends through
   };
   ────────────────────────────────────────────────────────── */

(function () {
  "use strict";
  const cfg = (window.__LASH_SCENE = window.__LASH_SCENE || {});
  const ASPECT      = cfg.aspect      != null ? cfg.aspect      : 5 / 16;
  const SCALE_MIN   = cfg.scaleMin    != null ? cfg.scaleMin    : 1.0;
  const SCALE_MAX   = cfg.scaleMax    != null ? cfg.scaleMax    : 1.2;
  const SPINE_SEL   = cfg.spineSelector || ".pager";

  const SVG_NS = "http://www.w3.org/2000/svg";
  let W = 1600;
  let H = 500;
  let HORIZON = H * 0.78;
  let PALETTE = {};

  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    PALETTE = {
      far:  { light: v("--range-far-light"),  shadow: v("--range-far-shadow")  },
      mid:  { light: v("--range-mid-light"),  shadow: v("--range-mid-shadow")  },
      near: { light: v("--range-near-light"), shadow: v("--range-near-shadow") },
      sodium:        v("--sodium"),
      dark:          v("--nun-robe"),
      chalk:         v("--nun-headdress"),
      chalkDim:      v("--nun-chin"),
      instrument:    v("--instrument-dark"),
      ground:        v("--ground"),
      ground2:       v("--ground2"),
      horizonLine:   v("--horizon-line"),
      haze:          v("--haze"),
      skyTop:        v("--sky-top"),
      skyMid:        v("--sky-mid"),
      skyBot:        v("--sky-bot"),
    };
  }

  function rand(min, max) { return min + Math.random() * (max - min); }
  function jitter(v, j) { return v + (Math.random() - 0.5) * 2 * j; }
  function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

  function jaggedPath(p1, p2, segments, roughness) {
    const out = [];
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      let x = p1.x + (p2.x - p1.x) * t;
      let y = p1.y + (p2.y - p1.y) * t;
      if (i > 0 && i < segments) {
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        const disp = dist * roughness;
        x += (Math.random() - 0.5) * disp;
        y += (Math.random() - 0.5) * disp;
      }
      out.push({ x, y });
    }
    return out;
  }
  function ptsToD(pts) {
    return pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  }
  function el(tag, attrs) {
    const e = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function bz(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return {
      x: u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x,
      y: u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y,
    };
  }
  function bzT(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return {
      x: 3*u*u*(p1.x-p0.x) + 6*u*t*(p2.x-p1.x) + 3*t*t*(p3.x-p2.x),
      y: 3*u*u*(p1.y-p0.y) + 6*u*t*(p2.y-p1.y) + 3*t*t*(p3.y-p2.y),
    };
  }
  function taperedPath(p0, p1, p2, p3, w0, w1, samples) {
    samples = samples || 28;
    const upper = [], lower = [];
    for (let i = 0; i <= samples; i++) {
      const t = i / samples;
      const p = bz(p0, p1, p2, p3, t);
      const d = bzT(p0, p1, p2, p3, t);
      const len = Math.hypot(d.x, d.y) || 1;
      const nx = -d.y / len;
      const ny =  d.x / len;
      const k = 1 - Math.pow(1 - t, 2);
      const w = (w0 + (w1 - w0) * k) / 2;
      upper.push({ x: p.x + nx * w, y: p.y + ny * w });
      lower.push({ x: p.x - nx * w, y: p.y - ny * w });
    }
    const u = upper.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const l = lower.slice().reverse().map(p => `L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    return `${u} ${l} Z`;
  }

  function makeMountain(parent, opts) {
    const { x, baseY, width, height, lightFill, shadowFill, peakOffset, complexity, roughness } = opts;
    const baseLeft  = { x: x,            y: baseY };
    const baseRight = { x: x + width,    y: baseY };
    const peak      = { x: x + width * peakOffset, y: baseY - height };
    const baseCenter = { x: x + width * peakOffset, y: baseY };
    const shadowPts = [ ...jaggedPath(baseLeft, peak, complexity, roughness), baseCenter ];
    const lightPts  = [ ...jaggedPath(peak, baseRight, complexity, roughness), baseCenter ];
    parent.appendChild(el("path", { d: ptsToD(shadowPts) + " Z", fill: shadowFill }));
    parent.appendChild(el("path", { d: ptsToD(lightPts)  + " Z", fill: lightFill  }));
    return { peakX: peak.x, peakY: peak.y, x, width, height, baseY };
  }
  function makeRange(parent, opts) {
    const { count, baseY, hRange, wRange, palette, complexity, roughness } = opts;
    const peaks = [];
    const stride = W / count;
    for (let i = 0; i < count; i++) {
      const w = rand(wRange[0], wRange[1]);
      const h = rand(hRange[0], hRange[1]);
      const x = i * stride - (w - stride) / 2 + rand(-stride * 0.3, stride * 0.3);
      const peakOffset = rand(0.38, 0.62);
      peaks.push(makeMountain(parent, {
        x, baseY, width: w, height: h,
        lightFill: palette.light, shadowFill: palette.shadow,
        peakOffset, complexity, roughness,
      }));
    }
    return peaks;
  }

  function makeNun(parent, baseX, baseY, scale) {
    const s = scale || 1;
    const g = el("g", { transform: `translate(${baseX} ${baseY}) scale(${s})` });
    g.appendChild(el("path", { d: "M -26 4 C -26 -16 -16 -50 0 -62 C 16 -50 26 -16 26 4 Z", fill: PALETTE.dark }));
    g.appendChild(el("path", { d: "M -12 -34 Q -26 -64 -18 -84 L -6 -84 Q -14 -58 0 -34 Z", fill: PALETTE.dark }));
    g.appendChild(el("path", { d: "M 12 -34 Q 26 -64 18 -84 L 6 -84 Q 14 -58 0 -34 Z", fill: PALETTE.dark }));
    g.appendChild(el("path", { d: "M -28 -14 Q -12 -42 0 -68 Q 12 -42 28 -14 Q 0 -6 -28 -14 Z", fill: PALETTE.chalk }));
    g.appendChild(el("ellipse", { cx: 3, cy: -46, rx: 5, ry: 7.5, fill: "#c9a892", transform: "rotate(20 3 -46)" }));
    g.appendChild(el("circle", { cx: 4.4, cy: -47.8, r: 0.9, fill: "#3a2f2a" }));
    g.appendChild(el("circle", { cx: -0.4, cy: -48.4, r: 0.9, fill: "#3a2f2a" }));
    g.appendChild(el("path", { d: "M -8 -33 Q 4 -28 13 -38 Q 11 -50 9 -53 Q 4 -40 -8 -33 Z", fill: PALETTE.chalkDim }));
    g.appendChild(el("path", { d: "M -7 -84 L 7 -84 L 4 -92 L -4 -92 Z", fill: PALETTE.instrument }));
    g.appendChild(el("path", { d: "M -32 -100 Q 0 -82 32 -100 L 22 -90 Q 0 -78 -22 -90 Z", fill: PALETTE.instrument }));
    g.appendChild(el("path", { d: "M -25 -97 Q 0 -82 25 -97 Q 0 -86 -25 -97 Z", fill: PALETTE.sodium, opacity: 0.92 }));
    g.appendChild(el("circle", { cx: 0, cy: -108, r: 5, fill: "none", stroke: PALETTE.sodium, "stroke-width": 0.9, opacity: 0.7 }));
    g.appendChild(el("circle", { cx: 0, cy: -108, r: 2.4, fill: "#ffffff", opacity: 0.95 }));
    parent.appendChild(g);
    return { x: baseX, y: baseY - 108 * s };
  }

  function makeBrazier(parent, baseX, baseY, scale) {
    const s = scale || 1;
    const g = el("g", { transform: `translate(${baseX} ${baseY}) scale(${s})` });
    g.appendChild(el("path", { d: "M -32 -8 Q 0 10 32 -8 L 22 2 Q 0 14 -22 2 Z", fill: PALETTE.instrument }));
    g.appendChild(el("path", { d: "M -26 -5 Q 0 9 26 -5 Q 0 1 -26 -5 Z", fill: PALETTE.sodium, opacity: 0.92 }));
    g.appendChild(el("path", { d: "M 0 -6 Q -6 -22 -2 -38 Q 0 -44 2 -38 Q 6 -22 0 -6 Z", fill: PALETTE.sodium, opacity: 0.92 }));
    g.appendChild(el("path", { d: "M 0 -10 Q -2 -22 0 -36 Q 2 -22 0 -10 Z", fill: "#ffffff", opacity: 0.35 }));
    parent.appendChild(g);
    return { x: baseX, y: baseY - 44 * s };
  }

  function makeSun(parent, x, y, r) {
    // wrap in a group so the whole sun is one click target — clicking
    // a sun regenerates the scene (delight). aria-label + tabindex
    // make it keyboard-accessible too.
    const g = el("g", {
      class: "sun",
      tabindex: 0,
      role: "button",
      "aria-label": "regenerate landscape",
    });
    g.appendChild(el("circle", { class: "sun__glow",   cx: x, cy: y, r: r * 2.0, fill: PALETTE.sodium, opacity: 0.10 }));
    g.appendChild(el("circle", { class: "sun__corona", cx: x, cy: y, r: r * 1.30, fill: PALETTE.sodium, opacity: 0.22 }));
    g.appendChild(el("circle", { class: "sun__disc",   cx: x, cy: y, r: r, fill: PALETTE.sodium, opacity: 0.92 }));
    g.addEventListener("click", (ev) => {
      ev.preventDefault();
      generate();
    });
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        generate();
      }
    });
    parent.appendChild(g);
  }

  function makeHaze(parent, yTop, yBot) {
    const gradId = "hazeGrad-" + Math.random().toString(36).slice(2, 8);
    const defs = el("defs");
    const grad = el("linearGradient", { id: gradId, x1: 0, y1: 0, x2: 0, y2: 1 });
    grad.appendChild(el("stop", { offset: "0%",   "stop-color": PALETTE.haze, "stop-opacity": "0" }));
    grad.appendChild(el("stop", { offset: "55%",  "stop-color": PALETTE.haze, "stop-opacity": "0.55" }));
    grad.appendChild(el("stop", { offset: "100%", "stop-color": PALETTE.haze, "stop-opacity": "0" }));
    defs.appendChild(grad);
    parent.appendChild(defs);
    parent.appendChild(el("rect", { x: 0, y: yTop, width: W, height: yBot - yTop, fill: `url(#${gradId})` }));
  }

  function makeGround(parent) {
    parent.appendChild(el("rect", { x: 0, y: HORIZON, width: W, height: H - HORIZON, fill: PALETTE.ground }));
    parent.appendChild(el("rect", { x: 0, y: H - 30, width: W, height: 30, fill: PALETTE.ground2 }));
    parent.appendChild(el("line", { x1: 0, y1: HORIZON, x2: W, y2: HORIZON, stroke: PALETTE.horizonLine, "stroke-width": 1 }));
  }

  function refreshSkyGradient() {
    const stops = document.querySelectorAll("#skyGrad stop");
    if (stops.length >= 3) {
      stops[0].setAttribute("stop-color", PALETTE.skyTop);
      stops[1].setAttribute("stop-color", PALETTE.skyMid);
      stops[2].setAttribute("stop-color", PALETTE.skyBot);
    }
  }

  function clearGroup(id) {
    const g = document.getElementById(id);
    if (!g) return;
    while (g.firstChild) g.removeChild(g.firstChild);
  }

  function rayToNun(parent, from, to, w0, w1) {
    const start = { x: from.x, y: from.y };
    const c1    = { x: jitter(from.x + (to.x - from.x) * 0.25, 40),
                    y: jitter(from.y + 70, 25) };
    const c2    = { x: jitter(to.x + Math.sign(from.x - to.x) * 25, 25),
                    y: jitter(to.y - 70, 25) };
    const end   = to;
    parent.appendChild(el("path", {
      d: taperedPath(start, c1, c2, end, w0 || 5.5, w1 || 1.0),
      fill: PALETTE.sodium,
    }));
  }

  function generate() {
    readPalette();

    const svgEl = document.getElementById("range");
    if (!svgEl) return;
    const docClientW = document.documentElement.clientWidth || window.innerWidth || 1600;
    W = Math.max(640, Math.round(docClientW));
    H = Math.max(280, Math.round(W * ASPECT));
    HORIZON = H * 0.78;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const skyRect = document.getElementById("sky-rect");
    if (skyRect) {
      skyRect.setAttribute("width", W);
      skyRect.setAttribute("height", H);
    }

    refreshSkyGradient();
    ["g-back", "g-mid", "g-lash-behind", "g-near", "g-lash-front", "g-ground"].forEach(clearGroup);

    const back   = document.getElementById("g-back");
    const mid    = document.getElementById("g-mid");
    const lashB  = document.getElementById("g-lash-behind");
    const near   = document.getElementById("g-near");
    const lashF  = document.getElementById("g-lash-front");
    const ground = document.getElementById("g-ground");

    const backCount = Math.max(5, Math.round(W / 220));
    const midCount  = Math.max(4, Math.round(W / 280));
    const nearCount = Math.max(3, Math.round(W / 360));

    const sunX = jitter(W - 320, 80);
    const sunY = jitter(H * 0.22, 40);
    makeSun(back, sunX, sunY, 18);

    makeRange(back, {
      count: backCount, baseY: HORIZON - 5,
      hRange: [H * 0.11, H * 0.22], wRange: [220, 360],
      palette: PALETTE.far, complexity: 6, roughness: 0.06,
    });
    makeHaze(back, HORIZON - H * 0.10, HORIZON + H * 0.06);

    makeRange(mid, {
      count: midCount, baseY: HORIZON + 10,
      hRange: [H * 0.20, H * 0.34], wRange: [320, 460],
      palette: PALETTE.mid, complexity: 8, roughness: 0.07,
    });

    const nearPeaks = makeRange(near, {
      count: nearCount, baseY: HORIZON + 40,
      hRange: [H * 0.32, H * 0.51], wRange: [380, 540],
      palette: PALETTE.near, complexity: 10, roughness: 0.08,
    });

    // spine arrival X — converges with whichever HTML spine is on this
    // page. On docs subpages the snake exposes its bottom anchor X via
    // window.__LASH_SNAKE_ANCHOR_X (viewport coords); on the landing
    // we compute it from the rail-column math.
    const spineX = (function () {
      const svgBoxNow = svgEl.getBoundingClientRect();
      if (svgBoxNow.width === 0) return W * 0.10;
      if (typeof window.__LASH_SNAKE_ANCHOR_X === "number") {
        return (window.__LASH_SNAKE_ANCHOR_X - svgBoxNow.left) * W / svgBoxNow.width;
      }
      const railEl = document.querySelector(SPINE_SEL)
                  || document.querySelector(".body");
      const railBox = railEl ? railEl.getBoundingClientRect() : null;
      if (!railBox) return W * 0.10;
      const docStyle = getComputedStyle(document.documentElement);
      const remPx    = parseFloat(docStyle.fontSize);
      const spaceXl  = 2 * remPx;
      const railRem  = parseFloat(docStyle.getPropertyValue("--rail")) || 9;
      const railPx   = railRem * remPx;
      const spineCenterInBody = spaceXl + railPx * 14 / 120 + 0.5;
      const spineCenterX = railBox.left + spineCenterInBody;
      return (spineCenterX - svgBoxNow.left) * W / svgBoxNow.width;
    })();
    // when no spine is on the page (landing), let the brazier strand
    // rise STRAIGHT UP from wherever the brazier ends up — match the
    // visual on docs subpages where the strand goes vertical into the
    // TOC's spine.
    const strandStraightUp = (cfg.strandStraightUp === true) ||
      (typeof window.__LASH_SNAKE_ANCHOR_X !== "number" &&
       !document.querySelector(SPINE_SEL) &&
       !document.querySelector(".body"));
    const convergence = { x: spineX, y: 0 };

    const brazierZone = nearPeaks.filter(p =>
      p.peakX > convergence.x + 80 && p.peakX < W * 0.50);
    const nunZone = nearPeaks.filter(p => p.peakX > W * 0.58);
    const brazierPeak = brazierZone.length
      ? pick(brazierZone)
      : nearPeaks.slice().sort((a, b) =>
          Math.abs(a.peakX - (convergence.x + W * 0.18))
        - Math.abs(b.peakX - (convergence.x + W * 0.18)))[0];
    const nunPeak = nunZone.length
      ? pick(nunZone)
      : nearPeaks.slice().sort((a, b) => b.peakX - a.peakX)[0];

    const brazier = {
      baseX: brazierPeak.peakX,
      baseY: brazierPeak.peakY + rand(2, 8),
      scale: rand(SCALE_MIN, SCALE_MAX),
    };
    const brazierFocal = {
      x: brazier.baseX,
      y: brazier.baseY - 44 * brazier.scale,
    };

    // snap the convergence X onto the brazier so the strand draws as
    // a vertical line. The brazier zone was already picked relative
    // to the OLD convergence X; only the strand endpoint moves now.
    if (strandStraightUp) {
      convergence.x = brazierFocal.x;
    }

    const nun = {
      baseX: nunPeak.peakX,
      baseY: nunPeak.peakY + rand(30, 50),
      scale: rand(SCALE_MIN, SCALE_MAX),
    };
    const nunFocal = {
      x: nun.baseX,
      y: nun.baseY - 108 * nun.scale,
    };

    // INPUT side: every visible sun sends a ray into the nun's dish.
    let rayCount = 1;
    rayToNun(lashB, { x: sunX, y: sunY }, nunFocal, 6.5, 1.0);

    const extraSunCount = Math.floor(rand(0, 2.999));
    for (let i = 0; i < extraSunCount; i++) {
      const ex = jitter(W * 0.55 + i * W * 0.18, 60);
      const ey = jitter(H * 0.20 + i * H * 0.06, 40);
      makeSun(back, ex, ey, rand(11, 15));
      rayToNun(lashB, { x: ex, y: ey }, nunFocal, rand(3.2, 4.8), rand(0.7, 1.0));
      rayCount++;
    }

    makeBrazier(lashF, brazier.baseX, brazier.baseY, brazier.scale);
    makeNun(lashF, nun.baseX, nun.baseY, nun.scale);

    // OUTPUT side: rayCount strands from brazier to spine convergence.
    // Skipped when an overlay (like the landing whip) is going to draw
    // the strand itself in viewport coords — we don't want a duplicate.
    if (!cfg.suppressOutputStrand) {
      const N = rayCount;
      const dx = convergence.x - brazierFocal.x;
      const dy = convergence.y - brazierFocal.y;
      const len = Math.hypot(dx, dy) || 1;
      const perpX = -dy / len;
      const perpY =  dx / len;
      const spread = 110 + N * 30;
      const verticalRunIn = Math.min(len * 0.40, 140);
      for (let i = 0; i < N; i++) {
        const t = N === 1 ? 0 : (i / (N - 1)) - 0.5;
        const fanOffset = t * spread;
        const c1x = brazierFocal.x + dx * 0.35 + perpX * fanOffset;
        const c1y = brazierFocal.y + dy * 0.35 + perpY * fanOffset;
        const c2x = convergence.x;
        const c2y = convergence.y + verticalRunIn;
        lashF.appendChild(el("path", {
          d: `M ${brazierFocal.x} ${brazierFocal.y} `
           + `C ${c1x} ${c1y}, ${c2x} ${c2y}, ${convergence.x} ${convergence.y}`,
          fill: "none",
          stroke: PALETTE.sodium,
          "stroke-width": 1,
          "stroke-linecap": "round",
          "vector-effect": "non-scaling-stroke",
        }));
      }
    }

    makeGround(ground);

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const revisit = document.documentElement.classList.contains("lash-revisit");
    if (!reduced && !revisit) {
      svgEl.style.opacity = "0";
      requestAnimationFrame(() => {
        svgEl.style.transition = "opacity 320ms cubic-bezier(0.2, 0.6, 0.2, 1)";
        svgEl.style.opacity = "1";
      });
    }
  }

  // expose so theme-init.js can repaint on theme switch
  window.__LASH_SCENE_RUN = generate;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", generate);
  } else {
    generate();
  }
  const regen = document.getElementById("regen");
  if (regen) regen.addEventListener("click", generate);
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(generate, 180);
  });
})();
