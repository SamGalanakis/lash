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
  // flushBottom: drop the ground floor entirely and plant mountains
  // at the bottom edge of the SVG so they bleed into the page bg.
  // Default true everywhere — explicitly opt out per page if needed.
  const FLUSH_BOTTOM = cfg.flushBottom !== false;
  // First call to generate() is the page's intro; later calls are
  // user-triggered regens. The intro paces slower so the eye can
  // register the scene as it composes itself.
  let isFirstRun = true;

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
    // a sun bursts in place, every lash recedes into its source sun,
    // and after a still pause the world re-slides and re-lights.
    const g = el("g", {
      class: "sun",
      tabindex: 0,
      role: "button",
      "aria-label": "regenerate landscape",
    });
    g.appendChild(el("circle", { class: "sun__glow",   cx: x, cy: y, r: r * 2.0, fill: PALETTE.sodium, opacity: 0.10 }));
    g.appendChild(el("circle", { class: "sun__corona", cx: x, cy: y, r: r * 1.30, fill: PALETTE.sodium, opacity: 0.22 }));
    g.appendChild(el("circle", { class: "sun__disc",   cx: x, cy: y, r: r, fill: PALETTE.sodium, opacity: 0.92 }));
    const burst = (ev) => {
      ev.preventDefault();
      if (g.classList.contains("sun--burst")) return;
      g.classList.add("sun--burst");
      // ── BEAT 1 / inhale ──────────────────────────────────
      // Every visible lash collapses to its own sun via its
      // pinned transform-origin. Inline transition overrides
      // the slower CSS default so the inhale ends crisply.
      const svgRoot = document.getElementById("range");
      if (svgRoot) {
        svgRoot.querySelectorAll(".ray-emit").forEach(r => {
          r.style.transition =
            "transform 280ms cubic-bezier(0.4, 0, 0.2, 1), " +
            "opacity 200ms cubic-bezier(0.4, 0, 0.2, 1)";
          r.style.transform = "scale(0.001)";
          r.style.opacity = "0";
        });
      }
      // 480ms ≈ 280 recede + 200 of stillness, so beat 1 is
      // visibly finished before mountains begin to slide.
      setTimeout(() => generate({ trigger: "sun", fromX: x }), 480);
    };
    g.addEventListener("click", burst);
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") burst(ev);
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

  // ─── parallax swap helpers ─────────────────────────────────
  // Each layer group keeps a single `.scene-enter` wrapper. On
  // regen, that wrapper is reclassed to `.scene-exit` (slides off
  // + fades) and a new `.scene-enter` is appended pre-translated
  // so a CSS transition can ease it home. Far ranges move less
  // than near ranges → parallax depth. Transforms live on inline
  // style because SVG <g> handles CSS !important on transform
  // unreliably; inline-wins is simpler.
  const ENTER_OFFSET = { back: 60, mid: 120, near: 220, figures: 220, ground: 40 };
  const EXIT_OFFSET  = { back: 90, mid: 170, near: 280, figures: 280, ground: 60 };
  function offsetCSS(depth, dir, table) {
    const off = table[depth] || 0;
    const sign = dir === "right" ? 1 : -1;
    return depth === "ground"
      ? `translateY(${sign * off}px)`
      : `translateX(${sign * off}px)`;
  }
  function reclassExit(parentId, depth, dir) {
    const parent = document.getElementById(parentId);
    if (!parent) return null;
    // kill any older exits — don't let them stack if the user mashes suns
    parent.querySelectorAll(":scope > .scene-exit").forEach(n => n.remove());
    const current = parent.querySelector(":scope > .scene-enter");
    if (!current) return null;
    current.classList.remove("scene-enter", "scene-enter--settled",
      "scene-enter--depth-back", "scene-enter--depth-mid",
      "scene-enter--depth-near", "scene-enter--depth-figures",
      "scene-enter--depth-ground");
    current.classList.add("scene-exit", `scene-exit--depth-${depth}`);
    current.style.transform = offsetCSS(depth, dir, EXIT_OFFSET);
    current.style.opacity = "0";
    setTimeout(() => { if (current && current.parentNode) current.remove(); }, 1100);
    return current;
  }
  function makeEnter(parentId, depth, dir) {
    const parent = document.getElementById(parentId);
    if (!parent) return null;
    const g = el("g", { class: `scene-enter scene-enter--depth-${depth}` });
    g.style.transform = offsetCSS(depth, dir, ENTER_OFFSET);
    g.style.opacity = "0";
    parent.appendChild(g);
    return g;
  }
  function settleEnter(g) {
    // Stale guard — a later generate may have reclassed this wrapper
    // to .scene-exit by the time the RAF fires; don't yank it back.
    if (!g || !g.classList.contains("scene-enter")) return;
    g.style.transform = "translate(0,0)";
    g.style.opacity = "1";
    g.classList.add("scene-enter--settled");
  }

  function rayToNun(parent, from, to, w0, w1) {
    const start = { x: from.x, y: from.y };
    const c1    = { x: jitter(from.x + (to.x - from.x) * 0.25, 40),
                    y: jitter(from.y + 70, 25) };
    const c2    = { x: jitter(to.x + Math.sign(from.x - to.x) * 25, 25),
                    y: jitter(to.y - 70, 25) };
    const end   = to;
    const path = el("path", {
      class: "ray-emit",
      d: taperedPath(start, c1, c2, end, w0 || 5.5, w1 || 1.0),
      fill: PALETTE.sodium,
    });
    // Anchor scale at the sun → on grow, ray extrudes toward the nun;
    // on recede (sun click), it collapses cleanly into its own source.
    path.style.transformBox = "view-box";
    path.style.transformOrigin = `${from.x}px ${from.y}px`;
    path.style.transform = "scale(0.001)";
    path.style.opacity = "0";
    parent.appendChild(path);
  }

  function generate(runOpts) {
    runOpts = runOpts || {};
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

    // FLUSH_BOTTOM: drop the visible ground strip by pushing the
    // horizon down to (and past) the bottom edge. Mountain bases sit
    // off-canvas, slopes carry to the bottom — page bg shows through
    // anywhere a mountain doesn't reach.
    if (FLUSH_BOTTOM) {
      HORIZON = H * 0.98;
    }

    refreshSkyGradient();

    // Decide slide directions. We want mountains to slide PAST each
    // other: alternate by depth, with a random global flip so it never
    // feels mechanical. If a sun was clicked, bias by click side so
    // the motion radiates away from the impact.
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const flip = (typeof runOpts.fromX === "number")
      ? (runOpts.fromX < W * 0.5 ? 1 : -1)
      : (Math.random() < 0.5 ? 1 : -1);
    const sideOf = (s) => s > 0 ? "right" : "left";
    const dirs = {
      back:   sideOf( flip),
      mid:    sideOf(-flip),
      lashB:  sideOf( flip),
      near:   sideOf(-flip),
      lashF:  sideOf(-flip),
      ground: sideOf( flip),
    };
    const exitDirOf = (d) => d === "right" ? "left" : "right";
    reclassExit("g-back",        "back",    exitDirOf(dirs.back));
    reclassExit("g-mid",         "mid",     exitDirOf(dirs.mid));
    reclassExit("g-lash-behind", "figures", exitDirOf(dirs.lashB));
    reclassExit("g-near",        "near",    exitDirOf(dirs.near));
    reclassExit("g-lash-front",  "figures", exitDirOf(dirs.lashF));
    reclassExit("g-ground",      "ground",  exitDirOf(dirs.ground));

    const back   = makeEnter("g-back",        "back",    dirs.back);
    const mid    = makeEnter("g-mid",         "mid",     dirs.mid);
    const lashB  = makeEnter("g-lash-behind", "figures", dirs.lashB);
    const near   = makeEnter("g-near",        "near",    dirs.near);
    const lashF  = makeEnter("g-lash-front",  "figures", dirs.lashF);
    const ground = makeEnter("g-ground",      "ground",  dirs.ground);

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
        const strand = el("path", {
          class: "ray-emit ray-emit--strand",
          d: `M ${brazierFocal.x} ${brazierFocal.y} `
           + `C ${c1x} ${c1y}, ${c2x} ${c2y}, ${convergence.x} ${convergence.y}`,
          fill: "none",
          stroke: PALETTE.sodium,
          "stroke-width": 1,
          "stroke-linecap": "round",
          "vector-effect": "non-scaling-stroke",
        });
        strand.style.transformBox = "view-box";
        strand.style.transformOrigin = `${brazierFocal.x}px ${brazierFocal.y}px`;
        strand.style.transform = "scale(0.001)";
        strand.style.opacity = "0";
        lashF.appendChild(strand);
      }
    }

    if (!FLUSH_BOTTOM) makeGround(ground);

    // Collect rays for the BEAT 2.b bloom (grow lashes out of the new
    // suns toward the nun) — fired after the slowest mountain lands.
    const rays = [];
    [lashB, lashF].forEach(w => {
      if (w) w.querySelectorAll(".ray-emit").forEach(r => rays.push(r));
    });
    const bloomRays = () => {
      rays.forEach(r => {
        const wrapper = r.parentNode;
        if (!wrapper || !wrapper.classList.contains("scene-enter")) return;
        // Clear any inline transition left from a recede so the bloom
        // uses the slower, more elegant CSS default.
        r.style.transition = "";
        r.style.transform = "scale(1)";
        r.style.opacity = "1";
      });
    };

    const enters = [back, mid, lashB, near, lashF, ground].filter(Boolean);
    // On the very first generate, mark the cover as `cover--intro`
    // so the CSS applies the slower, more cinematic pace. We strip
    // the class once the intro has had time to play. Revisits in
    // the same session (lash-revisit) skip the slow intro — they
    // use the snappier regen pace so the page feels responsive.
    const cover = svgEl.closest(".cover");
    const revisit = document.documentElement.classList.contains("lash-revisit");
    const intro = isFirstRun && !reduced && !revisit;
    if (intro && cover) cover.classList.add("cover--intro");
    if (reduced) {
      enters.forEach(e => {
        e.style.transition = "none";
        settleEnter(e);
      });
      rays.forEach(r => {
        r.style.transition = "none";
        r.style.transform = "scale(1)";
        r.style.opacity = "1";
      });
    } else {
      // setTimeout instead of rAF — rAF is throttled to zero in
      // hidden/background tabs, which would leave the scene stuck at
      // its offset entrance forever. Two staggered setTimeouts give
      // the browser a layout pass between "initial offset" and
      // "settled" so the CSS transition still observes the change.
      // Intro adds a small extra beat of "held" time so the page can
      // paint before motion begins.
      const settleAfter = intro ? 120 : 16;
      setTimeout(() => {
        void svgEl.getBoundingClientRect();
        setTimeout(() => {
          enters.forEach(settleEnter);
        }, 16);
      }, settleAfter);
      // Hold the bloom until the slowest layer has fully landed.
      // Intro slide is longer + stagger is wider, so push the bloom
      // out accordingly. Regen keeps the snappier ~1100ms.
      const bloomDelay = intro ? 1900 : 1100;
      setTimeout(bloomRays, bloomDelay);
      // Spine reveal — the brazier strand bloom takes ~980ms during
      // intro to grow from brazier up to the spine attachment. Fire
      // spine-revealed slightly BEFORE the strand fully lands so the
      // two motions overlap by a frame and read as one continuous
      // line. Idempotent — first set triggers the keyframe, later
      // regens no-op (class already present).
      if (intro) {
        setTimeout(() => {
          document.documentElement.classList.add("spine-revealed");
        }, bloomDelay + 850);
      }
      // strip the intro class after enough time that all transitions
      // have finished — re-applying it on a later generate would do
      // nothing (the wrappers are fresh each time).
      if (intro && cover) {
        setTimeout(() => cover.classList.remove("cover--intro"), 3600);
      }
    }
    isFirstRun = false;
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
