/* lash · docs · runtime-boundary turn trace
   ─────────────────────────────────────────────────────────────
   A single calm sodium pulse traces one turn through the runtime:
   plugin hooks compose the turn, the deterministic loop emits an
   effect, it crosses the typed boundary out to a host resource, the
   outcome returns, and once per turn the runtime commits atomically
   to the session store. One pass per ~8.4s with a rest before it
   repeats.

   Decorative only. The static SVG is the real diagram and is fully
   legible without JS, with reduced motion, or on a headless render —
   every pulse fades in over already-visible structure. The driver
   pauses itself whenever the figure scrolls out of view.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  const svg = document.querySelector(".stack__svg");
  const fig = document.querySelector(".stack");
  if (!svg || !fig || typeof svg.animate !== "function") return;

  const CYCLE = 8400; // ms — one full trace plus a calm rest
  const SEG = (name) => '[data-seg="' + name + '"]';
  const GLOW = (name) => '[data-glow="' + name + '"]';
  const HOOK = (i) => '[data-glow="hook"][data-i="' + i + '"]';

  // Each signal sweeps a short bright dash along a real connector
  // during [s, e] (fractions of the cycle), then parks invisibly.
  const SIGNALS = [
    { sel: SEG("cross-out"), s: 0.27, e: 0.43 }, // effect crosses out
    { sel: SEG("tools"), s: 0.31, e: 0.46 }, //      a second effect, to tools
    { sel: SEG("cross-back"), s: 0.50, e: 0.66 }, // outcome returns
    { sel: SEG("commit"), s: 0.72, e: 0.88 }, //     the turn commits
  ];

  // Nodes brighten as the pulse reaches them. `w` is the half-width of
  // each pulse in cycle fractions; hooks flash quicker than boxes.
  const GLOWS = [
    // plugin hooks compose the turn — a quick cascade into the core
    { sel: HOOK(0), peaks: [[0.05, 0.9]], w: 0.03 },
    { sel: HOOK(1), peaks: [[0.07, 0.9]], w: 0.03 },
    { sel: HOOK(2), peaks: [[0.09, 0.9]], w: 0.03 },
    { sel: HOOK(3), peaks: [[0.11, 0.9]], w: 0.03 },
    { sel: HOOK(4), peaks: [[0.13, 0.9]], w: 0.03 },
    { sel: HOOK(5), peaks: [[0.15, 0.9]], w: 0.03 },
    // the loop runs (emit), then again when the outcome lands
    { sel: GLOW("core"), peaks: [[0.22, 0.6], [0.69, 0.48]] },
    // the boundary flares as effects and the commit cross it
    { sel: GLOW("membrane"), peaks: [[0.34, 0.95], [0.57, 0.8], [0.79, 0.9]] },
    { sel: GLOW("provider"), peaks: [[0.45, 0.9]] },
    { sel: GLOW("tools"), peaks: [[0.47, 0.78]] },
    { sel: GLOW("store"), peaks: [[0.87, 0.9]] },
  ];

  const reduceMQ = window.matchMedia("(prefers-reduced-motion: reduce)");
  let anims = [];

  function signalFrames(s, e) {
    const eps = 0.006;
    const lit = Math.min(s + eps, e - 2 * eps);
    const out = Math.max(lit + eps, e - eps);
    return [
      { strokeDashoffset: 12, opacity: 0, offset: 0 },
      { strokeDashoffset: 12, opacity: 0, offset: s },
      { strokeDashoffset: 12, opacity: 1, offset: lit, easing: "cubic-bezier(.42,0,.58,1)" },
      { strokeDashoffset: -104, opacity: 1, offset: out },
      { strokeDashoffset: -104, opacity: 0, offset: e },
      { strokeDashoffset: -104, opacity: 0, offset: 1 },
    ];
  }

  function glowFrames(peaks, w) {
    const hw = w || 0.05;
    // Per-keyframe easing keeps each pulse soft; iteration easing stays
    // linear (below) so keyframe offsets map straight to cycle time.
    const f = [{ opacity: 0, offset: 0 }];
    peaks.forEach(([at, p]) => {
      f.push({ opacity: 0, offset: Math.max(0.0001, at - hw), easing: "ease-out" });
      f.push({ opacity: p, offset: at, easing: "ease-in" });
      f.push({ opacity: 0, offset: Math.min(0.9999, at + hw) });
    });
    f.push({ opacity: 0, offset: 1 });
    return f;
  }

  function build() {
    if (anims.length || reduceMQ.matches) return;
    SIGNALS.forEach((c) => {
      const el = svg.querySelector(c.sel);
      if (!el) return;
      anims.push(el.animate(signalFrames(c.s, c.e), {
        duration: CYCLE, iterations: Infinity, easing: "linear",
      }));
    });
    GLOWS.forEach((c) => {
      const el = svg.querySelector(c.sel);
      if (!el) return;
      anims.push(el.animate(glowFrames(c.peaks, c.w), {
        duration: CYCLE, iterations: Infinity, easing: "linear",
      }));
    });
    anims.forEach((a) => a.pause()); // resumed by the observer when on-screen
  }

  function teardown() {
    anims.forEach((a) => a.cancel());
    anims = [];
  }

  // Only run while the figure is on-screen — pause what can't be seen.
  let visible = false;
  const io = new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    anims.forEach((a) => (visible ? a.play() : a.pause()));
  }, { threshold: 0.12 });
  io.observe(fig);

  function sync() {
    if (reduceMQ.matches) {
      teardown();
    } else {
      build();
      anims.forEach((a) => (visible ? a.play() : a.pause()));
    }
  }

  if (typeof reduceMQ.addEventListener === "function") {
    reduceMQ.addEventListener("change", sync);
  }
  sync();
})();
