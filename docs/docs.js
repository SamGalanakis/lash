/* lash · docs · TOC builder + auto-knot + mermaid loader
   On pages with body.docs-page, build the multi-level TOC sidebar
   and highlight the current page. On the landing (no .docs-page),
   auto-knot each top-level .section/.stanza in .body. */

(function () {
  "use strict";

  // ── single global TOC structure for the docs ──────────────
  // Edit this when adding / renaming / reordering pages.
  // hrefs are relative to /docs/ (top level); architecture pages
  // use "architecture/<name>.html".
  // Items can be a plain link {name, href} OR a nested subgroup
  // {label, items: [...]} that renders as a small heading + indented
  // children. Subgroups exist so repeated prefixes like "lashlang ·"
  // collapse into one label.
  const TOC = [
    {
      label: "start",
      items: [
        { name: "quickstart",    href: "quickstart.html" },
        { name: "cli",           href: "cli.html" },
        { name: "agent service", href: "example-agent-service.html" },
      ],
    },
    {
      label: "lash api",
      items: [
        { name: "basics",   href: "embedding.html" },
        { name: "turns",    href: "embedding-turns.html" },
        { name: "prompts",  href: "embedding-prompts.html" },
        { name: "advanced", href: "embedding-advanced.html" },
      ],
    },
    {
      label: "plugins",
      items: [
        { name: "writing", href: "plugins.html" },
        { name: "tools",   href: "plugins-tools.html" },
        { name: "runtime", href: "plugins-runtime.html" },
      ],
    },
    {
      label: "lashlang",
      items: [
        { name: "language", href: "architecture/lashlang.html" },
        { name: "effects",  href: "architecture/lashlang-effects.html" },
        { name: "runtime",  href: "architecture/lashlang-runtime.html" },
      ],
    },
    {
      label: "architecture",
      items: [
        { name: "system",          href: "architecture/index.html" },
        { name: "overview",        href: "architecture/overview.html" },
        { name: "modules",         href: "architecture/modules.html" },
        { name: "flow",            href: "architecture/flow.html" },
        { name: "execution",       href: "architecture/execution.html" },
        { name: "execution modes", href: "architecture/execution-modes.html" },
        { name: "runtime",         href: "architecture/runtime.html" },
        { name: "abstractions",    href: "architecture/abstractions.html" },
      ],
    },
    {
      label: "persistence",
      items: [
        { name: "overview",   href: "persistence.html" },
        { name: "durability", href: "architecture/durability.html" },
      ],
    },
    {
      label: "tracing",
      items: [
        { name: "overview",     href: "tracing.html" },
        { name: "trace export", href: "trace-export-edges.html" },
      ],
    },
    {
      label: "reference",
      items: [
        { name: "providers",     href: "architecture/providers.html" },
        { name: "html exporter", href: "architecture/html-exporter.html" },
        { name: "deps",          href: "architecture/deps.html" },
        { name: "reference",     href: "architecture/reference.html" },
      ],
    },
  ];

  // Flatten the TOC into a single ordered list of {name, href} for
  // pager auto-fill and any other code that wants the linear order.
  function flatTOC() {
    const out = [];
    const visit = (item) => {
      if (item.items) {
        if (item.href) out.push({ name: item.label, href: item.href });
        item.items.forEach(visit);
      } else if (item.href) {
        out.push(item);
      }
    };
    TOC.forEach(visit);
    return out;
  }

  // ── helpers ─────────────────────────────────────────────
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function currentPath() {
    // strip leading slash + protocol + host; just keep the doc-relative bit
    let p = location.pathname;
    // find /docs/ in the path (works on GitHub Pages subpaths too) — fall
    // back to filename + parent dir
    const idx = p.indexOf("/docs/");
    if (idx >= 0) p = p.slice(idx + 6);
    else p = p.replace(/^\//, "");
    if (p === "" || p.endsWith("/")) p += "index.html";
    return p;
  }

  function tocBase() {
    // absolute URL path up to and including the /docs/ directory, so
    // TOC links resolve correctly from any current page after SPA
    // navigation (relative bases would rot once the URL changes
    // without rebuilding the TOC).
    let p = location.pathname;
    const idx = p.indexOf("/docs/");
    if (idx >= 0) return p.slice(0, idx + 6);
    // fallback for sites served at root
    return "/";
  }

  // ── TOC builder ─────────────────────────────────────────
  // Groups default to COLLAPSED — only the group containing the
  // active page auto-opens, plus any group the user has explicitly
  // expanded. Storage tracks the "expanded" set so user clicks
  // persist across navigations.
  const EXPANDED_KEY = "lash-toc-expanded";
  const CARET_SVG = '<svg class="toc__caret" viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><path d="M3.5 2 L8 6 L3.5 10" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function getExpandedGroups() {
    try {
      const raw = localStorage.getItem(EXPANDED_KEY);
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch (e) { return new Set(); }
  }
  function saveExpandedGroups(set) {
    try { localStorage.setItem(EXPANDED_KEY, JSON.stringify([...set])); }
    catch (e) { /* ignore quota / private mode */ }
  }

  function buildTOC() {
    const host = document.querySelector(".toc");
    if (!host) return;
    const cur = currentPath();
    const base = tocBase();
    const expanded = getExpandedGroups();

    // True if any direct or nested item under `group` matches `href`.
    function containsHref(group, href) {
      if (!group.items) return false;
      return group.items.some((it) =>
        it.items ? containsHref(it, href) : it.href === href
      );
    }

    // Render a single child of a group: either a leaf <li><a> or a
    // nested subgroup with its own small heading and indented list.
    function renderChild(item) {
      if (item.items) {
        const subActive = containsHref(item, cur);
        const subClasses = "toc__subgroup" + (subActive ? " is-active" : "");
        const lines = [];
        lines.push(`<li class="${subClasses}">`);
        lines.push(`<p class="toc__sub-label">${escapeHtml(item.label)}</p>`);
        lines.push(`<ol class="toc__subitems">`);
        for (const child of item.items) lines.push(renderChild(child));
        lines.push(`</ol>`);
        lines.push(`</li>`);
        return lines.join("");
      }
      const isActive = item.href === cur;
      return (
        `<li><a${isActive ? ' class="is-active" aria-current="page"' : ""} ` +
        `href="${escapeHtml(base + item.href)}">${escapeHtml(item.name)}</a></li>`
      );
    }

    const out = [];
    for (const group of TOC) {
      const containsActive = containsHref(group, cur);
      const groupActive = containsActive;
      const hasItems = group.items && group.items.length > 0;
      // Groups default to collapsed. Auto-open the group containing
      // the active page (so users see where they are), plus any
      // groups the reader has explicitly expanded previously.
      const isCollapsed = hasItems && !containsActive && !expanded.has(group.label);
      const classes = [
        "toc__group",
        hasItems ? "toc__group--collapsible" : "",
        isCollapsed ? "is-collapsed" : "",
      ].filter(Boolean).join(" ");
      out.push(`<nav class="${classes}" data-group="${escapeHtml(group.label)}">`);
      const titleClasses = "toc__group-title" + (groupActive ? " is-active" : "");
      // Title is a button when the group has items — the whole row
      // toggles collapse, not just the caret. The caret stays as a
      // visual indicator that rotates with the group's open state.
      if (hasItems) {
        out.push(
          `<button class="${titleClasses}" type="button" aria-expanded="${!isCollapsed}">` +
            `<span class="toc__caret-wrap" aria-hidden="true">${CARET_SVG}</span>` +
            `<span class="toc__group-title-text">${escapeHtml(group.label)}</span>` +
          `</button>`
        );
        out.push('<ol class="toc__items">');
        for (const item of group.items) out.push(renderChild(item));
        out.push("</ol>");
      } else {
        out.push(
          `<p class="${titleClasses}">` +
            `<span class="toc__caret-spacer" aria-hidden="true"></span>` +
            `<span class="toc__group-title-text">${escapeHtml(group.label)}</span>` +
          `</p>`
        );
      }
      out.push(`</nav>`);
    }
    host.innerHTML = out.join("\n");

    // wire title toggles — the whole title row is the toggle button
    host.querySelectorAll("button.toc__group-title").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const group = btn.closest(".toc__group");
        if (!group) return;
        const label = group.getAttribute("data-group");
        const nowCollapsed = group.classList.toggle("is-collapsed");
        btn.setAttribute("aria-expanded", String(!nowCollapsed));
        const current = getExpandedGroups();
        if (nowCollapsed) current.delete(label);
        else current.add(label);
        saveExpandedGroups(current);
        // redraw spine since group heights changed
        drawSpineSnake();
      });
    });
  }

  // ── pager auto-fill from neighbours in the flat page list ──
  function buildPager() {
    const host = document.querySelector(".pager__main");
    if (!host) return;
    // skip if the page already provided explicit prev/next links
    if (host.querySelector(".pager__prev, .pager__next")) return;
    const flat = flatTOC();
    const cur = currentPath();
    const idx = flat.findIndex(e => e.href === cur);
    if (idx < 0) return;
    const base = tocBase();
    const prev = flat[idx - 1];
    const next = flat[idx + 1];
    const prevHtml = prev
      ? `<a class="pager__prev" href="${escapeHtml(base + prev.href)}">` +
        `<span class="arrow" aria-hidden="true">←</span>` +
        `<span>previous</span><strong>${escapeHtml(prev.name)}</strong>` +
        `</a>`
      : '<span></span>';
    const nextHtml = next
      ? `<a class="pager__next" href="${escapeHtml(base + next.href)}">` +
        `<span>next</span><strong>${escapeHtml(next.name)}</strong>` +
        `<span class="arrow" aria-hidden="true">→</span>` +
        `</a>`
      : '<span></span>';
    host.innerHTML = prevHtml + nextHtml;
  }

  // ── auto-knot — only on the landing (no .docs-page class) ──
  function autoKnot() {
    if (document.body.classList.contains("docs-page")) return;
    const body = document.querySelector(".body");
    if (!body) return;
    const items = body.querySelectorAll(":scope > section, :scope > .section, :scope > .stanza");
    let idx = 0;
    items.forEach((item) => {
      if (item.querySelector(":scope > .stanza__rail, :scope > .section__rail")) return;
      idx += 1;
      let knotName = item.dataset.knot || "";
      if (!knotName) {
        const h2 = item.querySelector("h2");
        if (h2) knotName = h2.textContent.replace(/[\.•·…]+\s*$/u, "").trim();
      }
      const knotId = "§ " + String(idx).padStart(2, "0");
      const rail = document.createElement("div");
      rail.className = "stanza__rail";
      rail.innerHTML =
        '<div class="knot">' +
          '<span class="knot__bead" aria-hidden="true"></span>' +
          '<p class="knot__id">' + escapeHtml(knotId) + '</p>' +
          (knotName ? '<p class="knot__name">' + escapeHtml(knotName.toLowerCase()) + '</p>' : '') +
        '</div>';
      item.prepend(rail);
    });
  }

  // ── mermaid loader — only if the page contains .mermaid elements ──
  // bumped MERMAID_LOADER_VERSION when mermaid.js changes so static
  // caches don't keep serving the old loader
  const MERMAID_LOADER_VERSION = "3";
  function loadMermaidIfNeeded() {
    if (!document.querySelector(".mermaid")) return;
    if (window.__lashMermaid) return;
    const base = tocBase();
    const url = base + "mermaid.js?v=" + MERMAID_LOADER_VERSION;
    const existing = document.querySelector('script[data-lash-mermaid]');
    if (existing) return;
    const s = document.createElement("script");
    s.src = url;
    s.defer = true;
    s.setAttribute("data-lash-mermaid", "");
    document.head.appendChild(s);
  }

  // ── spine thread — a thin vertical sodium rail down the left edge
  //    of the TOC. No eyelets; active/hover affordances live on the
  //    item rows themselves now (background + bold text).
  function drawSpineSnake() {
    if (!document.body.classList.contains("docs-page")) return;
    const refToc = document.querySelector(".toc");
    if (!refToc) return;
    const prior = refToc.querySelector(":scope > .spine-snake");
    if (prior) prior.remove();

    const tocRect = refToc.getBoundingClientRect();
    // anchorX is the X (relative to .toc) where the spine sits. It is
    // a small fixed gutter inside the TOC's left edge — the spine acts
    // as a left rail, with everything (titles + items) sitting indented
    // to the right of it. Driven by a CSS variable so it can't drift.
    const anchorX = (function () {
      const rootCS = getComputedStyle(document.documentElement);
      const remPx = parseFloat(rootCS.fontSize) || 16;
      const railVar = rootCS.getPropertyValue("--toc-rail-x").trim();
      const rail = parseFloat(railVar);
      if (!Number.isNaN(rail) && rail > 0) {
        // value is in rem if it ends with rem, else px
        return railVar.endsWith("rem") ? rail * remPx : rail;
      }
      return 0.375 * remPx; // ~6px fallback
    })();
    const H = Math.max(refToc.scrollHeight, refToc.clientHeight);
    if (H < 100) return;

    const W = 28;
    const mainX = 6;

    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "spine-snake");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.left = (anchorX - mainX) + "px";
    svg.style.width = W + "px";
    svg.style.height = H + "px";

    const thread = document.createElementNS(NS, "path");
    thread.setAttribute("d", `M ${mainX} 0 L ${mainX} ${H.toFixed(2)}`);
    thread.setAttribute("class", "spine-thread");
    thread.setAttribute("fill", "none");
    thread.setAttribute("stroke", "currentColor");
    thread.setAttribute("stroke-width", "1");
    thread.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(thread);

    refToc.insertBefore(svg, refToc.firstChild);

    // brazier strand at the bottom of the cover still aims at this X
    window.__LASH_SNAKE_ANCHOR_X = tocRect.left + anchorX;
  }

  // ── floating "back to contents" button — appears after the TOC
  //    has scrolled out of view; clicking jumps to the top
  function mountTocJump() {
    if (!document.body.classList.contains("docs-page")) return;
    const toc = document.querySelector(".toc");
    if (!toc) return;
    let btn = document.getElementById("toc-jump");
    if (!btn) {
      btn = document.createElement("button");
      btn.id = "toc-jump";
      btn.className = "toc-jump";
      btn.type = "button";
      btn.setAttribute("aria-label", "back to contents");
      btn.innerHTML =
        '<span class="arrow" aria-hidden="true">↑</span>' +
        '<span>contents</span>';
      btn.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      document.body.appendChild(btn);
    }
    function tick() {
      const r = toc.getBoundingClientRect();
      // visible when the TOC's bottom is above the viewport
      const past = r.bottom < 24;
      btn.classList.toggle("is-visible", past);
    }
    tick();
    window.addEventListener("scroll", tick, { passive: true });
    window.addEventListener("resize", tick);
  }

  // ── SPA navigation — fetch + swap so the sidebar stays put ──
  //    Clicking a TOC link (or any internal docs link) fetches the
  //    target page, swaps the per-page sections in place, and updates
  //    the URL via pushState. The sticky TOC keeps its scroll and
  //    spine state; only the content area refreshes.
  function refreshTOCActive() {
    const cur = currentPath();
    document.querySelectorAll(".toc__items a").forEach((a) => {
      const href = a.getAttribute("href") || "";
      const tail = href.replace(/^.*\/docs\//, "");
      const match = tail === cur;
      a.classList.toggle("is-active", match);
      if (match) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    document.querySelectorAll(".toc__group").forEach((g) => {
      const title = g.querySelector(".toc__group-title");
      if (!title) return;
      const any = !!g.querySelector(".toc__items a.is-active") ||
        (title.querySelector("a") &&
          (title.querySelector("a").getAttribute("href") || "").replace(/^.*\/docs\//, "") === cur);
      title.classList.toggle("is-active", any);
    });
  }

  // mermaid.js owns its own theme + re-render lifecycle. After an SPA
  // content swap we just nudge it to pick up any newly-inserted .mermaid
  // blocks.
  function runMermaidIfReady() {
    if (typeof window.__lashMermaidRerender === "function") {
      window.__lashMermaidRerender();
    }
  }

  // pull window.__LASH_SCENE config from a fetched HTML string (each
  // page's <head> sets a per-page aspect / scale tuple)
  function extractSceneConfig(html) {
    const m = html.match(/window\.__LASH_SCENE\s*=\s*(\{[^}]+\})/);
    if (!m) return null;
    try { return new Function("return " + m[1])(); } catch (e) { return null; }
  }

  async function spaNavigate(url, pushHistory) {
    const targetUrl = new URL(url, location.href);
    // capture the sidebar's internal scroll up-front so we can restore
    // it after the swap — focus/reflow side effects in the middle of
    // the flow can reset it otherwise
    const tocEl = document.querySelector(".toc");
    const savedTocScroll = tocEl ? tocEl.scrollTop : 0;
    try {
      const res = await fetch(targetUrl.href, { credentials: "same-origin" });
      if (!res.ok) throw new Error("bad status");
      const html = await res.text();
      const doc = new DOMParser().parseFromString(html, "text/html");

      // sanity: only SPA-swap if both old and new are docs pages.
      // The .docs-layout shell is BUILT at runtime by mountShell(),
      // so it isn't present in fetched HTML — we have to look at
      // body.docs-page (which IS in the source) plus the per-page
      // pieces we're about to swap. Otherwise fall back to a full
      // navigation so the landing's different structure renders fresh.
      const newIsDocs = doc.body && doc.body.classList.contains("docs-page");
      const curIsDocs = document.body.classList.contains("docs-page");
      const newOpener = doc.querySelector(".opener");
      const newBody   = doc.querySelector(".body");
      if (!newIsDocs || !curIsDocs || !newOpener || !newBody) {
        location.href = targetUrl.href;
        return;
      }

      if (pushHistory) {
        history.pushState({ spa: true }, "", targetUrl.href);
      }

      document.title = doc.title || document.title;
      // body class — guard the docs-page flag (and any future variants)
      document.body.className = doc.body.className;

      // swap opener, body, pager — leave .toc and .cover svg alone
      const swap = (selector) => {
        const fresh = doc.querySelector(selector);
        const here = document.querySelector(selector);
        if (fresh && here) here.replaceWith(fresh.cloneNode(true));
      };
      swap(".opener");
      swap(".body");
      swap(".pager");
      // cover caption: keep the SVG, only swap the source/caption text
      const newCap = doc.querySelector(".cover__caption");
      const curCap = document.querySelector(".cover__caption");
      if (newCap && curCap) curCap.innerHTML = newCap.innerHTML;

      // per-page scene config (aspect ratio etc.)
      const scene = extractSceneConfig(html);
      if (scene) window.__LASH_SCENE = scene;

      // post-swap re-init — TOC active state, spine, mermaid, scene
      refreshTOCActive();
      drawSpineSnake();
      loadMermaidIfNeeded();
      runMermaidIfReady();
      if (typeof window.__LASH_SCENE_RUN === "function") {
        window.__LASH_SCENE_RUN();
      }

      // scroll: anchor if present, else top of content (sidebar keeps
      // its own scrollTop — that's the point of all this)
      if (targetUrl.hash) {
        const id = decodeURIComponent(targetUrl.hash.slice(1));
        const target = document.getElementById(id);
        if (target) {
          target.scrollIntoView({ block: "start" });
        } else {
          window.scrollTo({ top: 0 });
        }
      } else {
        window.scrollTo({ top: 0 });
      }
      // restore sidebar scroll AFTER everything else — the focus/reflow
      // chain above can reset it back to 0 on Chrome
      if (tocEl) {
        tocEl.scrollTop = savedTocScroll;
        // and again on the next frame in case a deferred layout still
        // wants to undo it
        requestAnimationFrame(() => { tocEl.scrollTop = savedTocScroll; });
      }
    } catch (err) {
      // anything blew up — fall back to a regular navigation
      location.href = targetUrl.href;
    }
  }

  // expose spaNavigate for the cmdk modal (and any future caller that
  // wants to do an SPA swap without simulating a click)
  window.__LASH_SPA = spaNavigate;

  function wireSpaNav() {
    document.addEventListener("click", (ev) => {
      if (ev.defaultPrevented) return;
      if (ev.button !== 0) return;
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
      const a = ev.target.closest && ev.target.closest("a[href]");
      if (!a) return;
      if (a.target && a.target !== "_self") return;
      if (a.hasAttribute("download")) return;
      const href = a.getAttribute("href");
      if (!href || href.startsWith("#")) return;
      const url = new URL(href, location.href);
      if (url.origin !== location.origin) return;
      // only intercept .html pages — leaves mailto:, tel:, raw asset
      // links, and external sites to the browser. The .docs-layout
      // check inside spaNavigate falls back to a full load when the
      // target turns out to be a landing-style page.
      if (!/\.html(?:\?|#|$)/.test(url.pathname)) return;
      ev.preventDefault();
      spaNavigate(url.href, true);
    });

    window.addEventListener("popstate", () => {
      spaNavigate(location.href, false);
    });
  }

  // ── cmdk search — modal triggered by ⌘K / Ctrl+K or the band button ──
  //    Walks the TOC, fetches every page once, extracts opener title +
  //    section h2/h3 headings, and offers a substring-matched dropdown.
  //    Selecting a result hands off to spaNavigate when possible.
  const SEARCH = {
    overlay: null,
    input: null,
    list: null,
    status: null,
    results: [],
    selected: 0,
    indexed: false,
    indexing: false,
  };

  function searchAllPages() {
    // flatten TOC into the unique page list, preserving order
    const seen = new Set();
    const pages = [];
    for (const { name, href } of flatTOC()) {
      if (!seen.has(href)) {
        seen.add(href);
        pages.push({ href, label: name });
      }
    }
    return pages;
  }

  async function buildSearchIndex() {
    if (window.__LASH_SEARCH_INDEX) {
      SEARCH.indexed = true;
      return window.__LASH_SEARCH_INDEX;
    }
    if (SEARCH.indexing) return null;
    SEARCH.indexing = true;
    const base = tocBase();
    const pages = searchAllPages();
    const entries = [];
    try {
      const docs = await Promise.all(pages.map((p) =>
        fetch(base + p.href, { credentials: "same-origin" })
          .then((r) => (r.ok ? r.text() : ""))
          .catch(() => "")
      ));
      pages.forEach((p, idx) => {
        const html = docs[idx];
        if (!html) return;
        const doc = new DOMParser().parseFromString(html, "text/html");
        const titleEl = doc.querySelector(".opener__title");
        const pageTitle = (titleEl ? titleEl.textContent : p.label || p.href)
          .replace(/\s+/g, " ").trim();
        entries.push({
          url: p.href,
          page_title: pageTitle,
          heading_text: "",
          kind: "page",
        });
        const body = doc.querySelector(".body");
        if (!body) return;
        const headings = body.querySelectorAll("h2, h3");
        headings.forEach((h) => {
          const text = (h.textContent || "").replace(/\s+/g, " ").trim();
          if (!text) return;
          // find the nearest ancestor with an id (.section / .stanza / etc.)
          let anchor = "";
          let node = h;
          while (node && node !== body) {
            if (node.id) { anchor = node.id; break; }
            node = node.parentElement;
          }
          entries.push({
            url: anchor ? p.href + "#" + anchor : p.href,
            page_title: pageTitle,
            heading_text: text,
            kind: "heading",
          });
        });
      });
    } finally {
      SEARCH.indexing = false;
    }
    window.__LASH_SEARCH_INDEX = entries;
    SEARCH.indexed = true;
    return entries;
  }

  function filterSearch(query) {
    const entries = window.__LASH_SEARCH_INDEX || [];
    const q = query.trim().toLowerCase();
    if (!q) return entries.slice(0, 30);
    const matches = [];
    for (const e of entries) {
      const hay = (e.page_title + " " + e.heading_text).toLowerCase();
      if (hay.indexOf(q) < 0) continue;
      // page-title hits rank above heading hits
      const titleHit = e.page_title.toLowerCase().indexOf(q) >= 0;
      const rank = (e.kind === "page" ? 0 : 1) + (titleHit ? 0 : 2);
      matches.push({ e, rank });
    }
    matches.sort((a, b) => a.rank - b.rank);
    return matches.slice(0, 30).map((m) => m.e);
  }

  function renderSearchResults() {
    if (!SEARCH.list) return;
    if (!SEARCH.indexed && SEARCH.indexing) {
      SEARCH.list.innerHTML = '<li class="cmdk__empty">indexing…</li>';
      return;
    }
    if (!SEARCH.results.length) {
      SEARCH.list.innerHTML = '<li class="cmdk__empty">no matches</li>';
      return;
    }
    const base = tocBase();
    const parts = [];
    SEARCH.results.forEach((r, i) => {
      const cls = "cmdk__row" + (i === SEARCH.selected ? " is-active" : "");
      const kindLabel = r.kind === "page" ? "page" : "section";
      const headingHtml = r.heading_text
        ? `<span class="cmdk__row-heading">${escapeHtml(r.heading_text)}</span>`
        : "";
      parts.push(
        `<li><a class="${cls}" data-idx="${i}" href="${escapeHtml(base + r.url)}">` +
          `<span class="cmdk__row-kind">${kindLabel}</span>` +
          `<span class="cmdk__row-body">` +
            `<span class="cmdk__row-title">${escapeHtml(r.page_title)}</span>` +
            headingHtml +
          `</span>` +
        `</a></li>`
      );
    });
    SEARCH.list.innerHTML = parts.join("");
  }

  function ensureSearchActiveVisible() {
    if (!SEARCH.list) return;
    const active = SEARCH.list.querySelector(".cmdk__row.is-active");
    if (active && typeof active.scrollIntoView === "function") {
      active.scrollIntoView({ block: "nearest" });
    }
  }

  function openSearch() {
    if (!SEARCH.overlay) mountSearchModal();
    if (SEARCH.overlay.classList.contains("is-open")) return;
    SEARCH.overlay.classList.add("is-open");
    document.documentElement.classList.add("cmdk-open");
    SEARCH.input.value = "";
    SEARCH.selected = 0;
    SEARCH.results = [];
    // kick off the index fetch if it hasn't been built yet
    if (!window.__LASH_SEARCH_INDEX) {
      renderSearchResults(); // show "indexing…" state
      buildSearchIndex().then(() => {
        if (!SEARCH.overlay.classList.contains("is-open")) return;
        SEARCH.results = filterSearch(SEARCH.input.value);
        SEARCH.selected = 0;
        renderSearchResults();
      });
    } else {
      SEARCH.results = filterSearch("");
      renderSearchResults();
    }
    // focus on next frame — focus() inside a click that opened us can
    // race against the click's own focus management
    requestAnimationFrame(() => { SEARCH.input.focus(); });
  }

  function closeSearch() {
    if (!SEARCH.overlay) return;
    SEARCH.overlay.classList.remove("is-open");
    document.documentElement.classList.remove("cmdk-open");
  }

  function activateSearchResult(idx) {
    const r = SEARCH.results[idx];
    if (!r) return;
    const base = tocBase();
    const url = base + r.url;
    closeSearch();
    if (typeof window.__LASH_SPA === "function") {
      window.__LASH_SPA(url, true);
    } else {
      location.href = url;
    }
  }

  function mountSearchModal() {
    const overlay = document.createElement("div");
    overlay.className = "cmdk";
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML =
      '<div class="cmdk__dimmer" data-cmdk-dismiss></div>' +
      '<div class="cmdk__card" role="dialog" aria-label="search docs">' +
        '<div class="cmdk__search-row">' +
          '<span class="cmdk__icon" aria-hidden="true">⌕</span>' +
          '<input class="cmdk__input" type="text" autocomplete="off" ' +
            'spellcheck="false" placeholder="search the docs…" aria-label="search">' +
          '<kbd class="cmdk__esc">esc</kbd>' +
        '</div>' +
        '<ul class="cmdk__list" role="listbox"></ul>' +
        '<div class="cmdk__hint">' +
          '<span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>' +
          '<span><kbd>↵</kbd> open</span>' +
          '<span><kbd>esc</kbd> close</span>' +
        '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    SEARCH.overlay = overlay;
    SEARCH.input = overlay.querySelector(".cmdk__input");
    SEARCH.list = overlay.querySelector(".cmdk__list");

    SEARCH.input.addEventListener("input", () => {
      if (!window.__LASH_SEARCH_INDEX) {
        renderSearchResults();
        return;
      }
      SEARCH.results = filterSearch(SEARCH.input.value);
      SEARCH.selected = 0;
      renderSearchResults();
    });
    SEARCH.input.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        if (SEARCH.results.length === 0) return;
        SEARCH.selected = (SEARCH.selected + 1) % SEARCH.results.length;
        renderSearchResults();
        ensureSearchActiveVisible();
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        if (SEARCH.results.length === 0) return;
        SEARCH.selected = (SEARCH.selected - 1 + SEARCH.results.length) % SEARCH.results.length;
        renderSearchResults();
        ensureSearchActiveVisible();
      } else if (ev.key === "Enter") {
        ev.preventDefault();
        activateSearchResult(SEARCH.selected);
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        closeSearch();
      }
    });
    overlay.addEventListener("click", (ev) => {
      if (ev.target && ev.target.hasAttribute && ev.target.hasAttribute("data-cmdk-dismiss")) {
        closeSearch();
        return;
      }
      const row = ev.target.closest && ev.target.closest(".cmdk__row");
      if (!row) return;
      ev.preventDefault();
      const idx = parseInt(row.getAttribute("data-idx"), 10);
      if (Number.isFinite(idx)) activateSearchResult(idx);
    });
    overlay.addEventListener("mousemove", (ev) => {
      const row = ev.target.closest && ev.target.closest(".cmdk__row");
      if (!row) return;
      const idx = parseInt(row.getAttribute("data-idx"), 10);
      if (!Number.isFinite(idx) || idx === SEARCH.selected) return;
      SEARCH.selected = idx;
      renderSearchResults();
    });
  }

  function wireSearchTrigger() {
    // global keydown — install once at boot; persists across spaNavigate
    document.addEventListener("keydown", (ev) => {
      const k = ev.key;
      if ((ev.metaKey || ev.ctrlKey) && (k === "k" || k === "K")) {
        ev.preventDefault();
        if (SEARCH.overlay && SEARCH.overlay.classList.contains("is-open")) {
          closeSearch();
        } else {
          openSearch();
        }
      }
    });
    // band-level click handler — the trigger button uses
    // data-cmdk-open so it survives any SPA swap that re-emits the
    // band markup
    document.addEventListener("click", (ev) => {
      const btn = ev.target.closest && ev.target.closest("[data-cmdk-open]");
      if (!btn) return;
      ev.preventDefault();
      openSearch();
    });
  }

  // ── shell mount ───────────────────────────────────────────
  // Each docs page used to ship its own copy of the band, layout
  // wrapper, toc aside, pager, and cover SVG. That's now built here so
  // the per-page HTML stays minimal. Legacy pages that still carry the
  // shell are detected and skipped.
  const GITHUB_ICON =
    '<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true">' +
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2 .37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>';

  // platform-aware label for the search shortcut chip — macOS sees
  // ⌘K, everyone else sees Ctrl+K (the symbol means nothing off-mac)
  function searchShortcutLabel() {
    const plat = (navigator.userAgentData && navigator.userAgentData.platform)
      || navigator.platform
      || "";
    return /mac|iphone|ipad|ipod/i.test(plat) ? "⌘K" : "Ctrl+K";
  }

  // The cover caption ships per-page as a data-attribute. The value is
  // a list of <code>path/to/file</code> spans joined by &middot;. Wrap
  // each path in a GH source link so the caption is navigable, not
  // inert prose.
  function linkifyCoverSource(html, repoBase) {
    if (!html) return "";
    return html.replace(
      /<code>([^<]+)<\/code>/g,
      (_, path) => {
        const url = repoBase + "/blob/main/" + path;
        return `<a class="cover__src" href="${url}" target="_blank" rel="noopener"><code>${path}</code></a>`;
      }
    );
  }

  function mountShell() {
    if (!document.body.classList.contains("docs-page")) return;
    if (document.querySelector(".docs-layout")) return; // legacy page; skip

    const base = tocBase();
    const body = document.body;
    const repoBase = "https://github.com/SamGalanakis/lash";

    // pull out the per-page pieces (opener, body, optional pager)
    const opener = body.querySelector(":scope > .opener");
    const main = body.querySelector(":scope > main.body, :scope > .body");
    let pager = body.querySelector(":scope > .pager");
    const coverSource = body.dataset.coverSource || "";

    if (!opener || !main) return; // shape doesn't match; bail

    // ── build skip-link ─────────────────────────────────────
    const skip = document.createElement("a");
    skip.className = "skip-link";
    skip.href = "#content";
    skip.textContent = "skip to content";

    // ── build band ──────────────────────────────────────────
    const kb = searchShortcutLabel();
    const kbAria = kb === "⌘K" ? "cmd + k" : "ctrl + k";
    const band = document.createElement("header");
    band.className = "band";
    band.innerHTML =
      `<span class="band__brand"><a href="${base}index.html">lash</a></span>` +
      '<div class="band__right">' +
        `<button class="band__btn band__btn--search" type="button" data-cmdk-open ` +
          `aria-label="search docs (${kbAria})" title="search docs (${kbAria})">` +
          '<span class="cmdk-label">search</span>' +
          `<span class="cmdk-kbd" aria-hidden="true">${kb}</span>` +
        '</button>' +
        `<a class="band__btn band__btn--icon" href="https://github.com/SamGalanakis/lash" ` +
          `target="_blank" rel="noopener" aria-label="lash on GitHub" title="lash on GitHub">${GITHUB_ICON}</a>` +
        '<button class="band__btn" id="theme-toggle" type="button" ' +
          'aria-label="toggle theme" title="toggle light / dark">' +
          '<span class="glyph" id="theme-glyph" aria-hidden="true">◐</span>' +
          '<span id="theme-label">light</span>' +
        '</button>' +
      '</div>';

    // ── build docs-layout shell ─────────────────────────────
    const layout = document.createElement("div");
    layout.className = "docs-layout";
    const toc = document.createElement("aside");
    toc.className = "toc";
    toc.setAttribute("aria-label", "documentation contents");
    const mainCol = document.createElement("div");
    mainCol.className = "docs-main";
    layout.appendChild(toc);
    layout.appendChild(mainCol);

    // ensure a pager exists (auto-filled later by buildPager)
    if (!pager) {
      pager = document.createElement("section");
      pager.className = "pager";
      pager.setAttribute("aria-label", "next chapter");
    }

    // ── build cover ─────────────────────────────────────────
    const cover = document.createElement("section");
    cover.className = "cover";
    cover.setAttribute("aria-label", "footer · twilight landscape");
    cover.innerHTML =
      '<svg id="range" class="cover__svg" preserveAspectRatio="xMidYMid slice" ' +
        'xmlns="http://www.w3.org/2000/svg" role="img" ' +
        'aria-label="A warm-iron twilight landscape across the full width of the page.">' +
        '<defs><linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">' +
          '<stop offset="0%" stop-color="#141412"/>' +
          '<stop offset="65%" stop-color="#1a1814"/>' +
          '<stop offset="100%" stop-color="#221d16"/>' +
        '</linearGradient></defs>' +
        '<rect id="sky-rect" width="100%" height="100%" fill="transparent"/>' +
        '<g id="g-back"></g>' +
        '<g id="g-mid"></g>' +
        '<g id="g-lash-behind"></g>' +
        '<g id="g-near"></g>' +
        '<g id="g-lash-front"></g>' +
        '<g id="g-ground"></g>' +
      '</svg>' +
      '<p class="cover__caption">' +
        (coverSource
          ? `<span>source &middot; ${linkifyCoverSource(coverSource, repoBase)}</span>`
          : '<span>&nbsp;</span>') +
        '<a class="cover__top" href="#top">back to top <span class="arrow" aria-hidden="true">↑</span></a>' +
      '</p>';

    // ── move pieces into the shell ──────────────────────────
    mainCol.appendChild(opener);
    mainCol.appendChild(main);
    mainCol.appendChild(pager);

    // clear body and re-populate in the right order
    body.id = body.id || "top";
    // remove anything else stranded at body root (data nodes etc.)
    [...body.children].forEach(c => c.remove());
    body.appendChild(skip);
    body.appendChild(band);
    body.appendChild(layout);
    body.appendChild(cover);

    // load the scene script if it isn't already on the page
    if (!document.querySelector('script[data-scene]')) {
      const s = document.createElement("script");
      s.src = base + "scene.js";
      s.defer = true;
      s.setAttribute("data-scene", "");
      document.head.appendChild(s);
    }
  }

  // On landing pages the band ships in static HTML; update the
  // hard-coded ⌘K chip so non-mac users see Ctrl+K too.
  function fixupSearchChip() {
    const kb = document.querySelector(".cmdk-kbd");
    if (kb) kb.textContent = searchShortcutLabel();
  }

  function bootstrap() {
    mountShell();
    fixupSearchChip();
    buildTOC();
    buildPager();
    autoKnot();
    drawSpineSnake();
    mountTocJump();
    loadMermaidIfNeeded();
    wireSpaNav();
    wireSearchTrigger();
    // re-run the scene now that we know the spine X — landscape's
    // strand needs to converge at the snake's bottom
    if (typeof window.__LASH_SCENE_RUN === "function") {
      window.__LASH_SCENE_RUN();
    }
  }

  // redraw the snake (and re-converge the strand) on resize
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawSpineSnake();
      if (typeof window.__LASH_SCENE_RUN === "function") {
        window.__LASH_SCENE_RUN();
      }
    }, 200);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
