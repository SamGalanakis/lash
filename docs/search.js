// docs/search.js — Cmd+K / "/" client-side search overlay backed by
// Pagefind's pre-built static index (see /pagefind/).
//
// We use Pagefind's JS API rather than its bundled UI so the chrome
// matches the docs design system.

(function () {
  // Resolve the pagefind module against the docs root. Same approach as
  // nav.js's rootPrefix(): read this script's own tag (search.js sits at the
  // docs root). Works at the host root AND when served under a project
  // subpath. `defer` scripts can't use `document.currentScript`, so we look
  // the tag up by filename.
  function indexUrl() {
    const tag = document.querySelector('script[src*="search.js"]');
    if (tag) {
      const url = new URL(tag.getAttribute("src"), location.href);
      const root = url.pathname.replace(/search\.js(\?.*)?$/, "");
      return root + "pagefind/pagefind.js";
    }
    // Fallback: legacy depth math.
    const slashes = (location.pathname.match(/\//g) || []).length;
    const depth = Math.max(0, slashes - 1);
    const prefix = depth === 0 ? "./" : "../".repeat(depth);
    return prefix + "pagefind/pagefind.js";
  }

  // Detect platform once for the keyboard-hint glyph.
  const isMac =
    /Mac|iPhone|iPad|iPod/.test(navigator.userAgentData?.platform || navigator.platform || "");
  const modKey = isMac ? "⌘" : "Ctrl";

  let pagefind = null;
  let overlay = null;
  let input = null;
  let resultsEl = null;
  let statusEl = null;
  let currentToken = 0;
  let lastResults = [];
  let activeIndex = -1;

  async function ensureLoaded() {
    if (pagefind) return pagefind;
    try {
      pagefind = await import(indexUrl());
      await pagefind.options({ excerptLength: 24 });
      await pagefind.init();
      return pagefind;
    } catch (err) {
      console.warn("pagefind load failed", err);
      return null;
    }
  }

  function buildOverlay() {
    overlay = document.createElement("div");
    overlay.className = "search-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Search");
    overlay.hidden = true;
    overlay.innerHTML = `
      <div class="search-backdrop" data-search-close></div>
      <div class="search-panel">
        <div class="search-input-row">
          <span class="search-icon" aria-hidden="true">⌕</span>
          <input class="search-input" type="search" placeholder="Search the docs…"
                 autocomplete="off" spellcheck="false" autocapitalize="off">
          <kbd class="search-esc">esc</kbd>
        </div>
        <div class="search-status" role="status" aria-live="polite"></div>
        <ol class="search-results" role="listbox"></ol>
      </div>
    `;
    document.body.appendChild(overlay);
    input = overlay.querySelector(".search-input");
    resultsEl = overlay.querySelector(".search-results");
    statusEl = overlay.querySelector(".search-status");

    overlay.addEventListener("click", (e) => {
      if (e.target.closest("[data-search-close]")) close();
    });
    overlay.addEventListener("keydown", onKey);
    input.addEventListener("input", onInput);
  }

  function open() {
    if (!overlay) buildOverlay();
    overlay.hidden = false;
    document.body.classList.add("search-open");
    input.value = "";
    resultsEl.innerHTML = "";
    statusEl.textContent = "";
    activeIndex = -1;
    setTimeout(() => input.focus(), 10);
    ensureLoaded().then((p) => {
      if (!p) statusEl.textContent = "Search index unavailable.";
    });
  }

  function close() {
    if (!overlay) return;
    overlay.hidden = true;
    document.body.classList.remove("search-open");
  }

  async function onInput() {
    const q = input.value.trim();
    const token = ++currentToken;
    if (!q) {
      resultsEl.innerHTML = "";
      statusEl.textContent = "";
      lastResults = [];
      activeIndex = -1;
      return;
    }
    const pf = await ensureLoaded();
    if (!pf) return;
    statusEl.textContent = "Searching…";
    const search = await pf.search(q);
    if (token !== currentToken) return; // stale

    // Expand each page result into the page-level hit plus its sub_results
    // (matching headings inside that page). This gives us heading-anchor
    // deep-linking.
    const pages = await Promise.all(
      search.results.slice(0, 8).map((r) => r.data()),
    );
    if (token !== currentToken) return;

    const flat = [];
    for (const page of pages) {
      // Best heading-level match — first sub_result with an anchor.
      const subs = (page.sub_results || []).filter((s) => s.anchor || s.title);
      if (subs.length) {
        // Lead with the strongest sub-result, then add up to 1 more sibling
        // from the same page so we don't drown the list.
        const best = subs[0];
        flat.push({
          url: best.url || page.url,
          title: page.meta?.title || page.url,
          section: prettyPath(page.url),
          excerpt: best.excerpt || page.excerpt,
          heading: best.title,
        });
        if (subs[1]) {
          flat.push({
            url: subs[1].url || page.url,
            title: page.meta?.title || page.url,
            section: prettyPath(page.url),
            excerpt: subs[1].excerpt,
            heading: subs[1].title,
          });
        }
      } else {
        flat.push({
          url: page.url,
          title: page.meta?.title || page.url,
          section: prettyPath(page.url),
          excerpt: page.excerpt,
          heading: null,
        });
      }
    }

    lastResults = flat;
    activeIndex = flat.length ? 0 : -1;
    if (!flat.length) {
      statusEl.textContent = `No results for "${q}".`;
      resultsEl.innerHTML = "";
      return;
    }
    statusEl.textContent = `${search.results.length} page${search.results.length === 1 ? "" : "s"}`;
    resultsEl.innerHTML = flat.map((d, i) => renderResult(d, i)).join("");
    updateActive();
  }

  function renderResult(d, i) {
    const heading = d.heading
      ? `<span class="search-result__heading">${escapeHTML(d.heading)}</span>`
      : "";
    return `
      <li>
        <a class="search-result" id="search-r${i}" href="${d.url}" data-idx="${i}" role="option">
          <div class="search-result__row">
            <span class="search-result__title">${escapeHTML(d.title)}${heading}</span>
            <span class="search-result__path">${escapeHTML(d.section)}</span>
          </div>
          <div class="search-result__excerpt">${d.excerpt || ""}</div>
        </a>
      </li>
    `;
  }

  function prettyPath(url) {
    return url
      .replace(/^\//, "")
      .replace(/\.html?$/, "")
      .replace(/index$/, "")
      .replace(/\/$/, "")
      .replace(/\//g, " · ");
  }

  function escapeHTML(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function updateActive() {
    const items = resultsEl.querySelectorAll(".search-result");
    items.forEach((el, i) => {
      el.classList.toggle("is-active", i === activeIndex);
      if (i === activeIndex) {
        el.scrollIntoView({ block: "nearest" });
      }
    });
  }

  function onKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === "ArrowDown") {
      if (!lastResults.length) return;
      e.preventDefault();
      activeIndex = (activeIndex + 1) % lastResults.length;
      updateActive();
    } else if (e.key === "ArrowUp") {
      if (!lastResults.length) return;
      e.preventDefault();
      activeIndex = (activeIndex - 1 + lastResults.length) % lastResults.length;
      updateActive();
    } else if (e.key === "Enter") {
      if (activeIndex < 0 || !lastResults[activeIndex]) return;
      e.preventDefault();
      location.href = lastResults[activeIndex].url;
    }
  }

  function shouldTriggerOpen(e) {
    if (e.defaultPrevented) return false;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") return true;
    if (e.key === "/" && !isTypingTarget(e.target)) return true;
    return false;
  }

  function isTypingTarget(t) {
    if (!t) return false;
    const tag = t.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
  }

  function init() {
    // Inject a search button into the topbar if it exists.
    const navHost = document.querySelector(".topbar__nav");
    if (navHost) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "search-trigger";
      btn.setAttribute("aria-label", `Search docs (${modKey}+K)`);
      btn.innerHTML =
        '<span class="search-trigger__icon" aria-hidden="true">⌕</span>' +
        '<span class="search-trigger__label">Search</span>' +
        `<kbd class="search-trigger__kbd">${modKey === "⌘" ? "⌘K" : "Ctrl K"}</kbd>`;
      btn.addEventListener("click", open);
      navHost.insertBefore(btn, navHost.firstChild);
    }

    window.addEventListener("keydown", (e) => {
      if (shouldTriggerOpen(e)) {
        e.preventDefault();
        open();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
