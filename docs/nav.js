// docs/nav.js — site-wide top bar with section nav. Injects a sticky
// header at the top of every docs page (excluding design-language.html
// which keeps the warm-iron showcase chrome). Highlights the active
// section based on the current URL.

(function () {
  // Path is something like "/architecture/flow.html" or "/cli.html" or "/".
  function classifySection(path) {
    if (path === "/" || /\/index\.html?$/.test(path)) {
      // landing — check whether under /architecture/
      if (/\/architecture\//.test(path)) return "architecture";
      return null; // landing itself, no section active
    }
    if (/\/architecture\//.test(path) || /\/architecture\.html?$/.test(path))
      return "architecture";
    if (/\/cli\.html?$/.test(path)) return "cli";
    if (/\/design-language\.html?$/.test(path)) return "design";
    // anything else in /docs root (quickstart, embedding, plugins, persistence,
    // tracing, example-agent-service, trace-export-edges) is part of the "lash"
    // guide section
    return "lash";
  }

  // Build a path back to docs root from the current page. `depth` is the
  // number of directory levels we are below the docs root; we emit one ".."
  // for each. Path "/index.html" → depth 0 → "./"; "/architecture/flow.html"
  // → depth 1 → "../".
  function rootPrefix() {
    const slashes = (location.pathname.match(/\//g) || []).length;
    const depth = Math.max(0, slashes - 1);
    return depth === 0 ? "./" : "../".repeat(depth);
  }

  function init() {
    if (document.querySelector(".topbar")) return; // already there

    const root = rootPrefix();
    const section = classifySection(location.pathname);

    const items = [
      { key: "lash", label: "Lash", href: root + "quickstart.html" },
      { key: "cli", label: "CLI", href: root + "cli.html" },
      { key: "architecture", label: "Architecture", href: root + "architecture/" },
      { key: "design", label: "Design", href: root + "design-language.html" },
    ];

    const bar = document.createElement("header");
    bar.className = "topbar";
    bar.setAttribute("role", "banner");
    bar.innerHTML = `
      <a class="topbar__logo" href="${root}index.html" aria-label="lash docs">
        <span class="topbar__mark">lash<span class="slash">/</span>docs</span>
      </a>
      <nav class="topbar__nav" aria-label="Sections">
        ${items
          .map(
            (it) =>
              `<a class="topbar__link${it.key === section ? " is-active" : ""}"
                  href="${it.href}"
                  ${it.key === section ? 'aria-current="page"' : ""}>${it.label}</a>`,
          )
          .join("")}
      </nav>
    `;

    // Insert before the skip-link's target if present, otherwise at body start.
    const skip = document.querySelector(".skip-link");
    if (skip && skip.nextSibling) {
      skip.parentNode.insertBefore(bar, skip.nextSibling);
    } else {
      document.body.insertBefore(bar, document.body.firstChild);
    }

    // Hide the redundant in-sidebar "← Docs" back link now that the top bar
    // handles cross-section navigation.
    document.querySelectorAll(".sidebar__exit").forEach((el) => {
      el.style.display = "none";
    });

    injectBreadcrumbs(section, root);
    enrichFooter(root);
  }

  function sectionLabel(key) {
    return (
      {
        lash: "Lash",
        cli: "CLI",
        architecture: "Architecture",
        design: "Design Language",
      }[key] || null
    );
  }

  function sectionIndexHref(key, root) {
    return (
      {
        lash: root + "quickstart.html",
        cli: root + "cli.html",
        architecture: root + "architecture/",
        design: root + "design-language.html",
      }[key] || null
    );
  }

  function injectBreadcrumbs(section, root) {
    // Skip on the landing — the topbar already says "lash/docs."
    if (!section) return;
    const hero = document.querySelector(".hero");
    if (!hero || hero.querySelector(".breadcrumbs")) return;

    const label = sectionLabel(section);
    const href = sectionIndexHref(section, root);
    // Resolve the page name from the hero's h1, falling back to <title>.
    const pageTitle =
      hero.querySelector("h1")?.textContent.replace(/#$/, "").trim() ||
      document.title.split(/[·|—-]/)[0].trim();

    // Skip the second crumb when we are on the section index itself.
    // Resolve the section-index href against the current page so we can
    // compare absolute pathnames (avoids "./cli.html" vs "/cli.html" mismatch).
    const hrefAbs = new URL(href, location.href).pathname.replace(/index\.html?$/, "");
    const hereAbs = location.pathname.replace(/index\.html?$/, "");
    const onSectionIndex = hrefAbs === hereAbs;

    const nav = document.createElement("nav");
    nav.className = "breadcrumbs";
    nav.setAttribute("aria-label", "Breadcrumb");
    const sep = '<span class="breadcrumbs__sep" aria-hidden="true">/</span>';
    if (onSectionIndex) {
      nav.innerHTML = `<span class="breadcrumbs__current">${label}</span>`;
    } else {
      nav.innerHTML =
        `<a href="${href}">${label}</a>${sep}` +
        `<span class="breadcrumbs__current">${escapeHTML(pageTitle)}</span>`;
    }
    // Insert before the eyebrow if present, otherwise at the very top of hero.
    const eyebrow = hero.querySelector(".eyebrow");
    if (eyebrow && eyebrow.parentElement === hero) {
      hero.insertBefore(nav, eyebrow);
    } else if (eyebrow) {
      eyebrow.parentElement.insertBefore(nav, eyebrow);
    } else {
      hero.insertBefore(nav, hero.firstChild);
    }
    // Mark the hero so CSS can hide the now-redundant eyebrow.
    hero.classList.add("hero--has-breadcrumb");
  }

  function enrichFooter(root) {
    const footer = document.querySelector(".footer");
    if (!footer || footer.querySelector(".footer__row")) return;
    const originalText = footer.innerHTML.trim();
    const repo = "https://github.com/SamGalanakis/lash";
    const pagePath = location.pathname.replace(/^\//, "");
    const editUrl = `${repo}/edit/staging/docs/${pagePath}`;
    footer.innerHTML = `
      <div class="footer__row">
        <div class="footer__source">${originalText}</div>
        <div class="footer__links">
          <a href="${editUrl}" target="_blank" rel="noopener">Edit this page</a>
          <a href="${repo}" target="_blank" rel="noopener">Repository</a>
        </div>
      </div>
    `;
  }

  function escapeHTML(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
