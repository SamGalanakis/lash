// docs/toc.js — auto-generate the on-page table-of-contents from h2/h3 in
// .content and nest it inside the sidebar, under the active chapter link.
// Also adds clickable `#` anchor markers to each heading and scroll-spies
// the active section.

(function () {
  function init() {
    const content = document.querySelector(".content");
    const sidebar = document.querySelector(".sidebar");
    if (!content || !sidebar) return;

    // Collect h2/h3 inside .content; skip diagram chrome and modals.
    const candidates = content.querySelectorAll("h2, h3");
    const headings = Array.from(candidates).filter((h) => {
      return !h.closest(".diagram, .mermaid, .diagram-modal");
    });
    if (headings.length < 2) return; // not worth a TOC

    // Ensure every heading has an id we can link to.
    const used = new Set();
    for (const h of headings) {
      if (!h.id) h.id = slugify(h.textContent, used);
      used.add(h.id);
      if (!h.querySelector(".heading-anchor")) {
        const a = document.createElement("a");
        a.className = "heading-anchor";
        a.href = "#" + h.id;
        a.setAttribute("aria-label", "Link to this section");
        a.textContent = "#";
        h.appendChild(a);
      }
    }

    // Build the nested TOC list.
    const tocList = document.createElement("ol");
    tocList.className = "sidebar__toc-list";
    tocList.setAttribute("aria-label", "On this page");
    for (const h of headings) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = textOf(h);
      a.dataset.depth = h.tagName === "H3" ? "3" : "2";
      a.dataset.target = h.id;
      li.appendChild(a);
      tocList.appendChild(li);
    }

    // Decide where to place it: nested under the active chapter link if one
    // exists in the sidebar; otherwise as a standalone block at the bottom.
    const activeChapter = sidebar.querySelector(
      ".sidebar__chapters a.active, .sidebar__chapters a[aria-current], .sidebar__chapters .active",
    );
    if (activeChapter) {
      const parentLi = activeChapter.closest("li");
      if (parentLi) {
        parentLi.appendChild(tocList);
      } else {
        activeChapter.insertAdjacentElement("afterend", tocList);
      }
    } else {
      // No active chapter — render the TOC as a standalone "On this page"
      // block at the bottom of the sidebar.
      const wrap = document.createElement("div");
      wrap.className = "sidebar__toc-block";
      const label = document.createElement("span");
      label.className = "sidebar__label";
      label.textContent = "On this page";
      wrap.appendChild(label);
      wrap.appendChild(tocList);
      sidebar.appendChild(wrap);
    }

    // Scroll-spy via IntersectionObserver.
    const links = new Map();
    tocList.querySelectorAll("a").forEach((a) => links.set(a.dataset.target, a));

    let active = null;
    const setActive = (id) => {
      if (id === active) return;
      if (active && links.get(active)) links.get(active).classList.remove("active");
      if (id && links.get(id)) links.get(id).classList.add("active");
      active = id;
    };

    // Track visible headings; the topmost one above the fold is "current".
    const visible = new Set();
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target);
          else visible.delete(entry.target);
        }
        // Pick the heading with the smallest y position that is still above
        // the bottom of its tracking window — i.e. the most recent heading
        // the reader has scrolled past or is currently in.
        let best = null;
        let bestY = -Infinity;
        for (const h of headings) {
          const r = h.getBoundingClientRect();
          if (r.top <= 80 && r.top > bestY) {
            best = h;
            bestY = r.top;
          }
        }
        if (best) setActive(best.id);
        else if (headings[0]) setActive(headings[0].id);
      },
      { rootMargin: "-80px 0px -70% 0px", threshold: [0, 1] },
    );
    for (const h of headings) io.observe(h);

    // Initial state — pick whatever's nearest at load.
    requestAnimationFrame(() => {
      let best = headings[0];
      let bestDist = Infinity;
      for (const h of headings) {
        const r = h.getBoundingClientRect();
        const d = Math.abs(r.top - 80);
        if (r.top <= 80 && r.top > best.getBoundingClientRect().top) best = h;
        else if (d < bestDist && bestDist === Infinity) {
          best = h;
          bestDist = d;
        }
      }
      setActive(best.id);
    });

    // Smooth-scroll on TOC click, adjusting for the sticky top bar.
    tocList.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-target]");
      if (!a) return;
      const id = a.dataset.target;
      const target = document.getElementById(id);
      if (!target) return;
      e.preventDefault();
      const offset = 80; // topbar + breathing room
      const y = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: y, behavior: "smooth" });
      history.replaceState(null, "", "#" + id);
    });
  }

  function textOf(h) {
    // Heading text minus the trailing anchor marker we may have added.
    const clone = h.cloneNode(true);
    clone.querySelectorAll(".heading-anchor").forEach((n) => n.remove());
    return clone.textContent.trim();
  }

  function slugify(text, taken) {
    let base = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-");
    if (!base) base = "section";
    let id = base;
    let i = 2;
    while (taken.has(id) || document.getElementById(id)) {
      id = base + "-" + i++;
    }
    return id;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
