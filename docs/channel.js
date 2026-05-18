/* lash · docs · channel switcher
   Detects the deployed channel from the URL path (`/main/` or
   `/staging/`) and injects a dropdown in the top-right band. Picking
   the other channel navigates to its homepage — no separate routing
   page. On local dev (no channel prefix) the dropdown is hidden. */
(function () {
  "use strict";

  function detectChannel() {
    const segments = location.pathname.split("/").filter(Boolean);
    const first = segments[0];
    if (first === "main" || first === "staging") return first;
    return null;
  }

  function urlFor(channel) {
    // Always land on the channel's homepage.
    return "/" + channel + "/";
  }

  function build() {
    const current = detectChannel();
    if (!current) return;
    const band = document.querySelector(".band__right");
    if (!band) return;
    // already built — re-init no-op
    if (band.querySelector(".channel")) return;

    const root = document.createElement("div");
    root.className = "channel";
    root.innerHTML =
      `<button class="band__btn channel__btn" type="button" ` +
        `aria-haspopup="menu" aria-expanded="false" ` +
        `aria-label="switch docs channel" title="switch docs channel">` +
        `<span class="channel__dot" aria-hidden="true"></span>` +
        `<span class="channel__label">${current}</span>` +
        `<span class="channel__caret" aria-hidden="true">▾</span>` +
      `</button>` +
      `<div class="channel__menu" role="menu" hidden>` +
        `<a class="channel__opt${current === "main"    ? " is-active" : ""}" role="menuitem" href="${urlFor("main")}">main</a>` +
        `<a class="channel__opt${current === "staging" ? " is-active" : ""}" role="menuitem" href="${urlFor("staging")}">staging</a>` +
      `</div>`;

    // place the channel switcher leftmost in the right-cluster so the
    // theme toggle stays the rightmost button (visual anchor).
    band.insertBefore(root, band.firstChild);

    const btn = root.querySelector(".channel__btn");
    const menu = root.querySelector(".channel__menu");
    const opts = root.querySelectorAll(".channel__opt");
    const close = () => {
      if (menu.hidden) return;
      menu.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    };
    const open = () => {
      menu.hidden = false;
      btn.setAttribute("aria-expanded", "true");
    };
    btn.addEventListener("click", (e) => {
      // toggle — and stop bubbling so the capture-phase outside-close
      // below doesn't immediately fire on the same click sequence.
      e.stopPropagation();
      if (menu.hidden) open(); else close();
    });
    // Picking an option: close the dropdown immediately, then let the
    // <a href> navigation happen normally. Browsers sometimes show the
    // open menu for a beat before navigating; without this the menu
    // appears to "stick" open.
    opts.forEach(opt => opt.addEventListener("click", () => close()));
    // Outside dismissal — capture phase + pointerdown so cover-SVG
    // sun handlers (which preventDefault on click) can't swallow it.
    const outsideHandler = (e) => {
      if (!root.contains(e.target)) close();
    };
    document.addEventListener("pointerdown", outsideHandler, true);
    document.addEventListener("click", outsideHandler, true);
    // Escape always closes
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") close();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
  // Expose for docs.js — it builds the band dynamically on docs pages,
  // and the static DOMContentLoaded hook fires before that markup
  // exists. docs.js calls this after mountShell().
  window.__LASH_CHANNEL_INIT = build;
})();
