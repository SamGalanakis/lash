/* lash · docs · channel switcher
   Detects the deployed channel from the URL path (`/main/` or
   `/staging/`) and injects a dropdown in the top-right band. Picking
   the other channel navigates to its homepage — no separate routing
   page. On local dev (no channel prefix) the dropdown is hidden.

   Uses the native HTML Popover API for open/close — the browser
   handles outside-click dismissal, Escape, and a11y wiring; no need
   for our own document listeners that could be intercepted by sun
   handlers in the cover SVG. */
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

    // Detect popover support. If unavailable, fall back to a manual
    // toggle (Safari < 17, older Firefox). Most users get the native
    // path — auto-close, Escape, light-dismiss are all free.
    const popoverSupported = HTMLElement.prototype.hasOwnProperty("popover");

    const popoverId = "channel-menu-" + Math.random().toString(36).slice(2, 8);
    const popoverAttr  = popoverSupported ? ' popover="auto"'                   : "";
    const popoverWire  = popoverSupported ? ` popovertarget="${popoverId}"`     : "";
    const hiddenAttr   = popoverSupported ? ""                                  : " hidden";

    const root = document.createElement("div");
    root.className = "channel";
    root.innerHTML =
      `<button class="band__btn channel__btn" type="button"${popoverWire} ` +
        `aria-haspopup="menu" aria-expanded="false" ` +
        `aria-label="switch docs channel" title="switch docs channel">` +
        `<span class="channel__dot" aria-hidden="true"></span>` +
        `<span class="channel__label">${current}</span>` +
        `<span class="channel__caret" aria-hidden="true">▾</span>` +
      `</button>` +
      `<div id="${popoverId}" class="channel__menu" role="menu"${popoverAttr}${hiddenAttr}>` +
        `<a class="channel__opt${current === "main"    ? " is-active" : ""}" role="menuitem" href="${urlFor("main")}">main</a>` +
        `<a class="channel__opt${current === "staging" ? " is-active" : ""}" role="menuitem" href="${urlFor("staging")}">staging</a>` +
      `</div>`;

    // Place the channel switcher leftmost in the right-cluster so the
    // theme toggle stays the rightmost button (visual anchor).
    band.insertBefore(root, band.firstChild);

    const btn  = root.querySelector(".channel__btn");
    const menu = root.querySelector(".channel__menu");
    const opts = root.querySelectorAll(".channel__opt");

    if (popoverSupported) {
      // Track aria-expanded against the popover toggle events the
      // browser fires for free.
      menu.addEventListener("toggle", (e) => {
        btn.setAttribute("aria-expanded", e.newState === "open" ? "true" : "false");
      });
      // Belt-and-suspenders outside dismissal. Popover "light dismiss"
      // works in modern Chrome but has been observed sticking open
      // when a descendant of an ancestor preventDefaults pointerdown
      // (the cover-SVG sun handlers do exactly that). We close
      // ourselves in capture phase before anyone can swallow it.
      const closeIfOutside = (e) => {
        if (menu.matches(":popover-open") && !root.contains(e.target)) {
          if (typeof menu.hidePopover === "function") menu.hidePopover();
        }
      };
      document.addEventListener("pointerdown", closeIfOutside, true);
      // Option click — close the popover before navigation begins so
      // it doesn't linger visibly for the page-load beat.
      opts.forEach(opt => opt.addEventListener("click", () => {
        if (typeof menu.hidePopover === "function") menu.hidePopover();
      }));
    } else {
      // Fallback: manual toggle + outside-click dismissal in capture
      // phase so other handlers (cover SVG suns) can't swallow it.
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
        e.stopPropagation();
        if (menu.hidden) open(); else close();
      });
      opts.forEach(opt => opt.addEventListener("click", () => close()));
      const outsideHandler = (e) => {
        if (!root.contains(e.target)) close();
      };
      document.addEventListener("pointerdown", outsideHandler, true);
      document.addEventListener("click", outsideHandler, true);
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") close();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
  // docs.js builds the band dynamically on docs pages; the static
  // DOMContentLoaded hook fires before that markup exists, so docs.js
  // calls this after mountShell(). Idempotent — second call no-ops.
  window.__LASH_CHANNEL_INIT = build;
})();
