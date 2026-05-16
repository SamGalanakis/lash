// docs/theme.js — light/dark theme toggle. Applies stored preference
// before paint to avoid a flash, then injects a floating toggle.

(function () {
  const KEY = "lash-docs-theme";
  const root = document.documentElement;

  const get = () => {
    try {
      return localStorage.getItem(KEY);
    } catch {
      return null;
    }
  };
  const set = (v) => {
    try {
      v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY);
    } catch {}
  };

  // Apply theme to html element. Pass null to clear (= auto/system).
  function apply(theme) {
    if (theme === "light" || theme === "dark") {
      root.setAttribute("data-theme", theme);
    } else {
      root.removeAttribute("data-theme");
    }
  }

  // Resolve "current visible theme" — explicit override or system.
  function effective() {
    const stored = get();
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  // Apply stored preference immediately to prevent flash of wrong theme.
  apply(get());

  function init() {
    if (document.querySelector(".theme-toggle")) return; // already injected

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    btn.setAttribute("aria-label", "Toggle color theme");
    btn.innerHTML =
      '<span class="theme-toggle__icon" aria-hidden="true"></span>' +
      '<span class="theme-toggle__label"></span>';

    function refresh() {
      const eff = effective();
      btn.querySelector(".theme-toggle__icon").textContent =
        eff === "dark" ? "☾" : "☀";
      btn.querySelector(".theme-toggle__label").textContent =
        eff === "dark" ? "Dark" : "Light";
      btn.dataset.theme = eff;
      btn.setAttribute(
        "aria-label",
        "Switch to " + (eff === "dark" ? "light" : "dark") + " theme",
      );
    }

    btn.addEventListener("click", () => {
      const next = effective() === "light" ? "dark" : "light";
      set(next);
      apply(next);
      refresh();
    });

    // Respond to OS theme changes when the user has no explicit override.
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => {
        if (!get()) refresh();
      });

    // Prefer hosting the toggle inside the top bar nav if it exists; otherwise
    // fall back to a floating position. Listen for late injection by nav.js.
    function place() {
      const navHost = document.querySelector(".topbar__nav");
      if (navHost) {
        btn.classList.add("topbar__theme");
        navHost.appendChild(btn);
      } else {
        document.body.appendChild(btn);
      }
    }
    place();
    if (!document.querySelector(".topbar__nav")) {
      // nav.js runs on DOMContentLoaded; wait a tick and re-place if needed.
      requestAnimationFrame(() => {
        if (
          !btn.classList.contains("topbar__theme") &&
          document.querySelector(".topbar__nav")
        ) {
          btn.remove();
          place();
        }
      });
    }
    refresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
