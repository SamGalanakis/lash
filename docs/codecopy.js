// docs/codecopy.js — drop a copy-to-clipboard button on every <pre> block.
(function () {
  function init() {
    const blocks = document.querySelectorAll("pre");
    blocks.forEach((pre) => {
      if (pre.dataset.copyAttached) return;
      pre.dataset.copyAttached = "1";

      // Wrap pre in a positioned container so the button can sit absolute.
      const wrap = document.createElement("div");
      wrap.className = "pre-wrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pre-copy";
      btn.setAttribute("aria-label", "Copy code");
      btn.textContent = "Copy";

      btn.addEventListener("click", async () => {
        const text = pre.innerText;
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = "Copied";
          btn.classList.add("is-success");
          setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("is-success");
          }, 1400);
        } catch (err) {
          btn.textContent = "Failed";
          setTimeout(() => (btn.textContent = "Copy"), 1400);
        }
      });
      wrap.appendChild(btn);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
