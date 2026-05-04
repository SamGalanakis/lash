// Conditional Mermaid loader — only fetches the library when the page
// actually has .mermaid blocks. Saves ~2 MB on diagram-free pages.
(function () {
  function init() {
    if (!document.querySelector(".mermaid")) return;

    const url = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    import(url)
      .then(({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          securityLevel: "loose",
          themeVariables: {
            background: "#141412",
            primaryColor: "#141412",
            primaryTextColor: "#e8e4d0",
            primaryBorderColor: "#e8a33c",
            lineColor: "#8a9e6c",
            secondaryColor: "#0e0d0b",
            tertiaryColor: "#1b1a17",
            clusterBkg: "#10100e",
            clusterBorder: "#5a5a50",
            edgeLabelBackground: "#080807",
            textColor: "#e8e4d0",
            fontFamily: "Chivo Mono, monospace",
            actorBorder: "#e8a33c",
            actorBkg: "#141412",
            actorTextColor: "#e8e4d0",
            signalColor: "#c8c4b8",
            signalTextColor: "#e8e4d0",
            labelBoxBkgColor: "#141412",
            labelBoxBorderColor: "#5a5a50",
            labelTextColor: "#e8e4d0",
            noteBkgColor: "#201e1a",
            noteBorderColor: "#8a9e6c",
            noteTextColor: "#e8e4d0",
          },
        });
        return mermaid.run({ querySelector: ".mermaid" });
      })
      .catch((err) => {
        console.error("Mermaid load failed:", err);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
