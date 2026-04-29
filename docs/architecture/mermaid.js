(function () {
  function initializeMermaid() {
    if (!window.mermaid) {
      console.error("Mermaid failed to load. Check network access to the Mermaid browser bundle.");
      return;
    }

    window.mermaid.initialize({
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
        noteTextColor: "#e8e4d0"
      }
    });

    window.mermaid.run({ querySelector: ".mermaid" });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeMermaid, { once: true });
  } else {
    initializeMermaid();
  }
})();
