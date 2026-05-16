// docs/syntax.js — syntax highlighting via highlight.js.
//
// Loads the library on demand, registers a small Lashlang grammar, and
// applies highlighting to every `<pre><code>` block on the page. If a block
// already has a `language-foo` class, that hint is used; otherwise the
// auto-detector picks the best fit.

(function () {
  if (!document.querySelector("pre code")) return;

  const CORE = "https://esm.sh/highlight.js@11.11.1/lib/core";
  const LANG = (name) =>
    `https://esm.sh/highlight.js@11.11.1/lib/languages/${name}`;

  // Languages to register. The keys are the canonical names; aliases that
  // appear on `<code class="language-xxx">` get mapped to these.
  const langs = [
    "rust",
    "javascript",
    "typescript",
    "python",
    "bash",
    "json",
    "yaml",
    "ini", // serves TOML too
    "css",
    "xml", // serves HTML
    "markdown",
    "diff",
    "plaintext",
  ];

  // Lashlang — Python/Lua-flavoured scripting surface used by RLM mode.
  function lashlang(hljs) {
    return {
      name: "Lashlang",
      aliases: ["lash", "lashlang"],
      keywords: {
        keyword:
          "if else elif while for return let fn def end then do break continue in",
        literal: "true false null nil None True False",
        built_in:
          "print call start submit spawn_agent llm_query continue_as len str int float list dict range",
      },
      contains: [
        hljs.COMMENT("//", "$"),
        hljs.COMMENT("#", "$"),
        hljs.QUOTE_STRING_MODE,
        hljs.APOS_STRING_MODE,
        hljs.NUMBER_MODE,
        {
          className: "title.function",
          begin: /\b[a-z_][\w]*(?=\s*\()/,
        },
        {
          className: "symbol",
          begin: /:[a-z_][\w]*/,
        },
      ],
    };
  }

  async function init() {
    try {
      const hljs = (await import(CORE)).default;
      const modules = await Promise.all(langs.map((n) => import(LANG(n))));
      modules.forEach((m, i) => hljs.registerLanguage(langs[i], m.default));
      hljs.registerLanguage("toml", (h) => h.getLanguage("ini"));
      hljs.registerLanguage("html", (h) => h.getLanguage("xml"));
      hljs.registerLanguage("sh", (h) => h.getLanguage("bash"));
      hljs.registerLanguage("shell", (h) => h.getLanguage("bash"));
      hljs.registerLanguage("text", (h) => h.getLanguage("plaintext"));
      hljs.registerLanguage("lashlang", lashlang);

      hljs.configure({ ignoreUnescapedHTML: true, throwUnescapedHTML: false });

      document.querySelectorAll("pre code").forEach((block) => {
        // Skip if the block already has a hljs class from a previous run.
        if (block.classList.contains("hljs")) return;
        try {
          hljs.highlightElement(block);
        } catch (err) {
          // best-effort; never let a single block break the rest
          console.warn("syntax highlight failed", err);
        }
      });
    } catch (err) {
      console.warn("highlight.js load failed", err);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
