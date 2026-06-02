/* lash · docs · code-block syntax highlighting
   Lazy-loads Prism from a CDN with its autoloader plugin, registers a
   small lashlang grammar, then walks every <pre> block on the page.
   For each block the language comes from data-lang when present, or a
   short content heuristic otherwise — and the resulting language is
   echoed back into data-lang so the corner label is always accurate.
   Re-runnable: docs.js calls highlightCodeBlocks() on bootstrap and
   after every SPA swap. */

(function () {
  "use strict";

  const PRISM_VERSION = "1.29.0";
  const PRISM_CDN = `https://cdn.jsdelivr.net/npm/prismjs@${PRISM_VERSION}`;

  // ── single shared loader promise, resolves to window.Prism ──
  let prismPromise = null;
  function ensurePrism() {
    if (prismPromise) return prismPromise;
    prismPromise = new Promise((resolve, reject) => {
      // Prism's autoloader needs Prism.manual = true set BEFORE the
      // core script runs, so we set it on window first.
      window.Prism = window.Prism || {};
      window.Prism.manual = true;

      function loadScript(src) {
        return new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = src;
          s.defer = true;
          s.onload = () => res();
          s.onerror = () => rej(new Error("failed: " + src));
          document.head.appendChild(s);
        });
      }

      loadScript(`${PRISM_CDN}/prism.min.js`)
        .then(() =>
          loadScript(`${PRISM_CDN}/plugins/autoloader/prism-autoloader.min.js`)
        )
        .then(() => {
          if (window.Prism && window.Prism.plugins && window.Prism.plugins.autoloader) {
            window.Prism.plugins.autoloader.languages_path = `${PRISM_CDN}/components/`;
          }
          defineLashlang(window.Prism);
          resolve(window.Prism);
        })
        .catch(reject);
    });
    return prismPromise;
  }

  // ── lashlang grammar — Prism doesn't ship with this one ──
  // Kept in sync with crates/lashlang: the lexer keywords plus the parser's
  // contextual keywords (process / start / finish / yield / wake / signal /
  // wait / run / with / trigger / while / break / continue / let /
  // enum), the builtins.rs registry, and the primitive type names. Comments
  // are `#` and `//`; strings are single- or triple-quoted ("""…""",
  // r"""…""", r'''…'''). There are no /* */ block comments in lashlang.
  function defineLashlang(Prism) {
    if (!Prism || !Prism.languages || Prism.languages.lashlang) return;
    Prism.languages.lashlang = {
      "triple-string": {
        pattern: /r?"""[\s\S]*?"""|r'''[\s\S]*?'''/,
        alias: "string",
        greedy: true,
      },
      "string": {
        pattern: /r?"(?:[^"\\]|\\.)*"/,
        greedy: true,
      },
      "comment": {
        pattern: /#.*|\/\/.*/,
        greedy: true,
      },
      "keyword": /\b(?:and|as|await|break|call|cancel|continue|each|else|enum|every|fail|false|finish|for|if|in|let|not|null|on|or|print|process|run|signal|sleep|start|submit|trigger|true|type|wait|wake|while|with|yield)\b/,
      "class-name": /\b[A-Z][A-Za-z0-9_]*\b/,
      "type": /\b(?:str|string|int|integer|float|number|bool|boolean|any|list)\b/,
      "builtin": /\b(?:ceil_div|contains|empty|ends_with|find|floor_div|format|grep_text|join|json_parse|keys|len|push|range|slice|split|starts_with|to_float|to_int|to_string|trim|validate|values)\b/,
      "function": /\b[a-z_][a-z0-9_]*(?=\s*\()/i,
      "number": /\b\d+(?:\.\d+)?\b/,
      "operator": /=>|->|::|[+\-*/%=<>!?:&|]+/,
      "punctuation": /[{}[\];(),.]/,
    };
    // shorthand alias used in fenced blocks
    Prism.languages.lash = Prism.languages.lashlang;
  }

  // ── language detection + mapping ──
  function mapDataLang(dl) {
    const d = (dl || "").trim().toLowerCase();
    if (!d) return null;
    if (d === "rs" || d === "rust") return "rust";
    if (d === "sh" || d === "shell" || d === "console") return "bash";
    if (d === "js") return "javascript";
    if (d === "ts") return "typescript";
    if (d === "py") return "python";
    if (d.includes("toml") || d === "cargo") return "toml";
    if (d === "lash" || d === "lashlang") return "lashlang";
    if (d === "yml") return "yaml";
    return d;
  }

  function detectLang(text) {
    const t = text || "";
    // Most-distinctive cues first.
    if (/^\s*\$\s/m.test(t) || /^\s*#\s*!.+/.test(t)) return "bash";
    if (/\bprocess\s+\w|\btrigger\s+\w|\bawait\s+[A-Z][A-Z0-9_]*\.\w+\.|\bstart\s+\w+\s*\(|```lashlang/.test(t)) return "lashlang";
    if (/^\s*[#/]{0,2}\[(?:dev-)?dependencies\]/m.test(t)) return "toml";
    if (/(?:^|\n)\s*\[[\w.\-]+\]\s*(?:\n|$)/.test(t) && /=\s*["'\d{[]/.test(t)) return "toml";
    if (/\bfn\s+\w|\bpub\s+(?:fn|struct|enum|use|mod|trait|type|const|static)\b|\buse\s+[a-z_][\w:]*::|#\[\w|\bimpl\s+\w/.test(t)) return "rust";
    if (/^\s*\{[\s\S]*"[\w-]+":\s/.test(t)) return "json";
    // Rust is the lingua franca of these docs — fall back to it so
    // colours always show up rather than rendering plain.
    return "rust";
  }

  // ── walk and highlight every pre > code on the page ──
  async function highlightCodeBlocks(root) {
    root = root || document;
    const blocks = root.querySelectorAll("pre > code");
    if (blocks.length === 0) return;

    let Prism;
    try {
      Prism = await ensurePrism();
    } catch (e) {
      // Network failure: leave the blocks as plain text rather than
      // crash. Better to ship readable code than no code.
      return;
    }

    blocks.forEach((code) => {
      if (code.dataset.lashHighlighted === "1") return;
      const pre = code.parentElement;
      if (!pre) return;
      const dataLang = pre.getAttribute("data-lang");
      const lang = mapDataLang(dataLang) || detectLang(code.textContent);

      // strip any prior language-* class so we don't double up across
      // SPA swaps; then set the new one.
      Array.from(code.classList)
        .filter((c) => c.startsWith("language-"))
        .forEach((c) => code.classList.remove(c));
      code.classList.add(`language-${lang}`);

      // echo the resolved language back as data-lang so the corner
      // label is always shown (it reads attr(data-lang) in CSS).
      if (!dataLang) pre.setAttribute("data-lang", lang);
      code.dataset.lashHighlighted = "1";

      try {
        Prism.highlightElement(code);
      } catch (_) {
        // grammar might not have loaded yet — autoloader is async.
        // Mark as needing a retry: clear the flag so a later run
        // picks it up.
        code.dataset.lashHighlighted = "";
        setTimeout(() => {
          if (code.dataset.lashHighlighted === "1") return;
          try {
            Prism.highlightElement(code);
            code.dataset.lashHighlighted = "1";
          } catch (_) {}
        }, 300);
      }
    });
  }

  window.__LASH_HIGHLIGHT = highlightCodeBlocks;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => highlightCodeBlocks());
  } else {
    highlightCodeBlocks();
  }
})();
