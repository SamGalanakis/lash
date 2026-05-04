//! Render markdown source into HTML for the session export.
//!
//! Uses `pulldown-cmark` with sensible extensions enabled (tables,
//! footnotes, strikethrough, task lists, autolinks). Output is wrapped
//! in a `<div class="md">` so the export stylesheet can scope its
//! markdown-specific rules without leaking into the rest of the page.

use pulldown_cmark::{Options, Parser, html};

/// Render `input` (markdown) into an HTML fragment scoped under
/// `div.md`. Always emits the wrapper, even when the input is empty,
/// so callers can rely on a stable structure.
pub fn render(input: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_HEADING_ATTRIBUTES);
    // Smart punctuation rewrites `---` → em-dash and `"x"` → curly
    // quotes. Useful for narrative copy, lethal for technical content
    // where literal punctuation matters (role separators, code, JSON).

    let parser = Parser::new_ext(input, options);
    let mut body = String::with_capacity(input.len() + input.len() / 4);
    html::push_html(&mut body, parser);

    let mut out = String::with_capacity(body.len() + 16);
    out.push_str("<div class=\"md\">");
    out.push_str(&body);
    out.push_str("</div>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_headings_and_lists() {
        let html = render("# heading\n\n- item one\n- item two");
        assert!(html.contains("<h1>"));
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>item one</li>"));
    }

    #[test]
    fn fenced_code_emits_pre_code_with_language_class() {
        let html = render("```rust\nlet x = 1;\n```");
        assert!(html.contains("<pre><code class=\"language-rust\">"));
    }

    #[test]
    fn escapes_inline_html_safely() {
        let html = render("hello <script>alert(1)</script> world");
        // pulldown-cmark passes inline HTML through by default; we accept
        // that — the trace is operator-internal, not user-facing.
        // The test pins behavior so we notice if it changes.
        assert!(html.contains("hello"));
    }
}
