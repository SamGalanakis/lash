use std::fmt::Write as _;

pub(crate) fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

pub(crate) fn escape_attr(input: &str) -> String {
    escape(input)
}

/// Escape and preserve newlines. Used for prose content where line breaks
/// matter but raw HTML must not pass through. We keep `\n` as `\n` and rely
/// on `white-space: pre-wrap` in CSS, which preserves both newlines and
/// soft-wraps long lines without dropping double-newline paragraph breaks.
pub(crate) fn escape_breaks(input: &str) -> String {
    escape(input)
}

// ─── tiny json syntax highlighter ────────────────────────────────────────────
// Walks the already-pretty-printed JSON character-by-character. Cheap, no
// regex, safe for arbitrary input. Output is HTML with span classes:
//   .j-key, .j-str, .j-num, .j-bool, .j-null, .j-punct.
pub(crate) fn json_highlight(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'"' => {
                // find end of string (respect escapes)
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                let raw = &s[start..i];
                // a key is a string immediately followed (after possible
                // whitespace) by ':'. peek ahead to decide the class.
                let mut j = i;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                let class = if j < bytes.len() && bytes[j] == b':' {
                    "j-key"
                } else {
                    "j-str"
                };
                let _ = write!(out, "<span class=\"{class}\">{}</span>", escape(raw));
            }
            b'{' | b'}' | b'[' | b']' | b',' | b':' => {
                let _ = write!(
                    out,
                    "<span class=\"j-punct\">{}</span>",
                    escape(&(c as char).to_string())
                );
                i += 1;
            }
            b't' | b'f' if matches_at(bytes, i, b"true") || matches_at(bytes, i, b"false") => {
                let len = if matches_at(bytes, i, b"true") { 4 } else { 5 };
                let _ = write!(
                    out,
                    "<span class=\"j-bool\">{}</span>",
                    escape(&s[i..i + len])
                );
                i += len;
            }
            b'n' if matches_at(bytes, i, b"null") => {
                out.push_str("<span class=\"j-null\">null</span>");
                i += 4;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                if bytes[i] == b'-' {
                    i += 1;
                }
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit()
                        || bytes[i] == b'.'
                        || bytes[i] == b'e'
                        || bytes[i] == b'E'
                        || bytes[i] == b'+'
                        || bytes[i] == b'-')
                {
                    i += 1;
                }
                let _ = write!(out, "<span class=\"j-num\">{}</span>", escape(&s[start..i]));
            }
            b'\n' => {
                out.push('\n');
                i += 1;
            }
            _ => {
                // pass through (whitespace, etc.) — escape just in case
                let ch = s[i..].chars().next().unwrap_or(' ');
                let len = ch.len_utf8();
                out.push_str(&escape(&ch.to_string()));
                i += len;
            }
        }
    }
    out
}

fn matches_at(bytes: &[u8], i: usize, needle: &[u8]) -> bool {
    bytes.len() >= i + needle.len() && &bytes[i..i + needle.len()] == needle
}

// ─── multi-view rendering ───────────────────────────────────────────────────
//
// A *view* is a chain of sessions joined by `continue_as` handoffs — the
// root and every subagent are heads of their own views. Inside a view,
// successor sessions are inlined behind a handoff divider; subagents
// become drill-in cards that switch the page to the subagent's view.

pub(crate) fn js_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ─── CSS ────────────────────────────────────────────────────────────────────
