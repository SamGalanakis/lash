/// Normalize streaming assistant text for display:
/// - drop leading/trailing blank lines
/// - collapse consecutive blank lines to a single blank line
pub fn normalize_stream_text(text: &str) -> String {
    let sanitized = strip_repl_fragments(text);
    let mut out = String::new();
    let mut started = false;
    let mut prev_blank = false;

    for line in sanitized.split('\n') {
        let is_blank = line.trim().is_empty();
        if !started {
            if is_blank {
                continue;
            }
            out.push_str(line);
            started = true;
            prev_blank = false;
            continue;
        }

        if is_blank {
            if !prev_blank {
                out.push('\n');
                prev_blank = true;
            }
        } else {
            out.push('\n');
            out.push_str(line);
            prev_blank = false;
        }
    }

    while out.ends_with('\n') {
        out.pop();
    }

    out
}

fn strip_repl_fragments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let tail = &text[i..];
        if let Some(consumed) = consume_repl_fragment_prefix(tail) {
            i += consumed;
            continue;
        }
        let mut chars = tail.chars();
        let ch = chars.next().expect("tail is non-empty");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn consume_repl_fragment_prefix(text: &str) -> Option<usize> {
    const PREFIXES: [&str; 2] = ["<repl", "</repl"];
    for prefix in PREFIXES {
        if !text.starts_with(prefix) {
            continue;
        }
        let next = text.as_bytes().get(prefix.len()).copied();
        let valid_suffix = next.is_none_or(|b| b == b'>' || b == b'/' || b.is_ascii_whitespace());
        if !valid_suffix {
            continue;
        }
        if let Some(end_idx) = text.find('>') {
            return Some(end_idx + 1);
        }
        return Some(prefix.len());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::normalize_stream_text;

    #[test]
    fn strips_repl_fragments() {
        let raw = "\n<repl>\nproc\n</repl>\n\nok\n";
        assert_eq!(normalize_stream_text(raw), "proc\n\nok");
    }
}
