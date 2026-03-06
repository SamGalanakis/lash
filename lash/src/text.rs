pub fn strip_repl_fragments(text: &str) -> String {
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
    use super::strip_repl_fragments;

    #[test]
    fn strips_dangling_and_complete_repl_fragments() {
        assert_eq!(strip_repl_fragments("<repl>hello</repl>"), "hello");
        assert_eq!(strip_repl_fragments("abc<repl"), "abc");
        assert_eq!(strip_repl_fragments("</repl>abc"), "abc");
    }
}
