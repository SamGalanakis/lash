pub(super) struct FenceExtraction {
    pub(super) code: String,
    pub(super) had_extra_fences: bool,
}

pub fn contains_closed_lashlang_fence(text: &str) -> bool {
    first_lashlang_fence_span(text).is_some_and(|span| span.close_len > 0)
}

pub fn project_visible_assistant_prose(text: &str) -> String {
    let Some(span) = first_lashlang_fence_span(text) else {
        return text.to_string();
    };
    text[..span.open_start].trim_end().to_string()
}

#[derive(Debug, Clone, Copy)]
struct FenceSpan {
    /// Byte offset of the first backtick of the opener.
    open_start: usize,
    /// Byte offset of the first byte of the body (after the opener
    /// line's terminating `\n`).
    body_start: usize,
    /// Byte offset of the first backtick of the closer (or `text.len()`
    /// when the body extends to EOF without a closer).
    body_end: usize,
    /// Number of backticks consumed by the closer. `0` when unclosed.
    /// Always ≤ `opener_len` — extra backticks in the closer run stay
    /// in the residual text so adjacent fences glued together (the
    /// "``````lashlang" pattern) still parse the way they always did.
    close_len: usize,
}

fn first_lashlang_fence_span(text: &str) -> Option<FenceSpan> {
    let bytes = text.as_bytes();
    let mut search_from = 0usize;
    while search_from < bytes.len() {
        let rel = text[search_from..].find("```")?;
        let open = search_from + rel;
        // CommonMark-style variable-length fences: count consecutive
        // backticks at the opener. N must be ≥3 (find("```") guarantees
        // that). Anything past N backticks is part of the opener run
        // and the matching closer must be at least N backticks long.
        let opener_len = bytes[open..].iter().take_while(|&&b| b == b'`').count();
        let after_open = open + opener_len;
        // The opening `\`\`\`` doesn't have to be on its own line —
        // an inline opener after prose still parses. The `lashlang`
        // language tag is distinctive enough that collisions with
        // inline prose are essentially impossible.
        let rest = &text[after_open..];
        let lang_end = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..lang_end].trim();
        if lang != "lashlang" {
            // Skip past this opener run and keep looking — a non-
            // lashlang language tag belongs to a different code block.
            search_from = after_open;
            continue;
        }
        let body_start = after_open + lang_end + 1;
        if body_start > text.len() {
            return None;
        }

        // Closer: the first run of ≥`opener_len` consecutive backticks
        // after `body_start`. We consume exactly `opener_len` of them
        // — leftover backticks in the run stay in the text so the
        // "``````lashlang" double-fence pattern still resolves into
        // two adjacent blocks.
        let (close, close_len) = match find_closing_fence(&bytes[body_start..], opener_len) {
            Some((rel, _run_len)) => (body_start + rel, opener_len),
            None => (text.len(), 0),
        };
        return Some(FenceSpan {
            open_start: open,
            body_start,
            body_end: close,
            close_len,
        });
    }
    None
}

/// Find the first run of consecutive backticks of length ≥ `min_len` in
/// `bytes`. Returns `(start_byte_offset, run_length)`.
fn find_closing_fence(bytes: &[u8], min_len: usize) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i;
            while i < bytes.len() && bytes[i] == b'`' {
                i += 1;
            }
            let len = i - start;
            if len >= min_len {
                return Some((start, len));
            }
        } else {
            i += 1;
        }
    }
    None
}

pub(super) fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    let span = first_lashlang_fence_span(text)?;
    let code = text[span.body_start..span.body_end]
        .trim_end_matches('\n')
        .to_string();
    let after_close = (span.body_end + span.close_len).min(text.len());
    let had_extra_fences = first_lashlang_fence_span(&text[after_close..]).is_some();
    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}
