//! Single source of truth for the ` ```lashlang ` fence grammar.
//!
//! The RLM protocol detects one CommonMark-style fenced `lashlang`
//! block per assistant turn. Two consumers need to agree, byte for
//! byte, on what counts as that block:
//!
//! - the **streaming mask** ([`crate::stream_mask`]) hides the fence
//!   body from the user-visible stream as it arrives, and
//! - the **finalize / non-streaming** path
//!   ([`crate::protocol::fence`]) re-parses the full assistant text to
//!   extract the code that gets executed.
//!
//! Before this module the grammar was copied three times (the
//! streaming detector, the finalize extractor, and the history
//! reasoning-stripper) with comments insisting they "stay in
//! lockstep". They didn't: the finalize path required the language tag
//! to be newline-terminated while the stream path accepted an opener
//! at end-of-text, so the same input could be a block on one path and
//! prose on the other. Everything now funnels through the functions
//! here.

/// The one canonical fence language tag. Aliases (`rlm`, `lash`, …)
/// are intentionally *not* recognized: the prompt contract asks for a
/// single ` ```lashlang ` block, so the parser accepts exactly that.
pub(crate) const FENCE_LANG: &str = "lashlang";

/// A located ` ```lashlang ` fence inside some text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FenceSpan {
    /// Byte offset of the first backtick of the opener.
    pub(crate) open_start: usize,
    /// Number of backticks in the opener run (≥3). The closer must be
    /// a run of at least this many backticks.
    pub(crate) opener_len: usize,
    /// Byte offset of the first byte of the body (after the opener
    /// line's terminating `\n`, or end-of-text when the opener is the
    /// final token with no newline yet).
    pub(crate) body_start: usize,
    /// Byte offset of the first backtick of the closer (or `text.len()`
    /// when the body extends to EOF without a closer).
    pub(crate) body_end: usize,
    /// Number of backticks consumed by the closer. `0` when unclosed.
    /// Capped at `opener_len` — extra backticks in the closing run stay
    /// in the residual text so adjacent fences glued together (the
    /// `` ``````lashlang `` pattern) still parse the way they always
    /// did.
    pub(crate) close_len: usize,
}

impl FenceSpan {
    pub(crate) fn is_closed(&self) -> bool {
        self.close_len > 0
    }
}

/// Locate the first ` ```lashlang ` fence in `text`, if any.
///
/// Accepts an opener whose language line is terminated by `\n` *or*
/// that sits at end-of-text with no newline yet (`"…```lashlang"`); the
/// latter yields an empty body anchored at EOF. This is the single
/// behavior both the streaming and finalize paths share.
pub(crate) fn first_lashlang_fence_span(text: &str) -> Option<FenceSpan> {
    let bytes = text.as_bytes();
    let mut search_from = 0usize;
    while search_from <= bytes.len() {
        let rel = text.get(search_from..)?.find("```")?;
        let open_start = search_from + rel;
        // CommonMark-style variable-length fences: count consecutive
        // backticks at the opener. N must be ≥3 (the `find("```")`
        // guarantees that). Anything past N backticks is part of the
        // opener run and the matching closer must be at least N long.
        let opener_len = bytes[open_start..]
            .iter()
            .take_while(|&&b| b == b'`')
            .count();
        let after_open = open_start + opener_len;

        // The opening ``` doesn't have to be on its own line — an
        // inline opener after prose still parses. The `lashlang`
        // language tag is distinctive enough that collisions with
        // inline prose are essentially impossible.
        let rest = &text[after_open..];
        let lang_end = rest.find('\n').unwrap_or(rest.len());
        let lang = rest[..lang_end].trim();
        if lang != FENCE_LANG {
            // Skip past this opener run and keep looking — a non-
            // lashlang language tag belongs to a different code block.
            search_from = after_open;
            continue;
        }

        // Body starts after the language line's terminating `\n`. When
        // the language line isn't newline-terminated (opener at EOF),
        // `lang_end == rest.len()` and the body is empty at EOF.
        let body_start = if lang_end < rest.len() {
            after_open + lang_end + 1
        } else {
            after_open + lang_end
        };

        let (body_end, close_len) = match find_closing_fence(&bytes[body_start..], opener_len) {
            Some((rel, _run_len)) => (body_start + rel, opener_len),
            None => (text.len(), 0),
        };
        return Some(FenceSpan {
            open_start,
            opener_len,
            body_start,
            body_end,
            close_len,
        });
    }
    None
}

/// Find the first run of consecutive backticks of length ≥ `min_len` in
/// `bytes`. Returns `(start_byte_offset, run_length)`.
pub(crate) fn find_closing_fence(bytes: &[u8], min_len: usize) -> Option<(usize, usize)> {
    if min_len == 0 {
        return None;
    }
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

/// `true` when `body` contains a closing run of at least `opener_len`
/// consecutive backticks. Thin wrapper over [`find_closing_fence`] used
/// by the streaming mask, which only needs the yes/no answer.
pub(crate) fn body_has_closing_fence(body: &str, opener_len: usize) -> bool {
    find_closing_fence(body.as_bytes(), opener_len).is_some()
}
