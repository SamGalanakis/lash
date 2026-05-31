//! Finalize / non-streaming fence handling. All grammar lives in
//! [`crate::fence_scan`]; this module only layers the
//! extraction-and-projection conveniences the driver needs on top of
//! the shared scanner.

use crate::fence_scan::first_lashlang_fence_span;

pub(super) struct FenceExtraction {
    pub(super) code: String,
    pub(super) had_extra_fences: bool,
}

pub fn contains_closed_lashlang_fence(text: &str) -> bool {
    first_lashlang_fence_span(text).is_some_and(|span| span.is_closed())
}

pub fn project_visible_assistant_prose(text: &str) -> String {
    let Some(span) = first_lashlang_fence_span(text) else {
        return text.to_string();
    };
    text[..span.open_start].trim_end().to_string()
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
