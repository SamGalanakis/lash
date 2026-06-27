//! Finalize / non-streaming Lashlang cell handling. All paired-tag grammar
//! lives in [`crate::cell_scan`]; this module only layers the
//! extraction-and-projection conveniences the driver needs.

use crate::cell_scan::first_lashlang_cell_span;

pub(super) struct CellExtraction {
    pub(super) prose: String,
    pub(super) code: String,
    pub(super) lashlang_cell_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CellExtractionError {
    MultipleCells,
    TrailingText,
}

impl CellExtractionError {
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::MultipleCells => {
                "Model response contained multiple `<lashlang>...</lashlang>` blocks. Reply with exactly one paired block."
            }
            Self::TrailingText => {
                "Model response contained non-whitespace text after the `</lashlang>` block. Put all prose before the block, or omit the block for a final prose answer when allowed."
            }
        }
    }
}

pub fn contains_lashlang_cell(text: &str) -> bool {
    first_lashlang_cell_span(text).is_some()
}

pub fn project_visible_assistant_prose(text: &str) -> String {
    let Some(span) = first_lashlang_cell_span(text) else {
        return text.to_string();
    };
    text[..span.start_tag_start].trim_end().to_string()
}

pub(super) fn extract_lashlang_cell(
    text: &str,
) -> Result<Option<CellExtraction>, CellExtractionError> {
    let Some(span) = first_lashlang_cell_span(text) else {
        return Ok(None);
    };
    let trailing = &text[span.end_tag_end..];
    if !trailing.trim().is_empty() {
        return Err(if first_lashlang_cell_span(trailing).is_some() {
            CellExtractionError::MultipleCells
        } else {
            CellExtractionError::TrailingText
        });
    }
    let code = text[span.body_start..span.body_end].to_string();
    Ok(Some(CellExtraction {
        prose: text[..span.start_tag_start].trim_end().to_string(),
        code,
        lashlang_cell_count: 1,
    }))
}
