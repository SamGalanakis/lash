use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

/// Insert zero-width non-joiners between common programming-ligature trigger runs.
/// This keeps terminal layout stable in fonts that aggressively shape operators,
/// while preserving the visible text and display width.
pub fn neutralize_ligatures(text: &str) -> String {
    const ZWNJ: char = '\u{200C}';

    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        out.push(ch);

        let next = chars.get(i + 1).copied();
        if let Some(next) = next
            && should_break_ligature(ch, next)
        {
            out.push(ZWNJ);
        }
        i += 1;
    }

    out
}

pub fn visible_width(text: &str) -> usize {
    UnicodeWidthStr::width(strip_invisible_shaping_controls(text).as_str())
}

pub fn line_visible_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| visible_width(span.content.as_ref()))
        .sum()
}

pub fn sanitize_span<'a>(
    content: impl Into<std::borrow::Cow<'a, str>>,
    style: ratatui::style::Style,
) -> Span<'static> {
    Span::styled(neutralize_ligatures(content.into().as_ref()), style)
}

pub fn strip_invisible_shaping_controls(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '\u{200C}' && *ch != '\u{200D}' && *ch != '\u{2060}')
        .collect()
}

fn should_break_ligature(a: char, b: char) -> bool {
    matches!(
        (a, b),
        ('=', '=')
            | ('!', '=')
            | ('<', '=')
            | ('>', '=')
            | ('-', '>')
            | ('=', '>')
            | ('<', '-')
            | ('|', '|')
            | ('&', '&')
            | ('+', '+')
            | ('-', '-')
            | ('/', '/')
            | ('*', '*')
            | ('|', '>')
            | (':', ':')
            | ('<', '<')
            | ('>', '>')
            | ('~', '>')
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutralizer_preserves_visible_text() {
        let original = "pub fn foo() -> bool { a != b && c == d }";
        let sanitized = neutralize_ligatures(original);
        assert_ne!(sanitized, original);
        assert_eq!(strip_invisible_shaping_controls(&sanitized), original);
    }

    #[test]
    fn neutralizer_preserves_display_width() {
        let original = "a -> b != c && d";
        let sanitized = neutralize_ligatures(original);
        assert_eq!(visible_width(original), visible_width(&sanitized));
    }
}
