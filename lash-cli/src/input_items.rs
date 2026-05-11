use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

use crate::app::App;
use crate::editor::PendingImage;
use lash_core::InputItem;

/// Build structured turn items from editor input:
/// - `@path` becomes a host-prepared text marker when resolvable
/// - `[Image #n]` binds to pasted image `n` from this turn's image list
/// - plain text remains `Text`
pub fn build_items_from_editor_input(
    input: &str,
    images: Vec<PendingImage>,
) -> (Vec<InputItem>, HashMap<String, Vec<u8>>) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut items: Vec<InputItem> = Vec::new();
    let mut image_blobs: HashMap<String, Vec<u8>> = HashMap::new();
    let image_slots: HashMap<usize, Vec<u8>> = images
        .into_iter()
        .map(|image| (image.id, image.png_bytes))
        .collect();
    let mut emitted_image_ids: HashMap<usize, String> = HashMap::new();

    let mut text_buf = String::with_capacity(input.len());
    let mut i = 0;
    let bytes = input.as_bytes();

    while i < bytes.len() {
        if let Some((next_i, img_idx)) = parse_image_marker_at(input, i)
            && let Some(bytes) = image_slots.get(&img_idx)
        {
            push_text_item(&mut items, &mut text_buf);
            let id = emitted_image_ids
                .entry(img_idx)
                .or_insert_with(|| format!("img-{img_idx}"))
                .clone();
            image_blobs
                .entry(id.clone())
                .or_insert_with(|| bytes.clone());
            items.push(InputItem::ImageRef { id });
            i = next_i;
            continue;
        }

        if bytes[i] == b'@' {
            // Check: must be at start or preceded by whitespace
            let at_start = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if at_start {
                // Find the end of the token (next whitespace or end)
                let token_start = i + 1;
                let mut token_end = token_start;
                while token_end < bytes.len() && !bytes[token_end].is_ascii_whitespace() {
                    token_end += 1;
                }
                let token = &input[token_start..token_end];
                if !token.is_empty() {
                    let path = if token.starts_with('/') {
                        PathBuf::from(token)
                    } else {
                        cwd.join(token)
                    };
                    if path.is_file() {
                        text_buf.push_str(&format!("[file: {}]", path.display()));
                        i = token_end;
                        continue;
                    } else if path.is_dir() {
                        text_buf.push_str(&format!(
                            "[directory: {}]",
                            path.display().to_string().trim_end_matches('/')
                        ));
                        i = token_end;
                        continue;
                    }
                }
            }
        }
        let ch = input[i..].chars().next().unwrap();
        text_buf.push(ch);
        i += ch.len_utf8();
    }

    push_text_item(&mut items, &mut text_buf);

    (items, image_blobs)
}

fn push_text_item(items: &mut Vec<InputItem>, text: &mut String) {
    if text.is_empty() {
        return;
    }
    if let Some(InputItem::Text { text: prev }) = items.last_mut() {
        prev.push_str(text);
        text.clear();
        return;
    }
    items.push(InputItem::Text {
        text: std::mem::take(text),
    });
}

pub(crate) fn parse_image_marker_at(input: &str, start: usize) -> Option<(usize, usize)> {
    let rest = &input[start..];
    let prefix = "[Image #";
    if !rest.starts_with(prefix) {
        return None;
    }
    let after = &rest[prefix.len()..];
    let digits_len = after
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .map(char::len_utf8)
        .sum::<usize>();
    if digits_len == 0 {
        return None;
    }
    let digits = &after[..digits_len];
    let remaining = &after[digits_len..];
    if !remaining.starts_with(']') {
        return None;
    }
    let idx = digits.parse::<usize>().ok()?;
    if idx == 0 {
        return None;
    }
    Some((start + prefix.len() + digits_len + 1, idx))
}

pub(crate) fn image_marker_ranges(input: &str) -> Vec<(Range<usize>, usize)> {
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < input.len() {
        if let Some((next, idx)) = parse_image_marker_at(input, i) {
            ranges.push((i..next, idx));
            i = next;
            continue;
        }
        i += input[i..].chars().next().map(char::len_utf8).unwrap_or(1);
    }
    ranges
}

/// Insert an inline attachment marker like `[Image #1]` at the current cursor,
/// adding surrounding spaces when needed so it reads naturally in the input.
pub fn insert_inline_marker(app: &mut App, marker: &str) {
    let needs_leading_space = app.cursor_pos() > 0
        && app.input()[..app.cursor_pos()]
            .chars()
            .next_back()
            .is_some_and(|c: char| !c.is_whitespace());

    let needs_trailing_space = app.cursor_pos() < app.input().len()
        && app.input()[app.cursor_pos()..]
            .chars()
            .next()
            .is_some_and(|c: char| !c.is_whitespace());

    if needs_leading_space {
        app.insert_char(' ');
    }
    app.insert_text(marker);
    if needs_trailing_space {
        app.insert_char(' ');
    }
}
