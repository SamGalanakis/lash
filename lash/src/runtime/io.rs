//! Input normalization and message-delta utilities used by the runtime.
//!
//! Extracted from `runtime/mod.rs`. These helpers are crate-internal; the
//! `InputItem` and `PathResolver` types they use keep their public paths
//! in `mod.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::session_model::Message;

use super::{InputItem, NormalizedItem, PathResolver};

pub(super) fn projection_message_delta_if_base_preserved<'a>(
    base: &[Message],
    next: impl IntoIterator<Item = &'a Message>,
) -> Option<Vec<Message>> {
    let mut base_index = 0;
    let mut appended = Vec::new();
    for message in next.into_iter().filter(|message| !message.is_transient()) {
        if let Some(base_message) = base.get(base_index) {
            if base_message.id != message.id {
                return None;
            }
            base_index += 1;
        } else {
            appended.push(message.clone());
        }
    }
    if base_index == base.len() {
        Some(appended)
    } else {
        None
    }
}

pub(super) fn normalize_input_items(
    items: &[InputItem],
    image_blobs: &HashMap<String, Vec<u8>>,
    base_dir: &Path,
    path_resolver: &dyn PathResolver,
) -> Result<Vec<NormalizedItem>, String> {
    let mut out: Vec<NormalizedItem> = Vec::new();
    for item in items {
        match item {
            InputItem::Text { text } => push_text(&mut out, text.clone()),
            InputItem::FileRef { path } => {
                let abs = resolve_existing_path(path, true, base_dir, path_resolver)?;
                push_text(&mut out, format!("[file: {}]", abs.display()));
            }
            InputItem::DirRef { path } => {
                let abs = resolve_existing_path(path, false, base_dir, path_resolver)?;
                push_text(
                    &mut out,
                    format!(
                        "[directory: {}]",
                        abs.to_string_lossy().trim_end_matches('/')
                    ),
                );
            }
            InputItem::ImageRef { id } => {
                if id.is_empty() {
                    return Err("Invalid image_ref: id cannot be empty".to_string());
                }
                let Some(blob) = image_blobs.get(id) else {
                    return Err(format!("Invalid image_ref: missing blob for id '{id}'"));
                };
                out.push(NormalizedItem::Image(blob.clone()));
            }
        }
    }
    Ok(out)
}

fn push_text(out: &mut Vec<NormalizedItem>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(NormalizedItem::Text(last)) = out.last_mut() {
        last.push_str(&text);
    } else {
        out.push(NormalizedItem::Text(text));
    }
}

fn resolve_existing_path(
    path: &str,
    expect_file: bool,
    base_dir: &Path,
    path_resolver: &dyn PathResolver,
) -> Result<PathBuf, String> {
    path_resolver.resolve(path, expect_file, base_dir)
}
