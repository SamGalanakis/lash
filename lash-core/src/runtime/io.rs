//! Input normalization and message-delta utilities used by the runtime.
//!
//! Extracted from `runtime/mod.rs`. These helpers are crate-internal.

use std::collections::HashMap;

use super::{InputItem, NormalizedItem};

pub(super) fn normalize_input_items(
    items: &[InputItem],
    image_blobs: &HashMap<String, Vec<u8>>,
    attachment_store: &dyn crate::AttachmentStore,
) -> Result<Vec<NormalizedItem>, String> {
    let mut out: Vec<NormalizedItem> = Vec::new();
    for item in items {
        match item {
            InputItem::Text { text } => push_text(&mut out, text.clone()),
            InputItem::ImageRef { id } => {
                if id.is_empty() {
                    return Err("Invalid image_ref: id cannot be empty".to_string());
                }
                let Some(blob) = image_blobs.get(id) else {
                    return Err(format!("Invalid image_ref: missing blob for id '{id}'"));
                };
                let meta = crate::AttachmentMeta::new(
                    crate::AttachmentId::new("pending"),
                    crate::MediaType::Image(crate::ImageMediaType::Png),
                    blob.len() as u64,
                    None,
                    None,
                    Some(id.clone()),
                );
                let reference = attachment_store
                    .put(blob.clone(), meta)
                    .map_err(|err| format!("Failed to store image_ref '{id}': {err}"))?;
                out.push(NormalizedItem::Image(reference));
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
