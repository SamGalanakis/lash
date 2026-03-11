use lash_core::strip_repl_fragments;

use crate::app::DisplayBlock;

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

pub fn push_assistant_text_block(blocks: &mut Vec<DisplayBlock>, text: &str) -> bool {
    let cleaned = normalize_stream_text(text);
    if cleaned.is_empty() {
        return false;
    }
    blocks.push(DisplayBlock::AssistantText(cleaned));
    true
}

#[derive(Default)]
pub struct AssistantReplay {
    pending_text: String,
    fallback_assistant_text: Option<String>,
}

impl AssistantReplay {
    pub fn push_text_delta(&mut self, text: &str) {
        self.pending_text.push_str(text);
    }

    pub fn remember_llm_response(&mut self, text: &str) {
        self.fallback_assistant_text = Some(text.to_string());
    }

    pub fn flush(&mut self, blocks: &mut Vec<DisplayBlock>) -> bool {
        let text = if self.pending_text.is_empty() {
            self.fallback_assistant_text.take().unwrap_or_default()
        } else {
            std::mem::take(&mut self.pending_text)
        };
        self.fallback_assistant_text = None;
        push_assistant_text_block(blocks, &text)
    }
}
