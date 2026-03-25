use lash::strip_repl_fragments;

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
