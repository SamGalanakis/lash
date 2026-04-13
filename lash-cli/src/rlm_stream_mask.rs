//! RLM stream masking plugin.
//!
//! When the outer LLM streams its assistant response in RLM mode, the
//! prose is useful (model "thinking out loud") but the lashlang code
//! body inside a fenced ` ```lashlang ` block is noise. This plugin
//! intercepts the `AssistantStreamHook` pipeline, detects the fence
//! opener (even when it arrives split across multiple streaming deltas),
//! and:
//!
//! - **Suppresses** the code body (returns empty chunks so no
//!   `TextDelta` reaches the UI).
//! - **Emits** a `PluginSurfaceEvent::Custom { name: "rlm_fence_start" }`
//!   when the fence is detected.
//!
//! An `AssistantResponseHook` resets state between LLM responses so
//! the pending buffer doesn't leak text from one response into the
//! next.
//!
//! Registered by lash-cli's bootstrap only when the session runs in
//! `ExecutionMode::Rlm`.

use std::sync::{Arc, Mutex};

use lash::plugin::{
    AssistantStreamHookContext, AssistantStreamTransform, PluginError, PluginFactory,
    PluginSessionContext, SessionPlugin,
};
use lash::{ExecutionMode, PluginRegistrar, PluginSurfaceEvent};

const PLUGIN_ID: &str = "rlm_stream_mask";

pub struct RlmStreamMaskPluginFactory;

impl PluginFactory for RlmStreamMaskPluginFactory {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmStreamMaskPlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Rlm),
        }))
    }
}

struct RlmStreamMaskPlugin {
    active: bool,
}

impl SessionPlugin for RlmStreamMaskPlugin {
    fn id(&self) -> &'static str {
        PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }

        let state = Arc::new(Mutex::new(FenceDetector::new()));

        let stream_state = Arc::clone(&state);
        reg.output()
            .stream(Arc::new(move |ctx: AssistantStreamHookContext| {
                let state = Arc::clone(&stream_state);
                Box::pin(async move {
                    let mut detector = state.lock().expect("fence detector lock");
                    Ok(detector.process_chunk(&ctx.chunk))
                })
            }));

        let response_state = Arc::clone(&state);
        reg.output().response(Arc::new(
            move |ctx: lash::plugin::AssistantResponseHookContext| {
                let state = Arc::clone(&response_state);
                Box::pin(async move {
                    state.lock().expect("fence detector lock").reset();
                    Ok(lash::plugin::AssistantResponseTransform {
                        response: ctx.response,
                        events: Vec::new(),
                    })
                })
            },
        ));

        Ok(())
    }
}

/// Tracks whether the accumulated stream has entered a ` ```lashlang `
/// or ` ```rlm ` fenced block. Accumulates a pending buffer across
/// streaming chunks to detect fence openers that are split across
/// token boundaries (e.g. `` ``` `` → `lash` → `lang` → `\n`).
/// The `AssistantResponseHook` calls `reset()` between LLM responses
/// to prevent cross-response leakage.
struct FenceDetector {
    pending: String,
    inside_fence: bool,
    emitted_start: bool,
}

impl FenceDetector {
    fn new() -> Self {
        Self {
            pending: String::new(),
            inside_fence: false,
            emitted_start: false,
        }
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.inside_fence = false;
        self.emitted_start = false;
    }

    fn process_chunk(&mut self, chunk: &str) -> AssistantStreamTransform {
        if self.inside_fence {
            return AssistantStreamTransform {
                chunk: String::new(),
                events: Vec::new(),
            };
        }

        self.pending.push_str(chunk);

        if let Some(fence_start) = find_fence_opener(&self.pending) {
            self.inside_fence = true;
            let prose_before = self.pending[..fence_start].to_string();
            self.pending.clear();

            let mut events = Vec::new();
            if !self.emitted_start {
                self.emitted_start = true;
                events.push(PluginSurfaceEvent::Custom {
                    name: "rlm_fence_start".to_string(),
                    payload: serde_json::json!({}),
                });
            }

            return AssistantStreamTransform {
                chunk: prose_before,
                events,
            };
        }

        // Flush everything up to the last newline. The remainder
        // (current incomplete line) is held as pending because it
        // could be the start of a fence opener split across chunks.
        let safe_len = match self.pending.rfind('\n') {
            Some(pos) => pos + 1,
            None => {
                if self.pending.len() <= 16 {
                    0
                } else {
                    // Walk backwards from the target to find a char
                    // boundary so we don't slice mid-multibyte.
                    let mut pos = self.pending.len() - 16;
                    while pos > 0 && !self.pending.is_char_boundary(pos) {
                        pos -= 1;
                    }
                    pos
                }
            }
        };

        if safe_len == 0 {
            return AssistantStreamTransform {
                chunk: String::new(),
                events: Vec::new(),
            };
        }

        let flushed = self.pending[..safe_len].to_string();
        self.pending = self.pending[safe_len..].to_string();
        AssistantStreamTransform {
            chunk: flushed,
            events: Vec::new(),
        }
    }
}

fn find_fence_opener(text: &str) -> Option<usize> {
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("```") {
        let pos = search_from + rel;
        let preceded_by_newline = pos == 0 || text.as_bytes().get(pos - 1).copied() == Some(b'\n');
        if !preceded_by_newline {
            search_from = pos + 3;
            continue;
        }
        let after = &text[pos + 3..];
        let line_end = after.find('\n').unwrap_or(after.len());
        let lang = after[..line_end].trim();
        if matches!(lang, "lashlang" | "rlm") {
            return Some(pos);
        }
        search_from = pos + 3;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_streams_through_before_fence() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Hello, here's my plan.\n\n");
        assert_eq!(t.chunk, "Hello, here's my plan.\n\n");
        assert!(t.events.is_empty());
    }

    #[test]
    fn fence_in_single_chunk() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Thinking...\n\n```lashlang\ncode\n```\n");
        assert_eq!(t.chunk, "Thinking...\n\n");
        assert_eq!(t.events.len(), 1);
        assert!(matches!(
            &t.events[0],
            PluginSurfaceEvent::Custom { name, .. } if name == "rlm_fence_start"
        ));
    }

    #[test]
    fn fence_split_across_chunks() {
        let mut d = FenceDetector::new();
        assert_eq!(d.process_chunk("Plan.\n\n").chunk, "Plan.\n\n");
        // ``` arrives alone — held as pending
        assert_eq!(d.process_chunk("```").chunk, "");
        // language tag arrives — now the opener is complete
        let t = d.process_chunk("lashlang\n");
        assert!(d.inside_fence);
        assert_eq!(t.events.len(), 1);
        // code body suppressed
        assert_eq!(d.process_chunk("code\n```\n").chunk, "");
    }

    #[test]
    fn token_by_token_streaming() {
        let mut d = FenceDetector::new();
        for tok in ["Let", " me", " check", ".\n\n"] {
            let t = d.process_chunk(tok);
            assert!(t.events.is_empty());
        }
        d.process_chunk("```");
        d.process_chunk("lash");
        // Fence detected as soon as the language tag completes (no need
        // to wait for the trailing newline — find_fence_opener accepts
        // end-of-buffer as the line boundary).
        let t = d.process_chunk("lang");
        assert!(d.inside_fence);
        assert_eq!(t.events.len(), 1);
        // Everything after is suppressed.
        assert_eq!(d.process_chunk("\n").chunk, "");
        assert_eq!(
            d.process_chunk("result = call exec { cmd: \"date\" }\n")
                .chunk,
            ""
        );
    }

    #[test]
    fn non_lashlang_fence_streams_through() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Example:\n\n```python\nprint('hi')\n```\n");
        assert!(!t.chunk.is_empty());
        assert!(t.events.is_empty());
    }

    #[test]
    fn rlm_alias_triggers_masking() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Check:\n\n```rlm\nobserve x\n```\n");
        assert_eq!(t.chunk, "Check:\n\n");
        assert_eq!(t.events.len(), 1);
    }

    #[test]
    fn inline_backticks_do_not_trigger() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Use ```lashlang in your code.\n");
        assert!(t.chunk.contains("lashlang"));
        assert!(t.events.is_empty());
    }

    #[test]
    fn reset_prevents_cross_response_leak() {
        let mut d = FenceDetector::new();
        d.process_chunk("Hi! How can I help you?");
        // Response hook fires between responses.
        d.reset();

        let t = d.process_chunk("New response.\n\n```lashlang\ncode\n```\n");
        assert!(t.chunk.starts_with("New response."));
        assert!(!t.chunk.contains("How can I help"));
    }

    #[test]
    fn reset_clears_fence_state() {
        let mut d = FenceDetector::new();
        d.process_chunk("Plan.\n\n```lashlang\ncode\n```\n");
        assert!(d.inside_fence);

        d.reset();
        assert!(!d.inside_fence);
        assert!(d.pending.is_empty());

        let t = d.process_chunk("Result.\n");
        assert_eq!(t.chunk, "Result.\n");
    }
}
