//! RLM stream mask: hooks that suppress the `lashlang` fence body as it
//! streams in and raise `AssistantStreamTransform.abort_stream` the
//! moment the fence closes, short-circuiting the LLM call.
//!
//! Registered from `RlmModePlugin::register` via
//! [`register_stream_mask`].

use std::sync::{Arc, Mutex};

use lash_core::PluginRuntimeEvent;
use lash_core::plugin::{
    AssistantStreamHookContext, AssistantStreamTransform, PluginError, PluginRegistrar,
};

/// Install the stream-mask / fence-close-abort hooks on the given
/// registrar. Called by [`crate::plugin::RlmModePlugin::register`] when
/// the session is active.
pub fn register_stream_mask(reg: &mut PluginRegistrar) -> Result<(), PluginError> {
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
        move |ctx: lash_core::plugin::AssistantResponseHookContext| {
            let state = Arc::clone(&response_state);
            Box::pin(async move {
                let mut response = ctx.response;
                // The stream hook suppresses the fence body from every
                // delta chunk so the UI doesn't see the raw code scroll
                // past. But the driver then inspects `response.full_text`
                // to decide whether to execute a block — and a suppressed
                // body leaves that text fence-free, so the driver would
                // fall into the "no fence → finish turn" branch. Before
                // we reset the detector for the next response, splice the
                // captured fence body back into the response so the
                // driver sees a complete ` ```lashlang … ``` ` block.
                {
                    let mut detector = state.lock().expect("fence detector lock");
                    if detector.inside_fence {
                        let mut spliced = response.full_text.clone();
                        if !spliced.is_empty() && !spliced.ends_with('\n') {
                            spliced.push('\n');
                        }
                        spliced.push_str("```lashlang\n");
                        spliced.push_str(&detector.fence_body);
                        if !detector.fence_closed {
                            // Stream aborted or ended before we saw a
                            // closing fence — append one ourselves so
                            // the driver can still parse and execute.
                            if !spliced.ends_with('\n') {
                                spliced.push('\n');
                            }
                            spliced.push_str("```");
                        }
                        response.full_text = spliced.clone();
                        // The RLM driver reads `response.parts` (not
                        // `full_text`) when it builds the assistant_text
                        // it parses for a fence. Mirror the spliced
                        // content into parts so a fenced-only reply
                        // (no prose lead-in) still carries a Text part;
                        // otherwise the driver trips its
                        // "Model returned no assistant text" guard on
                        // turn 2+ responses that are just a fence.
                        let needs_text_part = !response
                            .parts
                            .iter()
                            .any(|part| matches!(part, lash_core::LlmOutputPart::Text { .. }));
                        if needs_text_part {
                            response.parts.push(lash_core::LlmOutputPart::Text {
                                text: spliced,
                                response_meta: None,
                            });
                        }
                    }
                    detector.reset();
                }
                Ok(lash_core::plugin::AssistantResponseTransform {
                    response,
                    events: Vec::new(),
                })
            })
        },
    ));

    Ok(())
}

/// Tracks whether the accumulated stream has entered a ` ```lashlang `
/// fenced block. Only the `lashlang` opener is canonical — other
/// labels (e.g. `rlm`, `lash`) stream through as plain prose so the
/// prompt contract ("write one ` ```lashlang ` block") is unambiguous.
/// Accumulates a pending buffer across streaming chunks to detect
/// openers that are split across
/// token boundaries (e.g. `` ``` `` → `lash` → `lang` → `\n`).
/// The `AssistantResponseHook` calls `reset()` between LLM responses
/// to prevent cross-response leakage.
struct FenceDetector {
    pending: String,
    inside_fence: bool,
    emitted_start: bool,
    /// Number of backticks in the active opener (≥3). The closer must
    /// be at least this long. Always 0 when not `inside_fence`.
    opener_len: usize,
    /// Accumulated fence body (everything after the opener), used to
    /// detect a closing fence so we can raise `abort_stream` and end
    /// the LLM call the moment the model finishes its one block.
    fence_body: String,
    fence_closed: bool,
}

impl FenceDetector {
    fn new() -> Self {
        Self {
            pending: String::new(),
            inside_fence: false,
            emitted_start: false,
            opener_len: 0,
            fence_body: String::new(),
            fence_closed: false,
        }
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.inside_fence = false;
        self.emitted_start = false;
        self.opener_len = 0;
        self.fence_body.clear();
        self.fence_closed = false;
    }

    fn process_chunk(&mut self, chunk: &str) -> AssistantStreamTransform {
        if self.inside_fence {
            if self.fence_closed {
                // Already raised abort_stream — just keep swallowing.
                return AssistantStreamTransform {
                    chunk: String::new(),
                    reasoning_deltas: Vec::new(),
                    events: Vec::new(),
                    abort_stream: false,
                };
            }
            self.fence_body.push_str(chunk);
            if has_closing_fence(&self.fence_body, self.opener_len) {
                self.fence_closed = true;
                // Signal the runtime: stop the LLM stream, we have the
                // full block.
                return AssistantStreamTransform {
                    chunk: String::new(),
                    reasoning_deltas: Vec::new(),
                    events: Vec::new(),
                    abort_stream: true,
                };
            }
            return AssistantStreamTransform {
                chunk: String::new(),
                reasoning_deltas: Vec::new(),
                events: Vec::new(),
                abort_stream: false,
            };
        }

        self.pending.push_str(chunk);

        if let Some((fence_start, body_start, opener_len)) = find_fence_opener(&self.pending) {
            self.inside_fence = true;
            self.opener_len = opener_len;
            let prose_before = self.pending[..fence_start].to_string();
            // Preserve any body content that arrived in the same chunk as
            // the opener — `pending.clear()` would otherwise drop it.
            let initial_body = self.pending[body_start..].to_string();
            self.pending.clear();

            let mut events = Vec::new();
            if !self.emitted_start {
                self.emitted_start = true;
                events.push(PluginRuntimeEvent::Custom {
                    name: "rlm_fence_start".to_string(),
                    payload: serde_json::json!({}),
                });
            }

            if !initial_body.is_empty() {
                self.fence_body.push_str(&initial_body);
                if has_closing_fence(&self.fence_body, self.opener_len) {
                    self.fence_closed = true;
                    return AssistantStreamTransform {
                        chunk: String::new(),
                        reasoning_deltas: non_empty_reasoning_delta(prose_before),
                        events,
                        abort_stream: true,
                    };
                }
            }

            return AssistantStreamTransform {
                chunk: String::new(),
                reasoning_deltas: non_empty_reasoning_delta(prose_before),
                events,
                abort_stream: false,
            };
        }

        // Flush everything except a suffix that could still become a
        // split ```lashlang opener. This keeps prose-only final replies
        // from holding an arbitrary tail while still preserving cases
        // like "`" -> "``" -> "```lash" -> "lang".
        let safe_len = self.pending.len() - possible_fence_opener_suffix_len(&self.pending);

        if safe_len == 0 {
            return AssistantStreamTransform {
                chunk: String::new(),
                reasoning_deltas: Vec::new(),
                events: Vec::new(),
                abort_stream: false,
            };
        }

        let flushed = self.pending[..safe_len].to_string();
        self.pending = self.pending[safe_len..].to_string();
        AssistantStreamTransform {
            chunk: String::new(),
            reasoning_deltas: non_empty_reasoning_delta(flushed),
            events: Vec::new(),
            abort_stream: false,
        }
    }
}

fn non_empty_reasoning_delta(text: String) -> Vec<String> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![text]
    }
}

/// Return `true` when `text` (the accumulated fence body) contains a
/// closing run of at least `opener_len` consecutive backticks. Must
/// stay in lockstep with `first_lashlang_fence_span` in `protocol.rs`:
/// CommonMark variable-length fences — a 4-backtick opener requires a
/// 4+-backtick closer.
fn has_closing_fence(text: &str, opener_len: usize) -> bool {
    if opener_len == 0 {
        return false;
    }
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i;
            while i < bytes.len() && bytes[i] == b'`' {
                i += 1;
            }
            if i - start >= opener_len {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Locate a complete ` ```lashlang ` opener in `text`. Returns
/// `(fence_start, body_start, opener_len)` where `fence_start` is the
/// offset of the opening backtick run, `body_start` is the first byte
/// after the language tag's terminating `\n` (or end-of-text if the
/// tag isn't newline-terminated yet — in which case the body is still
/// empty), and `opener_len` is the number of backticks in the opener
/// (≥3, used to validate the closer length).
fn find_fence_opener(text: &str) -> Option<(usize, usize, usize)> {
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("```") {
        let pos = search_from + rel;
        let opener_len = text.as_bytes()[pos..]
            .iter()
            .take_while(|&&b| b == b'`')
            .count();
        let after_ticks = pos + opener_len;
        let after = &text[after_ticks..];
        let line_end = after.find('\n').unwrap_or(after.len());
        let lang = after[..line_end].trim();
        if lang == "lashlang" {
            // Skip the trailing `\n` so body_start lands on the first
            // byte of the actual body. When the lang line isn't
            // terminated yet, line_end == after.len() and body_start
            // sits at end-of-text, which is correct (empty initial
            // body).
            let body_start = if line_end < after.len() {
                after_ticks + line_end + 1
            } else {
                after_ticks + line_end
            };
            return Some((pos, body_start, opener_len));
        }
        search_from = after_ticks;
    }
    None
}

fn possible_fence_opener_suffix_len(text: &str) -> usize {
    text.char_indices()
        .find_map(|(idx, _)| {
            let suffix = &text[idx..];
            suffix_can_be_fence_opener_prefix(suffix).then_some(suffix.len())
        })
        .unwrap_or(0)
}

fn suffix_can_be_fence_opener_prefix(suffix: &str) -> bool {
    if suffix.is_empty() {
        return false;
    }
    const LANG: &str = "lashlang";

    // Count leading backticks in the suffix. If everything in the
    // suffix is backticks, it could still grow into a longer opener
    // run — preserve it. Variable-length opener support means a run
    // of any length ≥1 is a potential prefix.
    let backtick_count = suffix.chars().take_while(|&c| c == '`').count();
    let after_ticks: String = suffix.chars().skip(backtick_count).collect();

    if after_ticks.is_empty() {
        return backtick_count > 0;
    }
    if backtick_count < 3 {
        // Not enough backticks AND there's text after — not an opener.
        return false;
    }
    let after_padding = after_ticks.trim_start_matches(' ');
    after_padding.is_empty() || LANG.starts_with(after_padding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_streams_as_reasoning_before_fence() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Hello, here's my plan.\n\n");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Hello, here's my plan.\n\n"]);
        assert!(t.events.is_empty());
    }

    #[test]
    fn short_prose_without_newline_streams_immediately() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Hi - what can I help with?");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Hi - what can I help with?"]);
        assert!(d.pending.is_empty());
        assert!(t.events.is_empty());
    }

    #[test]
    fn only_possible_fence_suffix_is_held() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Plan. ```la");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Plan. "]);
        assert_eq!(d.pending, "```la");

        let t = d.process_chunk("shlang\n");
        assert_eq!(t.chunk, "");
        assert!(t.reasoning_deltas.is_empty());
        assert!(d.inside_fence);
        assert_eq!(t.events.len(), 1);
    }

    #[test]
    fn non_lashlang_fence_flushes_after_it_stops_matching() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Example: ``");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Example: "]);
        assert_eq!(d.pending, "``");

        let t = d.process_chunk("`python\n");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["```python\n"]);
        assert!(!d.inside_fence);
        assert!(d.pending.is_empty());
    }

    #[test]
    fn fence_in_single_chunk() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Thinking...\n\n```lashlang\ncode\n```\n");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Thinking...\n\n"]);
        assert_eq!(t.events.len(), 1);
        assert!(matches!(
            &t.events[0],
            PluginRuntimeEvent::Custom { name, .. } if name == "rlm_fence_start"
        ));
    }

    #[test]
    fn fence_split_across_chunks() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Plan.\n\n");
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Plan.\n\n"]);
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
            d.process_chunk("result = await TOOL.default.exec({ cmd: \"date\" })\n")
                .chunk,
            ""
        );
    }

    #[test]
    fn non_lashlang_fence_streams_as_reasoning_without_masking() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Example:\n\n```python\nprint('hi')\n```\n");
        assert!(t.chunk.is_empty());
        assert!(!t.reasoning_deltas.is_empty());
        assert!(t.events.is_empty());
    }

    #[test]
    fn rlm_alias_does_not_trigger_masking() {
        // Only ` ```lashlang ` is the canonical fence. `rlm` / `lash`
        // aliases were dropped so the prompt and the parser agree on
        // exactly one opener.
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Check:\n\n```rlm\nprint x\n```\n");
        assert!(t.chunk.is_empty());
        assert!(t.reasoning_deltas.join("").contains("```rlm"));
        assert!(t.events.is_empty());
    }

    #[test]
    fn inline_backticks_do_not_trigger() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("Use ```lashlang in your code.\n");
        assert!(t.chunk.is_empty());
        assert!(t.reasoning_deltas.join("").contains("lashlang"));
        assert!(t.events.is_empty());
    }

    #[test]
    fn reset_prevents_cross_response_leak() {
        let mut d = FenceDetector::new();
        d.process_chunk("Hi! How can I help you?");
        // Response hook fires between responses.
        d.reset();

        let t = d.process_chunk("New response.\n\n```lashlang\ncode\n```\n");
        assert_eq!(t.chunk, "");
        let reasoning = t.reasoning_deltas.join("");
        assert!(reasoning.starts_with("New response."));
        assert!(!reasoning.contains("How can I help"));
    }

    #[test]
    fn fence_opener_and_body_in_same_chunk_preserves_body() {
        // Reproduces: a single chunk like "```lashlang\nnow = await ..."
        // used to drop the part after the opener, so the spliced body
        // started after the assignment target instead of preserving the
        // full receiver operation.
        let mut d = FenceDetector::new();
        d.process_chunk(
            "```lashlang\nnow = await TOOL.default.exec({ cmd: \"date\" })?\nprint now.output\n",
        );
        assert!(d.inside_fence);
        assert!(
            d.fence_body
                .starts_with("now = await TOOL.default.exec({ cmd: \"date\" })?")
        );
    }

    #[test]
    fn fence_opener_with_body_and_close_in_same_chunk_aborts() {
        let mut d = FenceDetector::new();
        let t = d.process_chunk("```lashlang\nsubmit \"hi\"\n```");
        assert!(d.inside_fence);
        assert!(d.fence_closed);
        assert!(t.abort_stream);
        assert!(d.fence_body.contains("submit \"hi\""));
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
        assert_eq!(t.chunk, "");
        assert_eq!(t.reasoning_deltas, vec!["Result.\n"]);
    }
}
