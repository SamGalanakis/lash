//! RLM stream mask: hooks that suppress the `lashlang` fence body as it
//! streams in and raise `AssistantStreamTransform.abort_stream` the
//! moment the fence closes, short-circuiting the LLM call.
//!
//! Registered from `RlmProtocolPlugin::register` via
//! [`register_stream_mask`].

use std::sync::{Arc, Mutex};

use lash_core::PluginRuntimeEvent;
use lash_core::plugin::{
    AssistantStreamHookContext, AssistantStreamTransform, PluginError, PluginRegistrar,
};

use crate::fence_scan::{FENCE_LANG, body_has_closing_fence, first_lashlang_fence_span};

/// Install the stream-mask / fence-close-abort hooks on the given
/// registrar. Called by [`crate::plugin::RlmProtocolPlugin::register`] when
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
                        let spliced = detector.splice_into(&response.full_text);
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

    /// Reconstruct a complete ` ```lashlang … ``` ` block by splicing
    /// the suppressed fence body back onto the visible text. The
    /// streaming hook hides the body from every delta chunk, so the
    /// driver — which re-parses the assistant text to decide whether to
    /// execute — would otherwise see fence-free prose and fall into the
    /// "no fence → finish turn" branch.
    ///
    /// The opener is rebuilt with exactly `opener_len` backticks (not a
    /// hardcoded three): when the model opens a 4-backtick fence so its
    /// body can contain a literal ` ``` `, downgrading the splice to
    /// three backticks made the embedded triple prematurely close the
    /// reconstructed block, truncating the executed code.
    fn splice_into(&self, visible: &str) -> String {
        debug_assert!(self.inside_fence);
        let ticks = "`".repeat(self.opener_len.max(3));
        let mut spliced = visible.to_string();
        if !spliced.is_empty() && !spliced.ends_with('\n') {
            spliced.push('\n');
        }
        spliced.push_str(&ticks);
        spliced.push_str(FENCE_LANG);
        spliced.push('\n');
        spliced.push_str(&self.fence_body);
        if !self.fence_closed {
            // Stream aborted or ended before we saw a closing fence —
            // append one ourselves (matching the opener length) so the
            // driver can still parse and execute.
            if !spliced.ends_with('\n') {
                spliced.push('\n');
            }
            spliced.push_str(&ticks);
        }
        spliced
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
            if body_has_closing_fence(&self.fence_body, self.opener_len) {
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
                if body_has_closing_fence(&self.fence_body, self.opener_len) {
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

/// Locate a complete ` ```lashlang ` opener in `text`. Returns
/// `(fence_start, body_start, opener_len)` where `fence_start` is the
/// offset of the opening backtick run, `body_start` is the first byte
/// after the language tag's terminating `\n` (or end-of-text if the
/// tag isn't newline-terminated yet — in which case the body is still
/// empty), and `opener_len` is the number of backticks in the opener
/// (≥3, used to validate the closer length).
///
/// Delegates to [`first_lashlang_fence_span`] so the streaming mask and
/// the finalize path recognize *exactly* the same openers — including
/// an opener that sits at end-of-text with no trailing newline.
fn find_fence_opener(text: &str) -> Option<(usize, usize, usize)> {
    let span = first_lashlang_fence_span(text)?;
    Some((span.open_start, span.body_start, span.opener_len))
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
    after_padding.is_empty() || FENCE_LANG.starts_with(after_padding)
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
            d.process_chunk("result = await tools.exec({ cmd: \"date\" })\n")
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
        // full module operation.
        let mut d = FenceDetector::new();
        d.process_chunk(
            "```lashlang\nnow = await tools.exec({ cmd: \"date\" })?\nprint now.output\n",
        );
        assert!(d.inside_fence);
        assert!(
            d.fence_body
                .starts_with("now = await tools.exec({ cmd: \"date\" })?")
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

    /// Drive a detector across `chunks` exactly as the stream hook
    /// would, then run the response-hook splice. Returns the
    /// reconstructed `full_text` the driver re-parses (the visible
    /// stream stays empty because the mask suppresses every chunk once
    /// inside the fence, so we accumulate the reasoning deltas as the
    /// visible lead-in).
    fn stream_and_splice(chunks: &[&str]) -> String {
        let mut d = FenceDetector::new();
        let mut visible = String::new();
        for chunk in chunks {
            let t = d.process_chunk(chunk);
            for delta in t.reasoning_deltas {
                visible.push_str(&delta);
            }
        }
        if d.inside_fence {
            d.splice_into(&visible)
        } else {
            visible
        }
    }

    #[test]
    fn splice_reconstructs_plain_triple_backtick_block() {
        // 3-backtick round-trip: the spliced text must re-parse to the
        // exact code the model wrote.
        let spliced = stream_and_splice(&["Quick check.\n\n```lashlang\nprint \"hi\"\n```\n"]);
        let span = first_lashlang_fence_span(&spliced).expect("spliced block parses");
        let code = spliced[span.body_start..span.body_end].trim_end_matches('\n');
        assert_eq!(code, "print \"hi\"");
        assert!(span.is_closed());
    }

    #[test]
    fn splice_preserves_four_backtick_fence_with_embedded_triple() {
        // Regression: a 4-backtick opener lets the body carry a literal
        // ``` run. The splice used to hardcode a 3-backtick opener, so
        // re-parsing closed the block at the embedded triple and
        // truncated the executed code to `print "`. The opener length
        // must round-trip.
        let spliced =
            stream_and_splice(&["````lashlang\n", "print \"```\"\n", "submit 1\n", "````"]);
        let span = first_lashlang_fence_span(&spliced).expect("spliced block parses");
        assert_eq!(span.opener_len, 4, "opener length must survive the splice");
        let code = spliced[span.body_start..span.body_end].trim_end_matches('\n');
        assert_eq!(
            code, "print \"```\"\nsubmit 1",
            "embedded triple-backticks must not prematurely close the block"
        );
    }

    #[test]
    fn stream_and_finalize_agree_on_opener_at_end_of_text() {
        // Before the scanner was unified the two paths disagreed on a
        // language tag that sits at end-of-text with no trailing
        // newline ("…```lashlang"): the streaming detector entered the
        // fence, but the finalize extractor's `body_start > text.len()`
        // guard returned None, so the same response was a block on one
        // path and prose on the other. Both must now agree.
        let opener_at_eof = "Plan.\n\n```lashlang";

        // Streaming path: detector enters the fence.
        let mut d = FenceDetector::new();
        d.process_chunk(opener_at_eof);
        assert!(
            d.inside_fence,
            "streaming path must detect the end-of-text opener"
        );

        // Finalize path: the shared scanner detects the same opener.
        assert!(
            first_lashlang_fence_span(opener_at_eof).is_some(),
            "finalize path must detect the same end-of-text opener"
        );

        // And the spliced reconstruction is itself a valid block the
        // driver can extract.
        let spliced = d.splice_into("Plan.");
        assert!(first_lashlang_fence_span(&spliced).is_some());
    }

    #[test]
    fn splice_closes_unterminated_fence_with_matching_opener_length() {
        // Stream ended mid-block with a 4-backtick opener: the synthetic
        // closer must also be 4 backticks, or the embedded triple would
        // be mistaken for the closer.
        let spliced = stream_and_splice(&["````lashlang\n", "print \"```\"\n", "submit 1\n"]);
        let span = first_lashlang_fence_span(&spliced).expect("spliced block parses");
        assert_eq!(span.opener_len, 4);
        assert!(
            span.is_closed(),
            "unterminated stream gets a synthetic closer"
        );
        let code = spliced[span.body_start..span.body_end].trim_end_matches('\n');
        assert_eq!(code, "print \"```\"\nsubmit 1");
    }
}
