#![allow(clippy::result_large_err)]

use lash_core::LlmTransportError;
use lash_sansio::llm::types::{LlmEventSender, LlmStreamEvent, LlmUsage};

use std::time::Duration;

struct SseBuffer {
    pending: String,
    event_data: String,
}

impl SseBuffer {
    fn new() -> Self {
        Self {
            pending: String::new(),
            event_data: String::new(),
        }
    }

    fn push_chunk<F>(&mut self, chunk: &[u8], mut on_event: F) -> Result<(), LlmTransportError>
    where
        F: FnMut(&str) -> Result<(), LlmTransportError>,
    {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.pending.find('\n') {
            let line_end = if pos > 0 && self.pending.as_bytes()[pos - 1] == b'\r' {
                pos - 1
            } else {
                pos
            };
            let should_flush = {
                let line = &self.pending[..line_end];
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim();
                    if !self.event_data.is_empty() {
                        self.event_data.push('\n');
                    }
                    self.event_data.push_str(data);
                    false
                } else if line.starts_with("event:") {
                    false
                } else {
                    line.trim().is_empty()
                }
            };
            self.pending.drain(..=pos);
            if should_flush {
                self.flush_event(&mut on_event)?;
            }
        }
        Ok(())
    }

    fn finish<F>(&mut self, mut on_event: F) -> Result<(), LlmTransportError>
    where
        F: FnMut(&str) -> Result<(), LlmTransportError>,
    {
        {
            let pending = self.pending.trim();
            if !pending.is_empty()
                && let Some(data) = pending.strip_prefix("data:")
            {
                let data = data.trim();
                if !self.event_data.is_empty() {
                    self.event_data.push('\n');
                }
                self.event_data.push_str(data);
            }
        }
        self.pending.clear();
        self.flush_event(&mut on_event)
    }

    fn flush_event<F>(&mut self, on_event: &mut F) -> Result<(), LlmTransportError>
    where
        F: FnMut(&str) -> Result<(), LlmTransportError>,
    {
        if self.event_data.is_empty() {
            return Ok(());
        }
        let result = on_event(self.event_data.as_str());
        self.event_data.clear();
        result
    }
}

pub async fn drive_sse_response<F>(
    mut resp: reqwest::Response,
    timeout: Duration,
    read_timeout_message: &str,
    mut on_event: F,
) -> Result<(), LlmTransportError>
where
    F: FnMut(&str) -> Result<(), LlmTransportError>,
{
    let mut buffer = SseBuffer::new();
    loop {
        let chunk_opt = tokio::time::timeout(timeout, resp.chunk())
            .await
            .map_err(|_| LlmTransportError::new(read_timeout_message).retryable(true))?
            .map_err(|e| {
                LlmTransportError::new(format!("Stream read failed: {e}"))
                    .retryable(e.is_timeout() || e.is_connect() || e.is_body() || e.is_decode())
            })?;
        let Some(chunk) = chunk_opt else { break };
        buffer.push_chunk(&chunk, &mut on_event)?;
    }
    buffer.finish(on_event)
}

pub fn emit_stream_progress(
    tx: Option<&LlmEventSender>,
    added_deltas: impl IntoIterator<Item = String>,
    usage: &LlmUsage,
    prev_usage: &LlmUsage,
) {
    let Some(tx) = tx else {
        return;
    };
    if usage != prev_usage && usage != &LlmUsage::default() {
        tx.send(LlmStreamEvent::Usage(usage.clone()));
    }
    for piece in added_deltas {
        tx.send(LlmStreamEvent::Delta(piece));
    }
}
