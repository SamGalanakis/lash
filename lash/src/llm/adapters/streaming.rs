use tokio::sync::mpsc::UnboundedSender;

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmStreamEvent, LlmUsage};

use std::sync::OnceLock;
use std::time::Duration;

const DEFAULT_STREAM_CHUNK_TIMEOUT: Duration = Duration::from_secs(120);

pub fn stream_chunk_timeout() -> Duration {
    static CACHED: OnceLock<Duration> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LASH_LLM_STREAM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .filter(|d| !d.is_zero())
            .unwrap_or(DEFAULT_STREAM_CHUNK_TIMEOUT)
    })
}

pub struct SseBuffer {
    pending: String,
    event_lines: Vec<String>,
}

impl Default for SseBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl SseBuffer {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            event_lines: Vec::new(),
        }
    }

    pub fn push_chunk<F>(&mut self, chunk: &[u8], mut on_event: F) -> Result<(), LlmTransportError>
    where
        F: FnMut(String) -> Result<(), LlmTransportError>,
    {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        while let Some(pos) = self.pending.find('\n') {
            let mut line = self.pending[..pos].to_string();
            self.pending.drain(..=pos);
            if line.ends_with('\r') {
                line.pop();
            }
            self.handle_line(line, &mut on_event)?;
        }
        Ok(())
    }

    pub fn finish<F>(&mut self, mut on_event: F) -> Result<(), LlmTransportError>
    where
        F: FnMut(String) -> Result<(), LlmTransportError>,
    {
        if !self.pending.trim().is_empty()
            && let Some(data) = self.pending.trim().strip_prefix("data:")
        {
            self.event_lines.push(data.trim().to_string());
        }
        self.pending.clear();
        self.flush_event(&mut on_event)
    }

    fn handle_line<F>(&mut self, line: String, on_event: &mut F) -> Result<(), LlmTransportError>
    where
        F: FnMut(String) -> Result<(), LlmTransportError>,
    {
        if let Some(data) = line.strip_prefix("data:") {
            self.event_lines.push(data.trim().to_string());
            return Ok(());
        }
        if line.starts_with("event:") {
            return Ok(());
        }
        if line.trim().is_empty() {
            return self.flush_event(on_event);
        }
        Ok(())
    }

    fn flush_event<F>(&mut self, on_event: &mut F) -> Result<(), LlmTransportError>
    where
        F: FnMut(String) -> Result<(), LlmTransportError>,
    {
        if self.event_lines.is_empty() {
            return Ok(());
        }
        let raw = self.event_lines.join("\n");
        self.event_lines.clear();
        on_event(raw)
    }
}

pub async fn drive_sse_response<F>(
    mut resp: reqwest::Response,
    timeout: Duration,
    read_timeout_message: &str,
    mut on_event: F,
) -> Result<(), LlmTransportError>
where
    F: FnMut(String) -> Result<(), LlmTransportError>,
{
    let mut buffer = SseBuffer::new();
    loop {
        let chunk_opt = tokio::time::timeout(timeout, resp.chunk())
            .await
            .map_err(|_| LlmTransportError::new(read_timeout_message))?
            .map_err(|e| LlmTransportError::new(format!("Stream read failed: {e}")))?;
        let Some(chunk) = chunk_opt else { break };
        buffer.push_chunk(&chunk, &mut on_event)?;
    }
    buffer.finish(on_event)
}

pub fn emit_progress(
    tx: Option<&UnboundedSender<LlmStreamEvent>>,
    deltas: &[String],
    prev_len: usize,
    usage: &LlmUsage,
    prev_usage: &LlmUsage,
) {
    let Some(tx) = tx else {
        return;
    };
    for piece in deltas.iter().skip(prev_len) {
        let _ = tx.send(LlmStreamEvent::Delta(piece.clone()));
    }
    if usage != prev_usage && usage != &LlmUsage::default() {
        let _ = tx.send(LlmStreamEvent::Usage(usage.clone()));
    }
}
