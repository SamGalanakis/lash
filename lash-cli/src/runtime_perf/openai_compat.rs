use std::io;
use std::sync::Arc;

use anyhow::Context;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

use super::providers::BenchmarkStreamProfile;

pub(crate) struct OpenAiCompatBenchServer {
    pub(crate) base_url: String,
    shutdown: CancellationToken,
    accept_task: tokio::task::JoinHandle<()>,
}

impl OpenAiCompatBenchServer {
    pub(crate) async fn start(profile: BenchmarkStreamProfile) -> anyhow::Result<Self> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .context("bind local OpenAI-compatible perf server")?;
        let address = listener
            .local_addr()
            .context("resolve local OpenAI-compatible perf server address")?;
        let response_body = Arc::new(openai_compat_sse_body(&profile));
        let shutdown = CancellationToken::new();
        let accept_shutdown = shutdown.clone();
        let accept_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = accept_shutdown.cancelled() => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break; };
                        let body = Arc::clone(&response_body);
                        tokio::spawn(async move {
                            let _ = serve_openai_compat_connection(stream, body).await;
                        });
                    }
                }
            }
        });
        Ok(Self {
            base_url: format!("http://{address}/v1"),
            shutdown,
            accept_task,
        })
    }
}

impl Drop for OpenAiCompatBenchServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.accept_task.abort();
    }
}

pub(crate) fn openai_compat_sse_body(profile: &BenchmarkStreamProfile) -> Vec<u8> {
    let mut body = String::new();
    for delta in &profile.deltas {
        body.push_str("data: ");
        body.push_str(
            &serde_json::json!({
                "id": "chatcmpl-runtime-perf",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": delta,
                    },
                    "finish_reason": null,
                }],
            })
            .to_string(),
        );
        body.push_str("\n\n");
    }
    body.push_str("data: ");
    body.push_str(
        &serde_json::json!({
            "id": "chatcmpl-runtime-perf",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })
        .to_string(),
    );
    body.push_str("\n\n");
    body.push_str("data: ");
    body.push_str(
        &serde_json::json!({
            "id": "chatcmpl-runtime-perf",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 1024,
                "completion_tokens": 64,
                "prompt_tokens_details": {
                    "cached_tokens": 512,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 48,
                }
            },
        })
        .to_string(),
    );
    body.push_str("\n\n");
    body.push_str("data: [DONE]\n\n");
    body.into_bytes()
}

async fn serve_openai_compat_connection(
    mut stream: tokio::net::TcpStream,
    response_body: Arc<Vec<u8>>,
) -> io::Result<()> {
    drain_http_request(&mut stream).await?;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        response_body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(response_body.as_slice()).await?;
    stream.shutdown().await
}

async fn drain_http_request(stream: &mut tokio::net::TcpStream) -> io::Result<()> {
    let mut headers = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut content_length = None;
    let mut header_len = None;
    let mut total_read = 0usize;
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(());
        }
        total_read += read;
        if header_len.is_none() {
            headers.extend_from_slice(&chunk[..read]);
        }
        if header_len.is_none()
            && let Some(index) = find_bytes(&headers, b"\r\n\r\n")
        {
            let end = index + 4;
            let header_text = String::from_utf8_lossy(&headers);
            header_len = Some(end);
            content_length = Some(parse_content_length(&header_text).unwrap_or(0));
            headers.clear();
        }
        if let (Some(header_len), Some(content_length)) = (header_len, content_length)
            && total_read >= header_len + content_length
        {
            return Ok(());
        }
    }
}

fn parse_content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim().eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
