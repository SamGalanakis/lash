use super::support::*;

#[derive(Debug)]
pub struct ProviderRateLimiter {
    state: Mutex<ProviderRateLimiterState>,
}

#[derive(Debug)]
struct ProviderRateLimiterState {
    policy: ProviderRateLimitPolicy,
    semaphore: Option<Arc<tokio::sync::Semaphore>>,
    request_bucket: WindowBucket,
    token_bucket: WindowBucket,
}

#[derive(Clone, Debug)]
struct WindowBucket {
    used: u32,
    reset_at: Instant,
}

impl Default for WindowBucket {
    fn default() -> Self {
        Self {
            used: 0,
            reset_at: Instant::now(),
        }
    }
}

#[derive(Debug)]
pub struct ProviderRateLimitPermit {
    _concurrency: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl ProviderRateLimiter {
    pub fn new(policy: ProviderRateLimitPolicy) -> Self {
        let semaphore = policy
            .max_concurrency
            .filter(|limit| *limit > 0)
            .map(tokio::sync::Semaphore::new)
            .map(Arc::new);
        Self {
            state: Mutex::new(ProviderRateLimiterState {
                policy,
                semaphore,
                request_bucket: WindowBucket::default(),
                token_bucket: WindowBucket::default(),
            }),
        }
    }

    pub fn configure(&self, policy: ProviderRateLimitPolicy) {
        let mut state = self.state.lock().expect("provider rate limiter lock");
        if state.policy.max_concurrency != policy.max_concurrency {
            state.semaphore = policy
                .max_concurrency
                .filter(|limit| *limit > 0)
                .map(tokio::sync::Semaphore::new)
                .map(Arc::new);
        }
        state.policy = policy;
    }

    pub async fn admit(&self, request: &LlmRequest) -> ProviderRateLimitPermit {
        let semaphore = self
            .state
            .lock()
            .expect("provider rate limiter lock")
            .semaphore
            .clone();
        let concurrency = match semaphore {
            Some(semaphore) => Some(semaphore.acquire_owned().await.expect("semaphore open")),
            None => None,
        };
        self.wait_for_buckets(1, estimate_request_tokens(request))
            .await;
        ProviderRateLimitPermit {
            _concurrency: concurrency,
        }
    }

    async fn wait_for_buckets(&self, requests: u32, tokens: u32) {
        loop {
            let wait = {
                let mut state = self.state.lock().expect("provider rate limiter lock");
                let now = Instant::now();
                let policy = state.policy.clone();
                let request_wait = bucket_wait(
                    &mut state.request_bucket,
                    now,
                    policy.requests_per_window,
                    policy.request_window_ms,
                    requests,
                );
                let token_wait = bucket_wait(
                    &mut state.token_bucket,
                    now,
                    policy.tokens_per_window,
                    policy.token_window_ms,
                    tokens,
                );
                match (request_wait, token_wait) {
                    (None, None) => return,
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) | (None, Some(a)) => Some(a),
                }
            };
            if let Some(wait) = wait {
                tokio::time::sleep(wait).await;
            }
        }
    }
}

fn bucket_wait(
    bucket: &mut WindowBucket,
    now: Instant,
    limit: Option<u32>,
    window_ms: Option<u64>,
    cost: u32,
) -> Option<Duration> {
    let limit = limit.filter(|limit| *limit > 0)?;
    let window = Duration::from_millis(window_ms.unwrap_or(60_000).max(1));
    if now >= bucket.reset_at {
        bucket.used = 0;
        bucket.reset_at = now + window;
    }
    if bucket.used.saturating_add(cost.min(limit)) <= limit {
        bucket.used = bucket.used.saturating_add(cost.min(limit));
        None
    } else {
        Some(bucket.reset_at.saturating_duration_since(now))
    }
}

fn estimate_request_tokens(request: &LlmRequest) -> u32 {
    let mut chars = request.model.len();
    for message in &request.messages {
        for block in message.blocks.iter() {
            match block {
                LlmContentBlock::Text { text, .. } => chars += text.len(),
                LlmContentBlock::ToolCall { input_json, .. } => chars += input_json.len(),
                LlmContentBlock::ToolResult { content, .. } => chars += content.len(),
                LlmContentBlock::Reasoning { text, .. } => chars += text.len(),
                LlmContentBlock::Image { .. } => chars += 256,
            }
        }
    }
    chars = chars.saturating_add(request.attachments.iter().map(|a| a.data.len() / 4).sum());
    ((chars / 4).max(1)).try_into().unwrap_or(u32::MAX)
}
