use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use lash::llm::types::{LlmContentBlock, LlmRequest, LlmRole, LlmStreamEvent, LlmUsage};
use lash::{LlmOutputPart, LlmResponse, ProviderHandle};
#[cfg(test)]
use lash_embed::TurnResult;
use lash_embed::{
    Input, LashCore, LashSession, TurnEvent, TurnEventSink, message_role, message_text,
};
use tokio::sync::Mutex;

type ChatResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    TextDelta { text: String },
    ReasoningDelta { text: String },
    ToolCall { name: String, success: bool },
    Message { kind: String, text: String },
    Error { message: String },
    Done,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ChatTurn {
    pub conversation_id: String,
    pub reply: String,
    pub events: Vec<ChatEvent>,
    pub messages: Vec<ChatMessage>,
}

pub struct MockChatService {
    core: LashCore,
    conversations: Mutex<HashMap<String, LashSession>>,
}

impl MockChatService {
    pub fn new() -> Self {
        let provider = mock_provider();
        let core = LashCore::standard()
            .provider(provider)
            .model("mock-chat-model")
            .max_context_tokens(200_000)
            .build()
            .expect("mock chat service core should be valid");
        Self {
            core,
            conversations: Mutex::new(HashMap::new()),
        }
    }

    pub async fn send_message(&self, conversation_id: &str, text: &str) -> ChatResult<ChatTurn> {
        let runtime = self.runtime_for(conversation_id).await?;
        let events = RecordingChatEvents::default();
        let turn = runtime
            .turn(Input::text(text))
            .events(&events)
            .run()
            .await?;
        let messages = turn.messages.iter().map(chat_message_from_lash).collect();

        Ok(ChatTurn {
            conversation_id: conversation_id.to_string(),
            reply: turn.final_text,
            events: events.snapshot().await,
            messages,
        })
    }

    pub async fn messages(&self, conversation_id: &str) -> Vec<ChatMessage> {
        let Some(runtime) = self
            .conversations
            .lock()
            .await
            .get(conversation_id)
            .cloned()
        else {
            return Vec::new();
        };
        runtime
            .read_view()
            .await
            .messages()
            .iter()
            .map(chat_message_from_lash)
            .collect()
    }

    async fn runtime_for(&self, conversation_id: &str) -> ChatResult<LashSession> {
        if let Some(runtime) = self
            .conversations
            .lock()
            .await
            .get(conversation_id)
            .cloned()
        {
            return Ok(runtime);
        }

        let runtime = self.build_runtime(conversation_id).await?;
        let mut conversations = self.conversations.lock().await;
        Ok(conversations
            .entry(conversation_id.to_string())
            .or_insert_with(|| runtime.clone())
            .clone())
    }

    async fn build_runtime(&self, conversation_id: &str) -> ChatResult<LashSession> {
        Ok(self.core.session(conversation_id).standard().open().await?)
    }
}

impl Default for MockChatService {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Default)]
struct RecordingChatEvents {
    events: Arc<Mutex<Vec<ChatEvent>>>,
}

impl RecordingChatEvents {
    async fn snapshot(&self) -> Vec<ChatEvent> {
        self.events.lock().await.clone()
    }
}

#[async_trait]
impl TurnEventSink for RecordingChatEvents {
    async fn emit(&self, event: TurnEvent) {
        let mapped = match event {
            TurnEvent::TextDelta { content } => Some(ChatEvent::TextDelta { text: content }),
            TurnEvent::ReasoningDelta { content } => {
                Some(ChatEvent::ReasoningDelta { text: content })
            }
            TurnEvent::ToolCall { name, success, .. } => {
                Some(ChatEvent::ToolCall { name, success })
            }
            TurnEvent::Message { text, kind } => Some(ChatEvent::Message { kind, text }),
            TurnEvent::Error { message } => Some(ChatEvent::Error { message }),
            TurnEvent::Done => Some(ChatEvent::Done),
            TurnEvent::Usage { .. } => None,
        };

        if let Some(event) = mapped {
            self.events.lock().await.push(event);
        }
    }
}

fn mock_provider() -> ProviderHandle {
    let counter = Arc::new(AtomicUsize::new(0));
    lash::testing::TestProvider::builder()
        .kind("mock-chat")
        .default_model("mock-chat-model")
        .requires_streaming(true)
        .complete(move |request| {
            let counter = Arc::clone(&counter);
            async move {
                let turn_number = counter.fetch_add(1, Ordering::SeqCst) + 1;
                let user_text = last_user_text(&request);
                let reply = format!("Mock reply {turn_number}: you said `{user_text}`.");
                stream_reply(&request, &reply);
                Ok(LlmResponse {
                    full_text: reply.clone(),
                    parts: vec![LlmOutputPart::Text {
                        text: reply,
                        response_meta: None,
                    }],
                    usage: LlmUsage {
                        input_tokens: user_text.split_whitespace().count() as i64,
                        output_tokens: 8,
                        cached_input_tokens: 0,
                        reasoning_tokens: 0,
                    },
                    ..LlmResponse::default()
                })
            }
        })
        .build()
        .into_handle()
}

fn stream_reply(request: &LlmRequest, reply: &str) {
    let Some(events) = request.stream_events.as_ref() else {
        return;
    };
    for chunk in reply.as_bytes().chunks(12) {
        events.send(LlmStreamEvent::Delta(
            String::from_utf8_lossy(chunk).to_string(),
        ));
    }
}

fn last_user_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == LlmRole::User)
        .map(|message| {
            message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "(empty)".to_string())
}

fn chat_message_from_lash(message: &lash_embed::Message) -> ChatMessage {
    ChatMessage {
        role: message_role(message).to_string(),
        text: message_text(message),
    }
}

#[cfg(test)]
fn assert_successful_turn(turn: &TurnResult) {
    assert!(
        turn.errors.is_empty(),
        "mock chat turn should not emit errors: {:?}",
        turn.errors
    );
}

#[tokio::main]
async fn main() -> ChatResult<()> {
    let service = MockChatService::new();
    let first = service.send_message("demo", "hello lash").await?;
    let second = service.send_message("demo", "what did I just say?").await?;

    println!("{}", serde_json::to_string_pretty(&first)?);
    println!("{}", serde_json::to_string_pretty(&second)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn backend_service_runs_two_turns_without_cli_http_or_database() -> ChatResult<()> {
        let service = MockChatService::new();

        let first = service.send_message("conversation-1", "hello").await?;
        assert!(first.reply.contains("hello"));
        assert!(
            first
                .events
                .iter()
                .any(|event| matches!(event, ChatEvent::TextDelta { .. }))
        );
        assert_eq!(first.messages.len(), 2);

        let second = service
            .send_message("conversation-1", "remember me")
            .await?;
        assert!(second.reply.contains("remember me"));
        assert_eq!(second.messages.len(), 4);

        let messages = service.messages("conversation-1").await;
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");

        Ok(())
    }

    #[tokio::test]
    async fn raw_runtime_turn_still_exposes_current_embedding_shape() -> ChatResult<()> {
        let service = MockChatService::new();
        let runtime = service.runtime_for("contract").await?;
        let events = RecordingChatEvents::default();

        let turn = runtime
            .turn(Input::text("contract test"))
            .events(&events)
            .run()
            .await?;

        assert_successful_turn(&turn);
        assert!(turn.final_text.contains("contract test"));
        assert!(events.snapshot().await.contains(&ChatEvent::Done));
        Ok(())
    }
}
