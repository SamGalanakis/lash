use super::*;

pub(in crate::runtime) struct PendingPrompt {
    pub(in crate::runtime) request: crate::PromptRequest,
    pub(in crate::runtime) response_tx: std::sync::mpsc::Sender<crate::PromptResponse>,
}

#[derive(Clone, Default)]
pub(in crate::runtime) struct HostPromptBridge {
    sender: Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedSender<PendingPrompt>>>>,
}

impl HostPromptBridge {
    pub(in crate::runtime) fn new() -> Self {
        Self::default()
    }

    pub(in crate::runtime) fn set_sender(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<PendingPrompt>,
    ) {
        *self.sender.lock().expect("prompt bridge poisoned") = Some(tx);
    }

    pub(in crate::runtime) fn clear_sender(&self) {
        *self.sender.lock().expect("prompt bridge poisoned") = None;
    }

    pub(in crate::runtime::session_manager) async fn prompt(
        &self,
        request: crate::PromptRequest,
    ) -> Result<crate::PromptResponse, crate::PluginError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| crate::PluginError::Session("prompt bridge poisoned".to_string()))?
            .clone()
            .ok_or_else(|| {
                crate::PluginError::Session("user prompts are unavailable in this session".into())
            })?;
        let (response_tx, response_rx) = std::sync::mpsc::channel::<crate::PromptResponse>();
        sender
            .send(PendingPrompt {
                request,
                response_tx,
            })
            .map_err(|_| crate::PluginError::Session("prompt channel closed".to_string()))?;
        tokio::task::spawn_blocking(move || response_rx.recv())
            .await
            .map_err(|err| crate::PluginError::Session(format!("prompt task failed: {err}")))?
            .map_err(|_| crate::PluginError::Session("prompt response channel closed".to_string()))
    }
}
