use std::path::PathBuf;
use std::sync::Arc;

use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};

use super::process::ProcessRegistry;
use super::{
    InlineRuntimeEffectController, RuntimeEffectController, SessionStoreFactory, TerminationPolicy,
};

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub prompt: crate::PromptLayer,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
    pub termination: TerminationPolicy,
    pub effect_controller: Arc<dyn RuntimeEffectController>,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            prompt: crate::PromptLayer::new(),
            trace_sink: None,
            trace_level: TraceLevel::Standard,
            trace_context: TraceContext::default(),
            termination: TerminationPolicy::default(),
            effect_controller: Arc::new(InlineRuntimeEffectController::default()),
        }
    }
}

impl RuntimeCoreConfig {
    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.attachment_store = attachment_store;
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.prompt.template = Some(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_trace_jsonl_path(mut self, trace_path: Option<PathBuf>) -> Self {
        self.trace_sink =
            trace_path.map(|path| Arc::new(JsonlTraceSink::new(path)) as Arc<dyn TraceSink>);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.trace_sink = sink;
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = termination;
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.effect_controller = effect_controller;
        self
    }
}

/// Base host shape for embedded runtimes.
#[derive(Clone)]
pub struct EmbeddedRuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
}

impl EmbeddedRuntimeHost {
    pub fn new(core: RuntimeCoreConfig) -> Self {
        Self {
            core,
            session_store_factory: None,
        }
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
        self
    }
}

/// Host shape for runtimes that support background plugin work.
#[derive(Clone)]
pub struct ProcessRuntimeHost {
    pub embedded: EmbeddedRuntimeHost,
    pub process_registry: Arc<dyn ProcessRegistry>,
}

impl ProcessRuntimeHost {
    pub fn new(embedded: EmbeddedRuntimeHost, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            embedded,
            process_registry,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            process_registry: None,
        }
    }
}

impl From<ProcessRuntimeHost> for RuntimeHost {
    fn from(value: ProcessRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            process_registry: Some(value.process_registry),
        }
    }
}
