impl From<lash_core::TokenLedgerEntry> for RemoteTokenLedgerEntry {
    fn from(value: lash_core::TokenLedgerEntry) -> Self {
        let lash_core::TokenLedgerEntry {
            source,
            model,
            usage,
        } = value;
        Self {
            source,
            model,
            usage: usage.into(),
        }
    }
}

impl From<RemoteTokenLedgerEntry> for lash_core::TokenLedgerEntry {
    fn from(value: RemoteTokenLedgerEntry) -> Self {
        let RemoteTokenLedgerEntry {
            source,
            model,
            usage,
        } = value;
        Self {
            source,
            model,
            usage: usage.into(),
        }
    }
}

impl RemoteTurnResult {
    pub fn from_core(
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        turn: lash_core::AssembledTurn,
        activities: impl IntoIterator<Item = RemoteTurnActivity>,
    ) -> Self {
        // `state` is the local session snapshot; it never crosses the wire.
        let lash_core::AssembledTurn {
            state: _,
            outcome,
            assistant_output,
            execution,
            token_usage,
            children_usage,
            // ADR 0032 aggregation is local-only in this step. The remote
            // protocol mirror is intentionally deferred.
            llm_calls: _,
            tool_calls,
            errors,
        } = turn;
        let parent = RemoteUsage::from(token_usage);
        let children = children_usage
            .into_iter()
            .map(RemoteTokenLedgerEntry::from)
            .collect::<Vec<_>>();
        let mut total = parent.clone();
        for child in &children {
            total.add(&child.usage);
        }
        let outcome = RemoteTurnOutcome::from(outcome);
        let status = RemoteTurnStatus::from(&outcome);
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            status,
            outcome,
            assistant_output: assistant_output.into(),
            usage: RemoteTurnUsageSummary {
                parent,
                children,
                total,
            },
            execution: execution.into(),
            tool_calls: tool_calls.into_iter().map(Into::into).collect(),
            issues: errors.into_iter().map(Into::into).collect(),
            activities: activities.into_iter().collect(),
            metadata: HashMap::new(),
        }
    }
}

impl From<&RemoteTurnOutcome> for RemoteTurnStatus {
    fn from(value: &RemoteTurnOutcome) -> Self {
        match value {
            RemoteTurnOutcome::Finished { .. } | RemoteTurnOutcome::AgentFrameSwitch { .. } => {
                Self::Completed
            }
            RemoteTurnOutcome::Stopped {
                stop: RemoteTurnStop::Cancelled,
            } => Self::Cancelled,
            RemoteTurnOutcome::Stopped { .. } => Self::Failed,
        }
    }
}

impl From<lash_core::TurnOutcome> for RemoteTurnOutcome {
    fn from(value: lash_core::TurnOutcome) -> Self {
        match value {
            lash_core::TurnOutcome::Finished(finish) => Self::Finished {
                finish: finish.into(),
            },
            lash_core::TurnOutcome::AgentFrameSwitch { frame_id, task } => {
                Self::AgentFrameSwitch { frame_id, task }
            }
            lash_core::TurnOutcome::Stopped(stop) => Self::Stopped { stop: stop.into() },
        }
    }
}

impl From<lash_core::TurnFinish> for RemoteTurnFinish {
    fn from(value: lash_core::TurnFinish) -> Self {
        match value {
            lash_core::TurnFinish::AssistantMessage { text } => Self::AssistantMessage { text },
            lash_core::TurnFinish::FinalValue { value } => Self::FinalValue { value },
            lash_core::TurnFinish::ToolValue { tool_name, value } => {
                Self::ToolValue { tool_name, value }
            }
        }
    }
}

impl From<lash_core::TurnStop> for RemoteTurnStop {
    fn from(value: lash_core::TurnStop) -> Self {
        match value {
            lash_core::TurnStop::Cancelled => Self::Cancelled,
            lash_core::TurnStop::Incomplete => Self::Incomplete,
            lash_core::TurnStop::InvalidInput => Self::InvalidInput,
            lash_core::TurnStop::MaxTurns => Self::MaxTurns,
            lash_core::TurnStop::ToolFailure => Self::ToolFailure,
            lash_core::TurnStop::ProviderError => Self::ProviderError,
            lash_core::TurnStop::PluginAbort => Self::PluginAbort,
            lash_core::TurnStop::RuntimeError => Self::RuntimeError,
            lash_core::TurnStop::SubmittedError { value } => Self::SubmittedError { value },
            lash_core::TurnStop::ToolError { tool_name, value } => {
                Self::ToolError { tool_name, value }
            }
        }
    }
}

impl From<lash_core::AssistantOutput> for RemoteAssistantOutput {
    fn from(value: lash_core::AssistantOutput) -> Self {
        let lash_core::AssistantOutput {
            safe_text,
            raw_text,
            state,
        } = value;
        Self {
            safe_text,
            raw_text,
            state: state.into(),
        }
    }
}

impl From<lash_core::OutputState> for RemoteAssistantOutputState {
    fn from(value: lash_core::OutputState) -> Self {
        match value {
            lash_core::OutputState::Usable => Self::Usable,
            lash_core::OutputState::EmptyOutput => Self::EmptyOutput,
            lash_core::OutputState::TracebackOnly => Self::TracebackOnly,
            lash_core::OutputState::RecoveredFromError => Self::RecoveredFromError,
        }
    }
}

impl From<lash_core::ExecutionSummary> for RemoteExecutionSummary {
    fn from(value: lash_core::ExecutionSummary) -> Self {
        let lash_core::ExecutionSummary {
            had_tool_calls,
            had_code_execution,
            started_at_ms,
            duration_ms,
        } = value;
        Self {
            had_tool_calls,
            had_code_execution,
            started_at_ms,
            duration_ms,
        }
    }
}

impl From<lash_core::ToolCallRecord> for RemoteToolCallSummary {
    fn from(value: lash_core::ToolCallRecord) -> Self {
        let lash_core::ToolCallRecord {
            call_id,
            tool,
            args,
            output,
            duration_ms,
        } = value;
        Self {
            call_id,
            tool_name: tool,
            args,
            outcome: output.into(),
            duration_ms,
        }
    }
}

impl From<lash_core::ToolCallOutput> for RemoteToolCallOutcome {
    fn from(value: lash_core::ToolCallOutput) -> Self {
        // `control` is a local turn-control signal and never crosses the wire.
        let lash_core::ToolCallOutput {
            outcome,
            control: _,
        } = value;
        match outcome {
            lash_core::ToolCallOutcome::Success(value) => Self::Success(value.to_json_value()),
            lash_core::ToolCallOutcome::Failure(value) => Self::Failure(value.to_json_value()),
            lash_core::ToolCallOutcome::Cancelled(value) => Self::Cancelled(value.to_json_value()),
        }
    }
}

impl From<lash_core::TurnIssue> for RemoteTurnIssue {
    fn from(value: lash_core::TurnIssue) -> Self {
        let lash_core::TurnIssue {
            kind,
            code,
            terminal_reason,
            message,
            raw,
            retryable,
            provider_failure_kind,
        } = value;
        Self {
            kind,
            code,
            terminal_reason: terminal_reason.map(Into::into),
            message,
            raw,
            retryable,
            provider_failure_kind: provider_failure_kind.map(Into::into),
        }
    }
}
