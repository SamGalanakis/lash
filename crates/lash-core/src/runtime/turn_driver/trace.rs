use super::*;

impl RuntimeTurnDriver<'_> {
    pub(super) fn trace_context(&self, protocol_iteration: usize) -> lash_trace::TraceContext {
        lash_trace::TraceContext::default()
            .for_session(self.session_id.clone())
            .for_turn_index(self.turn_index)
            .for_protocol_iteration(protocol_iteration)
            .for_turn(self.turn_id.clone())
    }

    pub(super) fn llm_call_id(&mut self, protocol_iteration: usize) -> String {
        let ordinal = self.next_llm_ordinal;
        self.next_llm_ordinal += 1;
        format!(
            "{}:{}:{}:{}",
            self.session_id, self.turn_index, protocol_iteration, ordinal
        )
    }

    pub(super) fn emit_trace(&self, protocol_iteration: usize, event: lash_trace::TraceEvent) {
        crate::trace::emit_trace(
            &self.host.core.tracing.trace_sink,
            &self.host.core.tracing.trace_context,
            self.trace_context(protocol_iteration),
            event,
        );
    }

    pub(super) fn emit_tool_call_trace(
        &self,
        protocol_iteration: usize,
        record: &crate::ToolCallRecord,
    ) {
        self.emit_trace(
            protocol_iteration,
            lash_trace::TraceEvent::ToolCallCompleted {
                call_id: record.call_id.clone(),
                name: record.tool.clone(),
                args: record.args.clone(),
                output: crate::trace::trace_tool_call_output(&record.output),
                duration_ms: record.duration_ms,
            },
        );
    }

    pub(super) fn emit_tool_call_started_trace(
        &self,
        protocol_iteration: usize,
        call_id: Option<String>,
        name: String,
        args: serde_json::Value,
    ) {
        self.emit_trace(
            protocol_iteration,
            lash_trace::TraceEvent::ToolCallStarted {
                call_id,
                name,
                args,
            },
        );
    }

    pub(super) fn mark_phase_begin(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.begin(phase);
        }
    }

    pub(super) fn emit_protocol_diagnostic_trace(
        &self,
        protocol_iteration: usize,
        phase: &str,
        payload: serde_json::Value,
    ) {
        let protocol_event = crate::ProtocolEvent::typed(
            "runtime",
            serde_json::json!({
                "diagnostic": {
                    "phase": phase,
                    "payload": payload,
                }
            }),
        )
        .expect("protocol diagnostic event serializes");
        self.emit_trace(
            protocol_iteration,
            protocol_step_trace_event(&protocol_event),
        );
    }

    pub(super) fn mark_phase_end(&self, phase: RuntimeTurnPhase) {
        if let Some(probe) = self.turn_phase_probe.as_ref() {
            probe.end(phase);
        }
    }
}

pub(in crate::runtime) fn protocol_step_trace_event(
    protocol_event: &crate::ProtocolEvent,
) -> lash_trace::TraceEvent {
    lash_trace::TraceEvent::ProtocolStep {
        plugin_id: protocol_event.plugin_id.clone(),
        payload: protocol_event.payload.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_step_trace_event_preserves_protocol_payload() {
        let protocol_event = crate::ProtocolEvent::typed(
            "custom",
            serde_json::json!({
                "code": "print \"hi\"",
                "final_output": "done"
            }),
        )
        .expect("protocol event");

        let lash_trace::TraceEvent::ProtocolStep { plugin_id, payload } =
            protocol_step_trace_event(&protocol_event)
        else {
            panic!("expected protocol step trace event");
        };

        assert_eq!(plugin_id, "custom");
        assert_eq!(
            payload.get("code").and_then(serde_json::Value::as_str),
            Some("print \"hi\"")
        );
        assert_eq!(
            payload.get("final_output"),
            Some(&serde_json::json!("done"))
        );
    }
}
