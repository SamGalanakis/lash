pub mod gepa {
    use super::super::*;

    #[derive(Clone, Debug, Default)]
    pub struct ReflectiveGepaStrategy<P> {
        proposer: P,
    }

    impl<P> ReflectiveGepaStrategy<P> {
        pub fn new(proposer: P) -> Self {
            Self { proposer }
        }
    }

    #[async_trait]
    pub trait ReflectiveProposer: Send + Sync {
        async fn propose_json(
            &self,
            request: ReflectiveProposalRequest,
            cancellation: CancellationToken,
        ) -> Result<Value>;
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ReflectiveProposalRequest {
        pub run_id: String,
        pub experiment_id: String,
        pub generation: usize,
        pub artifact_dir: PathBuf,
        pub parent_candidate_id: String,
        pub mutable_components: BTreeMap<String, MutableComponent>,
        pub evidence: Value,
        pub output_schema: Value,
    }

    #[async_trait]
    impl<P> OptimizerStrategy for ReflectiveGepaStrategy<P>
    where
        P: ReflectiveProposer,
    {
        async fn propose(
            &self,
            request: StrategyRequest,
            cancellation: CancellationToken,
        ) -> Result<Vec<CandidateProposal>> {
            let reflective_request = ReflectiveProposalRequest {
                run_id: request.run_id,
                experiment_id: request.experiment_id,
                generation: request.generation,
                artifact_dir: request.artifact_dir,
                parent_candidate_id: request.evidence.parent.id.clone(),
                mutable_components: request.evidence.parent.mutable_components.clone(),
                evidence: render_reflective_evidence(&request.evidence.evaluated_examples),
                output_schema: proposal_output_schema(),
            };
            let value = self
                .proposer
                .propose_json(reflective_request, cancellation)
                .await?;
            parse_candidate_proposals(value)
        }
    }

    pub fn proposal_output_schema() -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["proposals"],
            "properties": {
                "proposals": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["parent_candidate_id", "patches"],
                        "properties": {
                            "parent_candidate_id": { "type": "string" },
                            "rationale": { "type": "string" },
                            "patches": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "required": ["kind", "component_id", "value"],
                                    "properties": {
                                        "kind": { "const": "replace_value" },
                                        "component_id": { "type": "string" },
                                        "value": {
                                            "oneOf": [
                                                {
                                                    "type": "object",
                                                    "additionalProperties": false,
                                                    "required": ["kind", "text"],
                                                    "properties": {
                                                        "kind": { "const": "text" },
                                                        "text": { "type": "string" }
                                                    }
                                                },
                                                {
                                                    "type": "object",
                                                    "additionalProperties": false,
                                                    "required": ["kind", "value"],
                                                    "properties": {
                                                        "kind": { "const": "json" },
                                                        "value": {}
                                                    }
                                                },
                                                {
                                                    "type": "object",
                                                    "additionalProperties": false,
                                                    "required": ["kind", "template"],
                                                    "properties": {
                                                        "kind": { "const": "prompt_template" },
                                                        "template": { "type": "object" }
                                                    }
                                                },
                                                {
                                                    "type": "object",
                                                    "additionalProperties": false,
                                                    "required": ["kind", "contribution"],
                                                    "properties": {
                                                        "kind": { "const": "prompt_contribution" },
                                                        "contribution": { "type": "object" }
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            },
                            "metadata": { "type": "object" }
                        }
                    }
                }
            }
        })
    }

    pub fn parse_candidate_proposals(value: Value) -> Result<Vec<CandidateProposal>> {
        let proposals = value
            .get("proposals")
            .cloned()
            .ok_or_else(|| HarnessOptError::InvalidProposal("missing proposals".to_string()))?;
        let proposals: Vec<CandidateProposal> = serde_json::from_value(proposals)
            .map_err(|error| HarnessOptError::InvalidProposal(error.to_string()))?;
        if proposals.is_empty() {
            return Err(HarnessOptError::InvalidProposal(
                "proposal list is empty".to_string(),
            ));
        }
        Ok(proposals)
    }

    pub fn render_reflective_evidence(runs: &[ExampleRun]) -> Value {
        let examples = runs
            .iter()
            .map(|run| {
                let records = run
                    .trace
                    .as_ref()
                    .map(|trace| {
                        trace
                            .records
                            .iter()
                            .map(render_trace_record)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                json!({
                    "example_id": run.example.id,
                    "split": run.example.split,
                    "score": run.result.score,
                    "passed": run.result.passed,
                    "feedback": run.result.feedback,
                    "turn_outcome": run.result.diagnostics.get("turn_outcome"),
                    "tool_call_count": run.result.diagnostics.get("tool_call_count"),
                    "error_count": run.result.diagnostics.get("error_count"),
                    "diary_behavior": run.result.diagnostics.get("diary_behavior"),
                    "schema_failure": run.result.diagnostics.get("schema_failure"),
                    "trace": records,
                })
            })
            .collect::<Vec<_>>();
        json!({ "examples": examples })
    }

    fn render_trace_record(record: &TraceRecord) -> Value {
        let event = match &record.event {
            TraceEvent::TurnStarted { .. } => "turn_started",
            TraceEvent::TurnCompleted { .. } => "turn_completed",
            TraceEvent::ToolCallStarted { .. } => "tool_call_started",
            TraceEvent::ToolCallCompleted { .. } => "tool_call_completed",
            TraceEvent::LlmCallFailed { .. } => "llm_call_failed",
            TraceEvent::LlmCallStarted { .. } => "llm_call_started",
            TraceEvent::LlmCallCompleted { .. } => "llm_call_completed",
            TraceEvent::PromptBuilt { .. } => "prompt_built",
            _ => "other",
        };
        json!({
            "event": event,
            "timestamp": record.timestamp,
            "context": {
                "run_id": record.context.run_id,
                "experiment_id": record.context.experiment_id,
                "candidate_id": record.context.candidate_id,
                "candidate_parent_id": record.context.candidate_parent_id,
                "example_id": record.context.example_id,
                "split": record.context.split,
            }
        })
    }
}
