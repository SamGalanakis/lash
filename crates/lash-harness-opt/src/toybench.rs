use super::*;

pub const MEMORY_GUIDANCE_COMPONENT: &str = "toybench.memory_guidance";
pub const PROMPT_TEMPLATE_COMPONENT: &str = "toybench.prompt_template";
pub const USER_DIRECTIVE_COMPONENT: &str = "toybench.user_directive";

pub const TOYBENCH_MEMORY_GUIDANCE: &str = r#"Use your persistent RLM REPL as memory. The following globals are bound for this turn:

- `iteration: int` — current benchmark iteration
- `current_query: str` — the task this turn must answer
- `current_feedback: str | null` — score/feedback from the previous action; `null` on the first iteration
- `diary: list` — persistent across iterations; each entry is `{ history_index: int, summary: str, learnings: str }`
- `history` — auto-bound read-only projection of past turn entries; index into it via `entry.history_index`

The current query and previous feedback are visible in the user turn and are also bound as `current_query` and `current_feedback` for exact access from lashlang. The shape your `finish` value must take is shown in the **Required output** block at the end of the user turn — consult it before building the action.

At the start of each turn:

- Use the visible current query and feedback first. Inspect `diary` only when you need details not already visible.
- Pull the matching `history[entry.history_index]` when an entry's `summary` is too compressed to act on.
- Apply prior learnings before choosing the benchmark action.

Before every `finish`, append exactly one diary record and finish a value matching the **Required output** shape:

<lashlang>
diary = push(diary, {
  history_index: len(history) - 1,
  summary: "brief task/action summary",
  learnings: "reusable lesson from this interaction"
})
finish answer
</lashlang>

`answer` must match the **Required output** block exactly — the shape varies per task and per step, so build the value to fit the announced contract rather than assuming a fixed wrapper. Keep diary entries short, factual, and reusable; do not duplicate old lessons; incorporate feedback and revise strategy in later entries."#;

pub const DEFAULT_USER_DIRECTIVE: &str = "Choose the next benchmark action.";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToybenchConfig {
    #[serde(default = "default_experiment_id")]
    pub experiment_id: String,
    #[serde(default)]
    pub train: Vec<HarnessExample>,
    #[serde(default)]
    pub val: Vec<HarnessExample>,
    #[serde(default)]
    pub test: Vec<HarnessExample>,
}

fn default_experiment_id() -> String {
    "toybench".to_string()
}

#[derive(Clone, Debug)]
pub struct ToybenchProject {
    config: ToybenchConfig,
}

impl ToybenchProject {
    pub fn new(config: ToybenchConfig) -> Self {
        Self { config }
    }

    pub fn seed_candidate_static() -> Candidate {
        Candidate {
            id: "seed".to_string(),
            parent_id: None,
            mutable_components: BTreeMap::new(),
            immutable_context: BTreeMap::from([(
                "tool_catalog".to_string(),
                json!([
                    "llm_query",
                    "spawn_agent",
                    "continue_as",
                    "list_process_handles"
                ]),
            )]),
            metadata: BTreeMap::from([("project".to_string(), json!("toybench"))]),
        }
        .with_component(MutableComponent {
            id: MEMORY_GUIDANCE_COMPONENT.to_string(),
            description: Some("Toybench persistent memory guidance".to_string()),
            value: ComponentValue::Text {
                text: TOYBENCH_MEMORY_GUIDANCE.to_string(),
            },
            constraints: ComponentConstraints {
                max_chars: Some(8_000),
                preserve_terms: vec![
                    "diary".to_string(),
                    "finish".to_string(),
                    "llm_query".to_string(),
                    "spawn_agent".to_string(),
                    "continue_as".to_string(),
                    "list_process_handles".to_string(),
                ],
                forbidden_terms: vec![
                    "exec_command".to_string(),
                    "read_file".to_string(),
                    "edit".to_string(),
                    "write".to_string(),
                ],
                format_hint: Some(
                    "Markdown guidance with a fenced lashlang diary example".to_string(),
                ),
            },
        })
        .with_component(MutableComponent {
            id: PROMPT_TEMPLATE_COMPONENT.to_string(),
            description: Some(
                "Toybench prompt-template section layout and static text".to_string(),
            ),
            value: ComponentValue::PromptTemplate {
                template: toybench_prompt_template(TOYBENCH_MEMORY_GUIDANCE),
            },
            constraints: ComponentConstraints {
                preserve_terms: vec!["Toy Memory".to_string(), "Execution".to_string()],
                ..Default::default()
            },
        })
        .with_component(MutableComponent {
            id: USER_DIRECTIVE_COMPONENT.to_string(),
            description: Some("Per-turn toybench next-action directive".to_string()),
            value: ComponentValue::Text {
                text: DEFAULT_USER_DIRECTIVE.to_string(),
            },
            constraints: ComponentConstraints {
                max_chars: Some(1_000),
                preserve_terms: vec!["finish".to_string()],
                ..Default::default()
            },
        })
    }

    pub fn prompt_template(candidate: &Candidate) -> Result<PromptTemplate> {
        match &candidate.component(PROMPT_TEMPLATE_COMPONENT)?.value {
            ComponentValue::PromptTemplate { template } => Ok(template.clone()),
            _ => Err(HarnessOptError::ConstraintViolation {
                component_id: PROMPT_TEMPLATE_COMPONENT.to_string(),
                reason: "expected prompt_template component".to_string(),
            }),
        }
    }

    pub fn user_directive(candidate: &Candidate) -> Result<String> {
        match &candidate.component(USER_DIRECTIVE_COMPONENT)?.value {
            ComponentValue::Text { text } => Ok(text.clone()),
            _ => Err(HarnessOptError::ConstraintViolation {
                component_id: USER_DIRECTIVE_COMPONENT.to_string(),
                reason: "expected text component".to_string(),
            }),
        }
    }

    pub async fn write_seed_candidate(path: &Path) -> Result<()> {
        let candidate = Self::seed_candidate_static();
        tokio::fs::write(path, serde_json::to_vec_pretty(&candidate)?).await?;
        Ok(())
    }
}

#[async_trait]
impl HarnessProject for ToybenchProject {
    async fn seed_candidate(&self) -> Result<Candidate> {
        Ok(Self::seed_candidate_static())
    }

    async fn trainset(&self) -> Result<Vec<HarnessExample>> {
        Ok(self.config.train.clone())
    }

    async fn valset(&self) -> Result<Vec<HarnessExample>> {
        Ok(self.config.val.clone())
    }

    async fn testset(&self) -> Result<Vec<HarnessExample>> {
        Ok(self.config.test.clone())
    }

    async fn evaluate_example(
        &self,
        run: &OptimizationRun,
        candidate: &Candidate,
        example: &HarnessExample,
        context: TraceContext,
        _cancellation: CancellationToken,
    ) -> Result<ExampleRun> {
        let example_dir = run
            .run_dir
            .join("examples")
            .join(&candidate.id)
            .join(&example.id);
        tokio::fs::create_dir_all(&example_dir).await?;
        let request_path = example_dir.join("request.json");
        let candidate_path = example_dir.join("candidate.json");
        let response_path = example_dir.join("response.json");
        let trace_path = example_dir.join("typed_trace.jsonl");
        let evidence_path = example_dir.join("rendered_evidence.json");
        let score_path = example_dir.join("score_report.json");
        let session_db = example_dir.join("session.db");

        let directive = Self::user_directive(candidate)?;
        let request = json!({
            "input": example.input,
            "expected": example.expected,
            "directive": directive,
            "trace_context": context,
        });
        tokio::fs::write(&request_path, serde_json::to_vec_pretty(&request)?).await?;
        tokio::fs::write(&candidate_path, serde_json::to_vec_pretty(candidate)?).await?;

        let expected = example.expected.as_ref();
        let score = expected
            .and_then(|expected| expected.get("score"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let feedback = expected
            .and_then(|expected| expected.get("feedback"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let result = EvaluationResult {
            example_id: example.id.clone(),
            split: example.split.clone(),
            score,
            passed: Some(score >= 1.0),
            feedback,
            metrics: BTreeMap::new(),
            diagnostics: BTreeMap::from([
                ("tool_call_count".to_string(), json!(0)),
                ("error_count".to_string(), json!(0)),
                ("turn_outcome".to_string(), json!("synthetic")),
            ]),
        };
        let trace = TraceBundle {
            example_id: example.id.clone(),
            records: vec![TraceRecord::new(
                context,
                TraceEvent::TurnStarted {
                    metadata: BTreeMap::from([("project".to_string(), json!("toybench"))]),
                },
            )],
        };
        let evidence = strategies::gepa::render_reflective_evidence(&[ExampleRun {
            example: example.clone(),
            result: result.clone(),
            trace: Some(trace.clone()),
            artifacts: RunArtifacts::default(),
            metric_calls: 1,
        }]);
        tokio::fs::write(
            &response_path,
            serde_json::to_vec_pretty(&json!({ "action": null }))?,
        )
        .await?;
        tokio::fs::write(
            &trace_path,
            trace
                .records
                .iter()
                .map(serde_json::to_string)
                .collect::<std::result::Result<Vec<_>, _>>()?
                .join("\n"),
        )
        .await?;
        tokio::fs::write(&evidence_path, serde_json::to_vec_pretty(&evidence)?).await?;
        tokio::fs::write(&score_path, serde_json::to_vec_pretty(&result)?).await?;

        Ok(ExampleRun {
            example: example.clone(),
            result,
            trace: Some(trace),
            artifacts: RunArtifacts {
                request_json: Some(request_path),
                candidate_json: Some(candidate_path),
                response_json: Some(response_path),
                session_db: Some(session_db),
                typed_trace_jsonl: Some(trace_path),
                rendered_evidence_json: Some(evidence_path),
                score_report_json: Some(score_path),
            },
            metric_calls: 1,
        })
    }
}

pub fn toybench_prompt_template(memory_guidance: &str) -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![PromptTemplateEntry::text(
            "You are being evaluated by Toy Bench, which tests whether an agent improves from feedback across sequential task instances.",
        )]),
        PromptTemplateSection::titled(
            "Execution",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
                PromptTemplateEntry::slot(PromptSlot::Execution),
            ],
        ),
        PromptTemplateSection::titled(
            "Toy Memory",
            vec![PromptTemplateEntry::text(memory_guidance)],
        ),
        PromptTemplateSection::titled(
            "Guidance",
            vec![
                PromptTemplateEntry::slot(PromptSlot::Guidance),
                PromptTemplateEntry::slot(PromptSlot::ProjectInstructions),
            ],
        ),
    ])
}
