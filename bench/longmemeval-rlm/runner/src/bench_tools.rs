use std::sync::Arc;

use lash::{ToolDefinition, ToolExecutionMode, ToolParam, ToolProvider, ToolResult};
use regex::RegexBuilder;
use serde_json::json;

use crate::dataset::{LongMemEvalQuestion, LongMemEvalTurn};

#[derive(Clone)]
pub struct BenchmarkQuestionContext {
    pub question: Arc<LongMemEvalQuestion>,
}

impl BenchmarkQuestionContext {
    pub fn new(question: LongMemEvalQuestion) -> Self {
        Self {
            question: Arc::new(question),
        }
    }

    fn session_count(&self) -> usize {
        self.question.haystack_sessions.len()
    }

    fn session_id(&self, zero_index: usize) -> Option<&str> {
        self.question
            .haystack_session_ids
            .get(zero_index)
            .map(String::as_str)
    }

    fn session_date(&self, zero_index: usize) -> Option<&str> {
        self.question
            .haystack_dates
            .get(zero_index)
            .map(String::as_str)
    }

    fn session_text(&self, zero_index: usize) -> String {
        let Some(session) = self.question.haystack_sessions.get(zero_index) else {
            return String::new();
        };
        render_session(session)
    }

    fn session_preview(&self, zero_index: usize) -> String {
        let rendered = self.session_text(zero_index);
        let trimmed = rendered.replace('\n', " ");
        if trimmed.chars().count() <= 220 {
            trimmed
        } else {
            format!("{}...", trimmed.chars().take(220).collect::<String>())
        }
    }

    fn list_sessions(&self) -> serde_json::Value {
        let sessions = (0..self.session_count())
            .map(|index| {
                json!({
                    "session_number": index + 1,
                    "session_id": self.session_id(index),
                    "date": self.session_date(index),
                    "turn_count": self.question.haystack_sessions.get(index).map_or(0, Vec::len),
                    "preview": self.session_preview(index),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "question_id": self.question.question_id,
            "session_count": self.session_count(),
            "sessions": sessions,
        })
    }

    fn get_session(&self, session_number: usize) -> ToolResult {
        if session_number == 0 {
            return ToolResult::err_fmt("session_number is 1-based and must be >= 1");
        }
        let index = session_number - 1;
        let Some(session) = self.question.haystack_sessions.get(index) else {
            return ToolResult::err_fmt(format!(
                "session_number {} out of range; there are {} sessions",
                session_number,
                self.session_count()
            ));
        };
        ToolResult::ok(json!({
            "session_number": session_number,
            "session_id": self.session_id(index),
            "date": self.session_date(index),
            "turn_count": session.len(),
            "session": session,
            "rendered": render_session(session),
        }))
    }

    fn search_sessions(&self, query: &str, limit: usize) -> ToolResult {
        let query = query.trim();
        if query.is_empty() {
            return ToolResult::err_fmt("query must not be empty");
        }
        let limit = limit.clamp(1, 50);
        let query_terms = query
            .split_whitespace()
            .map(|term| term.to_ascii_lowercase())
            .collect::<Vec<_>>();
        let mut matches = Vec::new();
        for index in 0..self.session_count() {
            let rendered = self.session_text(index);
            let haystack = rendered.to_ascii_lowercase();
            let score = query_terms
                .iter()
                .map(|term| haystack.matches(term).count())
                .sum::<usize>();
            if score == 0 {
                continue;
            }
            matches.push(json!({
                "session_number": index + 1,
                "session_id": self.session_id(index),
                "date": self.session_date(index),
                "score": score,
                "preview": self.session_preview(index),
            }));
        }
        matches.sort_by(|a, b| {
            let a_score = a.get("score").and_then(|value| value.as_u64()).unwrap_or(0);
            let b_score = b.get("score").and_then(|value| value.as_u64()).unwrap_or(0);
            b_score.cmp(&a_score)
        });
        matches.truncate(limit);
        ToolResult::ok(json!({
            "query": query,
            "match_count": matches.len(),
            "matches": matches,
        }))
    }

    fn grep_sessions(&self, pattern: &str, limit: usize) -> ToolResult {
        let regex = match RegexBuilder::new(pattern).case_insensitive(true).build() {
            Ok(regex) => regex,
            Err(err) => return ToolResult::err_fmt(format!("invalid regex: {err}")),
        };
        let limit = limit.clamp(1, 100);
        let mut matches = Vec::new();
        for index in 0..self.session_count() {
            let Some(session) = self.question.haystack_sessions.get(index) else {
                continue;
            };
            for (turn_index, turn) in session.iter().enumerate() {
                for mat in regex.find_iter(&turn.content) {
                    let preview_start =
                        floor_char_boundary(&turn.content, mat.start().saturating_sub(60));
                    let preview_end = ceil_char_boundary(
                        &turn.content,
                        usize::min(mat.end() + 60, turn.content.len()),
                    );
                    matches.push(json!({
                        "session_number": index + 1,
                        "session_id": self.session_id(index),
                        "date": self.session_date(index),
                        "turn_index": turn_index,
                        "role": turn.role,
                        "match": mat.as_str(),
                        "preview": turn.content[preview_start..preview_end].to_string(),
                    }));
                    if matches.len() >= limit {
                        break;
                    }
                }
                if matches.len() >= limit {
                    break;
                }
            }
            if matches.len() >= limit {
                break;
            }
        }
        ToolResult::ok(json!({
            "pattern": pattern,
            "match_count": matches.len(),
            "matches": matches,
        }))
    }
}

pub struct LongMemEvalSessionTools {
    ctx: BenchmarkQuestionContext,
}

impl LongMemEvalSessionTools {
    pub fn new(ctx: BenchmarkQuestionContext) -> Self {
        Self { ctx }
    }
}

#[async_trait::async_trait]
impl ToolProvider for LongMemEvalSessionTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "list_sessions".to_string(),
                description: "List benchmark sessions in chronological order with dates and previews.".to_string(),
                params: vec![],
                returns: "dict".to_string(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::callable(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "get_session".to_string(),
                description: "Fetch a full benchmark session by 1-based session number.".to_string(),
                params: vec![ToolParam::typed("session_number", "int")],
                returns: "dict".to_string(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::callable(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "search_sessions".to_string(),
                description: "Search sessions by plain-text query and return the best-matching session previews.".to_string(),
                params: vec![
                    ToolParam::typed("query", "str"),
                    ToolParam::optional("limit", "int"),
                ],
                returns: "dict".to_string(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::callable(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "grep_sessions".to_string(),
                description: "Search session contents with a regular expression and return matching snippets.".to_string(),
                params: vec![
                    ToolParam::typed("pattern", "str"),
                    ToolParam::optional("limit", "int"),
                ],
                returns: "dict".to_string(),
                examples: vec![],
                availability: lash::ToolAvailabilityConfig::callable(),
                activation: lash::ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Parallel,
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match name {
            "list_sessions" => ToolResult::ok(self.ctx.list_sessions()),
            "get_session" => {
                let Some(number) = args.get("session_number").and_then(|value| value.as_u64())
                else {
                    return ToolResult::err_fmt("session_number must be an integer");
                };
                self.ctx.get_session(number as usize)
            }
            "search_sessions" => {
                let Some(query) = args.get("query").and_then(|value| value.as_str()) else {
                    return ToolResult::err_fmt("query must be a string");
                };
                let limit = args
                    .get("limit")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize)
                    .unwrap_or(5);
                self.ctx.search_sessions(query, limit)
            }
            "grep_sessions" => {
                let pattern = args
                    .get("pattern")
                    .and_then(|value| value.as_str())
                    .or_else(|| args.get("query").and_then(|value| value.as_str()));
                let Some(pattern) = pattern else {
                    return ToolResult::err_fmt("pattern must be a string");
                };
                let limit = args
                    .get("limit")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize)
                    .unwrap_or(10);
                self.ctx.grep_sessions(pattern, limit)
            }
            other => ToolResult::err_fmt(format!("unknown tool: {other}")),
        }
    }
}

fn render_session(session: &[LongMemEvalTurn]) -> String {
    session
        .iter()
        .enumerate()
        .map(|(index, turn)| format!("{}: [{}] {}", index + 1, turn.role, turn.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::LongMemEvalQuestion;

    fn sample_context() -> BenchmarkQuestionContext {
        BenchmarkQuestionContext::new(LongMemEvalQuestion {
            question_id: "q1".to_string(),
            question: "What degree did I graduate with?".to_string(),
            answer: None,
            question_type: Some("single-session-user".to_string()),
            question_date: Some("2026-01-01".to_string()),
            haystack_dates: vec!["2025-01-01".to_string(), "2025-02-01".to_string()],
            haystack_session_ids: vec!["s1".to_string(), "s2".to_string()],
            haystack_sessions: vec![
                vec![
                    LongMemEvalTurn {
                        role: "user".to_string(),
                        content: "I graduated with Business Administration.".to_string(),
                    },
                    LongMemEvalTurn {
                        role: "assistant".to_string(),
                        content: "Congrats.".to_string(),
                    },
                ],
                vec![LongMemEvalTurn {
                    role: "user".to_string(),
                    content: "My pet bed cost $40.".to_string(),
                }],
            ],
        })
    }

    #[test]
    fn search_sessions_finds_matching_session() {
        let result = sample_context().search_sessions("Business", 5);
        assert!(result.success);
        let matches = result
            .result
            .get("matches")
            .and_then(|value| value.as_array())
            .expect("matches");
        assert_eq!(matches[0]["session_number"], 1);
    }

    #[test]
    fn grep_sessions_returns_snippet() {
        let result = sample_context().grep_sessions(r"\\$40|Business", 10);
        assert!(result.success);
        let matches = result
            .result
            .get("matches")
            .and_then(|value| value.as_array())
            .expect("matches");
        assert!(!matches.is_empty());
    }

    #[test]
    fn grep_sessions_handles_multibyte_preview_boundaries() {
        let ctx = BenchmarkQuestionContext::new(LongMemEvalQuestion {
            question_id: "q2".to_string(),
            question: "test".to_string(),
            answer: None,
            question_type: Some("single-session-user".to_string()),
            question_date: Some("2026-01-01".to_string()),
            haystack_dates: vec!["2025-01-01".to_string()],
            haystack_session_ids: vec!["s1".to_string()],
            haystack_sessions: vec![vec![LongMemEvalTurn {
                role: "assistant".to_string(),
                content: "아이디어 이름 후보는 바운시 앱 입니다".to_string(),
            }]],
        });
        let result = ctx.grep_sessions("바운시|앱", 10);
        assert!(result.success);
        let matches = result
            .result
            .get("matches")
            .and_then(|value| value.as_array())
            .expect("matches");
        assert!(!matches.is_empty());
    }
}
