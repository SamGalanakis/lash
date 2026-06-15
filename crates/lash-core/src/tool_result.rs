#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutBehavior {
    ErrorAsResult,
    FailTurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelHint {
    Ignore,
    CancelExternalWork,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PendingCompletion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<std::time::Duration>,
    pub on_timeout: TimeoutBehavior,
    pub on_cancel: CancelHint,
}

impl Default for PendingCompletion {
    fn default() -> Self {
        Self {
            deadline: None,
            on_timeout: TimeoutBehavior::ErrorAsResult,
            on_cancel: CancelHint::CancelExternalWork,
        }
    }
}

impl PendingCompletion {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_deadline(mut self, deadline: std::time::Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn fail_turn_on_timeout(mut self) -> Self {
        self.on_timeout = TimeoutBehavior::FailTurn;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolResult {
    Done(Box<crate::ToolCallOutput>),
    Pending(PendingCompletion),
}

impl ToolResult {
    pub fn from_output(output: crate::ToolCallOutput) -> Self {
        Self::Done(Box::new(output))
    }

    pub fn pending(pending: PendingCompletion) -> Self {
        Self::Pending(pending)
    }

    pub fn ok(result: serde_json::Value) -> Self {
        Self::from_output(crate::ToolCallOutput::success(result))
    }

    pub fn err(result: serde_json::Value) -> Self {
        let message = result
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| result.to_string());
        Self::from_output(crate::ToolCallOutput::failure(crate::ToolFailure {
            class: crate::ToolFailureClass::Execution,
            code: "tool_error".to_string(),
            message,
            source: crate::ToolFailureSource::Tool,
            retry: crate::ToolRetryDisposition::Never,
            raw: Some(crate::ToolValue::from(result)),
        }))
    }

    pub fn err_fmt(msg: impl std::fmt::Display) -> Self {
        Self::err(serde_json::json!(msg.to_string()))
    }

    pub fn failure(failure: crate::ToolFailure) -> Self {
        Self::from_output(crate::ToolCallOutput::failure(failure))
    }

    pub fn retryable_failure(
        class: crate::ToolFailureClass,
        code: impl Into<String>,
        message: impl Into<String>,
        after_ms: Option<u64>,
    ) -> Self {
        Self::failure(crate::ToolFailure::safe_retry(
            class, code, message, after_ms,
        ))
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::from_output(crate::ToolCallOutput::cancelled(
            crate::ToolCancellation::runtime(message),
        ))
    }

    pub fn cancelled_with_raw(message: impl Into<String>, raw: serde_json::Value) -> Self {
        let mut cancellation = crate::ToolCancellation::runtime(message);
        cancellation.raw = Some(crate::ToolValue::from(raw));
        Self::from_output(crate::ToolCallOutput::cancelled(cancellation))
    }

    pub fn with_control(mut self, control: crate::ToolControl) -> Self {
        if let Self::Done(output) = &mut self {
            output.as_mut().control = Some(control);
        }
        self
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Done(output) if output.is_success())
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending(_))
    }

    pub fn value_for_projection(&self) -> serde_json::Value {
        match &self
            .as_done_output()
            .expect("pending tool result has no projection value")
            .outcome
        {
            crate::ToolCallOutcome::Success(value) => value.to_json_value(),
            crate::ToolCallOutcome::Failure(failure) => failure
                .raw
                .as_ref()
                .map(crate::ToolValue::to_json_value)
                .unwrap_or_else(|| failure.to_json_value()),
            crate::ToolCallOutcome::Cancelled(cancellation) => cancellation
                .raw
                .as_ref()
                .map(crate::ToolValue::to_json_value)
                .unwrap_or_else(|| cancellation.to_json_value()),
        }
    }

    pub fn as_done_output(&self) -> Option<&crate::ToolCallOutput> {
        match self {
            Self::Done(output) => Some(output.as_ref()),
            Self::Pending(_) => None,
        }
    }

    pub fn as_output(&self) -> &crate::ToolCallOutput {
        self.as_done_output()
            .expect("pending tool result cannot be viewed as completed output")
    }

    pub fn into_done_output(self) -> Result<crate::ToolCallOutput, PendingCompletion> {
        match self {
            Self::Done(output) => Ok(*output),
            Self::Pending(pending) => Err(pending),
        }
    }
}

impl<T, E> From<Result<T, E>> for ToolResult
where
    T: serde::Serialize,
    E: std::fmt::Display,
{
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(value) => match serde_json::to_value(value) {
                Ok(value) => Self::ok(value),
                Err(err) => Self::err_fmt(format_args!("Failed to serialize tool result: {err}")),
            },
            Err(err) => Self::err_fmt(err),
        }
    }
}

pub(crate) fn tool_output_from_completion_resolution(
    resolution: crate::Resolution,
) -> crate::ToolCallOutput {
    match resolution {
        crate::Resolution::Ok(value) => crate::ToolCallOutput::success(value),
        crate::Resolution::Err(err) => {
            let mut failure =
                crate::ToolFailure::tool(crate::ToolFailureClass::Execution, err.code, err.message);
            failure.raw = err.raw.map(crate::ToolValue::from);
            crate::ToolCallOutput::failure(failure)
        }
        crate::Resolution::Timeout => crate::ToolCallOutput::failure(crate::ToolFailure::runtime(
            crate::ToolFailureClass::Timeout,
            "tool_completion_timeout",
            "pending tool completion timed out",
        )),
        crate::Resolution::Cancelled => crate::ToolCallOutput::cancelled(
            crate::ToolCancellation::runtime("pending tool completion cancelled"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde::ser::{Error as _, Serializer};

    use super::*;

    #[test]
    fn tool_result_from_result_serializes_success_values() {
        let result: ToolResult = Result::<_, std::io::Error>::Ok(vec!["alpha", "beta"]).into();
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection(),
            serde_json::json!(["alpha", "beta"])
        );
    }

    #[test]
    fn tool_result_from_result_formats_errors() {
        let result: ToolResult =
            Result::<serde_json::Value, _>::Err(std::io::Error::other("nope")).into();
        assert!(!result.is_success());
        assert_eq!(result.value_for_projection(), serde_json::json!("nope"));
        assert_eq!(
            result.as_output().value_for_projection()["message"],
            serde_json::json!("nope")
        );
    }

    #[test]
    fn tool_result_from_result_reports_serialize_failures() {
        struct BrokenValue;

        impl serde::Serialize for BrokenValue {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                Err(S::Error::custom("boom"))
            }
        }

        let result: ToolResult = Result::<BrokenValue, std::io::Error>::Ok(BrokenValue).into();
        assert!(!result.is_success());
        assert_eq!(
            result.value_for_projection(),
            serde_json::json!("Failed to serialize tool result: boom")
        );
    }

    #[test]
    fn pending_result_is_not_completed_output() {
        let result = ToolResult::pending(PendingCompletion::new());
        assert!(result.is_pending());
        assert!(result.as_done_output().is_none());
        assert!(result.into_done_output().is_err());
    }
}
