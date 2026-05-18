#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    output: Box<crate::ToolCallOutput>,
}

impl ToolResult {
    pub fn from_output(output: crate::ToolCallOutput) -> Self {
        Self {
            output: Box::new(output),
        }
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
        self.output.as_mut().control = Some(control);
        self
    }

    pub fn is_success(&self) -> bool {
        self.output.is_success()
    }

    pub fn value_for_projection(&self) -> serde_json::Value {
        match &self.output.outcome {
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

    pub fn as_output(&self) -> &crate::ToolCallOutput {
        self.output.as_ref()
    }

    pub fn into_output(self) -> crate::ToolCallOutput {
        *self.output
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
}
