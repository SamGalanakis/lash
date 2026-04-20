use super::{Record, Value};
use thiserror::Error;

pub trait ToolHost: Sync {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError>;

    fn call_batch(
        &self,
        _calls: &[(&str, &Record)],
        _push_result: &mut dyn FnMut(Result<Value, ToolHostError>),
    ) -> bool {
        false
    }

    fn start_call(&self, _name: &str, _args: &Record) -> Result<Value, ToolHostError> {
        Err(ToolHostError::new("async tool starts are unavailable"))
    }

    fn await_handle(&self, _handle: &Value) -> Result<Value, ToolHostError> {
        Err(ToolHostError::new("async tool handles are unavailable"))
    }

    fn cancel_handle(&self, _handle: &Value) -> Result<Value, ToolHostError> {
        Err(ToolHostError::new("async tool handles are unavailable"))
    }

    fn print(&self, _value: &Value) -> Result<(), ToolHostError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ToolHostError {
    message: String,
}

impl ToolHostError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
