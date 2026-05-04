use super::{Record, Value};
use futures_util::future::join_all;
use std::future::Future;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct ToolHostCall {
    pub name: String,
    pub args: Record,
}

pub trait ToolHost: Sync {
    fn call(
        &self,
        name: String,
        args: Record,
    ) -> impl Future<Output = Result<Value, ToolHostError>> + Send;

    fn call_batch(
        &self,
        calls: Vec<ToolHostCall>,
    ) -> impl Future<Output = Vec<Result<Value, ToolHostError>>> + Send {
        async move {
            join_all(
                calls
                    .into_iter()
                    .map(|call| self.call(call.name, call.args)),
            )
            .await
        }
    }

    fn start_call(
        &self,
        _name: String,
        _args: Record,
    ) -> impl Future<Output = Result<Value, ToolHostError>> + Send {
        async { Err(ToolHostError::new("async tool starts are unavailable")) }
    }

    fn await_handle(
        &self,
        _handle: Value,
    ) -> impl Future<Output = Result<Value, ToolHostError>> + Send {
        async { Err(ToolHostError::new("async tool handles are unavailable")) }
    }

    fn cancel_handle(
        &self,
        _handle: Value,
    ) -> impl Future<Output = Result<Value, ToolHostError>> + Send {
        async { Err(ToolHostError::new("async tool handles are unavailable")) }
    }

    fn print(&self, _value: Value) -> impl Future<Output = Result<(), ToolHostError>> + Send {
        async { Ok(()) }
    }

    fn yield_now(&self) -> impl Future<Output = ()> + Send {
        async {}
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
