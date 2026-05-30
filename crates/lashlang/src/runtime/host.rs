use crate::{ModuleRef, ProcessRef, RequiredSurfaceRef};

use super::{ExecutionScratch, ProfileReport, ProjectedBindings, Record, RuntimeFailure, Value};
use std::future::Future;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Clone, Debug)]
pub enum AbilityOp {
    ResourceOperation(ResourceOperation),
    Await(Value),
    Cancel(Value),
    Print(Value),
    Submit(Value),
    Finish(Value),
    Fail(Value),
    StartProcess(Box<ProcessStart>),
    ProcessEvent(ProcessEvent),
    Sleep(Sleep),
    WaitSignal,
    SignalRun(ProcessSignal),
}

#[derive(Clone, Debug)]
pub enum AbilityResult {
    Value(Value),
    Unit,
}

impl AbilityResult {
    pub fn into_value(self, op: &'static str) -> Result<Value, ExecutionHostError> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Unit => Err(ExecutionHostError::new(format!("{op} returned no value"))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProcessStart {
    pub module_ref: ModuleRef,
    pub process_ref: ProcessRef,
    pub required_surface_ref: RequiredSurfaceRef,
    pub process_name: String,
    pub args: Record,
}

#[derive(Clone, Debug)]
pub struct ResourceOperation {
    pub receiver: Value,
    pub operation: String,
    pub args: Vec<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessEventKind {
    Yield,
    Wake,
}

#[derive(Clone, Debug)]
pub struct ProcessEvent {
    pub kind: ProcessEventKind,
    pub value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepKind {
    For,
    Until,
}

#[derive(Clone, Debug)]
pub struct Sleep {
    pub kind: SleepKind,
    pub value: Value,
}

#[derive(Clone, Debug)]
pub struct ProcessSignal {
    pub run: Value,
    pub payload: Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    #[default]
    Foreground,
    Process,
}

pub trait ExecutionHost: Sync {
    fn perform(
        &self,
        op: AbilityOp,
    ) -> impl Future<Output = Result<AbilityResult, ExecutionHostError>> + Send;

    fn yield_now(&self) -> impl Future<Output = ()> + Send {
        async {}
    }

    fn execution_mode(&self) -> ExecutionMode {
        ExecutionMode::Foreground
    }

    fn projected_bindings(&self) -> ProjectedBindings {
        ProjectedBindings::default()
    }

    fn trace_runtime_errors(&self) -> bool {
        false
    }

    fn profile_execution(&self) -> bool {
        false
    }

    fn take_scratch(&self) -> Option<ExecutionScratch> {
        None
    }

    fn store_scratch(&self, _scratch: ExecutionScratch) {}

    fn observe_runtime_failure(&self, _failure: RuntimeFailure) {}

    fn observe_profile(&self, _profile: ProfileReport) {}
}

pub struct ExecutionEnvironment<'host, H: ExecutionHost> {
    host: &'host H,
    mode: ExecutionMode,
    projected: ProjectedBindings,
    scratch: Mutex<Option<ExecutionScratch>>,
    trace_runtime_errors: bool,
    profile_execution: bool,
    runtime_failure: Mutex<Option<RuntimeFailure>>,
    profile: Mutex<Option<ProfileReport>>,
}

impl<'host, H: ExecutionHost> ExecutionEnvironment<'host, H> {
    pub fn new(host: &'host H) -> Self {
        Self {
            host,
            mode: host.execution_mode(),
            projected: host.projected_bindings(),
            scratch: Mutex::new(host.take_scratch()),
            trace_runtime_errors: host.trace_runtime_errors(),
            profile_execution: host.profile_execution(),
            runtime_failure: Mutex::new(None),
            profile: Mutex::new(None),
        }
    }

    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn process(self) -> Self {
        self.with_mode(ExecutionMode::Process)
    }

    pub fn foreground(self) -> Self {
        self.with_mode(ExecutionMode::Foreground)
    }

    pub fn with_projected_bindings(mut self, projected: ProjectedBindings) -> Self {
        self.projected = projected;
        self
    }

    pub fn with_scratch(mut self, scratch: ExecutionScratch) -> Self {
        self.scratch = Mutex::new(Some(scratch));
        self
    }

    pub fn traced(mut self) -> Self {
        self.trace_runtime_errors = true;
        self
    }

    pub fn profiled(mut self) -> Self {
        self.profile_execution = true;
        self
    }

    pub fn take_runtime_failure(&self) -> Option<RuntimeFailure> {
        self.runtime_failure.lock().ok()?.take()
    }

    pub fn take_profile(&self) -> Option<ProfileReport> {
        self.profile.lock().ok()?.take()
    }

    pub fn take_recycled_scratch(&self) -> Option<ExecutionScratch> {
        self.scratch.lock().ok()?.take()
    }
}

impl<H: ExecutionHost> ExecutionHost for ExecutionEnvironment<'_, H> {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        self.host.perform(op).await
    }

    async fn yield_now(&self) {
        self.host.yield_now().await;
    }

    fn execution_mode(&self) -> ExecutionMode {
        self.mode
    }

    fn projected_bindings(&self) -> ProjectedBindings {
        self.projected.clone()
    }

    fn trace_runtime_errors(&self) -> bool {
        self.trace_runtime_errors
    }

    fn profile_execution(&self) -> bool {
        self.profile_execution
    }

    fn take_scratch(&self) -> Option<ExecutionScratch> {
        self.scratch.lock().ok()?.take()
    }

    fn store_scratch(&self, scratch: ExecutionScratch) {
        if let Ok(mut guard) = self.scratch.lock() {
            *guard = Some(scratch);
        }
    }

    fn observe_runtime_failure(&self, failure: RuntimeFailure) {
        self.host.observe_runtime_failure(failure.clone());
        if let Ok(mut guard) = self.runtime_failure.lock() {
            *guard = Some(failure);
        }
    }

    fn observe_profile(&self, profile: ProfileReport) {
        self.host.observe_profile(profile.clone());
        if let Ok(mut guard) = self.profile.lock() {
            *guard = Some(profile);
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ExecutionHostError {
    message: String,
}

impl ExecutionHostError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
