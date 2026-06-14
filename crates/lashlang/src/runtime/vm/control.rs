use std::time::Instant;

use super::super::{
    COOPERATIVE_YIELD_INSTRUCTION_BUDGET, ExecutionHost, ExecutionMode, ExecutionOutcome,
    RuntimeError, RuntimeFailure, Value,
};
use super::Vm;
use super::effects::VmEffect;

pub(super) enum VmStep {
    Continue,
    Effect(VmEffect),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VmMode {
    Foreground,
    Process,
}

impl From<ExecutionMode> for VmMode {
    fn from(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::Foreground => Self::Foreground,
            ExecutionMode::Process => Self::Process,
        }
    }
}

pub(super) enum VmOutcome {
    Continued,
    Finished(Value),
    ProcessFinished(Value),
    ProcessFailed(Value),
}

struct VmTrap {
    error: RuntimeError,
    instruction_ip: usize,
}

impl<H: ExecutionHost> Vm<'_, H> {
    pub(crate) async fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.run_raw().await? {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFinished(_) => {
                Err(RuntimeError::SessionProcessAdminOutsideProcess { keyword: "finish" })
            }
            VmOutcome::ProcessFailed(_) => {
                Err(RuntimeError::SessionProcessAdminOutsideProcess { keyword: "fail" })
            }
        }
    }

    pub(crate) async fn run_process(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.run_raw().await? {
            VmOutcome::Continued => Ok(ExecutionOutcome::Finished(Value::Null)),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(_) => {
                Err(RuntimeError::ForegroundControlInsideProcess { keyword: "submit" })
            }
        }
    }

    pub(crate) async fn run_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.mode {
            VmMode::Foreground => self.run().await,
            VmMode::Process => self.run_process().await,
        }
    }

    pub(crate) async fn run_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_raw_traced().await?;
        match result {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFinished(_) => Err(RuntimeFailure {
                error: RuntimeError::SessionProcessAdminOutsideProcess { keyword: "finish" },
                span: None,
            }),
            VmOutcome::ProcessFailed(_) => Err(RuntimeFailure {
                error: RuntimeError::SessionProcessAdminOutsideProcess { keyword: "fail" },
                span: None,
            }),
        }
    }

    pub(crate) async fn run_process_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_raw_traced().await?;
        match result {
            VmOutcome::Continued => Ok(ExecutionOutcome::Finished(Value::Null)),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(_) => Err(RuntimeFailure {
                error: RuntimeError::ForegroundControlInsideProcess { keyword: "submit" },
                span: None,
            }),
        }
    }

    pub(crate) async fn run_traced_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        match self.mode {
            VmMode::Foreground => self.run_traced().await,
            VmMode::Process => self.run_process_traced().await,
        }
    }

    async fn run_raw(&mut self) -> Result<VmOutcome, RuntimeError> {
        let result = self.run_loop().await.map_err(|trap| trap.error);
        self.unwind_iterators();
        result
    }

    async fn run_raw_traced(&mut self) -> Result<VmOutcome, RuntimeFailure> {
        let result = self.run_loop().await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: self.chunk.spans.get(trap.instruction_ip).copied().flatten(),
        });
        self.unwind_iterators();
        result
    }

    async fn run_loop(&mut self) -> Result<VmOutcome, VmTrap> {
        let mut budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
        while let Some(instruction) = self.chunk.code.get(self.ip).copied() {
            let instruction_ip = self.ip;
            self.ip += 1;
            let profile = self
                .profile
                .as_ref()
                .map(|_| (instruction.profile_tag(), Instant::now()));
            let step = match self.step_instruction_fast(instruction) {
                Ok(Some(step)) => Ok(step),
                Ok(None) => Box::pin(self.step_instruction(instruction)).await,
                Err(error) => Err(error),
            };
            let result = match step {
                Ok(VmStep::Continue) => Ok(None),
                Ok(VmStep::Effect(effect)) => self.resolve_effect(effect, instruction_ip).await,
                Err(error) => Err(error),
            };
            if let Some((tag, start)) = profile {
                self.record_instruction_profile(tag, start.elapsed().as_nanos());
            }
            match result {
                Ok(Some(outcome)) => return Ok(outcome),
                Ok(None) => {}
                Err(error) => {
                    return Err(VmTrap {
                        error,
                        instruction_ip,
                    });
                }
            }
            budget -= 1;
            if budget == 0 {
                self.host.yield_now().await;
                budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
            }
        }
        Ok(VmOutcome::Continued)
    }
}
