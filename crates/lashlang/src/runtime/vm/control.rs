use std::time::Instant;

use super::super::{
    COOPERATIVE_YIELD_INSTRUCTION_BUDGET, ExecutionHost, ExecutionMode, ExecutionOutcome,
    RuntimeError, RuntimeFailure, Value,
};
use super::effects::VmEffect;
use super::{Vm, VmRunOutcome};
use crate::lexer::Span;

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

impl From<VmMode> for ExecutionMode {
    fn from(mode: VmMode) -> Self {
        match mode {
            VmMode::Foreground => Self::Foreground,
            VmMode::Process => Self::Process,
        }
    }
}

pub(super) enum VmOutcome {
    Continued,
    EffectCompleted,
    Finished(Value),
    ProcessFinished(Value),
    ProcessFailed(Value),
    #[cfg(test)]
    Suspended,
}

struct VmTrap {
    error: RuntimeError,
    instruction_ip: usize,
    span: Option<Span>,
}

impl<H: ExecutionHost> Vm<'_, H> {
    pub(crate) async fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.run_raw().await? {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::EffectCompleted => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            #[cfg(test)]
            VmOutcome::Suspended => Ok(ExecutionOutcome::Continued),
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
            VmOutcome::EffectCompleted => Ok(ExecutionOutcome::Continued),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            #[cfg(test)]
            VmOutcome::Suspended => Ok(ExecutionOutcome::Continued),
        }
    }

    pub async fn run_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        match self.mode {
            VmMode::Foreground => self.run().await,
            VmMode::Process => self.run_process().await,
        }
    }

    pub async fn run_process_until_effect(&mut self) -> Result<VmRunOutcome, RuntimeError> {
        match self.run_raw_until_effect().await? {
            VmOutcome::EffectCompleted => Ok(VmRunOutcome::EffectCompleted),
            VmOutcome::Continued => Ok(VmRunOutcome::Complete(ExecutionOutcome::Finished(
                Value::Null,
            ))),
            VmOutcome::ProcessFinished(value) | VmOutcome::Finished(value) => {
                Ok(VmRunOutcome::Complete(ExecutionOutcome::Finished(value)))
            }
            VmOutcome::ProcessFailed(value) => {
                Ok(VmRunOutcome::Complete(ExecutionOutcome::Failed(value)))
            }
            #[cfg(test)]
            VmOutcome::Suspended => Ok(VmRunOutcome::EffectCompleted),
        }
    }

    pub async fn run_process_traced_until_effect(
        &mut self,
    ) -> Result<VmRunOutcome, RuntimeFailure> {
        match self.run_raw_traced_until_effect().await? {
            VmOutcome::EffectCompleted => Ok(VmRunOutcome::EffectCompleted),
            VmOutcome::Continued => Ok(VmRunOutcome::Complete(ExecutionOutcome::Finished(
                Value::Null,
            ))),
            VmOutcome::ProcessFinished(value) | VmOutcome::Finished(value) => {
                Ok(VmRunOutcome::Complete(ExecutionOutcome::Finished(value)))
            }
            VmOutcome::ProcessFailed(value) => {
                Ok(VmRunOutcome::Complete(ExecutionOutcome::Failed(value)))
            }
            #[cfg(test)]
            VmOutcome::Suspended => Ok(VmRunOutcome::EffectCompleted),
        }
    }

    pub(crate) async fn run_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_raw_traced().await?;
        match result {
            VmOutcome::Continued => Ok(ExecutionOutcome::Continued),
            VmOutcome::EffectCompleted => Ok(ExecutionOutcome::Continued),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            #[cfg(test)]
            VmOutcome::Suspended => Ok(ExecutionOutcome::Continued),
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
            VmOutcome::EffectCompleted => Ok(ExecutionOutcome::Continued),
            VmOutcome::ProcessFinished(value) => Ok(ExecutionOutcome::Finished(value)),
            VmOutcome::ProcessFailed(value) => Ok(ExecutionOutcome::Failed(value)),
            VmOutcome::Finished(value) => Ok(ExecutionOutcome::Finished(value)),
            #[cfg(test)]
            VmOutcome::Suspended => Ok(ExecutionOutcome::Continued),
        }
    }

    pub(crate) async fn run_traced_for_mode(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        match self.mode {
            VmMode::Foreground => self.run_traced().await,
            VmMode::Process => self.run_process_traced().await,
        }
    }

    async fn run_raw(&mut self) -> Result<VmOutcome, RuntimeError> {
        let result = self.run_loop(false).await.map_err(|trap| trap.error);
        #[cfg(test)]
        let suspended = matches!(result, Ok(VmOutcome::Suspended));
        #[cfg(not(test))]
        let suspended = false;
        if !suspended {
            self.unwind_iterators();
        }
        result
    }

    async fn run_raw_until_effect(&mut self) -> Result<VmOutcome, RuntimeError> {
        self.run_loop(true).await.map_err(|trap| trap.error)
    }

    async fn run_raw_traced(&mut self) -> Result<VmOutcome, RuntimeFailure> {
        let result = self.run_loop(false).await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: trap
                .span
                .or_else(|| self.chunk.spans.get(trap.instruction_ip).copied().flatten()),
        });
        #[cfg(test)]
        let suspended = matches!(result, Ok(VmOutcome::Suspended));
        #[cfg(not(test))]
        let suspended = false;
        if !suspended {
            self.unwind_iterators();
        }
        result
    }

    async fn run_raw_traced_until_effect(&mut self) -> Result<VmOutcome, RuntimeFailure> {
        self.run_loop(true).await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: trap
                .span
                .or_else(|| self.chunk.spans.get(trap.instruction_ip).copied().flatten()),
        })
    }

    async fn run_loop(&mut self, stop_after_effect: bool) -> Result<VmOutcome, VmTrap> {
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
            let completed_effect = matches!(&step, Ok(VmStep::Effect(_)));
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
                    let span = self.pending_error_span.take();
                    return Err(VmTrap {
                        error,
                        instruction_ip,
                        span,
                    });
                }
            }
            if stop_after_effect && completed_effect {
                return Ok(VmOutcome::EffectCompleted);
            }
            #[cfg(test)]
            if self.test_suspension.should_suspend(completed_effect) {
                return Ok(VmOutcome::Suspended);
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
