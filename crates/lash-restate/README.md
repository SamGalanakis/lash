# lash-restate

`lash-restate` adapts Lash's `RuntimeEffectController` boundary to Restate handlers.
Use it inside a Restate service, object, or workflow handler and pass the
resulting effect scope into Lash turn execution.

```rust,no_run
use lash_restate::RestateRuntimeEffectController;
use restate_sdk::prelude::*;

#[restate_sdk::workflow]
pub trait AgentTurnWorkflow {
    async fn run(req: Json<TurnRequest>) -> HandlerResult<Json<TurnResponse>>;
}

pub struct AgentTurnWorkflowImpl;

impl AgentTurnWorkflow for AgentTurnWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        Json(req): Json<TurnRequest>,
    ) -> HandlerResult<Json<TurnResponse>> {
        let effect_controller = RestateRuntimeEffectController::new(ctx, hooks);
        let turn_id = req.turn_id.clone();
        let effect_scope = effect_controller
            .effect_scope(&turn_id)
            .map_err(TerminalError::from_error)?;

        let output = req
            .session
            .turn(req.input)
            .run_with_effect_scope(effect_scope)
            .await
            .map_err(TerminalError::from_error)?;

        Ok(Json(TurnResponse::from(output)))
    }
}
```

The adapter journals Lash LLM calls, tool calls, direct completions,
checkpoints, execution-surface syncs, and exec effects with Restate `ctx.run`.
Runtime sleeps use Restate durable timers. `RestateRuntimeHooks` receives only
serialized Lash requests and background-task registrations; unsupported effects
must return an explicit host error instead of falling back to local execution.
