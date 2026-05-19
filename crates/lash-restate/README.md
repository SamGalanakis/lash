# lash-restate

`lash-restate` adapts Lash's `RuntimeEffectController` boundary to Restate handlers.
Use it inside a Restate service, object, or workflow handler and pass the
resulting effect scope into Lash turn execution or turn resume.

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

        let output = if req.resume {
            req.session
                .resume_turn(&turn_id)
                .run_with_effect_scope(effect_scope)
                .await
        } else {
            req.session
                .turn(req.input.with_trace_turn_id(turn_id))
                .run_with_effect_scope(effect_scope)
                .await
        }
        .map_err(TerminalError::from_error)?;

        Ok(Json(TurnResponse::from(output)))
    }
}
```

The adapter journals Lash LLM calls, tool calls, direct completions,
checkpoints, execution-surface syncs, and exec effects with Restate
`ctx.run(...).name(envelope.metadata.idempotency_key)`. Runtime sleeps use
Restate durable timers. Lash also saves the in-flight `RuntimeTurnCheckpoint`
and runtime effect journal in its configured `RuntimePersistence` under a
`RuntimeTurnLease`. The runtime renews the same lease token before checkpoint
and journal writes, abandons ownership on controller/runtime errors while
preserving resume data, and clears checkpoint plus journal rows only through
the final commit path. After worker movement or process loss, call
`resume_turn(turn_id)` with the same effect scope to claim ownership again,
reload the checkpoint, and replay recorded outcomes. `RestateRuntimeHooks`
receives only serialized Lash requests and background-task registrations;
unsupported effects must return an explicit host error instead of falling back
to local execution.
