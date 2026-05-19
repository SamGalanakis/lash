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
        let effect_controller = RestateRuntimeEffectController::new(ctx);
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
`ctx.run(...).name(envelope.metadata.idempotency_key)`, then runs the normal
Lash local executor inside that journaled block. Runtime sleeps use Restate
durable timers. Lash also saves the in-flight `RuntimeTurnCheckpoint` and
runtime effect journal in its configured `RuntimePersistence` under a
`RuntimeTurnLease`. The runtime renews the same lease token before checkpoint
and journal writes, abandons ownership on controller/runtime errors while
preserving resume data, and clears checkpoint plus journal rows only through a
lease-fenced final commit. A final commit with a missing, expired, or
superseded lease fails before mutating graph, checkpoint, usage, or in-flight
turn rows.

After worker movement or process loss, call `resume_turn(turn_id)` with the
same effect scope to claim ownership again, reload the checkpoint, and replay
recorded outcomes.

Background tasks are scheduled through the first-party
`LashBackgroundTaskWorkflow`. Bind it on your Restate endpoint with `.serve()`:

```rust,no_run
use std::sync::Arc;

use lash_restate::{
    LashBackgroundTaskWorkflow, LashBackgroundTaskWorkflowImpl, RestateBackgroundTaskRunner,
};
use restate_sdk::prelude::Endpoint;

fn endpoint<R>(runner: Arc<R>) -> restate_sdk::endpoint::Endpoint
where
    R: RestateBackgroundTaskRunner,
{
    Endpoint::builder()
        .bind(LashBackgroundTaskWorkflowImpl::new(runner).serve())
        .build()
}
```

The controller submits workflow `run` with workflow key
`BackgroundTaskRegistration.id` and sends cancellation to the workflow's shared
`cancel` handler.
