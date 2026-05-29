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
        let response = run_or_resume_lash_turn(effect_scope, req)
            .await
            .map_err(TerminalError::from_error)?;
        Ok(Json(response))
    }
}
```

The application owns `run_or_resume_lash_turn`: open the `LashSession` from
stable request data, call `session.resume_turn(turn_id).run_with_effect_scope`
when recovering a known in-flight turn, or call
`session.turn(input.with_trace_turn_id(turn_id)).run_with_effect_scope` for a
fresh durable turn.

The adapter journals Lash LLM calls, tool calls, direct completions,
checkpoints, execution-surface syncs, and exec effects with Restate
`ctx.run(...).name(envelope.invocation.replay.key)`, then runs the normal
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
`LashProcessWorkflow`. Bind it on your Restate endpoint with `.serve()`:

```rust,no_run
use std::sync::Arc;

use lash_restate::{
    LashProcessWorkflow, LashProcessWorkflowImpl, RestateCoreProcessRunner,
};
use restate_sdk::prelude::Endpoint;

fn endpoint(
    worker: lash_core::DurableProcessWorker,
    registry: Arc<dyn lash_core::ProcessRegistry>,
) -> restate_sdk::endpoint::Endpoint
{
    let runner = Arc::new(RestateCoreProcessRunner::new(worker));
    Endpoint::builder()
        .bind(LashProcessWorkflowImpl::new(runner, registry).serve())
        .build()
}
```

The controller submits workflow `run` with workflow key
`ProcessRegistration.id` and sends cancellation to the workflow's shared
`cancel` handler. The workflow runner should be built from the host's
deployment config: plugin factories, runtime core config, session-store
factory, process registry, attachment store, provider policy, and host profile.
Process rows carry the process input plus `ProcessProvenance`: owner scope,
host profile id, and optional causal parent. Workers do not parse grant keys to
recover the owner session.
