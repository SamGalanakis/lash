# lash-restate

`lash-restate` adapts Lash's scoped effect-controller boundary to Restate
handlers. Use it inside a Restate service, object, or workflow handler and pass
the resulting `ScopedEffectController` into Lash turn execution.

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
        let response = run_lash_turn(&effect_controller, req)
            .await
            .map_err(TerminalError::from_error)?;
        Ok(Json(response))
    }
}
```

The application owns `run_lash_turn`: open the `LashSession` from stable
request data and call
`session.turn(input).turn_id(turn_id).effects(&controller).run()`
for the Restate-backed turn. Restate recovery is handler replay with the same turn id
and request data, not a Lash-owned in-flight checkpoint reload.

The adapter records Lash LLM calls, tool calls, direct completions,
checkpoints, execution-surface syncs, and exec effects with Restate
`ctx.run(...).name(lash:<replay_key>)`, then runs the normal Lash local
executor inside that recorded block. Runtime sleeps use Restate durable timers.
Substrate-native Restate turns do not use store-side in-flight replay rows; Lash
only commits final session state through its turn-commit idempotency contract.
Replaying a handler with the same turn id returns Restate-recorded effect
outcomes, validates the current Lash envelope hash, and retries the final commit
without exposing partial session state.

For host-tier HTTP integration, use `RestateIngressClient` to submit `/send`
requests and capture the returned invocation id. The client accepts Restate's
`Accepted` and `PreviouslyAccepted` send statuses and returns the
`RestateInvocationId` from the response body, so the host can track a durable
turn invocation instead of modeling it as local in-process work.
`RestateAdminClient` cancels those active invocations through the Admin API,
queries invocation status, and exposes unfinished-invocation introspection for
tests and cleanup. `kill_invocation_for_test_cleanup` is intentionally reserved
for test/dev cleanup after graceful cancel fails. The Restate CLI remains a
useful operator tool, but Lash tests and examples use these HTTP APIs directly.

Deterministic contract failures are terminal handler errors, not retry loops.
If a replayed recorded effect no longer matches the current Lash envelope hash,
or a previously recorded Restate run completed with a terminal failure, the
adapter returns an explicit terminal error code to the handler. Hosts should
surface that failure and clear their running state rather than leaving the
invocation to back off forever.

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
deployment config: plugin factories, runtime host config, session-store
factory, process registry, attachment store, and provider policy.
Process rows carry the process input plus `ProcessProvenance`: originator
and optional causal parent. Tool and Lashlang rows also carry a
captured execution-environment reference, so workers do not parse grant keys or
rebuild origin sessions to recover execution context.
