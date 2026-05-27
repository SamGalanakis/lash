use std::future::Future;
use std::sync::Arc;

use futures_util::stream::{FuturesUnordered, StreamExt};

use crate::{ProgressSender, ToolCallRecord, ToolScheduling};

use super::context::ToolDispatchContext;
use super::preparation::dispatch_tool_call;

#[derive(Clone)]
pub struct ParallelToolCallSpec {
    pub index: usize,
    pub tool_name: String,
    pub args: serde_json::Value,
}

#[derive(Clone)]
pub struct ParallelToolCallOutcome {
    pub index: usize,
    pub record: ToolCallRecord,
}

pub(crate) async fn dispatch_parallel_tool_call(
    context: Arc<ToolDispatchContext<'_>>,
    spec: ParallelToolCallSpec,
    progress: Option<ProgressSender>,
) -> ParallelToolCallOutcome {
    let outcome = dispatch_tool_call(&context, spec.tool_name, spec.args, progress.as_ref()).await;
    ParallelToolCallOutcome {
        index: spec.index,
        record: outcome.record,
    }
}

/// Resolve the [`ToolScheduling`] declared on a tool's definition. Unknown
/// tool names default to [`ToolScheduling::Parallel`] — the dispatcher
/// will still surface an "unknown tool" error via the normal path.
pub(crate) fn resolve_tool_scheduling(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> ToolScheduling {
    context
        .surface
        .tools
        .iter()
        .find(|def| def.manifest.name == tool_name)
        .map(|def| def.manifest.scheduling)
        .unwrap_or_default()
}

/// Schedule a batch using Lash's tool execution policy.
///
/// Parallel-safe tools run concurrently first, then serial tools run
/// one-at-a-time in original index order. Returned outputs are sorted by the
/// same original index so callers keep their source/model ordering.
pub(crate) async fn schedule_tool_batch<T, O, IndexOf, SchedulingOf, Run, Fut>(
    items: Vec<T>,
    index_of: IndexOf,
    scheduling_of: SchedulingOf,
    run: Run,
) -> Vec<O>
where
    T: Send + 'static,
    O: Send + 'static,
    IndexOf: Fn(&T) -> usize,
    SchedulingOf: Fn(&T) -> ToolScheduling,
    Run: Fn(T) -> Fut,
    Fut: Future<Output = O> + Send,
{
    let mut parallel_items = Vec::new();
    let mut serial_items = Vec::new();
    for item in items {
        let index = index_of(&item);
        match scheduling_of(&item) {
            ToolScheduling::Parallel => parallel_items.push((index, item)),
            ToolScheduling::Serial => serial_items.push((index, item)),
        }
    }

    let mut outcomes = Vec::new();

    let mut pending = FuturesUnordered::new();
    for (index, item) in parallel_items {
        let future = run(item);
        pending.push(async move { (index, future.await) });
    }
    while let Some(outcome) = pending.next().await {
        outcomes.push(outcome);
    }

    serial_items.sort_by_key(|(index, _)| *index);
    for (index, item) in serial_items {
        outcomes.push((index, run(item).await));
    }

    outcomes.sort_by_key(|(index, _)| *index);
    outcomes.into_iter().map(|(_, outcome)| outcome).collect()
}

/// Dispatch a batch of tool calls produced by one model response.
pub async fn dispatch_parallel_tool_calls(
    context: Arc<ToolDispatchContext<'_>>,
    specs: Vec<ParallelToolCallSpec>,
    progress: Option<&ProgressSender>,
) -> Vec<ParallelToolCallOutcome> {
    let progress = progress.cloned();
    schedule_tool_batch(
        specs,
        |spec| spec.index,
        {
            let context = Arc::clone(&context);
            move |spec| resolve_tool_scheduling(&context, &spec.tool_name)
        },
        move |spec| dispatch_parallel_tool_call(Arc::clone(&context), spec, progress.clone()),
    )
    .await
}
