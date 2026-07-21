use std::future::Future;

use futures_util::stream::{FuturesUnordered, StreamExt};

/// Run a batch concurrently and report outcomes in original index order.
pub(crate) async fn schedule_tool_batch<T, O, IndexOf, Run, Fut>(
    items: Vec<T>,
    index_of: IndexOf,
    run: Run,
) -> Vec<O>
where
    T: Send + 'static,
    O: Send + 'static,
    IndexOf: Fn(&T) -> usize,
    Run: Fn(T) -> Fut,
    Fut: Future<Output = O> + Send,
{
    let mut pending = FuturesUnordered::new();
    for item in items {
        let index = index_of(&item);
        let future = run(item);
        pending.push(async move { (index, future.await) });
    }

    let mut outcomes = Vec::new();
    while let Some(outcome) = pending.next().await {
        outcomes.push(outcome);
    }

    outcomes.sort_by_key(|(index, _)| *index);
    outcomes.into_iter().map(|(_, outcome)| outcome).collect()
}
