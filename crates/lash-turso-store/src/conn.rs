/// Outcome a write flow returns to decide commit vs rollback while still
/// handing a value back to the caller.
pub(crate) enum TxOutcome<T> {
    Commit(T),
    Rollback(T),
}
