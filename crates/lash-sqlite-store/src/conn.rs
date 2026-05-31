//! Connection-access helpers that collapse the lock + transaction + commit +
//! error ceremony repeated across the store and the process registry.
//!
//! Two concerns are centralised here:
//!
//! * **Poison recovery.** A panic while a `Mutex<Connection>` guard is held
//!   would otherwise poison the lock and brick the store for the rest of the
//!   session. [`lock_conn`] recovers the guard via [`PoisonError::into_inner`]
//!   so a single panic can't permanently take the connection out of service.
//! * **`IMMEDIATE` write transactions.** rusqlite's `Connection::transaction`
//!   opens `BEGIN DEFERRED`, which only takes a write lock on the first write
//!   statement. A read-then-write path (head-revision CAS, lease fencing, the
//!   queued-work claim) running under `DEFERRED` lets two connections both run
//!   the read phase under a shared snapshot and only serialize at write time —
//!   defeating the cross-process CAS/lease guard the crate promises. Every
//!   read-then-write path therefore goes through [`Store::with_write_tx`] /
//!   [`SqliteProcessRegistry::with_write_tx`], which open
//!   `BEGIN IMMEDIATE` so the write lock is taken up front and a contending
//!   writer fails fast (or waits on the busy timeout) instead of reading a
//!   stale snapshot.

use std::sync::{Mutex, MutexGuard};

use rusqlite::{Connection, Transaction, TransactionBehavior};

use super::{StoreError, process_sqlite_error, sqlite_error};

/// Outcome a `with_write_tx_flow` closure returns to decide commit vs.
/// rollback while still handing a value back to the caller. Used for paths
/// that compute a result *and* may discover mid-transaction that the work must
/// not be persisted (e.g. a contended queued-work claim, where partially
/// claimed rows must be rolled back and the caller told nothing was claimed).
pub(crate) enum TxOutcome<T> {
    Commit(T),
    Rollback(T),
}

/// Lock a `Mutex<Connection>`, recovering the guard if the lock was poisoned by
/// a previous panic. Poisoning is not a durable-data problem here: the
/// connection itself remains valid, so surfacing the data and letting the next
/// transaction proceed is strictly better than panicking forever.
pub(crate) fn lock_conn(conn: &Mutex<Connection>) -> MutexGuard<'_, Connection> {
    conn.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl super::Store {
    /// Run `f` inside a `BEGIN IMMEDIATE` transaction, committing on `Ok` and
    /// rolling back (via drop) on `Err`. Use this for every read-then-write
    /// path so the write lock is acquired before the read phase.
    pub(crate) fn with_write_tx<T>(
        &self,
        f: impl FnOnce(&Transaction<'_>) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        let mut conn = lock_conn(&self.conn);
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sqlite_error)?;
        let value = f(&tx)?;
        tx.commit().map_err(sqlite_error)?;
        Ok(value)
    }

    /// Like [`with_write_tx`](Self::with_write_tx) but the closure decides
    /// whether to commit or roll back via [`TxOutcome`], in either case still
    /// returning a value. Lets a path keep transactional atomicity when it
    /// must abandon partially-applied writes.
    pub(crate) fn with_write_tx_flow<T>(
        &self,
        f: impl FnOnce(&Transaction<'_>) -> Result<TxOutcome<T>, StoreError>,
    ) -> Result<T, StoreError> {
        let mut conn = lock_conn(&self.conn);
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sqlite_error)?;
        match f(&tx)? {
            TxOutcome::Commit(value) => {
                tx.commit().map_err(sqlite_error)?;
                Ok(value)
            }
            TxOutcome::Rollback(value) => {
                tx.rollback().map_err(sqlite_error)?;
                Ok(value)
            }
        }
    }

    /// Run `f` against a shared connection for a read-only path. Stays
    /// `DEFERRED` (no explicit transaction) because read-only access does not
    /// need the up-front write lock.
    pub(crate) fn with_read<T>(
        &self,
        f: impl FnOnce(&Connection) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        let conn = lock_conn(&self.conn);
        f(&conn)
    }
}

impl super::SqliteProcessRegistry {
    /// Process-registry analogue of [`Store::with_write_tx`]. Opens
    /// `BEGIN IMMEDIATE` and surfaces failures on the `PluginError` channel the
    /// `ProcessRegistry` trait returns.
    pub(crate) fn with_write_tx<T>(
        &self,
        f: impl FnOnce(&Transaction<'_>) -> Result<T, lash_core::PluginError>,
    ) -> Result<T, lash_core::PluginError> {
        let mut conn = lock_conn(&self.conn);
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(process_sqlite_error)?;
        let value = f(&tx)?;
        tx.commit().map_err(process_sqlite_error)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Store;

    // Finding 6: a panic while a connection-mutex guard is held poisons the
    // lock. `lock_conn` must recover the guard via `into_inner` so a single
    // panic doesn't permanently brick the store. We poison the *real* private
    // `conn` mutex here (only reachable from inside the crate) and assert the
    // store keeps working through both `lock_conn` and a higher-level helper.
    #[test]
    fn lock_conn_recovers_a_poisoned_connection_mutex() {
        let store = Store::memory().expect("store");

        // Poison the mutex: hold a guard across a panic on a separate thread.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = store.conn.lock().expect("first lock is not poisoned");
            panic!("simulated panic while holding the connection guard");
        }))
        .is_err();
        assert!(panicked, "the guarded closure must have panicked");
        assert!(store.conn.lock().is_err(), "the mutex must now be poisoned");

        // Recovery: `lock_conn` returns a usable guard despite the poison, and
        // the write helper runs a transaction on it.
        {
            let conn = lock_conn(&store.conn);
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM blobs", [], |row| row.get(0))
                .expect("query on recovered guard");
            assert_eq!(count, 0);
        }
        store
            .with_write_tx(|tx| {
                tx.execute(
                    "INSERT OR IGNORE INTO blobs (hash, content) VALUES ('h', x'00')",
                    [],
                )
                .map_err(sqlite_error)?;
                Ok(())
            })
            .expect("write helper works after poison recovery");
    }
}
