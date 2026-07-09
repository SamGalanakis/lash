const RUNTIME_PERSISTENCE: &str = include_str!("../src/postgres/runtime_persistence.rs");

fn source_region(start: &str, end: &str) -> &'static str {
    let start_index = RUNTIME_PERSISTENCE
        .find(start)
        .unwrap_or_else(|| panic!("missing source marker `{start}`"));
    let region = &RUNTIME_PERSISTENCE[start_index..];
    let end_index = region
        .find(end)
        .unwrap_or_else(|| panic!("missing source marker `{end}` after `{start}`"));
    &region[..end_index]
}

#[test]
fn queued_work_and_pending_input_lease_decisions_use_the_postgres_clock() {
    let lease_sensitive_regions = [
        (
            "async fn claim_leading_ready_session_command(",
            "async fn claim_ready_queued_work(",
        ),
        (
            "async fn claim_ready_queued_work(",
            "async fn renew_queued_work_claim(",
        ),
        (
            "async fn renew_queued_work_claim(",
            "async fn abandon_queued_work_claim(",
        ),
        (
            "async fn cancel_queued_work_batch(",
            "async fn list_queued_work(",
        ),
        (
            "async fn list_pending_queued_work(",
            "impl TurnInputStore for PostgresSessionStore",
        ),
        (
            "async fn list_pending_turn_inputs(",
            "async fn cancel_pending_turn_inputs(",
        ),
        (
            "async fn cancel_pending_turn_inputs(",
            "async fn cancel_pending_turn_input_suffix(",
        ),
        (
            "async fn cancel_pending_turn_input_suffix(",
            "async fn claim_active_turn_inputs(",
        ),
        (
            "async fn claim_pending_turn_inputs_postgres(",
            "struct SessionExecutionLeaseRow",
        ),
    ];

    for (start, end) in lease_sensitive_regions {
        let region = source_region(start, end);
        assert!(
            region.contains("postgres_transaction_epoch_ms"),
            "`{start}` must derive lease validity from PostgreSQL"
        );
        assert!(
            !region.contains("current_epoch_ms()"),
            "`{start}` must not derive lease validity from the client wall clock"
        );
    }
}
