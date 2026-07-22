use super::process_registry::registration;
use super::*;

pub(super) async fn list_processes_filters_by_enriched_fields(registry: Arc<dyn ProcessRegistry>) {
    async fn filtered_ids(
        registry: &Arc<dyn ProcessRegistry>,
        filter: ProcessListFilter,
    ) -> Vec<String> {
        registry
            .list_processes(&filter)
            .await
            .expect("list processes")
            .into_iter()
            .map(|record| record.id)
            .collect()
    }

    let scope = SessionScope::for_agent_frame("filter-session", "filter-frame");
    let scope_id = scope.id().to_string();
    let target = registry
        .register_process(
            registration("proc-filter-target")
                .with_identity(ProcessIdentity::new("filter-kind").with_label(Some("target-label")))
                .with_process_provenance(ProcessProvenance::session(scope).with_caused_by(Some(
                    CausalRef::TriggerOccurrence {
                        occurrence_id: "occurrence-target".to_string(),
                        subscription_id: Some("subscription-target".to_string()),
                        subscription_incarnation: None,
                        subscription_revision: None,
                    },
                ))),
        )
        .await
        .expect("register target");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    registry
        .register_process(
            registration("proc-filter-other")
                .with_identity(ProcessIdentity::new("other-kind").with_label(Some("other-label"))),
        )
        .await
        .expect("register other");

    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                originator_scope_id: Some(scope_id),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()]
    );
    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                identity_kind: Some("filter-kind".to_string()),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()]
    );
    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                identity_label: Some("target-label".to_string()),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()]
    );
    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                caused_by_occurrence_id: Some("occurrence-target".to_string()),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()]
    );
    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                caused_by_subscription_id: Some("subscription-target".to_string()),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()]
    );
    assert_eq!(
        filtered_ids(
            &registry,
            ProcessListFilter {
                status: ProcessStatusFilter::Any,
                created_at_start_ms: Some(target.created_at_ms),
                created_at_end_ms: Some(target.created_at_ms.saturating_add(1)),
                ..ProcessListFilter::default()
            }
        )
        .await,
        vec!["proc-filter-target".to_string()],
        "created-at range is start-inclusive and end-exclusive"
    );
}
