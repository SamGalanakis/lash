//! [`AttachmentStore`] conformance: content addressing and round trips.

use super::*;

/// Run the full [`AttachmentStore`] conformance suite against the backend
/// produced by `make`. `make` must return a fresh, empty store on each call.
/// `expected_persistence` is the tier this backend declares (`Ephemeral` for
/// in-memory, `Durable` for file/Sqlite-backed).
pub async fn attachment_store<F>(make: F, expected_persistence: AttachmentStorePersistence)
where
    F: Fn() -> Arc<dyn AttachmentStore>,
{
    attachment_put_get_round_trips_bytes_and_meta(make()).await;
    attachment_is_content_addressed(make()).await;
    attachment_get_unknown_is_not_found(make()).await;
    attachment_reports_declared_persistence(make(), expected_persistence);
}

/// Run the full [`AttachmentStore`] suite plus durable reopen checks.
pub async fn attachment_store_reopenable<F>(
    make: F,
    expected_persistence: AttachmentStorePersistence,
) where
    F: Fn() -> ReopenableAttachmentStore,
{
    attachment_store(|| make().open, expected_persistence).await;
    attachment_store_survives_reopen(make()).await;
}

fn attachment_meta() -> AttachmentCreateMeta {
    AttachmentCreateMeta::new(
        MediaType::Image(ImageMediaType::Png),
        Some(7),
        Some(11),
        Some("pixel".to_string()),
    )
}

async fn attachment_put_get_round_trips_bytes_and_meta(store: Arc<dyn AttachmentStore>) {
    let bytes = vec![1u8, 2, 3, 4, 5];
    let reference = store
        .put(bytes.clone(), attachment_meta())
        .await
        .expect("put attachment");
    let stored = store.get(&reference.id).await.expect("get attachment");

    assert_eq!(stored.bytes, bytes, "bytes must round-trip unchanged");
    assert_eq!(stored.meta.id, reference.id);
    assert_eq!(stored.meta.byte_len, bytes.len() as u64);
    assert_eq!(
        stored.meta.media_type,
        MediaType::Image(ImageMediaType::Png)
    );
    assert_eq!(stored.meta.width, Some(7));
    assert_eq!(stored.meta.height, Some(11));
    assert_eq!(stored.meta.label.as_deref(), Some("pixel"));
}

async fn attachment_is_content_addressed(store: Arc<dyn AttachmentStore>) {
    let first = store
        .put(vec![9u8, 9, 9], attachment_meta())
        .await
        .expect("put first");
    let same = store
        .put(vec![9u8, 9, 9], attachment_meta())
        .await
        .expect("put identical bytes");
    let different = store
        .put(vec![9u8, 9, 8], attachment_meta())
        .await
        .expect("put different bytes");

    assert_eq!(
        first.id, same.id,
        "identical bytes must map to the same content-addressed id"
    );
    assert_ne!(
        first.id, different.id,
        "different bytes must map to different ids"
    );
}

async fn attachment_get_unknown_is_not_found(store: Arc<dyn AttachmentStore>) {
    let err = store
        .get(&AttachmentId::new("sha256:does-not-exist"))
        .await
        .expect_err("get of an unknown id must fail");
    assert!(
        matches!(err, AttachmentStoreError::NotFound(_)),
        "unknown id must map to NotFound, got {err:?}"
    );
}

fn attachment_reports_declared_persistence(
    store: Arc<dyn AttachmentStore>,
    expected: AttachmentStorePersistence,
) {
    assert_eq!(
        store.persistence(),
        expected,
        "persistence tier must match the backend's declared durability"
    );
}

async fn attachment_store_survives_reopen(factory: ReopenableAttachmentStore) {
    let reference = factory
        .open
        .put(vec![4u8, 3, 2, 1], attachment_meta())
        .await
        .expect("put attachment before reopen");
    let reopened = factory
        .reopen
        .get(&reference.id)
        .await
        .expect("get attachment after reopen");
    assert_eq!(reopened.bytes, vec![4u8, 3, 2, 1]);
    assert_eq!(reopened.meta.id, reference.id);
    assert_eq!(reopened.meta.byte_len, 4);
}
