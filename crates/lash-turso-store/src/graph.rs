//! Session-graph persistence and garbage collection on [`Store`].

use super::*;

impl Store {
    pub(crate) async fn load_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> lash_core::SessionGraph {
        let rows = match collect_rows(
            conn,
            "SELECT node_json FROM graph_nodes WHERE tombstoned = 0 ORDER BY seq ASC",
            (),
        )
        .await
        {
            Ok(rows) => rows,
            Err(err) => {
                tracing::warn!(error = %err, "failed to query graph rows");
                return lash_core::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let nodes = rows
            .into_iter()
            .filter_map(|row| {
                let node_json = row_string(&row, 0).ok()?;
                serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json).ok()
            })
            .collect();
        lash_core::SessionGraph::from_nodes(nodes, leaf_node_id)
    }

    pub(crate) async fn load_active_path_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> turso::Result<lash_core::SessionGraph> {
        let Some(leaf_node_id) = leaf_node_id else {
            return Ok(lash_core::SessionGraph::default());
        };
        let mut nodes = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut current = Some(leaf_node_id.clone());
        while let Some(node_id) = current {
            if !seen.insert(node_id.clone()) {
                break;
            }
            let Some(row) = optional_row(
                conn,
                "SELECT node_json FROM graph_nodes WHERE node_id = ?1 AND tombstoned = 0",
                params![node_id.as_str()],
            )
            .await?
            else {
                break;
            };
            let node_json = row_string(&row, 0)?;
            let Ok(node) = serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json) else {
                break;
            };
            current = node.parent_node_id.clone();
            nodes.push(node);
        }
        nodes.reverse();
        Ok(lash_core::SessionGraph::from_nodes(
            nodes,
            Some(leaf_node_id),
        ))
    }

    pub(crate) async fn maybe_auto_gc(&self) {
        let Some(interval) = self.options.gc_policy.auto_run_every_commits else {
            return;
        };
        let commits = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if interval != 0 && commits % interval == 0 {
            let _ = self.gc_unreachable().await;
        }
    }

    pub async fn replace_session_graph(&self, graph: &lash_core::SessionGraph) {
        let conn = self.conn.lock().await;
        if let Err(err) = conn.execute("BEGIN IMMEDIATE", ()).await {
            tracing::warn!(error = %err, "failed to begin graph replacement");
            return;
        }
        let result = async {
            conn.execute("DELETE FROM graph_nodes", ()).await?;
            for node in &graph.nodes {
                let node_json = encode_json(node);
                conn.execute(
                    "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                    params![node.node_id.clone(), node_json],
                )
                .await?;
            }
            Ok::<(), turso::Error>(())
        }
        .await;
        match result {
            Ok(()) => {
                if let Err(err) = conn.execute("COMMIT", ()).await {
                    tracing::warn!(error = %err, "failed to commit session graph replacement");
                }
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                tracing::warn!(error = %err, "failed to replace session graph");
            }
        }
    }

    pub async fn append_session_graph_nodes(&self, nodes: &[lash_core::SessionNodeRecord]) {
        if nodes.is_empty() {
            return;
        }
        let conn = self.conn.lock().await;
        if let Err(err) = conn.execute("BEGIN IMMEDIATE", ()).await {
            tracing::warn!(error = %err, "failed to begin graph append");
            return;
        }
        let result = async {
            for node in nodes {
                let node_json = encode_json(node);
                conn.execute(
                    "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                    params![node.node_id.clone(), node_json],
                )
                .await?;
            }
            Ok::<(), turso::Error>(())
        }
        .await;
        match result {
            Ok(()) => {
                if let Err(err) = conn.execute("COMMIT", ()).await {
                    tracing::warn!(error = %err, "failed to commit session graph append");
                }
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                tracing::warn!(error = %err, "failed to append session graph nodes");
            }
        }
    }

    pub async fn load_session_graph(&self) -> lash_core::SessionGraph {
        let conn = self.conn.lock().await;
        Self::load_session_graph_from_conn(&conn, None).await
    }

    pub async fn gc_unreachable(&self) -> GcReport {
        match self.try_gc_unreachable().await {
            Ok(report) => report,
            Err(err) => {
                tracing::warn!(error = %err, "gc_unreachable failed; retaining all blobs");
                GcReport::default()
            }
        }
    }

    async fn live_checkpoint_roots(
        conn: &Connection,
    ) -> Result<Vec<RetainedArtifactRef>, StoreError> {
        let mut roots = Vec::new();
        if let Some(checkpoint_ref) = load_session_head_meta_from_conn(conn)
            .await
            .as_ref()
            .and_then(|meta| meta.checkpoint_ref.as_ref())
            .cloned()
        {
            roots.push(RetainedArtifactRef {
                blob_ref: checkpoint_ref,
                kind: PersistedArtifactKind::CheckpointManifest,
            });
        }
        Ok(roots)
    }

    async fn try_gc_unreachable(&self) -> Result<GcReport, StoreError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(turso_error)?;
        let result = async {
            let mut roots = Self::live_checkpoint_roots(&conn).await?;
            for row in collect_rows(
                &conn,
                "SELECT blob_ref FROM artifact_refs ORDER BY artifact_ref",
                (),
            )
            .await
            .map_err(turso_error)?
            {
                roots.push(RetainedArtifactRef {
                    blob_ref: BlobRef(row_string(&row, 0).map_err(turso_error)?),
                    kind: PersistedArtifactKind::LashlangModule,
                });
            }

            let root_count = roots.len();
            let mut retained = std::collections::BTreeMap::<String, PersistedArtifactKind>::new();
            let mut stack = roots;
            while let Some(current) = stack.pop() {
                if retained
                    .insert(current.blob_ref.0.clone(), current.kind)
                    .is_some()
                {
                    continue;
                }
                if current.kind != PersistedArtifactKind::CheckpointManifest {
                    continue;
                }
                let bytes = optional_row(
                    &conn,
                    "SELECT content FROM blobs WHERE hash = ?1",
                    params![current.blob_ref.as_str()],
                )
                .await
                .map_err(turso_error)?
                .map(|row| row_blob(&row, 0).map_err(turso_error))
                .transpose()?;
                let Some(bytes) = bytes else {
                    continue;
                };
                let content = decode_artifact_blob(&bytes).unwrap_or(bytes);
                let checkpoint = decode_checkpoint(&content).ok_or_else(|| {
                    StoreError::Backend(format!(
                        "gc: live checkpoint manifest `{}` could not be decoded",
                        current.blob_ref
                    ))
                })?;
                stack.extend(retained_artifact_refs(&checkpoint));
            }

            let all_hashes = collect_rows(&conn, "SELECT hash FROM blobs ORDER BY hash ASC", ())
                .await
                .map_err(turso_error)?
                .into_iter()
                .map(|row| row_string(&row, 0).map_err(turso_error))
                .collect::<Result<Vec<_>, _>>()?;
            let mut deleted_blob_count = 0usize;
            for hash in &all_hashes {
                if retained.contains_key(hash) {
                    continue;
                }
                conn.execute("DELETE FROM blobs WHERE hash = ?1", params![hash.as_str()])
                    .await
                    .map_err(turso_error)?;
                deleted_blob_count += 1;
            }
            Ok(GcReport {
                root_count,
                retained_blob_count: retained.len(),
                deleted_blob_count,
            })
        }
        .await;
        match result {
            Ok(report) => {
                conn.execute("COMMIT", ()).await.map_err(turso_error)?;
                Ok(report)
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    }
}
