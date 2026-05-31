//! Session-graph persistence and garbage collection on [`Store`].

use super::*;

impl Store {
    pub(crate) fn load_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> lash_core::SessionGraph {
        // Tombstoned rows are physically still present until `vacuum()` is
        // called; the runtime view should never see them.
        let mut stmt = match conn
            .prepare("SELECT node_json FROM graph_nodes WHERE tombstoned = 0 ORDER BY seq ASC")
        {
            Ok(stmt) => stmt,
            Err(err) => {
                tracing::warn!(error = %err, "failed to prepare graph load statement");
                return lash_core::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
            Ok(rows) => rows,
            Err(err) => {
                tracing::warn!(error = %err, "failed to query graph rows");
                return lash_core::SessionGraph::from_nodes(Vec::new(), leaf_node_id);
            }
        };
        let mut nodes = Vec::new();
        for row in rows {
            let Ok(node_json) = row else {
                continue;
            };
            let Ok(node) = serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json) else {
                continue;
            };
            nodes.push(node);
        }
        lash_core::SessionGraph::from_nodes(nodes, leaf_node_id)
    }

    pub(crate) fn load_active_path_session_graph_from_conn(
        conn: &Connection,
        leaf_node_id: Option<String>,
    ) -> rusqlite::Result<lash_core::SessionGraph> {
        let Some(leaf_node_id) = leaf_node_id else {
            return Ok(lash_core::SessionGraph::default());
        };
        let mut stmt = conn.prepare(
            "WITH RECURSIVE active(node_id, node_json, parent_node_id, depth) AS (
                SELECT
                    node_id,
                    node_json,
                    json_extract(node_json, '$.parent_node_id'),
                    0
                FROM graph_nodes
                WHERE node_id = ?1 AND tombstoned = 0
              UNION ALL
                SELECT
                    g.node_id,
                    g.node_json,
                    json_extract(g.node_json, '$.parent_node_id'),
                    active.depth + 1
                FROM graph_nodes g
                JOIN active ON g.node_id = active.parent_node_id
                WHERE g.tombstoned = 0
            )
            SELECT node_json FROM active ORDER BY depth DESC",
        )?;
        let rows = stmt.query_map(params![leaf_node_id.as_str()], |row| {
            row.get::<_, String>(0)
        })?;
        let mut nodes = Vec::new();
        for row in rows {
            let node_json = row?;
            if let Ok(node) = serde_json::from_str::<lash_core::SessionNodeRecord>(&node_json) {
                nodes.push(node);
            }
        }
        Ok(lash_core::SessionGraph::from_nodes(
            nodes,
            Some(leaf_node_id),
        ))
    }

    pub(crate) fn maybe_auto_gc(&self) {
        let Some(interval) = self.options.gc_policy.auto_run_every_commits else {
            return;
        };
        let commits = self.commit_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if interval != 0 && commits % interval == 0 {
            let _ = self.gc_unreachable();
        }
    }

    pub fn replace_session_graph(&self, graph: &lash_core::SessionGraph) {
        let result = self.with_write_tx(|tx| {
            tx.execute("DELETE FROM graph_nodes", [])
                .map_err(sqlite_error)?;
            for node in &graph.nodes {
                let node_json = encode_json(node);
                tx.execute(
                    "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                    params![node.node_id, node_json],
                )
                .map_err(sqlite_error)?;
            }
            Ok(())
        });
        if let Err(err) = result {
            tracing::warn!(error = %err, "failed to replace session graph");
        }
    }

    pub fn append_session_graph_nodes(&self, nodes: &[lash_core::SessionNodeRecord]) {
        if nodes.is_empty() {
            return;
        }
        let result = self.with_write_tx(|tx| {
            for node in nodes {
                let node_json = encode_json(node);
                tx.execute(
                    "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                    params![node.node_id, node_json],
                )
                .map_err(sqlite_error)?;
            }
            Ok(())
        });
        if let Err(err) = result {
            tracing::warn!(error = %err, "failed to append session graph nodes");
        }
    }

    pub fn load_session_graph(&self) -> lash_core::SessionGraph {
        let conn = lock_conn(&self.conn);
        Self::load_session_graph_from_conn(&conn, None)
    }

    pub fn gc_unreachable(&self) -> GcReport {
        match self.try_gc_unreachable() {
            Ok(report) => report,
            Err(err) => {
                // GC is best-effort space reclamation. A backend failure must
                // never panic inside the commit and brick the store; log and
                // leave every blob in place (the conservative outcome).
                tracing::warn!(error = %err, "gc_unreachable failed; retaining all blobs");
                GcReport::default()
            }
        }
    }

    /// Collect the checkpoint-manifest roots that must survive GC.
    ///
    /// The session head's current `checkpoint_ref` is the live checkpoint; its
    /// manifest blob (and, transitively, the tool/plugin/execution snapshot
    /// blobs it references) is reachable and must be kept. Runtime turn
    /// checkpoints are stored inline as JSON in `runtime_turn_checkpoints`, not
    /// as content-addressed blobs, so they hold no blob references and need no
    /// rooting here.
    fn live_checkpoint_roots(tx: &Transaction<'_>) -> Result<Vec<RetainedArtifactRef>, StoreError> {
        let mut roots = Vec::new();
        if let Some(checkpoint_ref) = load_session_head_meta_from_conn(tx)
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

    fn try_gc_unreachable(&self) -> Result<GcReport, StoreError> {
        self.with_write_tx(|tx| {
            let mut roots = Self::live_checkpoint_roots(tx)?;
            {
                let mut stmt = tx
                    .prepare("SELECT blob_ref FROM artifact_refs ORDER BY artifact_ref")
                    .map_err(sqlite_error)?;
                let rows = stmt
                    .query_map([], |row| row.get::<_, String>(0))
                    .map_err(sqlite_error)?;
                for row in rows {
                    roots.push(RetainedArtifactRef {
                        blob_ref: BlobRef(row.map_err(sqlite_error)?),
                        kind: PersistedArtifactKind::LashlangModule,
                    });
                }
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
                // A rooted checkpoint manifest is *live*. If we cannot read or
                // decode it we must not silently drop the child blobs it
                // points at (tool/plugin/execution snapshots) — doing so would
                // delete blobs that belong to a live checkpoint. Skip a manifest
                // that simply isn't present (it may have been collected on a
                // prior run), but treat a present-yet-undecodable manifest as a
                // hard error so GC aborts rather than deleting live data.
                let bytes: Option<Vec<u8>> = tx
                    .query_row(
                        "SELECT content FROM blobs WHERE hash = ?1",
                        params![current.blob_ref.as_str()],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .optional()
                    .map_err(sqlite_error)?;
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
            let all_hashes = {
                let mut stmt = tx
                    .prepare("SELECT hash FROM blobs ORDER BY hash ASC")
                    .map_err(sqlite_error)?;
                let rows = stmt
                    .query_map([], |row| row.get::<_, String>(0))
                    .map_err(sqlite_error)?;
                rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_error)?
            };
            let mut deleted_blob_count = 0usize;
            for hash in &all_hashes {
                if retained.contains_key(hash) {
                    continue;
                }
                tx.execute("DELETE FROM blobs WHERE hash = ?1", params![hash])
                    .map_err(sqlite_error)?;
                deleted_blob_count += 1;
            }
            Ok(GcReport {
                root_count,
                retained_blob_count: retained.len(),
                deleted_blob_count,
            })
        })
    }
}
