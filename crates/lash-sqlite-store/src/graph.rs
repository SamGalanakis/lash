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
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(error = %err, "failed to begin graph replace transaction");
                return;
            }
        };
        if let Err(err) = tx.execute("DELETE FROM graph_nodes", []) {
            tracing::warn!(error = %err, "failed to clear graph rows");
            return;
        }
        for node in &graph.nodes {
            let node_json = encode_json(node);
            if let Err(err) = tx.execute(
                "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                params![node.node_id, node_json],
            ) {
                tracing::warn!(error = %err, node_id = %node.node_id, "failed to persist graph node");
                return;
            }
        }
        if let Err(err) = tx.commit() {
            tracing::warn!(error = %err, "failed to commit graph replace");
        }
    }

    pub fn append_session_graph_nodes(&self, nodes: &[lash_core::SessionNodeRecord]) {
        if nodes.is_empty() {
            return;
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = match conn.transaction() {
            Ok(tx) => tx,
            Err(err) => {
                tracing::warn!(error = %err, "failed to begin graph append transaction");
                return;
            }
        };
        for node in nodes {
            let node_json = encode_json(node);
            if let Err(err) = tx.execute(
                "INSERT INTO graph_nodes (node_id, node_json) VALUES (?1, ?2)",
                params![node.node_id, node_json],
            ) {
                tracing::warn!(error = %err, node_id = %node.node_id, "failed to append graph node");
                return;
            }
        }
        if let Err(err) = tx.commit() {
            tracing::warn!(error = %err, "failed to commit graph append");
        }
    }

    pub fn load_session_graph(&self) -> lash_core::SessionGraph {
        let conn = self.conn.lock().unwrap();
        Self::load_session_graph_from_conn(&conn, None)
    }

    pub fn gc_unreachable(&self) -> GcReport {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().expect("gc transaction");
        let head_meta = load_session_head_meta_from_conn(&tx);
        let mut roots = Vec::new();
        if let Some(checkpoint_ref) = head_meta
            .as_ref()
            .and_then(|meta| meta.checkpoint_ref.as_ref())
            .cloned()
        {
            roots.push(RetainedArtifactRef {
                blob_ref: checkpoint_ref,
                kind: PersistedArtifactKind::CheckpointManifest,
            });
        }
        if let Ok(mut stmt) = tx.prepare("SELECT blob_ref FROM artifact_refs ORDER BY artifact_ref")
            && let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0))
        {
            roots.extend(
                rows.filter_map(Result::ok)
                    .map(|blob_ref| RetainedArtifactRef {
                        blob_ref: BlobRef(blob_ref),
                        kind: PersistedArtifactKind::LashlangModule,
                    }),
            );
        }
        let mut retained = std::collections::BTreeMap::<String, PersistedArtifactKind>::new();
        let mut stack = roots.clone();
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
            let Some(bytes) = tx
                .query_row(
                    "SELECT content FROM blobs WHERE hash = ?1",
                    params![current.blob_ref.as_str()],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .ok()
            else {
                continue;
            };
            let Some(content) = decode_artifact_blob(&bytes).or(Some(bytes)) else {
                continue;
            };
            let Some(checkpoint) = decode_checkpoint(&content) else {
                continue;
            };
            stack.extend(retained_artifact_refs(&checkpoint));
        }
        let all_hashes = {
            let mut stmt = tx
                .prepare("SELECT hash FROM blobs ORDER BY hash ASC")
                .expect("prepare blob scan");
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .expect("query blob scan");
            rows.filter_map(Result::ok).collect::<Vec<_>>()
        };
        let mut deleted_blob_count = 0usize;
        for hash in &all_hashes {
            if retained.contains_key(hash) {
                continue;
            }
            tx.execute("DELETE FROM blobs WHERE hash = ?1", params![hash])
                .expect("delete unreachable blob");
            deleted_blob_count += 1;
        }
        tx.commit().expect("commit gc transaction");
        GcReport {
            root_count: roots.len(),
            retained_blob_count: retained.len(),
            deleted_blob_count,
        }
    }
}
