fn validate_unique_manifests<'a>(
    manifests: impl IntoIterator<Item = &'a ToolManifest>,
) -> Result<(), ReconfigureError> {
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for manifest in manifests {
        if manifest.id.as_str().trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool id cannot be empty".to_string(),
            ));
        }
        if !ids.insert(&manifest.id) {
            return Err(ReconfigureError::Validation(format!(
                "duplicate tool id `{}` in source",
                manifest.id
            )));
        }
        if manifest.name.trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool name cannot be empty".to_string(),
            ));
        }
        if !names.insert(manifest.name.as_str()) {
            return Err(ReconfigureError::Validation(format!(
                "duplicate tool name `{}` in source",
                manifest.name
            )));
        }
    }
    Ok(())
}

fn manifest_with_compact_contract(
    source: &dyn ToolSourceExecutor,
    mut manifest: ToolManifest,
) -> ToolManifest {
    if manifest.compact_contract.is_none()
        && let Some(contract) = source.resolve_contract_by_id(&manifest.id)
    {
        manifest.compact_contract = Some(contract.compact_contract(&manifest));
    }
    manifest
}

fn export_tool_state_entries(
    entries: &BTreeMap<ToolId, ToolRegistryEntry>,
) -> BTreeMap<ToolId, ToolStateEntry> {
    entries
        .iter()
        .map(|(id, entry)| (id.clone(), entry.export()))
        .collect()
}

/// How [`rebind_tool_state_entries`] treats a persisted tool that no
/// registered source resolves by id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RebindMode {
    /// Restore/fork: keep the entry as an orphan instead of failing. Sessions
    /// must stay openable when a dynamic source (e.g. an MCP server) is down.
    OrphanUnresolved,
    /// Explicit `apply_state`: fail on unresolved tools, except entries the
    /// snapshot already marks as orphaned — otherwise a host could never
    /// round-trip `export_state` → edit → `apply_state` while an orphan exists.
    RejectUnresolved,
}

struct ReboundTools {
    tools: BTreeMap<ToolId, ToolRegistryEntry>,
    orphaned: Vec<ToolId>,
}

fn rebind_tool_state_entries(
    entries: &BTreeMap<ToolId, ToolStateEntry>,
    sources: &BTreeMap<String, Arc<dyn ToolSourceExecutor>>,
    mode: RebindMode,
) -> Result<ReboundTools, ReconfigureError> {
    let mut rebound = BTreeMap::new();
    let mut orphaned = Vec::new();
    for (id, entry) in entries {
        if id != &entry.manifest.id {
            return Err(ReconfigureError::Validation(format!(
                "tool state key `{}` does not match manifest id `{}`",
                id, entry.manifest.id
            )));
        }

        let mut id_matches = Vec::new();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest_by_id(id) else {
                continue;
            };
            id_matches.push((
                source_id.clone(),
                manifest_with_compact_contract(source.as_ref(), manifest),
            ));
        }

        if id_matches.is_empty() {
            if mode == RebindMode::RejectUnresolved && !entry.orphaned {
                return Err(ReconfigureError::Validation(format!(
                    "no registered tool source resolves tool id `{id}`"
                )));
            }
            orphaned.push(id.clone());
            let mut orphan = ToolRegistryEntry::orphaned(entry.manifest.clone());
            orphan.member = entry.member;
            rebound.insert(id.clone(), orphan);
            continue;
        }

        if id_matches.len() == 1 {
            let (source_id, manifest) = id_matches
                .into_iter()
                .next()
                .expect("len checked above");
            let mut rebound_entry = ToolRegistryEntry::new(manifest, source_id);
            rebound_entry.member = entry.member;
            rebound.insert(id.clone(), rebound_entry);
        } else {
            return Err(ReconfigureError::Validation(format!(
                "tool id `{id}` is resolved by multiple registered sources"
            )));
        }
    }
    Ok(ReboundTools {
        tools: rebound,
        orphaned,
    })
}

fn validate_unique_manifest_entries<'a>(
    entries: impl IntoIterator<Item = &'a ToolStateEntry>,
) -> Result<(), ReconfigureError> {
    validate_unique_manifests(entries.into_iter().map(ToolStateEntry::stored_manifest))
}
