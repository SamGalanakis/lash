fn validate_unique_manifests(manifests: &[ToolManifest]) -> Result<(), ReconfigureError> {
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for manifest in manifests {
        if manifest.id.as_str().trim().is_empty() {
            return Err(ReconfigureError::Validation(
                "tool id cannot be empty".to_string(),
            ));
        }
        if !ids.insert(manifest.id.clone()) {
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
        if !names.insert(manifest.name.clone()) {
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
        && let Some(contract) = source.resolve_contract(&manifest.name)
    {
        manifest.compact_contract = Some(contract.compact_contract(&manifest));
    }
    manifest
}

fn export_tool_state_entries(
    entries: &BTreeMap<String, ToolRegistryEntry>,
) -> BTreeMap<String, ToolStateEntry> {
    entries
        .iter()
        .map(|(name, entry)| (name.clone(), entry.export()))
        .collect()
}

/// How [`rebind_tool_state_entries`] treats a persisted tool that no
/// registered source resolves by name.
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
    tools: BTreeMap<String, ToolRegistryEntry>,
    orphaned: Vec<String>,
}

fn rebind_tool_state_entries(
    entries: &BTreeMap<String, ToolStateEntry>,
    sources: &BTreeMap<String, Arc<dyn ToolSourceExecutor>>,
    mode: RebindMode,
) -> Result<ReboundTools, ReconfigureError> {
    let mut rebound = BTreeMap::new();
    let mut orphaned = Vec::new();
    for (name, entry) in entries {
        if name != &entry.manifest.name {
            return Err(ReconfigureError::Validation(format!(
                "tool state key `{}` does not match manifest name `{}`",
                name, entry.manifest.name
            )));
        }

        let mut name_matches = Vec::new();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            name_matches.push((
                source_id.clone(),
                manifest_with_compact_contract(source.as_ref(), manifest),
            ));
        }

        if name_matches.is_empty() {
            if mode == RebindMode::RejectUnresolved && !entry.orphaned {
                return Err(ReconfigureError::Validation(format!(
                    "no registered tool source resolves tool `{name}`"
                )));
            }
            orphaned.push(name.clone());
            rebound.insert(
                name.clone(),
                ToolRegistryEntry::orphaned(entry.manifest.clone()),
            );
            continue;
        }

        let matching_id = name_matches
            .iter()
            .filter(|(_, manifest)| manifest.id == entry.manifest.id)
            .collect::<Vec<_>>();

        if matching_id.len() == 1 {
            let source_id = matching_id[0].0.clone();
            rebound.insert(
                name.clone(),
                ToolRegistryEntry::new(entry.manifest.clone(), source_id),
            );
        } else if matching_id.is_empty() {
            let resolved_ids = name_matches
                .iter()
                .map(|(_, manifest)| manifest.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ReconfigureError::Validation(format!(
                "tool `{name}` resolved with id(s) `{resolved_ids}`, expected `{}`",
                entry.manifest.id
            )));
        } else {
            return Err(ReconfigureError::Validation(format!(
                "tool `{name}` with id `{}` is resolved by multiple registered sources",
                entry.manifest.id
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
    let manifests = entries
        .into_iter()
        .map(|entry| entry.stored_manifest().clone())
        .collect::<Vec<_>>();
    validate_unique_manifests(&manifests)
}
