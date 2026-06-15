impl ToolRegistry {
    /// Try to rebind an orphaned entry to a source that now resolves the same
    /// (name, id). Returns the live manifest on success. A source resolving
    /// the name with a *different* id is a replaced implementation — the
    /// orphan keeps the name and stays unavailable rather than silently
    /// swapping semantics under the session.
    fn try_rebind_orphan(&self, name: &str, orphan_id: &crate::ToolId) -> Option<ToolManifest> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(source_id, source)| (source_id.clone(), Arc::clone(source)))
            .collect::<Vec<_>>();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            if manifest.id != *orphan_id {
                continue;
            }
            let mut manifest = manifest_with_compact_contract(source.as_ref(), manifest);
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            let existing = state.tools.get(name)?;
            if !existing.is_orphaned() {
                return Some(existing.view_manifest());
            }
            manifest.availability_override = existing
                .manifest
                .availability_override
                .or(manifest.availability_override);
            state.tools.insert(
                name.to_string(),
                ToolRegistryEntry::new(manifest.clone(), source_id),
            );
            state.generation += 1;
            return Some(manifest);
        }
        None
    }

    /// Resolve the source for a registry entry, distinguishing "unknown tool"
    /// from "known but orphaned" so callers can fail with a precise error.
    fn resolve_execution_source(
        &self,
        name: &str,
    ) -> Result<Arc<dyn ToolSourceExecutor>, ToolResult> {
        if self.resolve_manifest(name).is_none() {
            return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}")));
        }
        let binding = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| entry.binding.clone())
        };
        let source_id = match binding {
            Some(ToolBinding::Bound(source_id)) => source_id,
            Some(ToolBinding::Orphaned) => {
                return Err(ToolResult::err_fmt(format_args!(
                    "Tool `{name}` is unavailable: it was restored from a persisted session \
                     but its source is not currently registered"
                )));
            }
            None => return Err(ToolResult::err_fmt(format_args!("Unknown tool: {name}"))),
        };
        let source = {
            self.sources
                .read()
                .expect("tool source lock poisoned")
                .get(&source_id)
                .cloned()
        };
        source.ok_or_else(|| {
            ToolResult::err_fmt(format_args!("Tool source missing for tool `{name}`"))
        })
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolRegistry {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        let state = self
            .state
            .read()
            .expect("tool registry state lock poisoned");
        state
            .tools
            .values()
            .map(ToolRegistryEntry::view_manifest)
            .collect()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        enum Known {
            Bound(ToolManifest),
            Orphaned(ToolManifest, crate::ToolId),
        }
        let known = {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state.tools.get(name).map(|entry| {
                if entry.is_orphaned() {
                    Known::Orphaned(entry.view_manifest(), entry.manifest.id.clone())
                } else {
                    Known::Bound(entry.manifest.clone())
                }
            })
        };
        match known {
            Some(Known::Bound(manifest)) => return Some(manifest),
            Some(Known::Orphaned(off_manifest, orphan_id)) => {
                // A source may have come back since the restore: rebind on
                // (name, id) match, otherwise keep serving the Off view.
                return Some(
                    self.try_rebind_orphan(name, &orphan_id)
                        .unwrap_or(off_manifest),
                );
            }
            None => {}
        }

        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(source_id, source)| (source_id.clone(), Arc::clone(source)))
            .collect::<Vec<_>>();
        for (source_id, source) in sources {
            let Some(manifest) = source.resolve_manifest(name) else {
                continue;
            };
            let mut manifest = manifest_with_compact_contract(source.as_ref(), manifest);
            let previous_override = {
                let state = self
                    .state
                    .read()
                    .expect("tool registry state lock poisoned");
                state
                    .tools
                    .get(&manifest.name)
                    .and_then(|entry| entry.manifest.availability_override)
            };
            manifest.availability_override = previous_override.or(manifest.availability_override);
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            if let Some(existing) = state.tools.get(&manifest.name) {
                return (existing.binding.source_id() == Some(source_id.as_str()))
                    .then(|| existing.view_manifest());
            }
            if let Some((_, existing)) = state
                .tools
                .iter()
                .find(|(_, entry)| entry.manifest.id == manifest.id)
            {
                return (existing.binding.source_id() == Some(source_id.as_str()))
                    .then(|| existing.view_manifest());
            }
            state.tools.insert(
                manifest.name.clone(),
                ToolRegistryEntry::new(manifest.clone(), source_id),
            );
            state.generation += 1;
            return Some(manifest);
        }
        None
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        let source_id = self.resolve_manifest(name).and_then(|_| {
            let state = self
                .state
                .read()
                .expect("tool registry state lock poisoned");
            state
                .tools
                .get(name)
                .and_then(|entry| entry.binding.source_id().map(str::to_string))
        })?;
        self.sources
            .read()
            .expect("tool source lock poisoned")
            .get(&source_id)?
            .resolve_contract(name)
    }

    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        let name = call.pending.tool_name.clone();
        let source = self.resolve_execution_source(&name)?;
        source.prepare_tool_call(call).await
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let name = call.name;
        let source = match self.resolve_execution_source(name) {
            Ok(source) => source,
            Err(result) => return result,
        };
        source
            .execute(name, call.args, call.context, call.progress)
            .await
    }
}
