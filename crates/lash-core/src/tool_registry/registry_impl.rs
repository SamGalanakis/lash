impl ToolRegistry {
    pub fn from_tool_provider(provider: Arc<dyn ToolProvider>) -> Result<Self, ReconfigureError> {
        let registry = Self::empty();
        registry.upsert_source(Arc::new(ToolProviderSource::new(
            PLUGIN_SOURCE_ID,
            provider,
        )))?;
        Ok(registry)
    }

    pub(crate) fn from_tool_providers(
        providers: Vec<Arc<dyn ToolProvider>>,
    ) -> Result<Self, ReconfigureError> {
        let registry = Self::empty();
        registry.upsert_source(Arc::new(ToolProviderGroupSource::new(
            PLUGIN_SOURCE_ID,
            providers,
        )))?;
        Ok(registry)
    }

    pub(crate) fn empty() -> Self {
        Self {
            sources: Arc::new(RwLock::new(BTreeMap::new())),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation: 0,
                tools: BTreeMap::new(),
                next_live_source_id: 0,
            })),
        }
    }

    pub fn generation(&self) -> u64 {
        self.state
            .read()
            .expect("tool registry state lock poisoned")
            .generation
    }

    pub fn export_state(&self) -> ToolState {
        let state = self
            .state
            .read()
            .expect("tool registry state lock poisoned");
        ToolState::new(state.generation, export_tool_state_entries(&state.tools))
    }

    pub fn apply_state(&self, next: ToolState) -> Result<u64, ReconfigureError> {
        let current_generation = self.generation();
        if next.generation != current_generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.generation,
                actual: current_generation,
            });
        }

        validate_unique_manifest_entries(next.entries().values())?;
        let rebound = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(next.entries(), &sources, RebindMode::RejectUnresolved)?
        };

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        if state.generation != next.generation {
            return Err(ReconfigureError::GenerationMismatch {
                expected: next.generation,
                actual: state.generation,
            });
        }
        state.tools = rebound.tools;
        state.generation += 1;
        Ok(state.generation)
    }

    /// Restore a persisted [`ToolState`] snapshot onto a freshly-built registry,
    /// adopting the snapshot's generation verbatim.
    ///
    /// Unlike [`apply_state`](Self::apply_state) — which applies an incremental
    /// *delta* expected at the current generation and bumps it by one — a
    /// restore reconstructs the exact persisted state regardless of the fresh
    /// registry's base generation, and does **not** bump. This is idempotent: a
    /// snapshot exported at generation `G` restores to generation `G`, so a
    /// re-export round-trips. Cold rebuilds (the durable process worker, session
    /// resume) restore a session whose tool catalog reached generation `G ≥ 2`
    /// onto a base registry at generation 1 — `apply_state` would reject that
    /// (`expected G, actual 1`); `restore_state` adopts `G`. Entries are still
    /// rebound to the live sources, so source identity is reconnected.
    ///
    /// Restore is tolerant of missing sources: a persisted tool that no current
    /// source resolves becomes an orphaned entry (kept with its last-known
    /// manifest, surfaced as `Off`, rebound when its source returns) instead of
    /// failing the whole restore. Tool id is the registry identity; the live
    /// manifest wins on rebind, with the persisted availability override
    /// preserved. Multiple sources resolving the same id or advertised name
    /// still fail because execution authority and model-facing names must both
    /// be unambiguous.
    pub fn restore_state(
        &self,
        snapshot: ToolState,
    ) -> Result<ToolRestoreReport, ReconfigureError> {
        validate_unique_manifest_entries(snapshot.entries().values())?;
        let rebound = {
            let sources = self.sources.read().expect("tool source lock poisoned");
            rebind_tool_state_entries(snapshot.entries(), &sources, RebindMode::OrphanUnresolved)?
        };

        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        state.tools = rebound.tools;
        state.generation = snapshot.generation();
        Ok(ToolRestoreReport {
            generation: state.generation,
            orphaned: rebound.orphaned,
        })
    }

    pub fn add_tool_provider(
        &self,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<ToolSourceHandle, ReconfigureError> {
        let source_id = {
            let mut state = self
                .state
                .write()
                .expect("tool registry state lock poisoned");
            state.next_live_source_id += 1;
            format!("live:{}", state.next_live_source_id)
        };
        self.upsert_source(Arc::new(ToolProviderSource::new(
            source_id.clone(),
            provider,
        )))?;
        Ok(ToolSourceHandle::new(source_id))
    }

    pub(crate) fn compose_session_catalog(
        &self,
        include_base_tools: bool,
        context_providers: Vec<Arc<dyn ToolProvider>>,
    ) -> Result<Self, ReconfigureError> {
        let registry = if include_base_tools {
            self.fork_with_state(self.export_state())?
        } else {
            Self::empty()
        };
        registry.upsert_overlay_source(Arc::new(ToolProviderGroupSource::new(
            "context",
            context_providers,
        )))?;
        Ok(registry)
    }

    pub(crate) fn upsert_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
    ) -> Result<u64, ReconfigureError> {
        self.reconcile_source(source, SourceReconcilePolicy::RejectExternalConflicts)
    }

    fn upsert_overlay_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
    ) -> Result<u64, ReconfigureError> {
        self.reconcile_source(source, SourceReconcilePolicy::OverlayReplacingConflicts)
    }

    fn reconcile_source(
        &self,
        source: Arc<dyn ToolSourceExecutor>,
        policy: SourceReconcilePolicy,
    ) -> Result<u64, ReconfigureError> {
        let source_id = source.id().to_string();
        let advertised_tools = source
            .advertised_tools()
            .into_iter()
            .map(|manifest| manifest_with_compact_contract(source.as_ref(), manifest))
            .collect::<Vec<_>>();
        validate_unique_manifests(advertised_tools.iter())?;
        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        let previous_overrides = state
            .tools
            .iter()
            .map(|(id, entry)| (id.clone(), entry.manifest.availability_override))
            .collect::<BTreeMap<_, _>>();
        {
            let advertised_names = advertised_tools
                .iter()
                .map(|manifest| manifest.name.as_str())
                .collect::<BTreeSet<_>>();
            let advertised_ids = advertised_tools
                .iter()
                .map(|manifest| &manifest.id)
                .collect::<BTreeSet<_>>();
            match policy {
                SourceReconcilePolicy::RejectExternalConflicts => {
                    // Orphans never conflict: a live advertisement supersedes a
                    // tool that is currently unresolvable. Matching id means the
                    // source came back, with its availability override preserved
                    // via `previous_overrides`.
                    for manifest in &advertised_tools {
                        if let Some(existing) = state.tools.get(&manifest.id)
                            && let Some(existing_source) = existing.binding.source_id()
                            && existing_source != source_id
                        {
                            return Err(ReconfigureError::Validation(format!(
                                "duplicate tool id `{}` from source `{}` conflicts with source `{}`",
                                manifest.id, source_id, existing_source
                            )));
                        }
                        if let Some((existing_id, existing_source)) =
                            state.tools.iter().find_map(|(id, entry)| {
                                let existing_source = entry.binding.source_id()?;
                                (existing_source != source_id && entry.manifest.name == manifest.name)
                                    .then(|| (id.clone(), existing_source.to_string()))
                            })
                        {
                            return Err(ReconfigureError::Validation(format!(
                                "duplicate tool name `{}` from source `{}` conflicts with tool id `{}` from source `{}`",
                                manifest.name, source_id, existing_id, existing_source
                            )));
                        }
                    }
                    state.tools.retain(|id, entry| match &entry.binding {
                        // Drop this source's previous surface; it is re-inserted
                        // from the fresh advertisement below.
                        ToolBinding::Bound(bound) => bound != &source_id,
                        // Drop orphans the fresh advertisement supersedes by id,
                        // or by name when a different grant now owns the alias.
                        ToolBinding::Orphaned => {
                            !advertised_ids.contains(id)
                                && !advertised_names.contains(&entry.manifest.name.as_str())
                        }
                    });
                }
                SourceReconcilePolicy::OverlayReplacingConflicts => {
                    state.tools.retain(|id, entry| {
                        entry.binding.source_id() != Some(source_id.as_str())
                            && !advertised_ids.contains(id)
                            && !advertised_names.contains(&entry.manifest.name.as_str())
                    });
                }
            }
        }
        for mut manifest in advertised_tools {
            let id = manifest.id.clone();
            manifest.availability_override = previous_overrides
                .get(&id)
                .copied()
                .flatten()
                .or(manifest.availability_override);
            state
                .tools
                .insert(id, ToolRegistryEntry::new(manifest, source_id.clone()));
        }
        self.sources
            .write()
            .expect("tool source lock poisoned")
            .insert(source_id, source);
        state.generation += 1;
        Ok(state.generation)
    }

    pub fn remove_source(&self, handle: &ToolSourceHandle) -> Result<u64, ReconfigureError> {
        self.remove_source_id(handle.as_str())
    }

    pub fn refresh_sources(&self) -> Result<u64, ReconfigureError> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut generation = self.generation();
        for source in sources {
            generation = self.upsert_source(source)?;
        }
        Ok(generation)
    }

    pub(crate) fn remove_source_id(&self, source_id: &str) -> Result<u64, ReconfigureError> {
        {
            let mut sources = self.sources.write().expect("tool source lock poisoned");
            if sources.remove(source_id).is_none() {
                return Err(ReconfigureError::UnknownSource(source_id.to_string()));
            }
        }
        let mut state = self
            .state
            .write()
            .expect("tool registry state lock poisoned");
        state
            .tools
            .retain(|_, entry| entry.binding.source_id() != Some(source_id));
        state.generation += 1;
        Ok(state.generation)
    }

    pub(crate) fn fork_with_state(&self, snapshot: ToolState) -> Result<Self, ReconfigureError> {
        let sources = self
            .sources
            .read()
            .expect("tool source lock poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect::<BTreeMap<_, _>>();
        validate_unique_manifest_entries(snapshot.entries().values())?;
        // Tolerant like `restore_state`: a fork mirrors whatever the parent
        // registry holds, including orphans whose sources are still absent.
        let rebound =
            rebind_tool_state_entries(snapshot.entries(), &sources, RebindMode::OrphanUnresolved)?;
        let generation = snapshot.generation.max(1);
        Ok(Self {
            sources: Arc::new(RwLock::new(sources)),
            state: Arc::new(RwLock::new(ToolRegistryState {
                generation,
                tools: rebound.tools,
                next_live_source_id: 0,
            })),
        })
    }
}
