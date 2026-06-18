#[derive(Clone)]
struct ToolRegistryEntry {
    manifest: ToolManifest,
    binding: ToolBinding,
}

impl ToolRegistryEntry {
    fn new(manifest: ToolManifest, source_id: impl Into<String>) -> Self {
        Self {
            manifest,
            binding: ToolBinding::Bound(source_id.into()),
        }
    }

    fn orphaned(manifest: ToolManifest) -> Self {
        Self {
            manifest,
            binding: ToolBinding::Orphaned,
        }
    }

    fn is_orphaned(&self) -> bool {
        self.binding == ToolBinding::Orphaned
    }

    /// The manifest as exposed to surfaces, catalogs, and availability checks.
    /// Orphaned entries are forced to `Off` in the view without mutating the
    /// stored manifest, so the persisted override survives a later rebind and
    /// export/restore round-trips stay byte-identical.
    fn view_manifest(&self) -> ToolManifest {
        let mut manifest = self.manifest.clone();
        if self.is_orphaned() {
            manifest.availability_override = Some(crate::ToolAvailability::Off);
        }
        manifest
    }

    fn export(&self) -> ToolStateEntry {
        ToolStateEntry {
            manifest: self.manifest.clone(),
            orphaned: self.is_orphaned(),
        }
    }
}

#[derive(Clone)]
struct ToolRegistryState {
    generation: u64,
    tools: BTreeMap<ToolId, ToolRegistryEntry>,
    next_live_source_id: u64,
}

/// Outcome of [`ToolRegistry::restore_state`]: the adopted generation plus the
/// ids of persisted tools that no registered source currently resolves.
/// Hosts should surface a non-empty `orphaned` list to the user — the session
/// opened, but those tools are `Off` until their source returns.
#[derive(Clone, Debug, Default)]
pub struct ToolRestoreReport {
    pub generation: u64,
    pub orphaned: Vec<ToolId>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReconfigureError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("unknown tool source: {0}")]
    UnknownSource(String),
    #[error("generation mismatch: expected {expected}, actual {actual}")]
    GenerationMismatch { expected: u64, actual: u64 },
}

#[derive(Clone)]
pub struct ToolRegistry {
    sources: Arc<RwLock<BTreeMap<String, Arc<dyn ToolSourceExecutor>>>>,
    state: Arc<RwLock<ToolRegistryState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceReconcilePolicy {
    RejectExternalConflicts,
    OverlayReplacingConflicts,
}
