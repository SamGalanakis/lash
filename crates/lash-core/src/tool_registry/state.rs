const PLUGIN_SOURCE_ID: &str = "plugins";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolSourceHandle {
    id: String,
}

impl ToolSourceHandle {
    pub(crate) fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolStateEntry {
    manifest: ToolManifest,
    /// True when this tool was not resolvable from any registered source at
    /// export time (e.g. a detached MCP server). Orphaned entries keep their
    /// last-known manifest, surface as [`crate::ToolAvailability::Off`], and
    /// rebind automatically when a source re-advertises the same (name, id).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    orphaned: bool,
}

impl ToolStateEntry {
    #[cfg(test)]
    pub(crate) fn new(manifest: ToolManifest) -> Self {
        Self {
            manifest,
            orphaned: false,
        }
    }

    /// The manifest as exposed to callers. Orphaned entries are known to the
    /// session but unavailable until their source returns, so their public
    /// availability is always forced to `Off`.
    pub fn manifest(&self) -> ToolManifest {
        let mut manifest = self.manifest.clone();
        if self.orphaned {
            manifest.availability_override = Some(crate::ToolAvailability::Off);
        }
        manifest
    }

    fn stored_manifest(&self) -> &ToolManifest {
        &self.manifest
    }

    pub fn is_orphaned(&self) -> bool {
        self.orphaned
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolState {
    generation: u64,
    tools: Arc<BTreeMap<String, ToolStateEntry>>,
}

impl ToolState {
    pub(crate) fn new(generation: u64, tools: BTreeMap<String, ToolStateEntry>) -> Self {
        Self {
            generation,
            tools: Arc::new(tools),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    pub fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tools.values().map(ToolStateEntry::manifest).collect()
    }

    pub fn get(&self, name: &str) -> Option<&ToolStateEntry> {
        self.tools.get(name)
    }

    pub fn manifest_mut(&mut self, name: &str) -> Option<&mut ToolManifest> {
        Arc::make_mut(&mut self.tools)
            .get_mut(name)
            .map(|entry| &mut entry.manifest)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ToolStateEntry)> {
        self.tools
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
    }

    pub fn set_availability(
        &mut self,
        name: &str,
        availability: Option<crate::ToolAvailability>,
    ) -> Result<(), ReconfigureError> {
        let Some(entry) = Arc::make_mut(&mut self.tools).get_mut(name) else {
            return Err(ReconfigureError::Validation(format!(
                "unknown tool `{name}`"
            )));
        };
        entry.manifest.availability_override = availability;
        Ok(())
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&str, &ToolStateEntry) -> bool) {
        Arc::make_mut(&mut self.tools).retain(|name, entry| keep(name, entry));
    }

    pub fn remove(&mut self, name: &str) -> Option<ToolStateEntry> {
        Arc::make_mut(&mut self.tools).remove(name)
    }

    pub(crate) fn entries(&self) -> &BTreeMap<String, ToolStateEntry> {
        self.tools.as_ref()
    }
}

impl Serialize for ToolState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct ToolStateRef<'a> {
            generation: u64,
            tools: &'a BTreeMap<String, ToolStateEntry>,
        }

        ToolStateRef {
            generation: self.generation,
            tools: self.tools.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ToolStateOwned {
            generation: u64,
            tools: BTreeMap<String, ToolStateEntry>,
        }

        let owned = ToolStateOwned::deserialize(deserializer)?;
        Ok(Self {
            generation: owned.generation,
            tools: Arc::new(owned.tools),
        })
    }
}

#[async_trait::async_trait]
pub(crate) trait ToolSourceExecutor: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn advertised_tools(&self) -> Vec<ToolManifest>;
    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.advertised_tools()
            .into_iter()
            .find(|manifest| manifest.name == name)
    }
    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>>;
    async fn prepare_tool_call(
        &self,
        call: ToolPrepareCall<'_>,
    ) -> Result<PreparedToolCall, ToolResult> {
        Ok(PreparedToolCall::identity(call.pending))
    }
    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        context: &ToolContext<'_>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult;
}
