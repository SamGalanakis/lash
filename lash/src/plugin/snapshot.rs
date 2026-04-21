//! Plugin snapshot types: how plugins persist their mode-specific state
//! across session reloads.
//!
//! Split out of `plugin/mod.rs` purely for file size; `pub use` in
//! `plugin/mod.rs` preserves the external path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginSessionSnapshot {
    #[serde(default)]
    pub plugins: BTreeMap<String, PluginSnapshotEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotEntry {
    pub meta: PluginSnapshotMeta,
    #[serde(default)]
    pub artifacts: Vec<PluginSnapshotArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotMeta {
    pub plugin_id: String,
    pub plugin_version: String,
    #[serde(default)]
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginSnapshotArtifact {
    pub name: String,
    pub data: Vec<u8>,
}

pub trait SnapshotWriter {
    fn write_blob(&mut self, name: String, data: Vec<u8>);
}

pub trait SnapshotReader {
    fn read_blob(&self, name: &str) -> Option<&[u8]>;
}

#[derive(Default)]
pub(crate) struct InMemorySnapshotWriter {
    artifacts: Vec<PluginSnapshotArtifact>,
}

impl InMemorySnapshotWriter {
    pub(crate) fn finish(self) -> Vec<PluginSnapshotArtifact> {
        self.artifacts
    }
}

impl SnapshotWriter for InMemorySnapshotWriter {
    fn write_blob(&mut self, name: String, data: Vec<u8>) {
        self.artifacts.push(PluginSnapshotArtifact { name, data });
    }
}

pub(crate) struct InMemorySnapshotReader<'a> {
    pub(crate) entry: &'a PluginSnapshotEntry,
}

impl SnapshotReader for InMemorySnapshotReader<'_> {
    fn read_blob(&self, name: &str) -> Option<&[u8]> {
        self.entry
            .artifacts
            .iter()
            .find(|artifact| artifact.name == name)
            .map(|artifact| artifact.data.as_slice())
    }
}
