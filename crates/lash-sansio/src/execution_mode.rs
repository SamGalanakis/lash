/// Stable string id for the execution backend that owns a session turn.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionMode(std::sync::Arc<str>);

impl ExecutionMode {
    pub fn new(id: impl Into<std::sync::Arc<str>>) -> Self {
        Self(id.into())
    }

    pub fn standard() -> Self {
        Self::new("standard")
    }

    pub fn plugin_id(&self) -> &str {
        &self.0
    }
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::standard()
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.plugin_id())
    }
}

impl serde::Serialize for ExecutionMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.plugin_id())
    }
}

impl<'de> serde::Deserialize<'de> for ExecutionMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = <String as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::new(id))
    }
}

pub fn execution_mode_supported(_mode: &ExecutionMode) -> bool {
    true
}

pub fn default_execution_mode() -> ExecutionMode {
    ExecutionMode::default()
}
