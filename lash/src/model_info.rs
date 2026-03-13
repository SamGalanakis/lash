use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelInfo {
    pub context_window: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedModelSpec {
    pub configured_model: String,
    pub resolved_model: String,
    pub catalog_model_id: String,
    pub info: ModelInfo,
}

impl ResolvedModelSpec {
    pub fn context_window(&self) -> u64 {
        self.info.context_window
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCatalog {
    entries: HashMap<String, ModelInfo>,
}

impl ModelCatalog {
    pub fn get(&self, model_id: &str) -> Option<&ModelInfo> {
        self.entries.get(model_id)
    }

    pub fn into_entries(self) -> HashMap<String, ModelInfo> {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn from_models_dev_json(raw: &str) -> Result<Self, String> {
        let providers = serde_json::from_str::<serde_json::Value>(raw)
            .map_err(|err| format!("failed to parse models catalog JSON: {err}"))?;
        let mut entries = HashMap::new();
        let Some(obj) = providers.as_object() else {
            return Err("models catalog root is not an object".to_string());
        };
        for (provider, provider_info) in obj {
            let Some(models) = provider_info.get("models").and_then(|m| m.as_object()) else {
                continue;
            };
            for (model_id, info) in models {
                let Some(context_window) = info
                    .get("limit")
                    .and_then(|l| l.get("context"))
                    .and_then(|c| c.as_u64())
                else {
                    continue;
                };
                let max_input_tokens = info
                    .get("limit")
                    .and_then(|l| l.get("input"))
                    .and_then(|c| c.as_u64());
                let max_output_tokens = info
                    .get("limit")
                    .and_then(|l| l.get("output"))
                    .and_then(|c| c.as_u64());
                entries.insert(
                    format!("{provider}/{model_id}"),
                    ModelInfo {
                        context_window,
                        max_input_tokens,
                        max_output_tokens,
                    },
                );
            }
        }
        Ok(Self { entries })
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryModelCatalogStore {
    raw: Arc<RwLock<Option<String>>>,
    modified_at: Arc<RwLock<Option<SystemTime>>>,
}

impl MemoryModelCatalogStore {
    pub fn new(raw: Option<String>) -> Self {
        Self {
            raw: Arc::new(RwLock::new(raw)),
            modified_at: Arc::new(RwLock::new(None)),
        }
    }
}

impl ModelCatalogStore for MemoryModelCatalogStore {
    fn load(&self) -> Result<Option<String>, String> {
        self.raw
            .read()
            .map(|raw| raw.clone())
            .map_err(|_| "model catalog memory store lock poisoned".to_string())
    }

    fn save(&self, raw: &str) -> Result<(), String> {
        self.raw
            .write()
            .map_err(|_| "model catalog memory store lock poisoned".to_string())
            .map(|mut slot| *slot = Some(raw.to_string()))?;
        self.modified_at
            .write()
            .map_err(|_| "model catalog memory store lock poisoned".to_string())
            .map(|mut slot| *slot = Some(SystemTime::now()))
    }

    fn modified_at(&self) -> Result<Option<SystemTime>, String> {
        self.modified_at
            .read()
            .map(|value| *value)
            .map_err(|_| "model catalog memory store lock poisoned".to_string())
    }
}

#[async_trait]
pub trait ModelCatalogSource: Send + Sync {
    async fn fetch(&self) -> Result<String, String>;
}

pub trait ModelCatalogStore: Send + Sync {
    fn load(&self) -> Result<Option<String>, String>;
    fn save(&self, raw: &str) -> Result<(), String>;
    fn modified_at(&self) -> Result<Option<SystemTime>, String>;
}

#[derive(Clone, Debug)]
pub struct FileModelCatalogStore {
    path: PathBuf,
}

impl FileModelCatalogStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn default_models_dev() -> Self {
        Self::new(crate::lash_cache_dir().join("models.json"))
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl ModelCatalogStore for FileModelCatalogStore {
    fn load(&self) -> Result<Option<String>, String> {
        match std::fs::read_to_string(&self.path) {
            Ok(raw) => Ok(Some(raw)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(format!("failed to read model catalog cache: {err}")),
        }
    }

    fn save(&self, raw: &str) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create model catalog cache directory: {err}"))?;
        }
        std::fs::write(&self.path, raw)
            .map_err(|err| format!("failed to write model catalog cache: {err}"))
    }

    fn modified_at(&self) -> Result<Option<SystemTime>, String> {
        let metadata = match std::fs::metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(format!("failed to stat model catalog cache: {err}")),
        };
        metadata
            .modified()
            .map(Some)
            .map_err(|err| format!("failed to read model catalog cache mtime: {err}"))
    }
}

#[derive(Clone, Debug)]
pub struct ModelsDevHttpSource {
    url: String,
    user_agent: String,
    timeout: Duration,
}

impl ModelsDevHttpSource {
    pub fn new(url: impl Into<String>, user_agent: impl Into<String>, timeout: Duration) -> Self {
        Self {
            url: url.into(),
            user_agent: user_agent.into(),
            timeout,
        }
    }

    pub fn default_models_dev() -> Self {
        Self::new(
            MODELS_DEV_URL,
            format!("lash/{}", crate::VERSION),
            Duration::from_secs(10),
        )
    }
}

#[async_trait]
impl ModelCatalogSource for ModelsDevHttpSource {
    async fn fetch(&self) -> Result<String, String> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .user_agent(self.user_agent.clone())
            .build()
            .map_err(|err| format!("failed to build model catalog client: {err}"))?;
        let response = client
            .get(&self.url)
            .send()
            .await
            .map_err(|err| format!("failed to fetch model catalog: {err}"))?;
        let response = response
            .error_for_status()
            .map_err(|err| format!("model catalog source returned an error: {err}"))?;
        response
            .text()
            .await
            .map_err(|err| format!("failed to read model catalog response: {err}"))
    }
}

#[derive(Clone)]
pub struct CachedModelCatalog {
    catalog: Arc<RwLock<ModelCatalog>>,
    store: Arc<dyn ModelCatalogStore>,
    source: Option<Arc<dyn ModelCatalogSource>>,
}

impl CachedModelCatalog {
    pub fn new(
        store: Arc<dyn ModelCatalogStore>,
        source: Option<Arc<dyn ModelCatalogSource>>,
        bundled_snapshot: &'static str,
    ) -> Result<Self, String> {
        let catalog = if let Some(raw) = store.load()?
            && let Ok(parsed) = ModelCatalog::from_models_dev_json(&raw)
            && !parsed.entries.is_empty()
        {
            parsed
        } else {
            ModelCatalog::from_models_dev_json(bundled_snapshot).unwrap_or_default()
        };
        Ok(Self {
            catalog: Arc::new(RwLock::new(catalog)),
            store,
            source,
        })
    }

    pub fn models_dev(
        store: Arc<dyn ModelCatalogStore>,
        source: Option<Arc<dyn ModelCatalogSource>>,
    ) -> Result<Self, String> {
        Self::new(store, source, bundled_models_dev_snapshot())
    }

    pub fn snapshot(&self) -> ModelCatalog {
        self.catalog
            .read()
            .map(|catalog| catalog.clone())
            .unwrap_or_default()
    }

    pub fn get(&self, model_id: &str) -> Option<ModelInfo> {
        self.catalog
            .read()
            .ok()
            .and_then(|catalog| catalog.get(model_id).cloned())
    }

    pub async fn refresh_if_stale(&self, max_age: Duration) -> Result<bool, String> {
        if self.cache_is_fresh(max_age)? {
            return Ok(false);
        }
        let Some(source) = self.source.as_ref() else {
            return Ok(false);
        };
        let raw = source.fetch().await?;
        let parsed = ModelCatalog::from_models_dev_json(&raw)?;
        self.store.save(&raw)?;
        let mut guard = self
            .catalog
            .write()
            .map_err(|_| "model catalog lock poisoned".to_string())?;
        *guard = parsed;
        Ok(true)
    }

    fn cache_is_fresh(&self, max_age: Duration) -> Result<bool, String> {
        let Some(modified) = self.store.modified_at()? else {
            return false_result();
        };
        let Ok(age) = SystemTime::now().duration_since(modified) else {
            return false_result();
        };
        Ok(age <= max_age)
    }
}

fn false_result() -> Result<bool, String> {
    Ok(false)
}

pub fn bundled_models_dev_snapshot() -> &'static str {
    include_str!(concat!(env!("OUT_DIR"), "/models_snapshot.json"))
}

#[cfg(test)]
mod tests {
    use super::{CachedModelCatalog, ModelCatalog, ModelCatalogStore};
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

    #[test]
    fn parse_context_map_reads_provider_prefixed_limits() {
        let raw = r#"{
          "anthropic": {
            "models": {
              "claude-opus-4-6": { "limit": { "context": 123456, "output": 32000 } }
            }
          },
          "openai": {
            "models": {
              "gpt-4.1": { "limit": { "context": 1047576, "input": 900000, "output": 32768 } }
            }
          }
        }"#;
        let map = ModelCatalog::from_models_dev_json(raw).expect("parse context map");
        assert_eq!(
            map.get("anthropic/claude-opus-4-6")
                .map(|info| info.context_window),
            Some(123456)
        );
        assert_eq!(
            map.get("openai/gpt-4.1")
                .and_then(|info| info.max_input_tokens),
            Some(900000)
        );
    }

    struct MemoryStore {
        raw: Mutex<Option<String>>,
        modified: Mutex<Option<SystemTime>>,
    }

    impl MemoryStore {
        fn new(raw: Option<String>) -> Self {
            Self {
                raw: Mutex::new(raw),
                modified: Mutex::new(None),
            }
        }
    }

    impl ModelCatalogStore for MemoryStore {
        fn load(&self) -> Result<Option<String>, String> {
            Ok(self.raw.lock().unwrap().clone())
        }

        fn save(&self, raw: &str) -> Result<(), String> {
            *self.raw.lock().unwrap() = Some(raw.to_string());
            *self.modified.lock().unwrap() = Some(SystemTime::now());
            Ok(())
        }

        fn modified_at(&self) -> Result<Option<SystemTime>, String> {
            Ok(*self.modified.lock().unwrap())
        }
    }

    #[test]
    fn cached_catalog_uses_store_before_snapshot() {
        let store = Arc::new(MemoryStore::new(Some(
            r#"{"anthropic":{"models":{"claude-opus-4-6":{"limit":{"context":42}}}}}"#.to_string(),
        )));
        let cache = CachedModelCatalog::new(store, None, "{}").expect("cached catalog");
        assert_eq!(
            cache
                .get("anthropic/claude-opus-4-6")
                .map(|info| info.context_window),
            Some(42)
        );
    }
}
