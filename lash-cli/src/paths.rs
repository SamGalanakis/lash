//! Filesystem layout conventions used by the lash CLI.
//!
//! These functions live in lash-cli rather than lash core: where a
//! user's config/cache lives is a CLI-application decision, not part
//! of the library surface. lash-core accepts paths explicitly from the
//! caller (via `RuntimeEnvironment.base_dir`, `InstructionLoaderConfig`,
//! `FileModelCatalogStore::new`, etc.); lash-cli is the concrete host
//! that wires these values from the `~/.lash/` conventions below.

use std::path::PathBuf;

/// Root data directory for lash. `LASH_HOME` env var overrides;
/// otherwise `~/.lash/`.
pub fn lash_home() -> PathBuf {
    if let Ok(dir) = std::env::var("LASH_HOME") {
        PathBuf::from(dir)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lash")
    }
}

/// Cache directory. `$LASH_HOME/cache` when set, else `~/.cache/lash/`.
pub fn lash_cache_dir() -> PathBuf {
    if std::env::var("LASH_HOME").is_ok() {
        lash_home().join("cache")
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("lash")
    }
}

/// Preferred repo-local directory for lash artifacts.
pub fn repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".agents").join("lash")
}

/// Legacy repo-local directory for lash artifacts.
pub fn legacy_repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".lash")
}

/// Skill search directories, lowest to highest priority.
pub fn default_skill_dirs() -> Vec<PathBuf> {
    vec![
        lash_home().join("skills"),
        legacy_repo_local_lash_dir().join("skills"),
        repo_local_lash_dir().join("skills"),
    ]
}

/// Path to the CLI's provider config JSON file.
pub fn config_file() -> PathBuf {
    lash_home().join("config.json")
}

pub fn attachments_dir() -> PathBuf {
    lash_home().join("attachments")
}

/// Path to the CLI's model catalog cache.
pub fn model_catalog_cache_file() -> PathBuf {
    lash_cache_dir().join("models.json")
}
