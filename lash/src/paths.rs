use std::path::PathBuf;

/// Return the root data directory for lash.
///
/// Checks `LASH_HOME` env var first, falling back to `~/.lash/`.
pub fn lash_home() -> PathBuf {
    if let Ok(dir) = std::env::var("LASH_HOME") {
        PathBuf::from(dir)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lash")
    }
}

/// Return the cache directory for lash.
///
/// When `LASH_HOME` is set: `$LASH_HOME/cache`.
/// Otherwise: `~/.cache/lash/` (via `dirs::cache_dir`).
pub fn lash_cache_dir() -> PathBuf {
    if std::env::var("LASH_HOME").is_ok() {
        lash_home().join("cache")
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("lash")
    }
}

/// Return the preferred repo-local directory for lash artifacts.
pub fn repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".agents").join("lash")
}

/// Return the legacy repo-local directory for lash artifacts.
pub fn legacy_repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".lash")
}

/// Return skill search directories in override order from lowest to highest priority.
pub fn default_skill_dirs() -> Vec<PathBuf> {
    vec![
        lash_home().join("skills"),
        legacy_repo_local_lash_dir().join("skills"),
        repo_local_lash_dir().join("skills"),
    ]
}
