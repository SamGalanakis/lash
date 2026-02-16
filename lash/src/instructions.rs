use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Loads project instruction files (AGENTS.md, CLAUDE.md) with deduplication.
///
/// System-level instructions are computed once at construction (global + ancestor walk).
/// Context-aware instructions are resolved per file read, walking from the file's
/// directory up to the project root.
pub struct InstructionLoader {
    /// Cached system instructions text (computed once).
    system_text: String,
    /// Paths already included in system instructions (for dedup).
    system_paths: HashSet<PathBuf>,
    /// Project root boundary (cwd at construction time).
    project_root: PathBuf,
    /// Context-aware paths already loaded (dedup across file reads).
    loaded_context: Mutex<HashSet<PathBuf>>,
}

impl Default for InstructionLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionLoader {
    /// Create a new loader. Computes system-level instructions immediately.
    pub fn new() -> Self {
        let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut system_paths = HashSet::new();
        let mut parts = Vec::new();

        // 1. Global: ~/.lash/AGENT.md
        if let Some(home) = dirs::home_dir() {
            let global = home.join(".lash").join("AGENT.md");
            if let Some(text) = load_with_prefix(&global) {
                system_paths.insert(global);
                parts.push(text);
            }
        }

        // 2. Walk ancestors root → cwd (most-specific-last)
        let ancestors: Vec<_> = project_root.ancestors().collect();
        for dir in ancestors.into_iter().rev() {
            // First match wins per directory: AGENTS.md > CLAUDE.md
            if let Some(path) = find_instruction_file(dir)
                && let Some(text) = load_with_prefix(&path)
            {
                system_paths.insert(path);
                parts.push(text);
            }
        }

        Self {
            system_text: parts.join("\n\n"),
            system_paths,
            project_root,
            loaded_context: Mutex::new(HashSet::new()),
        }
    }

    /// Return the cached system-level instructions.
    pub fn system_instructions(&self) -> &str {
        &self.system_text
    }

    /// Resolve context-aware instructions for a file being read.
    ///
    /// Walks from `filepath`'s parent directory up to the project root.
    /// Returns formatted instruction text if a new (not yet loaded) instruction
    /// file is found, or None.
    pub fn resolve(&self, filepath: &str) -> Option<String> {
        let file_path = Path::new(filepath);
        let start_dir = file_path.parent()?;

        // Only resolve within the project root
        if !start_dir.starts_with(&self.project_root) {
            return None;
        }

        // Walk UP from start_dir toward project_root
        let mut dir = start_dir.to_path_buf();
        let mut parts = Vec::new();

        loop {
            if let Some(path) = find_instruction_file(&dir) {
                // Skip if already in system instructions
                if !self.system_paths.contains(&path) {
                    let mut loaded = self.loaded_context.lock().unwrap();
                    // Skip if already loaded via a previous context-aware resolve
                    if !loaded.contains(&path)
                        && let Some(text) = load_with_prefix(&path)
                    {
                        loaded.insert(path);
                        parts.push(text);
                    }
                }
            }

            // Stop at project root
            if dir == self.project_root {
                break;
            }
            match dir.parent() {
                Some(parent) if parent != dir => dir = parent.to_path_buf(),
                _ => break,
            }
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

/// Check for instruction files in priority order. First match wins.
fn find_instruction_file(dir: &Path) -> Option<PathBuf> {
    let candidates = [dir.join("AGENTS.md"), dir.join("CLAUDE.md")];
    candidates.into_iter().find(|p| p.is_file())
}

/// Read a file and prefix with its source path.
fn load_with_prefix(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!(
        "# Instructions from: {}\n\n{}",
        path.display(),
        trimmed
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn find_instruction_file_prefers_agents() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agents").unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude").unwrap();
        let found = find_instruction_file(dir.path()).unwrap();
        assert!(found.ends_with("AGENTS.md"));
    }

    #[test]
    fn find_instruction_file_falls_back_to_claude() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude").unwrap();
        let found = find_instruction_file(dir.path()).unwrap();
        assert!(found.ends_with("CLAUDE.md"));
    }

    #[test]
    fn find_instruction_file_none() {
        let dir = TempDir::new().unwrap();
        assert!(find_instruction_file(dir.path()).is_none());
    }

    #[test]
    fn load_with_prefix_nonempty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("TEST.md");
        std::fs::write(&path, "hello world").unwrap();
        let result = load_with_prefix(&path).unwrap();
        assert!(result.contains("# Instructions from:"));
        assert!(result.contains("hello world"));
    }

    #[test]
    fn load_with_prefix_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("EMPTY.md");
        std::fs::write(&path, "   \n  ").unwrap();
        assert!(load_with_prefix(&path).is_none());
    }

    #[test]
    fn load_with_prefix_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("NOPE.md");
        assert!(load_with_prefix(&path).is_none());
    }

    #[test]
    fn resolve_finds_instruction_in_parent() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("AGENTS.md"), "sub instructions").unwrap();

        // Build a loader with project_root = dir, no system paths
        let loader = InstructionLoader {
            system_text: String::new(),
            system_paths: HashSet::new(),
            project_root: dir.path().to_path_buf(),
            loaded_context: Mutex::new(HashSet::new()),
        };

        let file = sub.join("code.rs");
        let result = loader.resolve(file.to_str().unwrap());
        assert!(result.is_some());
        assert!(result.unwrap().contains("sub instructions"));
    }

    #[test]
    fn resolve_dedup_same_file() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("AGENTS.md"), "instructions").unwrap();

        let loader = InstructionLoader {
            system_text: String::new(),
            system_paths: HashSet::new(),
            project_root: dir.path().to_path_buf(),
            loaded_context: Mutex::new(HashSet::new()),
        };

        let file = sub.join("code.rs");
        let r1 = loader.resolve(file.to_str().unwrap());
        assert!(r1.is_some());
        let r2 = loader.resolve(file.to_str().unwrap());
        assert!(r2.is_none()); // dedup: already loaded
    }

    #[test]
    fn resolve_outside_project_root() {
        let dir = TempDir::new().unwrap();
        let loader = InstructionLoader {
            system_text: String::new(),
            system_paths: HashSet::new(),
            project_root: dir.path().to_path_buf(),
            loaded_context: Mutex::new(HashSet::new()),
        };
        assert!(loader.resolve("/tmp/outside/file.rs").is_none());
    }
}
