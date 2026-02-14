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
            if let Some(path) = find_instruction_file(dir) {
                if let Some(text) = load_with_prefix(&path) {
                    system_paths.insert(path);
                    parts.push(text);
                }
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
                    if !loaded.contains(&path) {
                        if let Some(text) = load_with_prefix(&path) {
                            loaded.insert(path);
                            parts.push(text);
                        }
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
    let candidates = [
        dir.join("AGENTS.md"),
        dir.join("CLAUDE.md"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Read a file and prefix with its source path.
fn load_with_prefix(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(format!("# Instructions from: {}\n\n{}", path.display(), trimmed))
}
