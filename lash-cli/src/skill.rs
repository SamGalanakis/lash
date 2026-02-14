use std::path::PathBuf;

/// A skill loaded from a markdown file with YAML frontmatter.
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
}

impl Skill {
    /// Parse a skill from a markdown file.
    /// Expected format:
    /// ```
    /// ---
    /// name: my-skill
    /// description: One-liner for autocomplete
    /// ---
    /// Markdown body (instructions for the LLM)
    /// ```
    fn parse(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let text = text.trim_start();

        // Must start with frontmatter delimiter
        if !text.starts_with("---") {
            return None;
        }

        // Find the closing delimiter
        let after_open = &text[3..];
        let close_idx = after_open.find("\n---")?;
        let frontmatter = &after_open[..close_idx];
        let body_start = 3 + close_idx + 4; // skip "---" + "\n---"
        let content = text[body_start..].trim().to_string();

        let mut name = None;
        let mut description = None;

        for line in frontmatter.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("name:") {
                name = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("description:") {
                description = Some(val.trim().to_string());
            }
        }

        let name = name.filter(|n| !n.is_empty())?;

        // Validate name: only lowercase alphanumeric and hyphens
        if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            return None;
        }

        Some(Skill {
            name,
            description: description.unwrap_or_default(),
            content,
        })
    }
}

/// Registry of all loaded skills.
pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    /// Load skills from global (~/.lash/skills/) and project-local (.lash/skills/) directories.
    /// Project-local skills override global ones by name.
    pub fn load() -> Self {
        let mut skills = Vec::new();

        // Global skills
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let global_dir = PathBuf::from(home).join(".lash").join("skills");
        Self::load_dir(&global_dir, &mut skills);

        // Project-local skills (override global by name)
        let local_dir = PathBuf::from(".lash").join("skills");
        Self::load_dir(&local_dir, &mut skills);

        skills.sort_by(|a, b| a.name.cmp(&b.name));

        SkillRegistry { skills }
    }

    fn load_dir(dir: &std::path::Path, skills: &mut Vec<Skill>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if let Some(skill) = Skill::parse(&path) {
                // Remove any existing skill with the same name (project-local overrides global)
                skills.retain(|s| s.name != skill.name);
                skills.push(skill);
            }
        }
    }

    /// Return skills matching a prefix, as ("/name", "description") tuples.
    pub fn completions(&self, prefix: &str) -> Vec<(String, String)> {
        let cmd_prefix = prefix.strip_prefix('/').unwrap_or(prefix);
        self.skills
            .iter()
            .filter(|s| s.name.starts_with(cmd_prefix))
            .map(|s| (format!("/{}", s.name), s.description.clone()))
            .collect()
    }

    /// Look up a skill by exact name.
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    /// Whether any skills are loaded.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Iterate over all skills.
    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.iter()
    }
}
