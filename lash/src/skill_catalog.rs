use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedSkill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub path_to_skill_md: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct SkillCatalog {
    skills: Vec<LoadedSkill>,
}

impl SkillCatalog {
    pub fn load() -> Self {
        Self::from_dirs(&crate::default_skill_dirs())
    }

    pub fn from_dirs(skill_dirs: &[PathBuf]) -> Self {
        let mut skills = Vec::new();

        for dir in skill_dirs {
            let entries = match std::fs::read_dir(dir) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let skill_dir = entry.path();
                if !skill_dir.is_dir() {
                    continue;
                }
                let skill_md = skill_dir.join("SKILL.md");
                if !skill_md.is_file() {
                    continue;
                }
                let text = match std::fs::read_to_string(&skill_md) {
                    Ok(text) => text,
                    Err(_) => continue,
                };
                let (mut name, description, instructions) = match parse_frontmatter(&text) {
                    Some(parsed) => parsed,
                    None => continue,
                };
                if name.is_empty() {
                    let Some(dir_name) = skill_dir.file_name().and_then(|name| name.to_str())
                    else {
                        continue;
                    };
                    name = dir_name.to_string();
                }
                if !is_valid_skill_name(&name) {
                    continue;
                }

                let path_to_skill_md = match skill_md.canonicalize() {
                    Ok(path) => path,
                    Err(_) => continue,
                };

                skills.retain(|skill: &LoadedSkill| skill.name != name);
                skills.push(LoadedSkill {
                    name,
                    description,
                    instructions,
                    path_to_skill_md,
                });
            }
        }

        skills.sort_by(|left, right| left.name.cmp(&right.name));
        Self { skills }
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&LoadedSkill> {
        self.skills.iter().find(|skill| skill.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &LoadedSkill> {
        self.skills.iter()
    }
}

fn parse_frontmatter(text: &str) -> Option<(String, String, String)> {
    let text = text.trim_start();
    if !text.starts_with("---") {
        return None;
    }

    let after_open = &text[3..];
    let close_idx = after_open.find("\n---")?;
    let frontmatter = &after_open[..close_idx];
    let body_start = 3 + close_idx + 4;
    let body = text[body_start..].trim().to_string();

    let mut name = String::new();
    let mut description = String::new();

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            name = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("description:") {
            description = value.trim().to_string();
        }
    }

    Some((name, description, body))
}

fn is_valid_skill_name(name: &str) -> bool {
    name.chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_skill(root: &std::path::Path, dir_name: &str, body: &str) {
        let skill_dir = root.join(dir_name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), body).unwrap();
    }

    #[test]
    fn later_directories_override_earlier_skills_by_name() {
        let global = TempDir::new().unwrap();
        let local = TempDir::new().unwrap();
        write_skill(
            global.path(),
            "demo",
            "---\nname: demo\ndescription: global\n---\n# global\n",
        );
        write_skill(
            local.path(),
            "demo",
            "---\nname: demo\ndescription: local\n---\n# local\n",
        );

        let catalog =
            SkillCatalog::from_dirs(&[global.path().to_path_buf(), local.path().to_path_buf()]);

        assert_eq!(
            catalog.get("demo").map(|skill| skill.description.as_str()),
            Some("local")
        );
    }

    #[test]
    fn stores_canonical_skill_path() {
        let root = TempDir::new().unwrap();
        write_skill(
            root.path(),
            "demo",
            "---\nname: demo\ndescription: demo\n---\n# demo\n",
        );

        let catalog = SkillCatalog::from_dirs(&[root.path().to_path_buf()]);
        let skill = catalog.get("demo").expect("skill");
        assert!(skill.path_to_skill_md.ends_with("SKILL.md"));
        assert!(skill.path_to_skill_md.is_absolute());
    }
}
