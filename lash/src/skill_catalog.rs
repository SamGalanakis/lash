use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedSkill {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub argument_options: Vec<String>,
    pub instructions: String,
    pub path_to_skill_md: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct SkillCatalog {
    skills: Vec<LoadedSkill>,
}

struct ParsedFrontmatter {
    name: String,
    description: String,
    argument_hint: Option<String>,
    argument_options: Option<Vec<String>>,
    instructions: String,
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
                let ParsedFrontmatter {
                    mut name,
                    description,
                    argument_hint,
                    argument_options: explicit_argument_options,
                    instructions,
                } = match parse_frontmatter(&text) {
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

                let argument_options = explicit_argument_options
                    .filter(|options| !options.is_empty())
                    .unwrap_or_else(|| {
                        argument_hint
                            .as_deref()
                            .map(parse_argument_options)
                            .unwrap_or_default()
                    });

                skills.retain(|skill: &LoadedSkill| skill.name != name);
                skills.push(LoadedSkill {
                    name,
                    description,
                    argument_hint,
                    argument_options,
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

    pub fn argument_hint(&self, name: &str) -> Option<&str> {
        self.get(name)
            .and_then(|skill| skill.argument_hint.as_deref())
    }

    pub fn argument_options(&self, name: &str) -> &[String] {
        self.get(name)
            .map(|skill| skill.argument_options.as_slice())
            .unwrap_or(&[])
    }
}

fn parse_frontmatter(text: &str) -> Option<ParsedFrontmatter> {
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
    let mut argument_hint = None;
    let mut argument_options = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("name:") {
            name = parse_frontmatter_value(value);
        } else if let Some(value) = line.strip_prefix("description:") {
            description = parse_frontmatter_value(value);
        } else if let Some(value) = line.strip_prefix("argument-hint:") {
            let parsed = parse_frontmatter_value(value);
            if !parsed.is_empty() {
                argument_hint = Some(parsed);
            }
        } else if let Some(value) = line.strip_prefix("argument-options:") {
            let parsed = parse_argument_options(&parse_frontmatter_value(value));
            if !parsed.is_empty() {
                argument_options = Some(parsed);
            }
        }
    }

    Some(ParsedFrontmatter {
        name,
        description,
        argument_hint,
        argument_options,
        instructions: body,
    })
}

fn parse_frontmatter_value(value: &str) -> String {
    let trimmed = value.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(trimmed);
    unquoted.trim().to_string()
}

fn parse_argument_options(value: &str) -> Vec<String> {
    let trimmed = value.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(trimmed);
    let separator = if inner.contains('|') {
        '|'
    } else if inner.contains(',') {
        ','
    } else {
        return Vec::new();
    };
    inner
        .split(separator)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
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

    #[test]
    fn parses_argument_hint_from_frontmatter() {
        let root = TempDir::new().unwrap();
        write_skill(
            root.path(),
            "demo",
            "---\nname: demo\ndescription: demo\nargument-hint: \"[alpha|beta]\"\n---\n# demo\n",
        );

        let catalog = SkillCatalog::from_dirs(&[root.path().to_path_buf()]);
        let skill = catalog.get("demo").expect("skill");
        assert_eq!(skill.argument_hint.as_deref(), Some("[alpha|beta]"));
        assert_eq!(catalog.argument_hint("demo"), Some("[alpha|beta]"));
        assert_eq!(
            catalog.argument_options("demo"),
            &["alpha".to_string(), "beta".to_string()]
        );
    }

    #[test]
    fn explicit_argument_options_override_derived_hint_options() {
        let root = TempDir::new().unwrap();
        write_skill(
            root.path(),
            "demo",
            "---\nname: demo\ndescription: demo\nargument-hint: \"[placeholder]\"\nargument-options: \"alpha, beta\"\n---\n# demo\n",
        );

        let catalog = SkillCatalog::from_dirs(&[root.path().to_path_buf()]);
        assert_eq!(
            catalog.argument_options("demo"),
            &["alpha".to_string(), "beta".to_string()]
        );
    }
}
