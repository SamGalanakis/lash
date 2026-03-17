use std::collections::HashSet;

use crate::{LoadedSkill, SkillCatalog};

pub fn collect_skill_mentions(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut mentions = Vec::new();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx] != b'$' {
            idx += 1;
            continue;
        }

        if idx > 0 && is_skill_name_byte(bytes[idx - 1]) {
            idx += 1;
            continue;
        }

        let start = idx + 1;
        let mut end = start;
        while end < bytes.len() && is_skill_name_byte(bytes[end]) {
            end += 1;
        }
        if end > start {
            mentions.push(text[start..end].to_string());
            idx = end;
        } else {
            idx += 1;
        }
    }

    mentions
}

pub fn append_skill_blocks(text: &str, catalog: &SkillCatalog) -> String {
    if catalog.is_empty() {
        return text.to_string();
    }

    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for name in collect_skill_mentions(text) {
        if seen.insert(name.clone())
            && let Some(skill) = catalog.get(&name)
        {
            selected.push(skill);
        }
    }
    if selected.is_empty() {
        return text.to_string();
    }

    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(
        &selected
            .into_iter()
            .map(render_skill_block)
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    out
}

fn is_skill_name_byte(byte: u8) -> bool {
    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
}

fn render_skill_block(skill: &LoadedSkill) -> String {
    format!(
        "<skill>\n<name>{}</name>\n<path>{}</path>\n{}\n</skill>",
        skill.name,
        skill.path_to_skill_md.display(),
        skill.instructions
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill_catalog_with(names: &[(&str, &str)]) -> SkillCatalog {
        let root = std::env::temp_dir().join(format!("lash-skill-prompt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        for (name, description) in names {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("skill dir");
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {description}\n---\n\nbody for {name}\n"),
            )
            .expect("skill file");
        }
        let catalog = SkillCatalog::from_dirs(&[root.clone()]);
        let _ = std::fs::remove_dir_all(root);
        catalog
    }

    #[test]
    fn collects_skill_mentions_from_plain_text() {
        assert_eq!(
            collect_skill_mentions("Use $frontend-design and then $wholehog."),
            vec!["frontend-design".to_string(), "wholehog".to_string()]
        );
    }

    #[test]
    fn appends_selected_skills_as_blocks() {
        let catalog = skill_catalog_with(&[("frontend-design", "demo")]);
        let expanded = append_skill_blocks("Use $frontend-design for this page.", &catalog);

        assert!(expanded.contains("<skill>\n<name>frontend-design</name>"));
        assert!(expanded.contains("body for frontend-design"));
    }

    #[test]
    fn ignores_unknown_mentions_and_deduplicates_known_ones() {
        let catalog = skill_catalog_with(&[("demo", "demo")]);
        let expanded = append_skill_blocks("$demo then $unknown then $demo again", &catalog);

        assert_eq!(expanded.matches("<skill>").count(), 1);
        assert!(expanded.contains("<name>demo</name>"));
    }
}
