use std::collections::HashSet;

use crate::{LoadedSkill, SkillCatalog};

pub fn collect_skill_mentions(text: &str) -> Vec<String> {
    collect_skill_mentions_with_ranges(text)
        .into_iter()
        .map(|(_, name)| name)
        .collect()
}

pub fn collect_skill_mentions_with_ranges(text: &str) -> Vec<(std::ops::Range<usize>, String)> {
    let bytes = text.as_bytes();
    let mut mentions = Vec::new();
    let mut idx = 0usize;

    while idx < bytes.len() {
        let sigil = match bytes[idx] {
            b'/' => bytes[idx],
            _ => {
                idx += 1;
                continue;
            }
        };

        if !is_valid_skill_mention_start(bytes, idx, sigil) {
            idx += 1;
            continue;
        }

        let start = idx + 1;
        let mut end = start;
        while end < bytes.len() && is_skill_name_byte(bytes[end]) {
            end += 1;
        }
        if end > start && is_valid_skill_mention_end(bytes, end, sigil) {
            mentions.push((idx..end, text[start..end].to_string()));
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

fn is_valid_skill_mention_start(bytes: &[u8], idx: usize, sigil: u8) -> bool {
    if idx == 0 {
        return true;
    }
    let prev = bytes[idx - 1];
    match sigil {
        b'/' => {
            prev.is_ascii_whitespace()
                || matches!(prev, b'(' | b'[' | b'{' | b'<' | b'"' | b'\'' | b'`')
        }
        _ => false,
    }
}

fn is_valid_skill_mention_end(bytes: &[u8], end: usize, sigil: u8) -> bool {
    !matches!((sigil, bytes.get(end).copied()), (b'/', Some(b'/')))
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
        let catalog = SkillCatalog::from_dirs(std::slice::from_ref(&root));
        let _ = std::fs::remove_dir_all(root);
        catalog
    }

    #[test]
    fn ignores_dollar_prefixed_text() {
        assert!(collect_skill_mentions("Use $frontend-design and then $wholehog.").is_empty());
    }

    #[test]
    fn collects_skill_mentions_with_ranges_including_sigil() {
        let mentions =
            collect_skill_mentions_with_ranges("Use /frontend-design and then /wholehog.");

        assert_eq!(mentions.len(), 2);
        assert_eq!(mentions[0].0, 4..20);
        assert_eq!(mentions[0].1, "frontend-design");
        assert_eq!(mentions[1].0, 30..39);
        assert_eq!(mentions[1].1, "wholehog");
    }

    #[test]
    fn collects_inline_slash_skill_mentions() {
        assert_eq!(
            collect_skill_mentions("See /localref opencode, then (/wholehog) again."),
            vec!["localref".to_string(), "wholehog".to_string()]
        );
    }

    #[test]
    fn ignores_path_like_slash_segments() {
        assert!(collect_skill_mentions("/localref/opencode and https://example.com").is_empty());
    }

    #[test]
    fn appends_selected_skills_as_blocks() {
        let catalog = skill_catalog_with(&[("frontend-design", "demo"), ("wholehog", "demo")]);
        let expanded = append_skill_blocks(
            "Use /frontend-design for this page and /wholehog for the cutover.",
            &catalog,
        );

        assert!(expanded.contains("<skill>\n<name>frontend-design</name>"));
        assert!(expanded.contains("<skill>\n<name>wholehog</name>"));
        assert!(expanded.contains("body for frontend-design"));
    }

    #[test]
    fn ignores_unknown_mentions_and_deduplicates_known_ones() {
        let catalog = skill_catalog_with(&[("demo", "demo")]);
        let expanded = append_skill_blocks("/demo then $unknown then /demo again", &catalog);

        assert_eq!(expanded.matches("<skill>").count(), 1);
        assert!(expanded.contains("<name>demo</name>"));
    }
}
