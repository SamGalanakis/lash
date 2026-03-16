use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::plugin::{
    MessageMutatorHook, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::{LoadedSkill, Message, MessageRole, Part, PartKind, PromptContribution, SkillCatalog};

#[derive(Clone)]
pub struct SkillsPluginFactory {
    skill_dirs: Vec<PathBuf>,
}

struct SkillsPlugin {
    skill_dirs: Vec<PathBuf>,
}

impl SkillsPluginFactory {
    pub fn new(skill_dirs: Vec<PathBuf>) -> Self {
        Self { skill_dirs }
    }
}

impl PluginFactory for SkillsPluginFactory {
    fn id(&self) -> &'static str {
        "skills"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(SkillsPlugin {
            skill_dirs: self.skill_dirs.clone(),
        }))
    }
}

pub(crate) fn skills_prompt_contributions(catalog: &SkillCatalog) -> Vec<PromptContribution> {
    if catalog.is_empty() {
        return Vec::new();
    }

    vec![PromptContribution::guidance(
        "### Skills\nWhen the user explicitly invokes a skill, lash injects a `<skill>` block containing that skill's instructions for the current turn. Follow injected skill blocks directly instead of trying to rediscover skill files yourself.",
    )]
}

impl SessionPlugin for SkillsPlugin {
    fn id(&self) -> &'static str {
        "skills"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let prompt_skill_dirs = self.skill_dirs.clone();
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let prompt_skill_dirs = prompt_skill_dirs.clone();
            Box::pin(async move {
                let catalog = SkillCatalog::from_dirs(&prompt_skill_dirs);
                Ok(skills_prompt_contributions(&catalog))
            })
        }));

        let message_skill_dirs = self.skill_dirs.clone();
        reg.messages().mutator(
            MessageMutatorHook::BeforeTurn,
            Arc::new(move |_ctx, messages| {
                let message_skill_dirs = message_skill_dirs.clone();
                Box::pin(async move {
                    let catalog = SkillCatalog::from_dirs(&message_skill_dirs);
                    inject_skill_blocks(messages, &catalog)
                })
            }),
        )?;
        Ok(())
    }
}

fn inject_skill_blocks(
    mut messages: Vec<Message>,
    catalog: &SkillCatalog,
) -> Result<Vec<Message>, PluginError> {
    if catalog.is_empty() {
        return Ok(messages);
    }

    let Some(message_idx) = messages
        .iter()
        .rposition(|message| message.role == MessageRole::User)
    else {
        return Ok(messages);
    };
    let message = &mut messages[message_idx];
    let skills = collect_selected_skills(message, catalog);
    if skills.is_empty() {
        return Ok(messages);
    }

    for skill in skills {
        let part_id = format!("{}.p{}", message.id, message.parts.len());
        message.parts.push(Part {
            id: part_id,
            kind: PartKind::Text,
            content: render_skill_block(skill),
            tool_call_id: None,
            tool_name: None,
            prune_state: crate::PruneState::Intact,
        });
    }

    Ok(messages)
}

fn collect_selected_skills<'a>(
    message: &Message,
    catalog: &'a SkillCatalog,
) -> Vec<&'a LoadedSkill> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for part in &message.parts {
        if part.kind != PartKind::Text {
            continue;
        }
        for name in collect_skill_mentions(&part.content) {
            if seen.insert(name.clone())
                && let Some(skill) = catalog.get(&name)
            {
                selected.push(skill);
            }
        }
    }

    selected
}

fn collect_skill_mentions(text: &str) -> Vec<String> {
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
    use tempfile::TempDir;

    fn write_skill(root: &std::path::Path, dir_name: &str, name: &str) {
        let skill_dir = root.join(dir_name);
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: demo\n---\n\nbody for {name}\n"),
        )
        .expect("skill");
    }

    fn user_message(text: &str) -> Message {
        Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: text.to_string(),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        }
    }

    #[test]
    fn prompt_contributions_are_omitted_when_no_skills_exist() {
        let catalog = SkillCatalog::default();
        assert!(skills_prompt_contributions(&catalog).is_empty());
    }

    #[test]
    fn collects_skill_mentions_from_plain_text() {
        assert_eq!(
            collect_skill_mentions("Use $frontend-design and then $wholehog."),
            vec!["frontend-design".to_string(), "wholehog".to_string()]
        );
    }

    #[test]
    fn injects_selected_skills_into_last_user_message() {
        let dir = TempDir::new().expect("tmp");
        write_skill(dir.path(), "frontend-design", "frontend-design");
        let catalog = SkillCatalog::from_dirs(&[dir.path().to_path_buf()]);

        let messages = vec![user_message("Use $frontend-design for this page.")];
        let mutated = inject_skill_blocks(messages, &catalog).expect("mutated");
        let last = mutated.last().expect("message");

        assert_eq!(last.parts.len(), 2);
        assert!(
            last.parts[1]
                .content
                .contains("<skill>\n<name>frontend-design</name>")
        );
        assert!(last.parts[1].content.contains("body for frontend-design"));
    }

    #[test]
    fn ignores_unknown_mentions_and_deduplicates_known_ones() {
        let dir = TempDir::new().expect("tmp");
        write_skill(dir.path(), "demo", "demo");
        let catalog = SkillCatalog::from_dirs(&[dir.path().to_path_buf()]);

        let messages = vec![user_message("$demo then $unknown then $demo again")];
        let mutated = inject_skill_blocks(messages, &catalog).expect("mutated");
        let last = mutated.last().expect("message");

        assert_eq!(last.parts.len(), 2);
        assert!(last.parts[1].content.contains("<name>demo</name>"));
    }
}
