use std::path::PathBuf;

use anyhow::anyhow;
use lash::session_model::{PromptOverrideMode, PromptSectionName, PromptSectionOverride};

use crate::Args;

fn parse_section(raw: &str) -> anyhow::Result<PromptSectionName> {
    raw.parse::<PromptSectionName>().map_err(anyhow::Error::msg)
}

fn parse_target(raw: &str) -> anyhow::Result<(PromptSectionName, Option<String>)> {
    let raw = raw.trim();
    let (section_raw, block_raw) = match raw.split_once('.') {
        Some((section_raw, block_raw)) => (section_raw.trim(), Some(block_raw.trim())),
        None => (raw, None),
    };
    let section = parse_section(section_raw)?;
    let block = match block_raw {
        Some("") => {
            return Err(anyhow!(
                "Expected SECTION[.BLOCK], got `{raw}`. Valid sections: {}",
                PromptSectionName::names_csv()
            ));
        }
        Some(block) => Some(block.to_string()),
        None => None,
    };
    Ok((section, block))
}

fn parse_target_kv(raw: &str) -> anyhow::Result<(PromptSectionName, Option<String>, String)> {
    let Some((target_raw, value)) = raw.split_once('=') else {
        return Err(anyhow!(
            "Expected TARGET=VALUE, got `{raw}`. Valid sections: {}",
            PromptSectionName::names_csv()
        ));
    };
    let (section, block) = parse_target(target_raw)?;
    Ok((section, block, value.to_string()))
}

fn parse_target_file_kv(raw: &str) -> anyhow::Result<(PromptSectionName, Option<String>, PathBuf)> {
    let Some((target_raw, path_raw)) = raw.split_once('=') else {
        return Err(anyhow!(
            "Expected TARGET=PATH, got `{raw}`. Valid sections: {}",
            PromptSectionName::names_csv()
        ));
    };
    let (section, block) = parse_target(target_raw)?;
    Ok((section, block, PathBuf::from(path_raw)))
}

fn read_override_file(path: &std::path::Path, target: &str) -> anyhow::Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        anyhow!(
            "Failed reading override file {} for target {}: {}",
            path.display(),
            target,
            e
        )
    })
}

pub fn resolve_prompt_overrides(args: &Args) -> anyhow::Result<Vec<PromptSectionOverride>> {
    let mut overrides = Vec::new();

    for raw in &args.prompt_replace {
        let (section, block, content) = parse_target_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Replace,
            content,
        });
    }
    for raw in &args.prompt_replace_file {
        let (section, block, path) = parse_target_file_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Replace,
            content: read_override_file(&path, raw)?,
        });
    }

    for raw in &args.prompt_prepend {
        let (section, block, content) = parse_target_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Prepend,
            content,
        });
    }
    for raw in &args.prompt_prepend_file {
        let (section, block, path) = parse_target_file_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Prepend,
            content: read_override_file(&path, raw)?,
        });
    }

    for raw in &args.prompt_append {
        let (section, block, content) = parse_target_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Append,
            content,
        });
    }
    for raw in &args.prompt_append_file {
        let (section, block, path) = parse_target_file_kv(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Append,
            content: read_override_file(&path, raw)?,
        });
    }

    for raw in &args.prompt_disable {
        let (section, block) = parse_target(raw)?;
        overrides.push(PromptSectionOverride {
            section,
            block,
            mode: PromptOverrideMode::Disable,
            content: String::new(),
        });
    }

    Ok(overrides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_target_accepts_section_only_and_block_targets() {
        assert_eq!(
            parse_target("guidance").unwrap(),
            (PromptSectionName::Guidance, None)
        );
        assert_eq!(
            parse_target("guidance.project_instructions").unwrap(),
            (
                PromptSectionName::Guidance,
                Some("project_instructions".to_string())
            )
        );
    }
}
