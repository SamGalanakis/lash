use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde_json::json;

use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs};
use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

use super::run_blocking;

type SkillCatalogEntry = (String, String, usize);
type SkillCatalogCache = Arc<RwLock<Option<Vec<SkillCatalogEntry>>>>;

#[derive(Clone)]
pub struct StateStore {
    skill_dirs: Vec<PathBuf>,
    skill_catalog_cache: SkillCatalogCache,
}

impl StateStore {
    pub fn new(skill_dirs: Vec<PathBuf>) -> Self {
        Self {
            skill_dirs,
            skill_catalog_cache: Arc::new(RwLock::new(None)),
        }
    }

    fn parse_skill_frontmatter(text: &str) -> Option<(String, String)> {
        let text = text.trim_start();
        if !text.starts_with("---") {
            return None;
        }
        let after_open = &text[3..];
        let close_idx = after_open.find("\n---")?;
        let frontmatter = &after_open[..close_idx];
        let mut name = String::new();
        let mut description = String::new();
        for line in frontmatter.lines() {
            let line = line.trim();
            if let Some(v) = line.strip_prefix("name:") {
                name = v.trim().to_string();
            } else if let Some(v) = line.strip_prefix("description:") {
                description = v.trim().to_string();
            }
        }
        Some((name, description))
    }

    fn build_skill_catalog(&self) -> Vec<(String, String, usize)> {
        let mut by_name: HashMap<String, (String, usize)> = HashMap::new();
        for dir in &self.skill_dirs {
            let entries = match std::fs::read_dir(dir) {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let skill_md = path.join("SKILL.md");
                if !skill_md.is_file() {
                    continue;
                }
                let text = match std::fs::read_to_string(&skill_md) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let Some((mut name, description)) = Self::parse_skill_frontmatter(&text) else {
                    continue;
                };
                if name.is_empty() {
                    name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default()
                        .to_string();
                }
                if name.is_empty() {
                    continue;
                }
                if !name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                {
                    continue;
                }
                let file_count = ignore::WalkBuilder::new(&path)
                    .hidden(false)
                    .build()
                    .filter_map(Result::ok)
                    .filter(|e| e.path().is_file())
                    .count()
                    .saturating_sub(1);
                by_name.insert(name, (description, file_count));
            }
        }
        let mut out: Vec<(String, String, usize)> = by_name
            .into_iter()
            .map(|(name, (description, file_count))| (name, description, file_count))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn discover_skills(&self) -> Vec<(String, String, usize)> {
        if let Ok(cache) = self.skill_catalog_cache.read()
            && let Some(skills) = cache.as_ref()
        {
            return skills.clone();
        }

        let skills = self.build_skill_catalog();
        if let Ok(mut cache) = self.skill_catalog_cache.write() {
            *cache = Some(skills.clone());
        }
        skills
    }

    fn fallback_tool_catalog(&self) -> Vec<serde_json::Value> {
        self.definitions()
            .into_iter()
            .filter(|d| !d.hidden && !d.description_for(crate::ExecutionMode::Standard).is_empty())
            .map(|d| {
                let p = d.project(crate::ExecutionMode::Standard);
                json!({
                    "name": p.name,
                    "description": p.description,
                    "examples": p.examples,
                    "inject_into_prompt": p.inject_into_prompt,
                    "hidden": d.hidden,
                })
            })
            .collect()
    }

    fn tool_catalog(&self, args: &serde_json::Value) -> Vec<serde_json::Value> {
        args.get("catalog")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_else(|| self.fallback_tool_catalog())
    }

    fn list_tools(&self, args: &serde_json::Value) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .map(|v| v.to_ascii_lowercase());
        let injected_only = args.get("injected_only").and_then(|v| v.as_bool());
        let catalog = self.tool_catalog(args);

        let items: Vec<serde_json::Value> = catalog
            .into_iter()
            .filter(|t| {
                if t.get("hidden").and_then(|v| v.as_bool()).unwrap_or(false) {
                    return false;
                }
                if let Some(flag) = injected_only {
                    let inject = t
                        .get("inject_into_prompt")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if inject != flag {
                        return false;
                    }
                }
                if let Some(needle) = &query {
                    let name = t
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    let desc = t
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    return name.contains(needle) || desc.contains(needle);
                }
                true
            })
            .map(|t| {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                let desc = t
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let oneliner = desc.lines().next().unwrap_or("").trim();
                let signature = match (
                    t.get("params").and_then(|v| v.as_array()),
                    t.get("returns").and_then(|v| v.as_str()),
                ) {
                    (Some(params), Some(returns)) => {
                        let rendered = params
                            .iter()
                            .map(|param| {
                                let name = param
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default();
                                let ty =
                                    param.get("type").and_then(|v| v.as_str()).unwrap_or("any");
                                let required = param
                                    .get("required")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(true);
                                if required {
                                    format!("{name}: {ty}")
                                } else {
                                    format!("{name}: {ty} = None")
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{name}({rendered}) -> {returns}")
                    }
                    _ => String::new(),
                };
                json!({
                    "name": name,
                    "oneliner": oneliner,
                    "signature": signature,
                })
            })
            .collect();

        ToolResult::ok(json!(items))
    }

    fn search_tools(&self, args: &serde_json::Value) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = limit_from_args(args);
        let injected_only = args.get("injected_only").and_then(|v| v.as_bool());
        let catalog = self.tool_catalog(args);

        let mut filtered = Vec::new();
        for t in &catalog {
            if t.get("hidden").and_then(|v| v.as_bool()).unwrap_or(false) {
                continue;
            }
            if let Some(injected) = injected_only {
                let inject = t
                    .get("inject_into_prompt")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if injected != inject {
                    continue;
                }
            }
            filtered.push(t.clone());
        }

        let docs: Vec<SearchDoc> = filtered
            .iter()
            .map(|t| {
                let examples = t
                    .get("examples")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let mut fields = HashMap::new();
                fields.insert(
                    "name",
                    t.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "description",
                    t.get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert("examples", examples);
                SearchDoc { fields }
            })
            .collect();
        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[("name", 4.0), ("description", 2.0), ("examples", 1.0)],
        );
        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, _)| {
                let mut tool = filtered[idx].clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.insert("score".to_string(), json!(score));
                }
                tool
            })
            .collect();
        ToolResult::ok(json!(items))
    }

    fn search_skills(&self, args: &serde_json::Value) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = limit_from_args(args);
        let skills = self.discover_skills();
        let docs: Vec<SearchDoc> = skills
            .iter()
            .map(|(name, description, _)| {
                let mut fields = HashMap::new();
                fields.insert("name", name.clone());
                fields.insert("description", description.clone());
                SearchDoc { fields }
            })
            .collect();
        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[("name", 4.0), ("description", 2.0)],
        );
        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, _)| {
                let (name, description, file_count) = &skills[idx];
                json!({
                    "__type__": "skill_summary",
                    "name": name,
                    "description": description,
                    "file_count": file_count,
                    "score": score,
                })
            })
            .collect();
        ToolResult::ok(json!(items))
    }
}

#[async_trait::async_trait]
impl ToolProvider for StateStore {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![
            ToolDefinition {
                name: "list_tools".into(),
                description: vec![crate::ToolText::new(
                    "List available tools with compact summaries. Use search_tools for ranked or filtered results.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::optional("query", "str"),
                    ToolParam::optional("injected_only", "bool"),
                ],
                returns: "list".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
            ToolDefinition {
                name: "search_tools".into(),
                description: vec![crate::ToolText::new(
                    "Search tools using hybrid/literal/regex matching. Results include relevance scores.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::typed("query", "str"),
                    ToolParam::optional("mode", "str"),
                    ToolParam::optional("regex", "str"),
                    ToolParam::optional("limit", "int"),
                    ToolParam::optional("injected_only", "bool"),
                ],
                returns: "list".into(),
                examples: vec![],
                hidden: false,
                inject_into_prompt: false,
            },
        ];
        defs.push(ToolDefinition {
            name: "search_skills".into(),
            description: vec![crate::ToolText::new(
                "Search installed skills using hybrid/literal/regex matching.",
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam::typed("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
            ],
            returns: "list[SkillSummary]".into(),
            examples: vec![],
            hidden: false,
            inject_into_prompt: false,
        });
        defs
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let this = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "list_tools" => this.list_tools(&args),
            "search_tools" => this.search_tools(&args),
            "search_skills" => this.search_skills(&args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> StateStore {
        StateStore::new(Vec::new())
    }

    #[test]
    fn tool_search_uses_catalog() {
        let p = provider();
        let result = p.search_tools(&json!({
            "query":"patch",
            "mode":"hybrid",
            "limit":10,
            "catalog":[
                {"name":"read_file","description":"Read file","examples":[],"inject_into_prompt":true},
                {"name":"apply_patch","description":"Apply structured patch","examples":[],"inject_into_prompt":true}
            ]
        }));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert!(!items.is_empty());
        assert_eq!(
            items[0].get("name").and_then(|v| v.as_str()),
            Some("apply_patch")
        );
    }

    #[test]
    fn list_tools_works_without_catalog_arg() {
        let p = provider();
        let result = p.list_tools(&json!({
            "query":"list_tools"
        }));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert!(!items.is_empty());
        assert_eq!(
            items[0].get("name").and_then(|v| v.as_str()),
            Some("list_tools")
        );
        assert!(items[0].get("signature").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn search_tools_works_without_catalog_arg() {
        let p = provider();
        let result = p.search_tools(&json!({
            "query":"search tools",
            "mode":"hybrid",
            "limit":10
        }));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert!(!items.is_empty());
    }

    #[test]
    fn state_store_definitions_exclude_plugin_owned_tools() {
        let p = provider();
        let names: Vec<String> = p.definitions().into_iter().map(|def| def.name).collect();
        assert!(!names.iter().any(|name| name == "search_history"));
        assert!(!names.iter().any(|name| name == "search_mem"));
        assert!(!names.iter().any(|name| name == "mem_set"));
        assert!(!names.iter().any(|name| name == "history_add_turn"));
        assert!(!names.iter().any(|name| name == "mem_load"));
    }

    #[test]
    fn skills_search_is_defined() {
        let p = StateStore::new(Vec::new());
        let names: Vec<String> = p.definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "search_skills"));
    }
}
