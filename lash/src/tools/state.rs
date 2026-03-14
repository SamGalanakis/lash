use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde_json::json;

use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs};
use crate::{ToolDefinition, ToolParam, ToolPromptContext, ToolProvider, ToolResult};

use super::skills::discover_installed_skills;
use super::{INTERNAL_TOOL_CATALOG_ARG, run_blocking};

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

    fn build_skill_catalog(&self) -> Vec<(String, String, usize)> {
        discover_installed_skills(&self.skill_dirs)
            .into_iter()
            .map(|skill| (skill.name, skill.description, skill.files.len()))
            .collect()
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

    fn tool_catalog(&self, args: &serde_json::Value) -> Result<Vec<serde_json::Value>, ToolResult> {
        args.get(INTERNAL_TOOL_CATALOG_ARG)
            .and_then(|v| v.as_array())
            .cloned()
            .ok_or_else(|| {
                ToolResult::err(json!(
                    "search_tools requires the active session tool catalog; call it through the runtime or plugin host"
                ))
            })
    }

    fn search_tools(&self, args: &serde_json::Value) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let injected_only = args.get("injected_only").and_then(|v| v.as_bool());
        let catalog = match self.tool_catalog(args) {
            Ok(catalog) => catalog,
            Err(err) => return err,
        };

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

        if browse_all {
            filtered.sort_by(|left, right| {
                let left_name = left
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let right_name = right
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                left_name.cmp(right_name)
            });
            return ToolResult::ok(json!(filtered.into_iter().take(limit).collect::<Vec<_>>()));
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
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let skills = self.discover_skills();
        if browse_all {
            return ToolResult::ok(json!(
                skills
                    .into_iter()
                    .take(limit)
                    .map(|(name, description, file_count)| {
                        json!({
                            "name": name,
                            "description": description,
                            "file_count": file_count,
                        })
                    })
                    .collect::<Vec<_>>()
            ));
        }
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
        let mut defs = vec![ToolDefinition {
            name: "search_tools".into(),
            description: vec![crate::ToolText::new(
                "Discover available tools. With a focused `query`, returns ranked matches using hybrid/literal/regex search. With no `query`, returns the full active tool catalog in stable name order.",
                [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
            )],
            params: vec![
                ToolParam::optional("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("injected_only", "bool"),
            ],
            returns: "list".into(),
            examples: vec![
                crate::ToolText::new(
                    "call search_tools { query: \"task planning\" }",
                    [crate::ExecutionMode::Repl],
                ),
                crate::ToolText::new(
                    "search_tools(query=\"task planning\")",
                    [crate::ExecutionMode::Standard],
                ),
                crate::ToolText::new("call search_tools {}", [crate::ExecutionMode::Repl]),
                crate::ToolText::new("search_tools()", [crate::ExecutionMode::Standard]),
            ],
            hidden: false,
            inject_into_prompt: false,
        }];
        if !self.discover_skills().is_empty() {
            defs.push(ToolDefinition {
                name: "search_skills".into(),
                description: vec![crate::ToolText::new(
                    "Discover installed skills. With a focused `query`, returns ranked matches using hybrid/literal/regex search. With no `query`, returns the full installed skill catalog in stable name order.",
                    [crate::ExecutionMode::Repl, crate::ExecutionMode::Standard],
                )],
                params: vec![
                    ToolParam::optional("query", "str"),
                    ToolParam::optional("mode", "str"),
                    ToolParam::optional("regex", "str"),
                    ToolParam::optional("limit", "int"),
                ],
                returns: "list[SkillSummary]".into(),
                examples: vec![
                    crate::ToolText::new(
                        "call search_skills { query: \"benchmarking\" }",
                        [crate::ExecutionMode::Repl],
                    ),
                    crate::ToolText::new(
                        "search_skills(query=\"benchmarking\")",
                        [crate::ExecutionMode::Standard],
                    ),
                    crate::ToolText::new(
                        "call search_skills {}",
                        [crate::ExecutionMode::Repl],
                    ),
                    crate::ToolText::new(
                        "search_skills()",
                        [crate::ExecutionMode::Standard],
                    ),
                ],
                hidden: false,
                inject_into_prompt: true,
            });
        }
        defs
    }

    fn prompt_guides(&self, _context: &ToolPromptContext) -> Vec<String> {
        Vec::new()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let this = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
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
    use tempfile::TempDir;

    fn provider() -> StateStore {
        StateStore::new(Vec::new())
    }

    fn write_skill(root: &std::path::Path, dir_name: &str, name: &str, description: &str) {
        let skill_dir = root.join(dir_name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n# {name}\n"),
        )
        .unwrap();
    }

    #[test]
    fn tool_search_uses_internal_catalog() {
        let p = provider();
        let result = p.search_tools(&json!({
            "query":"patch",
            "mode":"hybrid",
            "limit":10,
            "__tool_catalog":[
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
    fn search_tools_errors_without_session_catalog() {
        let p = provider();
        let result = p.search_tools(&json!({
            "query":"search tools",
            "mode":"hybrid",
            "limit":10
        }));
        assert!(!result.success);
        assert!(
            result
                .result
                .as_str()
                .unwrap_or_default()
                .contains("active session tool catalog")
        );
    }

    #[test]
    fn search_tools_lists_all_without_query() {
        let p = provider();
        let result = p.search_tools(&json!({
            "__tool_catalog":[
                {"name":"search_tools","description":"Discover tools","examples":[],"inject_into_prompt":true},
                {"name":"read_file","description":"Read file","examples":[],"inject_into_prompt":true}
            ]
        }));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].get("name").and_then(|v| v.as_str()),
            Some("read_file")
        );
        assert_eq!(
            items[1].get("name").and_then(|v| v.as_str()),
            Some("search_tools")
        );
        assert!(items[0].get("score").is_none());
    }

    #[test]
    fn search_tools_hybrid_matches_broader_query_terms() {
        let p = provider();
        let result = p.search_tools(&json!({
            "query":"asking users",
            "mode":"hybrid",
            "limit":10,
            "__tool_catalog":[
                {"name":"ask","description":"Pause and ask the user a targeted question, then wait for the answer before continuing.","examples":[],"inject_into_prompt":true},
                {"name":"read_file","description":"Read file contents from disk.","examples":[],"inject_into_prompt":true}
            ]
        }));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert!(!items.is_empty());
        assert_eq!(items[0].get("name").and_then(|v| v.as_str()), Some("ask"));
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
    fn skills_search_is_omitted_when_no_skills_exist() {
        let p = StateStore::new(Vec::new());
        let names: Vec<String> = p.definitions().into_iter().map(|d| d.name).collect();
        assert!(!names.iter().any(|n| n == "search_skills"));
    }

    #[test]
    fn skills_search_is_defined_when_skills_exist() {
        let dir = TempDir::new().unwrap();
        write_skill(dir.path(), "alpha", "alpha", "Alpha skill");
        let p = StateStore::new(vec![dir.path().to_path_buf()]);
        let defs = p.definitions();
        assert!(
            defs.iter()
                .any(|def| def.name == "search_skills" && def.inject_into_prompt)
        );
    }

    #[test]
    fn search_skills_lists_all_without_query() {
        let dir = TempDir::new().unwrap();
        write_skill(dir.path(), "alpha", "alpha", "Alpha skill");
        write_skill(dir.path(), "beta", "beta", "Beta skill");
        let p = StateStore::new(vec![dir.path().to_path_buf()]);
        let result = p.search_skills(&json!({}));
        assert!(result.success);
        let items = result.result.as_array().cloned().unwrap_or_default();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].get("name").and_then(|v| v.as_str()), Some("alpha"));
        assert_eq!(items[1].get("name").and_then(|v| v.as_str()), Some("beta"));
        assert!(items[0].get("score").is_none());
    }
}
