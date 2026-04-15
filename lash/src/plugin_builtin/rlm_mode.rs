use std::sync::Arc;

#[path = "rlm_support.rs"]
mod rlm_support;

use crate::plugin::{
    ModeExecutionPlugin, ModeExecutionPreamble, ModeNativeToolsPlugin, ModeSessionPlugin,
    ModeTurnConfig, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::{
    ExecutionMode, ProgressSender, SessionError, ToolResult, ToolResultProjectionPluginConfig,
};

use self::rlm_support::{
    SearchToolsProvider, bound_variables_prompt_contributions, tool_discovery_prompt_contributions,
    tool_surface_contribution,
};

const RLM_EXECUTION_SECTION_BASE: &str = r#"In this mode you write `lashlang` code inside your prose response and the runtime executes it. There is no native tool-call envelope — you embed code directly.

Format every work step like this:

````
Brief reasoning here in plain prose (one or two sentences is fine).

```lashlang
result = call tool_name { arg: value }
observe result
```
````

- Wrap each work step in **exactly one** ` ```lashlang ` fenced block. Only the first block runs per turn — additional blocks are ignored.
- Plain prose alongside the block becomes your reasoning trace; keep it short.
- After each execution result, decide whether to write another fenced block (more work to do) or finish the turn in pure prose with no fenced block (task complete).
- When the task is complete, reply with prose only — no fenced lashlang block — and that ends the turn.
- Work iteratively: inspect, act, observe, continue. Most tasks take multiple lashlang steps, not one large step.
- Verify the concrete end state before finalizing in prose when possible.
- Do not describe what you would do instead of doing it.

### RLM Language

`lashlang` is a small workflow language for tool orchestration.

- Values are null, booleans, numbers, strings, lists, and records.
- List and record literals use comma-separated entries: `[a, b]`, `{ a: 1, b: 2 }`. Tool-argument records follow the same rule.
- Assign with `name = expr`. Variables persist across iterations — anything you bind in one fenced block is still in scope on the next.
- If the prompt includes a **Bound Variables** section, those names are already in scope. Access them directly in lashlang instead of rebuilding them from prose.
- Bare expressions are valid statements. In `parallel { ... }`, a bare-expression branch contributes that value to the result list.
- Call tools with `call tool_name { arg: expr }`.
- Start any tool call in the background with `start call tool_name { arg: expr }`. This returns a handle value.
- Resolve a background handle with `await handle`. If you already have a list of handles, `await handles` returns a list of results in order.
- Stop a background handle with `cancel handle`. Cancellation is best-effort: Lash always stops waiting locally, and cooperative tools are also asked to stop their underlying work.
- Use `parallel { ... }` only for independent tool calls. If one call needs another call's output, do not put them in the same `parallel { ... }`.
- `parallel { ... }` returns a list of branch results in order.
- Use ternary expressions for inline branching: `cond ? yes : no`. There is no expression-form `if`.
- Control flow is limited to statement `if` and `for`.
- Boolean negation supports both `!cond` and `not cond`.
- Use `observe expr` to inspect a value and continue execution.
- `observe` output and tool results are fed back into the next iteration (your context), so inspect first and refine on the next step if needed.
- You must explicitly use `observe` to inspect values and make progress based on them. Do not rely on implicit inspection through tool results or execution errors.

### Decomposition

- Break large tasks into smaller, self-contained steps.
- Prefer narrow checks over brute-force scanning when the input is large.
- Use focused intermediate observations to verify subquestions before finalizing.
- Keep each step concrete and bounded instead of attempting the whole task at once.
- Use `start`/`await` when a long-running tool can make progress in the background while you do other work. This is especially useful for `delegate`.

Example fanout pattern:

```lashlang
h1 = start call delegate { task: "Read chunk 1 and extract the key claim", intelligence: "low", output: { claim: "str" } }
h2 = start call delegate { task: "Read chunk 2 and extract the key claim", intelligence: "low", output: { claim: "str" } }
handles = [h1, h2]
results = await handles
finish results
```"#;

fn rlm_execution_section(observe_projection: &ToolResultProjectionPluginConfig) -> String {
    let mut prompt = String::from(RLM_EXECUTION_SECTION_BASE);
    prompt.push_str("\n\n");
    prompt.push_str(&format!(
        "Observe output is capped before reinjection using the current RLM observe limit (mode: `{}`, limit: {}, max_lines: {}). If you see a cap/truncation note, narrow the expression and inspect specific fields or slices instead of dumping the whole value.",
        match observe_projection.mode {
            crate::ToolResultProjectionMode::Bytes => "bytes",
            crate::ToolResultProjectionMode::Tokens => "tokens",
        },
        observe_projection.limit,
        observe_projection.max_lines,
    ));
    prompt
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
}

pub struct BuiltinRlmModePluginFactory {
    config: RlmModePluginConfig,
}

impl BuiltinRlmModePluginFactory {
    pub fn new(config: RlmModePluginConfig) -> Self {
        Self { config }
    }
}

impl Default for BuiltinRlmModePluginFactory {
    fn default() -> Self {
        Self::new(RlmModePluginConfig::default())
    }
}

impl PluginFactory for BuiltinRlmModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Rlm),
            provider: Arc::new(SearchToolsProvider::new()),
            config: self.config.clone(),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    provider: Arc<SearchToolsProvider>,
    config: RlmModePluginConfig,
}

impl SessionPlugin for RlmModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().execution(Arc::new(RlmModeExecution {
            config: self.config.clone(),
        }))?;
        reg.mode().session(Arc::new(RlmModeSession {
            config: self.config.clone(),
        }))?;
        reg.mode().native_tools(Arc::new(RlmModeNativeTools))?;
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn crate::ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move {
                let mut contributions = tool_discovery_prompt_contributions(&ctx.prompt);
                contributions.extend(bound_variables_prompt_contributions(&ctx));
                Ok(contributions)
            })
        }));
        reg.surface()
            .contribute(Arc::new(|ctx| Ok(tool_surface_contribution(&ctx))));
        Ok(())
    }
}

struct RlmModeExecution {
    config: RlmModePluginConfig,
}

impl ModeExecutionPlugin for RlmModeExecution {
    fn build_execution_preamble(
        &self,
        surface: &crate::plugin::ExecutionSurface,
    ) -> ModeExecutionPreamble {
        let enabled_tools = surface.enabled_tools();
        let omitted_tool_count =
            crate::session_model::count_prompt_omitted_tools(enabled_tools.as_slice());
        let mut prompt_contributions = Vec::new();
        let mut tool_list =
            crate::ToolDefinition::format_tool_docs(surface.prompt_tools().as_slice());
        for note in &surface.tool_list_notes {
            if !tool_list.is_empty() {
                tool_list.push_str("\n\n");
            }
            tool_list.push_str(note);
        }
        if !tool_list.trim().is_empty() {
            prompt_contributions.push(crate::PromptContribution::execution(
                "Available Tools",
                tool_list,
            ));
        }
        ModeExecutionPreamble {
            tool_specs: Arc::new(Vec::new()),
            tool_names: enabled_tools.iter().map(|tool| tool.name.clone()).collect(),
            omitted_tool_count,
            execution_prompt: rlm_execution_section(&self.config.observe_projection),
            prompt_contributions,
        }
    }

    fn turn_config(&self) -> ModeTurnConfig {
        ModeTurnConfig {
            protocol: std::sync::Arc::new(crate::modes::RlmDriver),
            sync_execution_surface: true,
        }
    }
}

struct RlmModeSession {
    config: RlmModePluginConfig,
}

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmModeSession {
    async fn initialize_session(
        &self,
        session: &mut crate::Session,
        session_id: &str,
    ) -> Result<(), SessionError> {
        session.set_rlm_observe_projection_config(self.config.observe_projection.clone());
        session.start_rlm_runtime(session_id).await
    }

    async fn restore_session(
        &self,
        session: &mut crate::Session,
        state: &crate::runtime::PersistedSessionState,
    ) -> Result<(), SessionError> {
        if let Some(snapshot) = state.execution_state_snapshot.clone() {
            session.restore_execution_state(&snapshot).await?;
        }
        for body in state
            .session_graph
            .active_path_plugins(crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE)
        {
            let patch = serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node: {err}"))
                })?;
            session.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        session: &mut crate::Session,
        nodes: &[crate::SessionAppendNode],
    ) -> Result<(), SessionError> {
        for node in nodes {
            let crate::SessionAppendNode::Plugin { plugin_type, body } = node else {
                continue;
            };
            if plugin_type != crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE {
                continue;
            }
            let patch = serde_json::from_value::<crate::RlmGlobalsPatchPluginBody>(body.clone())
                .map_err(|err| {
                    SessionError::Protocol(format!("invalid RLM globals patch node body: {err}"))
                })?;
            session.apply_rlm_globals_patch(&patch).await?;
        }
        Ok(())
    }

    fn configure_runtime_from_request(
        &self,
        runtime: &mut crate::runtime::LashRuntime,
        request: &crate::SessionCreateRequest,
    ) {
        if let crate::ModeExtras::Rlm(extras) = &request.mode_extras {
            runtime.set_repl_termination(extras.termination.clone());
        }
    }
}

struct RlmModeNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for RlmModeNativeTools {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        Vec::new()
    }

    async fn execute(
        &self,
        _context: &crate::tool_dispatch::ToolDispatchContext,
        _name: &str,
        _args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_prompt_mentions_observe_cap_guidance() {
        let prompt = rlm_execution_section(&ToolResultProjectionPluginConfig {
            mode: crate::ToolResultProjectionMode::Tokens,
            limit: 321,
            max_lines: 17,
        });
        assert!(prompt.contains("Observe output is capped before reinjection"));
        assert!(prompt.contains("mode: `tokens`, limit: 321, max_lines: 17"));
        assert!(prompt.contains("narrow the expression"));
    }
}
