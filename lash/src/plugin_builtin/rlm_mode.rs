use std::sync::Arc;

#[path = "rlm_support.rs"]
mod rlm_support;

use crate::plugin::{
    ModeExecutionPlugin, ModeExecutionPreamble, ModeNativeToolsPlugin, ModeSessionPlugin,
    ModeTurnConfig, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::{ExecutionMode, ProgressSender, SessionError, ToolResult};

use self::rlm_support::{
    SearchToolsProvider, bound_variables_prompt_contributions, tool_discovery_prompt_contributions,
    tool_surface_contribution,
};

const RLM_EXECUTION_SECTION: &str = r#"In this mode you write `lashlang` code inside your prose response and the runtime executes it. There is no native tool-call envelope — you embed code directly.

Format every work step like this:

````
Brief reasoning here in plain prose (one or two sentences is fine).

```lashlang
files = call ls { path: "." }
observe files
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
- Resolve a background handle with `await handle`.
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
- Use `start`/`await` when a long-running tool can make progress in the background while you do other work. This is especially useful for `delegate`."#;

pub(crate) struct RlmModePluginFactory;

impl PluginFactory for RlmModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Rlm),
            provider: Arc::new(SearchToolsProvider::new()),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    provider: Arc<SearchToolsProvider>,
}

impl SessionPlugin for RlmModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().execution(Arc::new(RlmModeExecution))?;
        reg.mode().session(Arc::new(RlmModeSession))?;
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

struct RlmModeExecution;

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
            prompt_contributions.push(crate::PromptContribution::block(
                crate::PromptSectionName::Execution,
                "available_tools",
                "Available Tools",
                tool_list,
            ));
        }
        ModeExecutionPreamble {
            tool_specs: Arc::new(Vec::new()),
            tool_names: enabled_tools.iter().map(|tool| tool.name.clone()).collect(),
            omitted_tool_count,
            execution_prompt: RLM_EXECUTION_SECTION.to_string(),
            prompt_contributions,
        }
    }

    fn turn_config(&self) -> ModeTurnConfig {
        ModeTurnConfig {
            protocol: crate::sansio::TurnProtocol::Rlm,
            sync_execution_surface: true,
        }
    }
}

struct RlmModeSession;

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmModeSession {
    async fn initialize_session(
        &self,
        session: &mut crate::Session,
        session_id: &str,
    ) -> Result<(), SessionError> {
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
