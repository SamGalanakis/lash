//! The in-process implementation of [`SubagentHost`]: `LocalSubagentHost`.
//!
//! This is the biggest piece of the crate and owns the end-to-end
//! orchestration of child subagents:
//!   * turn launch + completion (`prepare_turn_launch`, `start_next_turn`,
//!     `finish_turn`)
//!   * request handlers for the [`SubagentHost`] trait (`spawn_agent`,
//!     `wait_agent`, `close_agent`)
//!   * forced teardown paths (`force_close_subtree`) used when the
//!     parent session cancels a subagent via `tasks_stop`
//!   * registry integration: each subagent registers a background task
//!     on its owner session so `tasks_list`/`tasks_stop` can see it, and
//!     live-state transitions mirror the agent's actual activity
//!     (Running/Idle/Cancelled/Completed)
//!
//! The inherent impl block contains internal helpers (`state_lock`,
//! `prepare_turn_launch`, `start_next_turn`, `finish_turn`,
//! `drain_matching_events`, etc.) plus the
//! `pub` entry points. The [`SubagentHost`] trait impl is a thin
//! layer on top of those.
//!
//! Internal state primitives live in [`crate::queue`]; pure routing
//! and path helpers live in [`crate::routing`]; API-facing types live
//! in [`crate::types`]. The trait itself is in [`crate::host`].

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use lash_core::{
    AssembledTurn, ManagedRunState, ManagedTaskKind, ManagedTaskSpec, SessionToolAccess,
    SubagentSessionAuthority, ToolHookHost,
};
use std::collections::BTreeSet;

use crate::host::SubagentHost;
use crate::queue::{
    ActiveTurn, ActiveTurnKind, AgentLocator, AgentRecord, AgentTree, HostState,
    PreparedTurnLaunch, QueuedTask, QueuedTurn, task_completion_event,
};
use crate::routing::{
    completed_agents, event_matches, event_visible_for_until, normalize_agent_name, wait_response,
    wait_until_satisfied,
};
use crate::shared::{MAX_SUBAGENT_DEPTH, SUBAGENT_SUITE_DENY};
use crate::types::{
    AgentMetadata, CloseAgentRequest, CloseAgentResponse, SpawnAgentRequest, SpawnAgentResponse,
    WaitAgentEvent, WaitAgentPending, WaitAgentRequest, WaitAgentResponse, WaitUntil,
};

#[derive(Clone, Default)]
pub struct LocalSubagentHost {
    state: Arc<std::sync::Mutex<HostState>>,
}

fn wait_agent_satisfied(
    events: &[WaitAgentEvent],
    pending: &BTreeMap<String, WaitAgentPending>,
    until: WaitUntil,
    all: bool,
) -> bool {
    if all {
        pending.is_empty()
    } else {
        wait_until_satisfied(events, until)
    }
}

fn is_same_or_descendant(path: &str, prefix: &str) -> bool {
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

impl LocalSubagentHost {
    fn state_lock(&self) -> Result<std::sync::MutexGuard<'_, HostState>, String> {
        self.state
            .lock()
            .map_err(|_| "subagent host state poisoned".to_string())
    }

    fn prepare_turn_launch(
        &self,
        root_session_id: &str,
        path: &str,
    ) -> Result<Option<PreparedTurnLaunch>, String> {
        let mut state = self.state_lock()?;
        let tree = match state.trees.get_mut(root_session_id) {
            Some(tree) => tree,
            None => return Ok(None),
        };
        let agent = match tree.agents.get_mut(path) {
            Some(agent) => agent,
            None => return Ok(None),
        };
        if agent.closing || agent.active_turn.is_some() {
            return Ok(None);
        }
        let Some(queued) = agent.queued_turns.pop_front() else {
            return Ok(None);
        };
        let (kind, turn_input) = match queued {
            QueuedTurn::Task(task) => (
                ActiveTurnKind::Task {
                    task: task.task.clone(),
                },
                task.turn_input,
            ),
        };
        Ok(Some(PreparedTurnLaunch {
            session_id: agent.session_id.clone(),
            kind,
            turn_input,
            notify: tree.notify.clone(),
        }))
    }

    fn ensure_current_agent_locked(state: &mut HostState, session_id: &str) -> AgentLocator {
        if let Some(locator) = state.session_agents.get(session_id) {
            return locator.clone();
        }

        let root_session_id = session_id.to_string();
        let path = "/root".to_string();
        let tree = state.trees.entry(root_session_id.clone()).or_default();
        tree.agents
            .entry(path.clone())
            .or_insert_with(|| AgentRecord {
                session_id: session_id.to_string(),
                agent_name: "root".to_string(),
                parent_session_id: None,
                capability: None,
                model: String::new(),
                model_variant: None,
                active_turn: None,
                queued_turns: VecDeque::new(),
                closing: false,
                last_task_state: None,
                last_iterations: None,
                last_tool_calls: None,
                last_token_usage: None,
                owner_session_id: session_id.to_string(),
            });
        let locator = AgentLocator {
            root_session_id,
            path,
            depth: 0,
        };
        state
            .session_agents
            .insert(session_id.to_string(), locator.clone());
        locator
    }

    /// Thin wrapper around [`crate::queue::queue_event`] preserved so
    /// tests (and any out-of-tree integrators) can still reach the
    /// purge-stale-on-TaskStarted behavior via the historical path
    /// `LocalSubagentHost::queue_event`.
    fn queue_event(tree: &mut AgentTree, event: WaitAgentEvent) {
        crate::queue::queue_event(tree, event);
    }

    fn resolve_direct_child_locked(
        state: &HostState,
        parent_session_id: &str,
        agent_name: &str,
    ) -> Result<(String, String), String> {
        let normalized = normalize_agent_name(agent_name)?;
        let child_session_id = state
            .children_by_parent_session
            .get(parent_session_id)
            .and_then(|children| children.get(&normalized))
            .cloned()
            .ok_or_else(|| format!("unknown agent_name `{normalized}`"))?;
        let locator = state
            .session_agents
            .get(&child_session_id)
            .ok_or_else(|| format!("unknown agent_name `{normalized}`"))?;
        Ok((locator.root_session_id.clone(), locator.path.clone()))
    }

    fn internal_child_path(parent_path: &str, agent_name: &str) -> String {
        if parent_path == "/root" {
            format!("/root/{agent_name}")
        } else {
            format!("{parent_path}/{agent_name}")
        }
    }

    fn remove_agent_locked(state: &mut HostState, root_session_id: &str, path: &str) {
        if let Some(tree) = state.trees.get_mut(root_session_id) {
            if let Some(agent) = tree.agents.remove(path) {
                if let Some(parent_session_id) = &agent.parent_session_id
                    && let Some(children) =
                        state.children_by_parent_session.get_mut(parent_session_id)
                {
                    children.remove(&agent.agent_name);
                    if children.is_empty() {
                        state.children_by_parent_session.remove(parent_session_id);
                    }
                }
                state.session_agents.remove(&agent.session_id);
            }
            if tree.agents.is_empty() {
                state.trees.remove(root_session_id);
            }
        }
    }

    fn owner_session_id_for_path(
        state: &HostState,
        root_session_id: &str,
        path: &str,
    ) -> Option<String> {
        state
            .trees
            .get(root_session_id)
            .and_then(|tree| tree.agents.get(path))
            .map(|agent| agent.owner_session_id.clone())
    }

    fn agent_has_pending_task(agent: &AgentRecord) -> bool {
        agent
            .active_turn
            .as_ref()
            .is_some_and(|turn| matches!(turn.kind, ActiveTurnKind::Task { .. }))
            || agent
                .queued_turns
                .iter()
                .any(|turn| matches!(turn, QueuedTurn::Task(_)))
    }

    fn pending_task_agents(
        tree: &AgentTree,
        parent_session_id: &str,
        agents: &[String],
        events: &[WaitAgentEvent],
    ) -> BTreeMap<String, WaitAgentPending> {
        let finished = completed_agents(events);
        let candidates = if agents.is_empty() {
            tree.agents
                .values()
                .filter(|agent| agent.parent_session_id.as_deref() == Some(parent_session_id))
                .map(|agent| agent.agent_name.clone())
                .collect::<Vec<_>>()
        } else {
            agents.to_vec()
        };
        candidates
            .into_iter()
            .filter(|agent_name| !finished.contains(agent_name))
            .filter_map(|agent_name| {
                let agent = tree.agents.values().find(|agent| {
                    agent.parent_session_id.as_deref() == Some(parent_session_id)
                        && agent.agent_name == agent_name
                })?;
                Self::agent_has_pending_task(agent).then(|| {
                    let task = agent.active_turn.as_ref().map(|turn| match &turn.kind {
                        ActiveTurnKind::Task { task } => task.clone(),
                    });
                    (
                        agent_name.clone(),
                        WaitAgentPending {
                            agent_name,
                            task,
                            status: if agent.active_turn.is_some() {
                                "running".to_string()
                            } else {
                                "queued".to_string()
                            },
                        },
                    )
                })
            })
            .collect()
    }

    /// Close an agent subtree by absolute path without needing a
    /// ToolContext. Used by the background task registry when
    /// `tasks_stop` targets a subagent.
    pub async fn force_close_subtree(
        &self,
        host: Arc<dyn ToolHookHost>,
        root_session_id: &str,
        target: &str,
    ) -> Result<Vec<String>, String> {
        let (paths, turn_ids, immediate_closes) = {
            let mut state = self.state_lock()?;
            let tree = state
                .trees
                .get_mut(root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            let paths = tree
                .agents
                .keys()
                .filter(|path| is_same_or_descendant(path, target))
                .cloned()
                .collect::<Vec<_>>();
            if paths.is_empty() {
                return Err(format!("unknown agent `{target}`"));
            }
            let mut turn_ids = Vec::new();
            let mut immediate_closes = Vec::new();
            for path in &paths {
                if let Some(agent) = tree.agents.get_mut(path) {
                    agent.closing = true;
                    agent.queued_turns.clear();
                    if let Some(active) = &agent.active_turn {
                        turn_ids.push(active.turn_id.clone());
                    } else {
                        immediate_closes.push((
                            path.clone(),
                            agent.session_id.clone(),
                            agent.owner_session_id.clone(),
                        ));
                    }
                }
            }
            (paths, turn_ids, immediate_closes)
        };
        for turn_id in turn_ids {
            let _ = host.cancel_turn(&turn_id).await;
        }
        for (path, session_id, owner_session_id) in &immediate_closes {
            let _ = host.close_session(session_id).await;
            {
                let mut state = self.state_lock()?;
                if let Some(tree) = state.trees.get_mut(root_session_id) {
                    LocalSubagentHost::queue_event(
                        tree,
                        WaitAgentEvent::AgentClosed {
                            agent_name: tree
                                .agents
                                .get(path)
                                .map(|a| a.agent_name.clone())
                                .unwrap_or_else(|| path.to_string()),
                            parent_session_id: tree
                                .agents
                                .get(path)
                                .and_then(|a| a.parent_session_id.clone())
                                .unwrap_or_default(),
                        },
                    );
                }
                Self::remove_agent_locked(&mut state, root_session_id, path);
            }
            host.complete_background_task(
                owner_session_id,
                &format!("subagent:{path}"),
                ManagedRunState::Cancelled,
            )
            .await;
        }
        Ok(paths)
    }

    async fn start_next_turn(
        &self,
        manager: Arc<dyn ToolHookHost>,
        root_session_id: String,
        path: String,
    ) -> Result<(), String> {
        let Some(launch) = self.prepare_turn_launch(&root_session_id, &path)? else {
            return Ok(());
        };
        let turn = manager
            .start_turn_stream(&launch.session_id, launch.turn_input)
            .await
            .map_err(|err| format!("failed to start child turn: {err}"))?;
        let turn_id = turn.turn_id.clone();
        drop(turn.events);

        {
            let mut state = self.state_lock()?;
            let tree = state
                .trees
                .get_mut(&root_session_id)
                .ok_or_else(|| "subagent tree disappeared".to_string())?;
            let event = {
                let agent = tree
                    .agents
                    .get_mut(&path)
                    .ok_or_else(|| "subagent disappeared".to_string())?;
                let event = match &launch.kind {
                    ActiveTurnKind::Task { task } => WaitAgentEvent::TaskStarted {
                        agent_name: agent.agent_name.clone(),
                        parent_session_id: agent.parent_session_id.clone().unwrap_or_default(),
                        task: task.clone(),
                        session_id: launch.session_id.clone(),
                    },
                };
                agent.active_turn = Some(ActiveTurn {
                    kind: launch.kind,
                    turn_id: turn_id.clone(),
                });
                Some(event)
            };
            if let Some(event) = event {
                LocalSubagentHost::queue_event(tree, event);
            }
        }
        launch.notify.notify_waiters();

        // If this turn is starting on an already-registered subagent
        // (i.e. a follow-up that flipped the registry to `Idle`),
        // transition back to `Running` so `tasks_list` reflects the
        // active turn.
        let owner_session_id = if let Ok(state) = self.state_lock() {
            Self::owner_session_id_for_path(&state, &root_session_id, &path)
        } else {
            None
        };
        if let Some(owner) = owner_session_id {
            manager
                .transition_background_task_live_state(
                    &owner,
                    &format!("subagent:{path}"),
                    ManagedRunState::Running,
                )
                .await;
        }

        let host = self.clone();
        tokio::spawn(async move {
            let outcome = manager.await_turn(&turn_id).await;
            host.finish_turn(root_session_id, path, outcome, manager)
                .await;
        });
        Ok(())
    }

    async fn finish_turn(
        &self,
        root_session_id: String,
        path: String,
        outcome: Result<AssembledTurn, lash_core::PluginError>,
        manager: Arc<dyn ToolHookHost>,
    ) {
        let mut close_session_id = None;
        let mut start_next = false;
        let terminal_state = subagent_terminal_state(&outcome);
        if let Ok(mut state) = self.state_lock()
            && let Some(tree) = state.trees.get_mut(&root_session_id)
            && tree.agents.contains_key(&path)
        {
            let (events, closing_session_id, queued_more) = {
                let agent = tree
                    .agents
                    .get_mut(&path)
                    .expect("checked contains_key above");
                let active = agent.active_turn.take();
                let events = match active.map(|active| active.kind) {
                    Some(ActiveTurnKind::Task { task }) => {
                        vec![task_completion_event(agent, task, &outcome)]
                    }
                    None => Vec::new(),
                };
                let queued_more = !agent.queued_turns.is_empty();
                let closing_session_id = if agent.closing || !queued_more {
                    Some(agent.session_id.clone())
                } else {
                    None
                };
                (events, closing_session_id, queued_more)
            };

            for event in events {
                LocalSubagentHost::queue_event(tree, event);
            }
            close_session_id = closing_session_id;
            start_next = close_session_id.is_none() && queued_more;
        }

        if let Some(session_id) = close_session_id {
            let _ = manager.close_session(&session_id).await;
            let owner_session_id = {
                if let Ok(mut state) = self.state_lock() {
                    let owner = Self::owner_session_id_for_path(&state, &root_session_id, &path);
                    if let Some(tree) = state.trees.get_mut(&root_session_id) {
                        LocalSubagentHost::queue_event(
                            tree,
                            WaitAgentEvent::AgentClosed {
                                agent_name: tree
                                    .agents
                                    .get(&path)
                                    .map(|a| a.agent_name.clone())
                                    .unwrap_or_else(|| path.clone()),
                                parent_session_id: tree
                                    .agents
                                    .get(&path)
                                    .and_then(|a| a.parent_session_id.clone())
                                    .unwrap_or_default(),
                            },
                        );
                    }
                    Self::remove_agent_locked(&mut state, &root_session_id, &path);
                    owner
                } else {
                    None
                }
            };
            if let Some(owner) = owner_session_id {
                manager
                    .complete_background_task(&owner, &format!("subagent:{path}"), terminal_state)
                    .await;
            }
            return;
        }

        if start_next {
            let host = self.clone();
            tokio::task::block_in_place(move || {
                let handle = tokio::runtime::Handle::current();
                let _ = handle.block_on(host.start_next_turn(manager, root_session_id, path));
            });
        }
    }

    fn drain_matching_events(
        tree: &mut AgentTree,
        parent_session_id: &str,
        agents: &[String],
    ) -> Vec<WaitAgentEvent> {
        let mut drained = Vec::new();
        let mut kept = VecDeque::new();
        while let Some(event) = tree.events.pop_front() {
            if event_matches(&event, parent_session_id, agents) {
                drained.push(event);
            } else {
                kept.push_back(event);
            }
        }
        tree.events = kept;
        drained
    }
}

fn subagent_terminal_state(outcome: &Result<AssembledTurn, lash_core::PluginError>) -> ManagedRunState {
    let Ok(turn) = outcome else {
        return ManagedRunState::Failed;
    };
    match &turn.outcome {
        lash_core::TurnOutcome::Finished(_) | lash_core::TurnOutcome::Handoff { .. } => {
            ManagedRunState::Completed
        }
        lash_core::TurnOutcome::Stopped(lash_core::TurnStop::Cancelled) => ManagedRunState::Cancelled,
        lash_core::TurnOutcome::Stopped(_) => ManagedRunState::Failed,
    }
}

#[async_trait]
impl SubagentHost for LocalSubagentHost {
    fn agent_metadata(&self, session_id: &str, agent_name: &str) -> Option<AgentMetadata> {
        let state = self.state.lock().ok()?;
        let (_, path) = Self::resolve_direct_child_locked(&state, session_id, agent_name).ok()?;
        let agent = state
            .trees
            .values()
            .find_map(|tree| tree.agents.get(&path))?;
        let run_state = if agent.closing {
            "closed"
        } else if agent.active_turn.is_some() {
            "running"
        } else {
            "idle"
        };
        Some(AgentMetadata {
            session_id: agent.session_id.clone(),
            parent_session_id: agent.parent_session_id.clone(),
            capability: agent.capability.clone(),
            run_state: run_state.to_string(),
            model: agent.model.clone(),
            model_variant: agent.model_variant.clone(),
            last_iterations: agent.last_iterations,
            last_tool_calls: agent.last_tool_calls,
            last_token_usage: agent.last_token_usage.clone(),
        })
    }

    async fn spawn_agent(
        &self,
        context: &lash_core::ToolContext,
        mut request: SpawnAgentRequest,
    ) -> Result<SpawnAgentResponse, String> {
        if context
            .host()
            .tool_catalog(context.session_id())
            .await
            .ok()
            .is_some_and(|catalog| {
                catalog.iter().all(|tool| {
                    tool.get("name").and_then(serde_json::Value::as_str) != Some("spawn_agent")
                })
            })
        {
            return Err("subagent spawning is unavailable in this session".to_string());
        }
        let session_id = request
            .create_request
            .session_id
            .clone()
            .ok_or_else(|| "child session id is required".to_string())?;

        let normalized_agent_name = normalize_agent_name(&request.agent_name)?;
        let agent_name_note = (normalized_agent_name != request.agent_name).then(|| {
            format!(
                "agent_name `{original}` was normalized to `{normalized}` (lowercase letters, digits, and underscores only)",
                original = request.agent_name,
                normalized = normalized_agent_name,
            )
        });
        let (root_session_id, path, child_depth, parent_session_id) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, context.session_id());
            let parent_depth = locator.depth;
            let child_depth = parent_depth.saturating_add(1);
            if child_depth > MAX_SUBAGENT_DEPTH {
                return Err(format!(
                    "subagent recursion depth exceeded: max depth is {MAX_SUBAGENT_DEPTH}"
                ));
            }
            if state
                .children_by_parent_session
                .get(context.session_id())
                .is_some_and(|children| children.contains_key(&normalized_agent_name))
            {
                return Err(format!(
                    "agent_name `{normalized_agent_name}` already exists for this parent"
                ));
            }
            let path = Self::internal_child_path(&locator.path, &normalized_agent_name);
            let tree = state
                .trees
                .get_mut(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            tree.agents.insert(
                path.clone(),
                AgentRecord {
                    session_id: session_id.clone(),
                    agent_name: normalized_agent_name.clone(),
                    parent_session_id: Some(context.session_id().to_string()),
                    capability: Some(request.capability.clone()),
                    model: request
                        .create_request
                        .policy
                        .as_ref()
                        .map(|policy| policy.model.clone())
                        .unwrap_or_default(),
                    model_variant: request
                        .create_request
                        .policy
                        .as_ref()
                        .and_then(|policy| policy.model_variant.clone()),
                    active_turn: None,
                    queued_turns: VecDeque::new(),
                    closing: false,
                    last_task_state: None,
                    last_iterations: None,
                    last_tool_calls: None,
                    last_token_usage: None,
                    owner_session_id: context.session_id().to_string(),
                },
            );
            state
                .children_by_parent_session
                .entry(context.session_id().to_string())
                .or_default()
                .insert(normalized_agent_name.clone(), session_id.clone());
            (
                locator.root_session_id.clone(),
                path.clone(),
                child_depth,
                context.session_id().to_string(),
            )
        };
        let mut hidden_tools = request.hidden_tools.clone();
        if child_depth >= MAX_SUBAGENT_DEPTH {
            hidden_tools.extend(SUBAGENT_SUITE_DENY.iter().map(|name| name.to_string()));
        }
        request.create_request.tool_access = SessionToolAccess {
            tools: request.create_request.tool_access.tools.clone(),
            hidden_tools: hidden_tools.into_iter().collect::<BTreeSet<_>>(),
        };
        request.create_request.subagent = Some(SubagentSessionAuthority {
            agent_name: normalized_agent_name.clone(),
            parent_session_id: parent_session_id.clone(),
            capability: request.capability.clone(),
            depth: child_depth,
            max_depth: MAX_SUBAGENT_DEPTH,
        });

        let session = match context.host().create_session(request.create_request).await {
            Ok(session) => session,
            Err(err) => {
                let mut state = self.state_lock()?;
                Self::remove_agent_locked(&mut state, &root_session_id, &path);
                return Err(format!("failed to create child session: {err}"));
            }
        };

        // Register with the parent session's background task registry so it
        // shows up in `tasks_list` and can be cancelled via `tasks_stop`.
        let cancel_host = context.host().clone();
        let cancel_self = self.clone();
        let cancel_root = root_session_id.clone();
        let cancel_path = path.clone();
        let cancel: lash_core::ManagedTaskCancel = Arc::new(move || {
            let host = Arc::clone(&cancel_host);
            let this = cancel_self.clone();
            let root = cancel_root.clone();
            let target = cancel_path.clone();
            Box::pin(async move {
                if let Err(err) = this.force_close_subtree(host, &root, &target).await {
                    tracing::warn!(
                        error = %err,
                        agent_path = %target,
                        "failed to close subagent subtree from tasks_stop"
                    );
                }
            })
        });
        if let Err(err) = context
            .host()
            .register_background_task(
                context.session_id(),
                ManagedTaskSpec {
                    id: format!("subagent:{path}"),
                    kind: ManagedTaskKind::Subagent,
                    producer: "subagent",
                },
                Some(cancel),
            )
            .await
        {
            let _ = context.host().close_session(&session.session_id).await;
            let mut state = self.state_lock()?;
            Self::remove_agent_locked(&mut state, &root_session_id, &path);
            return Err(format!(
                "failed to register subagent background task: {err}"
            ));
        }

        {
            let mut state = self.state_lock()?;
            let tree = state
                .trees
                .get_mut(&root_session_id)
                .ok_or_else(|| "subagent tree disappeared".to_string())?;
            let agent = tree
                .agents
                .get_mut(&path)
                .ok_or_else(|| "subagent disappeared".to_string())?;
            agent.session_id = session.session_id.clone();
            agent.parent_session_id = session.parent_session_id.clone();
            agent.model = session.policy.model.clone();
            agent.model_variant = session.policy.model_variant.clone();
            agent.queued_turns.push_back(QueuedTurn::Task(QueuedTask {
                task: request.task.clone(),
                turn_input: request.turn_input,
            }));
            state.session_agents.insert(
                session.session_id.clone(),
                AgentLocator {
                    root_session_id: root_session_id.clone(),
                    path: path.clone(),
                    depth: child_depth,
                },
            );
        }

        if let Err(err) = self
            .start_next_turn(
                context.host().clone(),
                root_session_id.clone(),
                path.clone(),
            )
            .await
        {
            context
                .host()
                .unregister_background_task(context.session_id(), &format!("subagent:{path}"))
                .await;
            let _ = context.host().close_session(&session.session_id).await;
            let mut state = self.state_lock()?;
            Self::remove_agent_locked(&mut state, &root_session_id, &path);
            return Err(err);
        }

        Ok(SpawnAgentResponse {
            agent_name: normalized_agent_name.clone(),
            task_id: format!("subagent:{normalized_agent_name}"),
            agent_name_note,
        })
    }

    async fn wait_agent(
        &self,
        context: &lash_core::ToolContext,
        request: WaitAgentRequest,
    ) -> Result<WaitAgentResponse, String> {
        if request.all
            && !matches!(
                request.until,
                WaitUntil::TaskCompleted | WaitUntil::Terminal
            )
        {
            return Err(
                "`wait_agent(all=true)` is only valid with until=\"task_completed\" or until=\"terminal\""
                    .to_string(),
            );
        }
        let (root_session_id, parent_session_id, agents, notify) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, context.session_id());
            let agents = if request.agents.is_empty() {
                state
                    .children_by_parent_session
                    .get(context.session_id())
                    .map(|children| children.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default()
            } else {
                request
                    .agents
                    .iter()
                    .map(|agent| {
                        let normalized = normalize_agent_name(agent)?;
                        Self::resolve_direct_child_locked(
                            &state,
                            context.session_id(),
                            &normalized,
                        )?;
                        Ok::<String, String>(normalized)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            let tree = state
                .trees
                .get(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            (
                locator.root_session_id,
                context.session_id().to_string(),
                agents,
                Arc::clone(&tree.notify),
            )
        };

        let cancellation = context.cancellation_token().cloned();
        let mut events = Vec::new();

        // Two modes:
        //   * `timeout_ms = Some(ms)` — block up to `ms` then return whatever
        //     state we have (possibly with `timed_out = true`).
        //   * `timeout_ms = None` — block until satisfied, the parent's
        //     cancellation token fires, or `ms == 0` (an explicit "poll once").
        //
        // The default at the wire layer is `None`: fan-outs should block
        // until every spawned agent completes, not race a 30-second timer.
        let deadline = request
            .timeout_ms
            .map(|ms| tokio::time::Instant::now() + std::time::Duration::from_millis(ms));
        let poll_once = matches!(request.timeout_ms, Some(0));

        loop {
            // Register for the next notification before draining the queue,
            // otherwise an event that lands between drain and await can be
            // missed by notify_waiters.
            let notified = notify.notified();
            {
                let mut state = self.state_lock()?;
                let tree = state
                    .trees
                    .get_mut(&root_session_id)
                    .ok_or_else(|| "subagent tree missing".to_string())?;
                events.extend(
                    Self::drain_matching_events(tree, &parent_session_id, &agents)
                        .into_iter()
                        .filter(|event| event_visible_for_until(event, request.until)),
                );
                let pending = Self::pending_task_agents(tree, &parent_session_id, &agents, &events);
                if wait_agent_satisfied(&events, &pending, request.until, request.all) {
                    return Ok(wait_response(false, events, pending));
                }
            }

            if poll_once {
                let pending = {
                    let state = self.state_lock()?;
                    let tree = state
                        .trees
                        .get(&root_session_id)
                        .ok_or_else(|| "subagent tree missing".to_string())?;
                    Self::pending_task_agents(tree, &parent_session_id, &agents, &events)
                };
                return Ok(wait_response(true, events, pending));
            }

            if cancellation
                .as_ref()
                .is_some_and(|token| token.is_cancelled())
            {
                let pending = {
                    let state = self.state_lock()?;
                    let tree = state
                        .trees
                        .get(&root_session_id)
                        .ok_or_else(|| "subagent tree missing".to_string())?;
                    Self::pending_task_agents(tree, &parent_session_id, &agents, &events)
                };
                return Ok(wait_response(true, events, pending));
            }

            let cancelled = async {
                if let Some(token) = cancellation.as_ref() {
                    token.cancelled().await;
                } else {
                    std::future::pending::<()>().await;
                }
            };

            let woken = match deadline {
                Some(deadline) => {
                    if tokio::time::Instant::now() >= deadline {
                        false
                    } else {
                        tokio::select! {
                            biased;
                            _ = cancelled => false,
                            res = tokio::time::timeout_at(deadline, notified) => res.is_ok(),
                        }
                    }
                }
                None => {
                    tokio::select! {
                        biased;
                        _ = cancelled => false,
                        _ = notified => true,
                    }
                }
            };

            if !woken {
                let mut state = self.state_lock()?;
                let tree = state
                    .trees
                    .get_mut(&root_session_id)
                    .ok_or_else(|| "subagent tree missing".to_string())?;
                events.extend(
                    Self::drain_matching_events(tree, &parent_session_id, &agents)
                        .into_iter()
                        .filter(|event| event_visible_for_until(event, request.until)),
                );
                let pending = Self::pending_task_agents(tree, &parent_session_id, &agents, &events);
                let timed_out =
                    !wait_agent_satisfied(&events, &pending, request.until, request.all);
                return Ok(wait_response(timed_out, events, pending));
            }
        }
    }

    async fn close_agent(
        &self,
        context: &lash_core::ToolContext,
        request: CloseAgentRequest,
    ) -> Result<CloseAgentResponse, String> {
        let (root_session_id, target, agent_name) = {
            let mut state = self.state_lock()?;
            Self::ensure_current_agent_locked(&mut state, context.session_id());
            let (root_session_id, target) = Self::resolve_direct_child_locked(
                &state,
                context.session_id(),
                &request.agent_name,
            )?;
            let agent_name = normalize_agent_name(&request.agent_name)?;
            (root_session_id, target, agent_name)
        };
        let paths = self
            .force_close_subtree(context.host().clone(), &root_session_id, &target)
            .await?;
        let closed = paths
            .into_iter()
            .map(|path| {
                path.rsplit('/')
                    .next()
                    .map(str::to_string)
                    .unwrap_or_else(|| agent_name.clone())
            })
            .collect();
        Ok(CloseAgentResponse { closed })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn normalize_agent_name_slugifies_mixed_case_and_hyphens() {
        assert_eq!(
            normalize_agent_name("Task-Lifecycle Test").unwrap(),
            "task_lifecycle_test"
        );
        assert_eq!(normalize_agent_name("InspectAuth").unwrap(), "inspectauth");
        assert_eq!(normalize_agent_name("foo__bar").unwrap(), "foo_bar");
    }

    #[test]
    fn normalize_agent_name_rejects_empty_names() {
        assert!(normalize_agent_name("---").is_err());
        assert!(normalize_agent_name("").is_err());
    }

    #[test]
    fn queue_event_task_started_purges_stale_completion_for_same_agent_name() {
        let mut tree = AgentTree::default();
        LocalSubagentHost::queue_event(
            &mut tree,
            WaitAgentEvent::TaskCompleted {
                agent_name: "worker".to_string(),
                parent_session_id: "root".to_string(),
                task: "old".to_string(),
                status: "completed".to_string(),
                result: Value::Null,
                error: None,
            },
        );
        LocalSubagentHost::queue_event(
            &mut tree,
            WaitAgentEvent::AgentClosed {
                agent_name: "worker".to_string(),
                parent_session_id: "root".to_string(),
            },
        );
        LocalSubagentHost::queue_event(
            &mut tree,
            WaitAgentEvent::TaskCompleted {
                agent_name: "other".to_string(),
                parent_session_id: "root".to_string(),
                task: "done".to_string(),
                status: "completed".to_string(),
                result: Value::Null,
                error: None,
            },
        );
        LocalSubagentHost::queue_event(
            &mut tree,
            WaitAgentEvent::TaskStarted {
                agent_name: "worker".to_string(),
                parent_session_id: "root".to_string(),
                task: "new".to_string(),
                session_id: "worker-session".to_string(),
            },
        );

        let remaining = tree
            .events
            .iter()
            .map(|event| match event {
                WaitAgentEvent::TaskCompleted { agent_name, .. } => {
                    format!("completed:{agent_name}")
                }
                WaitAgentEvent::AgentClosed { agent_name, .. } => format!("closed:{agent_name}"),
                WaitAgentEvent::TaskStarted { agent_name, .. } => format!("started:{agent_name}"),
            })
            .collect::<Vec<_>>();

        assert_eq!(remaining, vec!["completed:other", "started:worker"]);
    }
}
