//! The in-process implementation of [`SubagentHost`]: `LocalSubagentHost`.
//!
//! This is the biggest piece of the crate and owns the end-to-end
//! orchestration of child subagents:
//!   * turn launch + completion (`prepare_turn_launch`, `start_next_turn`,
//!     `finish_turn`)
//!   * request handlers for the [`SubagentHost`] trait (`spawn_agent`,
//!     `send_message`, `followup_task`, `wait_agent`, `close_agent`,
//!     `list_agents`)
//!   * forced teardown paths (`force_close_subtree`) used when the
//!     parent session cancels a subagent via `tasks_stop`
//!   * registry integration: each subagent registers a background task
//!     on its owner session so `tasks_list`/`tasks_stop` can see it, and
//!     live-state transitions mirror the agent's actual activity
//!     (Running/Idle/Cancelled/Completed)
//!
//! The inherent impl block contains internal helpers (`state_lock`,
//! `prepare_turn_launch`, `start_next_turn`, `finish_turn`,
//! `drain_matching_events`, `queue_message_turn`, etc.) plus the
//! `pub` entry points. The [`SubagentHost`] trait impl is a thin
//! layer on top of those.
//!
//! Internal state primitives live in [`crate::queue`]; pure routing
//! and path helpers live in [`crate::routing`]; API-facing types live
//! in [`crate::types`]. The trait itself is in [`crate::host`].

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use lash::{
    AssembledTurn, ManagedRunState, ManagedTaskKind, ManagedTaskSpec, MessageRole, PluginMessage,
    SessionManager,
};

use crate::host::SubagentHost;
use crate::queue::{
    ActiveTurn, ActiveTurnKind, AgentLocator, AgentRecord, AgentTree, HostState,
    PreparedTurnLaunch, QueuedMessage, QueuedTask, QueuedTurn, message_response_event,
    message_turn_input, task_completion_event,
};
use crate::routing::{
    event_matches, event_visible_for_until, is_same_or_descendant, join_path,
    normalize_absolute_path, normalize_relative_path, wait_response, wait_until_satisfied,
};
use crate::types::{
    AgentSummary, CloseAgentRequest, CloseAgentResponse, DeliveryMode, FollowupTaskRequest,
    FollowupTaskResponse, ListAgentsRequest, ListAgentsResponse, SendMessageRequest,
    SendMessageResponse, SessionAgentInfo, SpawnAgentRequest, SpawnAgentResponse, WaitAgentEvent,
    WaitAgentRequest, WaitAgentResponse,
};

#[derive(Clone, Default)]
pub struct LocalSubagentHost {
    state: Arc<std::sync::Mutex<HostState>>,
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
            QueuedTurn::Message(message) => {
                let turn_input = message_turn_input(&message.from, &message.message);
                (ActiveTurnKind::Message { from: message.from }, turn_input)
            }
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
                parent_session_id: None,
                parent_path: None,
                capability: None,
                model: String::new(),
                model_variant: None,
                active_turn: None,
                queued_turns: VecDeque::new(),
                closing: false,
                last_task_state: None,
                owner_session_id: session_id.to_string(),
            });
        let locator = AgentLocator {
            root_session_id,
            path,
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

    fn resolve_target(current_path: &str, target: &str) -> Result<String, String> {
        let target = target.trim();
        if target.is_empty() {
            return Err("target must not be empty".to_string());
        }
        if target == "root" || target == "/root" {
            return Ok("/root".to_string());
        }
        if let Some(rest) = target.strip_prefix('/') {
            return normalize_absolute_path(rest);
        }
        let relative = normalize_relative_path(target)?;
        Ok(join_path(current_path, &relative))
    }

    fn remove_agent_locked(state: &mut HostState, root_session_id: &str, path: &str) {
        if let Some(tree) = state.trees.get_mut(root_session_id) {
            if let Some(agent) = tree.agents.remove(path) {
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

    /// Close an agent subtree by absolute path without needing a
    /// ToolExecutionContext. Used by the background task registry when
    /// `tasks_stop` targets a subagent.
    pub async fn force_close_subtree(
        &self,
        host: Arc<dyn SessionManager>,
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
                    Self::queue_event(
                        tree,
                        WaitAgentEvent::AgentClosed {
                            target: path.clone(),
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
        manager: Arc<dyn SessionManager>,
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
                    ActiveTurnKind::Task { task } => Some(WaitAgentEvent::TaskStarted {
                        target: path.clone(),
                        task: task.clone(),
                        session_id: launch.session_id.clone(),
                        capability: agent
                            .capability
                            .clone()
                            .unwrap_or_else(|| "root".to_string()),
                        model: agent.model.clone(),
                        model_variant: agent.model_variant.clone(),
                    }),
                    ActiveTurnKind::Message { .. } => None,
                };
                agent.active_turn = Some(ActiveTurn {
                    kind: launch.kind,
                    turn_id: turn_id.clone(),
                });
                event
            };
            if let Some(event) = event {
                Self::queue_event(tree, event);
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
        outcome: Result<AssembledTurn, lash::PluginError>,
        manager: Arc<dyn SessionManager>,
    ) {
        let mut close_session_id = None;
        let mut start_next = false;
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
                        vec![task_completion_event(agent, &path, task, &outcome)]
                    }
                    Some(ActiveTurnKind::Message { from }) => {
                        vec![message_response_event(&path, from, &outcome)]
                    }
                    None => Vec::new(),
                };
                let closing_session_id = if agent.closing {
                    Some(agent.session_id.clone())
                } else {
                    None
                };
                (events, closing_session_id, !agent.queued_turns.is_empty())
            };

            for event in events {
                Self::queue_event(tree, event);
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
                        Self::queue_event(
                            tree,
                            WaitAgentEvent::AgentClosed {
                                target: path.clone(),
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
                    .complete_background_task(
                        &owner,
                        &format!("subagent:{path}"),
                        ManagedRunState::Completed,
                    )
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
        } else {
            // Task finished, session kept alive, nothing queued — the
            // subagent is now idle, awaiting a follow-up. Reflect that
            // in the background-task registry so `tasks_list` can
            // distinguish idle subagents from actively running ones.
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
                        ManagedRunState::Idle,
                    )
                    .await;
            }
        }
    }

    fn drain_matching_events(
        tree: &mut AgentTree,
        current_path: &str,
        targets: &[String],
    ) -> Vec<WaitAgentEvent> {
        let mut drained = Vec::new();
        let mut kept = VecDeque::new();
        while let Some(event) = tree.events.pop_front() {
            if event_matches(&event, current_path, targets) {
                drained.push(event);
            } else {
                kept.push_back(event);
            }
        }
        tree.events = kept;
        drained
    }

    fn queue_message_turn(
        &self,
        root_session_id: &str,
        path: &str,
        message: QueuedMessage,
        front: bool,
    ) -> Result<bool, String> {
        let mut state = self.state_lock()?;
        let Some(tree) = state.trees.get_mut(root_session_id) else {
            return Ok(false);
        };
        let Some(agent) = tree.agents.get_mut(path) else {
            return Ok(false);
        };
        if agent.closing {
            return Ok(false);
        }
        let should_start = agent.active_turn.is_none() && agent.queued_turns.is_empty();
        if front {
            agent.queued_turns.push_front(QueuedTurn::Message(message));
        } else {
            agent.queued_turns.push_back(QueuedTurn::Message(message));
        }
        Ok(should_start)
    }
}

#[async_trait]
impl SubagentHost for LocalSubagentHost {
    fn session_info(&self, session_id: &str) -> Option<SessionAgentInfo> {
        let state = self.state.lock().ok()?;
        let locator = state.session_agents.get(session_id)?;
        let tree = state.trees.get(&locator.root_session_id)?;
        let agent = tree.agents.get(&locator.path)?;
        Some(SessionAgentInfo {
            path: locator.path.clone(),
            capability: agent.capability.clone(),
        })
    }

    async fn spawn_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: SpawnAgentRequest,
    ) -> Result<SpawnAgentResponse, String> {
        let session_id = request
            .create_request
            .session_id
            .clone()
            .ok_or_else(|| "child session id is required".to_string())?;

        let normalized_task_name = normalize_relative_path(&request.task_name)?;
        let task_name_note = (normalized_task_name != request.task_name).then(|| {
            format!(
                "task_name `{original}` was normalized to `{normalized}` (lowercase letters, digits, and underscores only)",
                original = request.task_name,
                normalized = normalized_task_name,
            )
        });
        let (root_session_id, path) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, &context.session_id);
            let path = join_path(&locator.path, &normalized_task_name);
            let tree = state
                .trees
                .get_mut(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            if tree.agents.contains_key(&path) {
                return Err(format!("agent path `{path}` already exists"));
            }
            tree.agents.insert(
                path.clone(),
                AgentRecord {
                    session_id: session_id.clone(),
                    parent_session_id: request.create_request.parent_session_id.clone(),
                    parent_path: Some(locator.path.clone()),
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
                    owner_session_id: context.session_id.clone(),
                },
            );
            (locator.root_session_id.clone(), path.clone())
        };

        let session = match context.host.create_session(request.create_request).await {
            Ok(session) => session,
            Err(err) => {
                let mut state = self.state_lock()?;
                Self::remove_agent_locked(&mut state, &root_session_id, &path);
                return Err(format!("failed to create child session: {err}"));
            }
        };

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
                },
            );
        }

        if let Err(err) = self
            .start_next_turn(
                Arc::clone(&context.host),
                root_session_id.clone(),
                path.clone(),
            )
            .await
        {
            let _ = context.host.close_session(&session.session_id).await;
            let mut state = self.state_lock()?;
            Self::remove_agent_locked(&mut state, &root_session_id, &path);
            return Err(err);
        }

        // Register with the parent session's background task registry so it
        // shows up in `tasks_list` and can be cancelled via `tasks_stop`.
        let cancel_host = Arc::clone(&context.host);
        let cancel_self = self.clone();
        let cancel_root = root_session_id.clone();
        let cancel_path = path.clone();
        let cancel: lash::ManagedTaskCancel = Arc::new(move || {
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
            .host
            .register_background_task(
                &context.session_id,
                ManagedTaskSpec {
                    id: format!("subagent:{path}"),
                    label: request.task_name.clone(),
                    kind: ManagedTaskKind::Subagent,
                    producer: "subagent",
                },
                Some(cancel),
            )
            .await
        {
            tracing::warn!(
                error = %err,
                agent_path = %path,
                "failed to register subagent with background task registry"
            );
        }

        Ok(SpawnAgentResponse {
            task_name: normalized_task_name,
            task_id: format!("subagent:{path}"),
            target: path,
            session_id: session.session_id,
            run_state: "running".to_string(),
            capability: request.capability,
            model: session.policy.model,
            model_variant: session.policy.model_variant,
            task_name_note,
        })
    }

    async fn send_message(
        &self,
        context: &lash::ToolExecutionContext,
        request: SendMessageRequest,
    ) -> Result<SendMessageResponse, String> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let message = QueuedMessage {
            from: String::new(),
            message: request.message.clone(),
        };
        let (root_session_id, from, to, status, active_session_id, turn_to_cancel, should_start) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, &context.session_id);
            let to = Self::resolve_target(&locator.path, &request.target)?;
            let tree = state
                .trees
                .get_mut(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            let target = tree
                .agents
                .get_mut(&to)
                .ok_or_else(|| format!("unknown agent `{to}`"))?;
            if target.closing {
                return Err(format!("agent `{to}` is closing"));
            }

            let mut message = message.clone();
            message.from = locator.path.clone();
            let active_turn_id = target
                .active_turn
                .as_ref()
                .map(|active| active.turn_id.clone());
            let active_session_id = active_turn_id.as_ref().map(|_| target.session_id.clone());
            let (status, turn_to_cancel, should_start) =
                if to == "/root" && active_turn_id.is_none() {
                    ("notified", None, false)
                } else {
                    match (active_turn_id, request.delivery) {
                        (Some(_), DeliveryMode::NextPossible) => ("delivering", None, false),
                        (Some(turn_id), DeliveryMode::Interrupt) => {
                            target
                                .queued_turns
                                .push_front(QueuedTurn::Message(message.clone()));
                            ("queued_after_interrupt", Some(turn_id), false)
                        }
                        (Some(_), DeliveryMode::NextTurn) => {
                            target
                                .queued_turns
                                .push_back(QueuedTurn::Message(message.clone()));
                            ("queued", None, false)
                        }
                        (None, DeliveryMode::NextTurn) => {
                            target
                                .queued_turns
                                .push_back(QueuedTurn::Message(message.clone()));
                            ("queued", None, false)
                        }
                        (None, DeliveryMode::NextPossible | DeliveryMode::Interrupt) => {
                            target
                                .queued_turns
                                .push_back(QueuedTurn::Message(message.clone()));
                            ("started", None, true)
                        }
                    }
                };
            Self::queue_event(
                tree,
                WaitAgentEvent::Message {
                    from: locator.path.clone(),
                    to: to.clone(),
                    message: request.message.clone(),
                },
            );
            (
                locator.root_session_id,
                locator.path,
                to,
                status.to_string(),
                active_session_id,
                turn_to_cancel,
                should_start,
            )
        };

        if let Some(session_id) = active_session_id
            && request.delivery == DeliveryMode::NextPossible
        {
            let text = format!("## Message from {from}\n\n{}", request.message);
            let plugin_message = PluginMessage {
                role: MessageRole::User,
                content: text,
                parts: Vec::new(),
                images: Vec::new(),
                user_input: None,
            };
            let host = self.clone();
            let manager = Arc::clone(&context.host);
            let fallback = QueuedMessage {
                from: from.clone(),
                message: request.message.clone(),
            };
            let root_for_fallback = root_session_id.clone();
            let target_for_fallback = to.clone();
            tokio::spawn(async move {
                if let Err(err) = manager.inject_turn_input(&session_id, plugin_message).await {
                    let should_start = host
                        .queue_message_turn(
                            &root_for_fallback,
                            &target_for_fallback,
                            fallback,
                            true,
                        )
                        .unwrap_or(false);
                    if should_start {
                        let _ = host
                            .start_next_turn(
                                Arc::clone(&manager),
                                root_for_fallback,
                                target_for_fallback.clone(),
                            )
                            .await;
                    }
                    tracing::debug!(
                        target = %target_for_fallback,
                        error = %err,
                        "send_message bridge injection failed; queued message turn"
                    );
                }
            });
        }

        if let Some(turn_id) = turn_to_cancel {
            let manager = Arc::clone(&context.host);
            tokio::spawn(async move {
                let _ = manager.cancel_turn(&turn_id).await;
            });
        }
        if should_start {
            self.start_next_turn(Arc::clone(&context.host), root_session_id, to.clone())
                .await?;
        }

        Ok(SendMessageResponse {
            from,
            to,
            message_id,
            delivery: request.delivery.as_str().to_string(),
            status,
        })
    }

    async fn followup_task(
        &self,
        context: &lash::ToolExecutionContext,
        request: FollowupTaskRequest,
    ) -> Result<FollowupTaskResponse, String> {
        let (root_session_id, target_path, disposition, turn_to_cancel, should_start) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, &context.session_id);
            let target_path = Self::resolve_target(&locator.path, &request.target)?;
            if target_path == "/root" {
                return Err("cannot assign followup work to /root".to_string());
            }
            let tree = state
                .trees
                .get_mut(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            let agent = tree
                .agents
                .get_mut(&target_path)
                .ok_or_else(|| format!("unknown agent `{target_path}`"))?;
            if agent.closing {
                return Err(format!("agent `{target_path}` is closing"));
            }
            let active_turn = agent.active_turn.as_ref().map(|turn| turn.turn_id.clone());
            let queued = QueuedTurn::Task(QueuedTask {
                task: request.task.clone(),
                turn_input: request.turn_input,
            });
            let (disposition, turn_to_cancel, should_start) = match (active_turn, request.delivery)
            {
                (Some(turn_id), DeliveryMode::Interrupt) => {
                    agent.queued_turns.push_front(queued);
                    ("queued_after_interrupt", Some(turn_id), false)
                }
                (Some(_), DeliveryMode::NextPossible | DeliveryMode::NextTurn) => {
                    agent.queued_turns.push_back(queued);
                    ("queued", None, false)
                }
                (None, DeliveryMode::NextTurn) => {
                    agent.queued_turns.push_back(queued);
                    ("queued", None, false)
                }
                (None, DeliveryMode::NextPossible | DeliveryMode::Interrupt) => {
                    agent.queued_turns.push_back(queued);
                    ("started", None, true)
                }
            };
            (
                locator.root_session_id,
                target_path,
                disposition.to_string(),
                turn_to_cancel,
                should_start,
            )
        };

        if let Some(turn_id) = turn_to_cancel {
            let manager = Arc::clone(&context.host);
            tokio::spawn(async move {
                let _ = manager.cancel_turn(&turn_id).await;
            });
        }
        if should_start {
            self.start_next_turn(
                Arc::clone(&context.host),
                root_session_id,
                target_path.clone(),
            )
            .await?;
        }

        Ok(FollowupTaskResponse {
            task_id: format!("subagent:{target_path}"),
            target: target_path,
            delivery: request.delivery.as_str().to_string(),
            status: disposition,
        })
    }

    async fn wait_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: WaitAgentRequest,
    ) -> Result<WaitAgentResponse, String> {
        let (root_session_id, current_path, targets, notify) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, &context.session_id);
            let targets = request
                .targets
                .iter()
                .map(|target| Self::resolve_target(&locator.path, target))
                .collect::<Result<Vec<_>, _>>()?;
            let tree = state
                .trees
                .get(&locator.root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            (
                locator.root_session_id,
                locator.path,
                targets,
                Arc::clone(&tree.notify),
            )
        };

        let timeout_ms = request.timeout_ms.unwrap_or(30_000);
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let mut events = Vec::new();

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
                    Self::drain_matching_events(tree, &current_path, &targets)
                        .into_iter()
                        .filter(|event| event_visible_for_until(event, request.until)),
                );
                if wait_until_satisfied(&events, request.until) {
                    return Ok(wait_response(false, events));
                }
            }

            if timeout_ms == 0 || tokio::time::Instant::now() >= deadline {
                return Ok(wait_response(true, events));
            }

            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                let mut state = self.state_lock()?;
                let tree = state
                    .trees
                    .get_mut(&root_session_id)
                    .ok_or_else(|| "subagent tree missing".to_string())?;
                events.extend(
                    Self::drain_matching_events(tree, &current_path, &targets)
                        .into_iter()
                        .filter(|event| event_visible_for_until(event, request.until)),
                );
                let timed_out = !wait_until_satisfied(&events, request.until);
                return Ok(wait_response(timed_out, events));
            }
        }
    }

    async fn close_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: CloseAgentRequest,
    ) -> Result<CloseAgentResponse, String> {
        let (root_session_id, target) = {
            let mut state = self.state_lock()?;
            let locator = Self::ensure_current_agent_locked(&mut state, &context.session_id);
            let target = Self::resolve_target(&locator.path, &request.target)?;
            if target == "/root" {
                return Err("cannot close /root".to_string());
            }
            (locator.root_session_id, target)
        };
        let paths = self
            .force_close_subtree(Arc::clone(&context.host), &root_session_id, &target)
            .await?;
        Ok(CloseAgentResponse { closed: paths })
    }

    async fn list_agents(
        &self,
        context: &lash::ToolExecutionContext,
        request: ListAgentsRequest,
    ) -> Result<ListAgentsResponse, String> {
        let state = self.state_lock()?;
        let locator = state
            .session_agents
            .get(&context.session_id)
            .cloned()
            .unwrap_or_else(|| AgentLocator {
                root_session_id: context.session_id.clone(),
                path: "/root".to_string(),
            });
        let Some(tree) = state.trees.get(&locator.root_session_id) else {
            return Ok(ListAgentsResponse { agents: Vec::new() });
        };
        let prefix = match request.path_prefix {
            Some(prefix) => Self::resolve_target(&locator.path, &prefix)?,
            None => locator.path,
        };
        let agents = tree
            .agents
            .iter()
            .filter(|(path, _)| is_same_or_descendant(path, &prefix))
            .map(|(path, agent)| AgentSummary {
                target: path.clone(),
                task_id: format!("subagent:{path}"),
                session_id: agent.session_id.clone(),
                parent_target: agent.parent_path.clone(),
                capability: agent.capability.clone(),
                agent_state: if agent.closing {
                    "closed"
                } else if agent.active_turn.is_some() {
                    "running"
                } else {
                    "idle"
                }
                .to_string(),
                current_task: agent
                    .active_turn
                    .as_ref()
                    .and_then(|turn| match &turn.kind {
                        ActiveTurnKind::Task { task } => Some(task.clone()),
                        ActiveTurnKind::Message { .. } => None,
                    }),
                current_task_state: agent.active_turn.as_ref().map(|_| "running".to_string()),
                last_task_state: agent.last_task_state.clone(),
                queued_tasks: agent
                    .queued_turns
                    .iter()
                    .filter(|turn| matches!(turn, QueuedTurn::Task(_)))
                    .count(),
                queued_messages: agent
                    .queued_turns
                    .iter()
                    .filter(|turn| matches!(turn, QueuedTurn::Message(_)))
                    .count(),
                model: agent.model.clone(),
                model_variant: agent.model_variant.clone(),
            })
            .collect();
        Ok(ListAgentsResponse { agents })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex as StdMutex};

    use async_trait::async_trait;
    use lash::{InputItem, TurnInput};
    use serde_json::{Value, json};

    use super::*;
    use crate::routing::{normalize_relative_path, validate_segment};
    use crate::types::{WaitAgentSessionSummary, WaitUntil};

    struct NoopSessionManager;

    #[async_trait]
    impl lash::SessionManager for NoopSessionManager {
        async fn snapshot_current(&self) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::SessionSnapshot, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: lash::SessionCreateRequest,
        ) -> Result<lash::SessionHandle, lash::PluginError> {
            Ok(lash::SessionHandle {
                session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.parent_session_id,
                policy: request.policy.unwrap_or_default(),
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: lash::TurnInput,
        ) -> Result<lash::SessionTurnHandle, lash::PluginError> {
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(lash::SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                policy: lash::SessionPolicy::default(),
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<lash::AssembledTurn, lash::PluginError> {
            Err(lash::PluginError::Session("not used".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }
    }

    struct BlockingInjectManager {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl lash::SessionManager for BlockingInjectManager {
        async fn snapshot_current(&self) -> Result<lash::PersistedSessionState, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::PersistedSessionState, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: lash::SessionCreateRequest,
        ) -> Result<lash::plugin::SessionHandle, lash::PluginError> {
            Ok(lash::plugin::SessionHandle {
                session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.parent_session_id,
                policy: request.policy.unwrap_or_default(),
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: lash::TurnInput,
        ) -> Result<lash::SessionTurnHandle, lash::PluginError> {
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(lash::SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                policy: lash::SessionPolicy::default(),
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<lash::AssembledTurn, lash::PluginError> {
            std::future::pending().await
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }

        async fn inject_turn_input(
            &self,
            _session_id: &str,
            _message: PluginMessage,
        ) -> Result<(), lash::PluginError> {
            self.started.notify_waiters();
            self.release.notified().await;
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingStartManager {
        inputs: StdMutex<Vec<TurnInput>>,
    }

    #[async_trait]
    impl lash::SessionManager for RecordingStartManager {
        async fn snapshot_current(&self) -> Result<lash::PersistedSessionState, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash::PersistedSessionState, lash::PluginError> {
            Ok(lash::PersistedSessionState::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: lash::SessionCreateRequest,
        ) -> Result<lash::plugin::SessionHandle, lash::PluginError> {
            Ok(lash::plugin::SessionHandle {
                session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.parent_session_id,
                policy: request.policy.unwrap_or_default(),
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            input: lash::TurnInput,
        ) -> Result<lash::SessionTurnHandle, lash::PluginError> {
            self.inputs.lock().expect("inputs").push(input);
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(lash::SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                policy: lash::SessionPolicy::default(),
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<lash::AssembledTurn, lash::PluginError> {
            std::future::pending().await
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), lash::PluginError> {
            Ok(())
        }
    }

    fn test_context() -> lash::ToolExecutionContext {
        test_context_with_host(Arc::new(NoopSessionManager))
    }

    fn test_context_with_host(host: Arc<dyn lash::SessionManager>) -> lash::ToolExecutionContext {
        test_context_for_session("root", host)
    }

    fn test_context_for_session(
        session_id: &str,
        host: Arc<dyn lash::SessionManager>,
    ) -> lash::ToolExecutionContext {
        lash::ToolExecutionContext {
            session_id: session_id.to_string(),
            host,
            cancellation_token: None,
            async_task_id: None,
        }
    }

    fn test_agent(
        session_id: &str,
        parent_path: Option<&str>,
        active_turn: Option<ActiveTurn>,
    ) -> AgentRecord {
        AgentRecord {
            session_id: session_id.to_string(),
            parent_session_id: parent_path.map(|_| "root".to_string()),
            parent_path: parent_path.map(str::to_string),
            capability: parent_path.map(|_| "low".to_string()),
            model: "mock-model".to_string(),
            model_variant: None,
            active_turn,
            queued_turns: VecDeque::new(),
            closing: false,
            last_task_state: None,
            owner_session_id: "root".to_string(),
        }
    }

    fn test_completed_event(path: &str) -> WaitAgentEvent {
        WaitAgentEvent::TaskCompleted {
            target: path.to_string(),
            task: "work".to_string(),
            status: "completed".to_string(),
            result: Value::String("done".to_string()),
            error: None,
            session: WaitAgentSessionSummary {
                id: "worker-session".to_string(),
                parent_session_id: Some("root".to_string()),
                task: "work".to_string(),
                iterations: 1,
                tool_calls: 0,
                model: "mock-model".to_string(),
                model_variant: None,
                token_usage: json!({
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                }),
            },
        }
    }

    fn seed_started_worker(host: &LocalSubagentHost) {
        let mut state = host.state_lock().expect("state");
        let mut tree = AgentTree::default();
        tree.agents
            .insert("/root".to_string(), test_agent("root", None, None));
        tree.agents.insert(
            "/root/worker".to_string(),
            test_agent(
                "worker-session",
                Some("/root"),
                Some(ActiveTurn {
                    kind: ActiveTurnKind::Task {
                        task: "work".to_string(),
                    },
                    turn_id: "turn-1".to_string(),
                }),
            ),
        );
        tree.events.push_back(WaitAgentEvent::TaskStarted {
            target: "/root/worker".to_string(),
            task: "work".to_string(),
            session_id: "worker-session".to_string(),
            capability: "low".to_string(),
            model: "mock-model".to_string(),
            model_variant: None,
        });
        state.trees.insert("root".to_string(), tree);
        state.session_agents.insert(
            "root".to_string(),
            AgentLocator {
                root_session_id: "root".to_string(),
                path: "/root".to_string(),
            },
        );
        state.session_agents.insert(
            "worker-session".to_string(),
            AgentLocator {
                root_session_id: "root".to_string(),
                path: "/root/worker".to_string(),
            },
        );
    }

    #[test]
    fn slugifies_mixed_case_and_hyphens() {
        assert_eq!(
            validate_segment("Task-Lifecycle Test").unwrap(),
            "task_lifecycle_test"
        );
        assert_eq!(validate_segment("InspectAuth").unwrap(), "inspectauth");
        assert_eq!(validate_segment("foo__bar").unwrap(), "foo_bar");
        assert_eq!(validate_segment("foo-").unwrap(), "foo");
    }

    #[test]
    fn rejects_segment_with_no_alphanumerics() {
        assert!(validate_segment("---").is_err());
        assert!(validate_segment("").is_err());
    }

    #[test]
    fn passes_through_already_valid_segment() {
        assert_eq!(validate_segment("foo_bar").unwrap(), "foo_bar");
    }

    #[test]
    fn normalize_relative_path_handles_multiple_segments() {
        assert_eq!(
            normalize_relative_path("Auth Flow/Stage 1").unwrap(),
            "auth_flow/stage_1"
        );
    }

    #[test]
    fn wait_until_defaults_to_task_completed() {
        assert_eq!(WaitUntil::parse(None).unwrap(), WaitUntil::TaskCompleted);
        assert_eq!(
            WaitUntil::parse(Some("task_completed")).unwrap(),
            WaitUntil::TaskCompleted
        );
        assert!(WaitUntil::parse(Some("started")).is_err());
    }

    #[tokio::test]
    async fn wait_agent_does_not_complete_on_task_started_only() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);

        let response = host
            .wait_agent(
                &test_context(),
                WaitAgentRequest {
                    targets: vec!["/root/worker".to_string()],
                    until: WaitUntil::TaskCompleted,
                    timeout_ms: Some(1),
                },
            )
            .await
            .expect("wait response");

        assert!(response.timed_out);
        assert!(response.events.is_empty());
        assert!(response.completion.is_none());
    }

    #[tokio::test]
    async fn wait_agent_defaults_to_task_completion_not_message() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);
        {
            let mut state = host.state_lock().expect("state");
            let tree = state.trees.get_mut("root").expect("tree");
            LocalSubagentHost::queue_event(
                tree,
                WaitAgentEvent::Message {
                    from: "/root/worker".to_string(),
                    to: "/root".to_string(),
                    message: "progress".to_string(),
                },
            );
        }

        let response = host
            .wait_agent(
                &test_context(),
                WaitAgentRequest {
                    targets: vec!["/root/worker".to_string()],
                    until: WaitUntil::TaskCompleted,
                    timeout_ms: Some(1),
                },
            )
            .await
            .expect("wait response");

        assert!(response.timed_out);
        assert!(response.events.is_empty());
        assert!(response.completion.is_none());
    }

    #[tokio::test]
    async fn wait_agent_message_target_ignores_callers_outbound_echo() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);
        {
            let mut state = host.state_lock().expect("state");
            let tree = state.trees.get_mut("root").expect("tree");
            LocalSubagentHost::queue_event(
                tree,
                WaitAgentEvent::Message {
                    from: "/root".to_string(),
                    to: "/root/worker".to_string(),
                    message: "outbound".to_string(),
                },
            );
        }

        let response = host
            .wait_agent(
                &test_context(),
                WaitAgentRequest {
                    targets: vec!["/root/worker".to_string()],
                    until: WaitUntil::Message,
                    timeout_ms: Some(1),
                },
            )
            .await
            .expect("wait response");

        assert!(response.timed_out);
        assert!(response.events.is_empty());
    }

    #[tokio::test]
    async fn wait_agent_message_recipient_receives_inbound_message() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);
        {
            let mut state = host.state_lock().expect("state");
            let tree = state.trees.get_mut("root").expect("tree");
            LocalSubagentHost::queue_event(
                tree,
                WaitAgentEvent::Message {
                    from: "/root".to_string(),
                    to: "/root/worker".to_string(),
                    message: "inbound".to_string(),
                },
            );
        }

        let response = host
            .wait_agent(
                &test_context_for_session("worker-session", Arc::new(NoopSessionManager)),
                WaitAgentRequest {
                    targets: Vec::new(),
                    until: WaitUntil::Message,
                    timeout_ms: Some(1),
                },
            )
            .await
            .expect("wait response");

        assert!(!response.timed_out);
        assert_eq!(
            response
                .message
                .as_ref()
                .map(|message| message.message.as_str()),
            Some("inbound")
        );
    }

    #[tokio::test]
    async fn wait_agent_returns_started_with_later_completion() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);
        let notifier = {
            let state = host.state_lock().expect("state");
            state.trees.get("root").expect("tree").notify.clone()
        };
        let finishing_host = host.clone();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let mut state = finishing_host.state_lock().expect("state");
            let tree = state.trees.get_mut("root").expect("tree");
            LocalSubagentHost::queue_event(tree, test_completed_event("/root/worker"));
            notifier.notify_waiters();
        });

        let response = host
            .wait_agent(
                &test_context(),
                WaitAgentRequest {
                    targets: vec!["/root/worker".to_string()],
                    until: WaitUntil::TaskCompleted,
                    timeout_ms: Some(500),
                },
            )
            .await
            .expect("wait response");

        assert!(!response.timed_out);
        assert_eq!(response.events.len(), 1);
        assert!(matches!(
            response.events[0],
            WaitAgentEvent::TaskCompleted { .. }
        ));
        assert_eq!(
            response
                .completion
                .as_ref()
                .map(|completion| &completion.result),
            Some(&Value::String("done".to_string()))
        );
    }

    #[tokio::test]
    async fn wait_agent_any_event_keeps_intermediate_events() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);

        let response = host
            .wait_agent(
                &test_context(),
                WaitAgentRequest {
                    targets: vec!["/root/worker".to_string()],
                    until: WaitUntil::AnyEvent,
                    timeout_ms: Some(1),
                },
            )
            .await
            .expect("wait response");

        assert!(!response.timed_out);
        assert_eq!(response.events.len(), 1);
        assert!(matches!(
            response.events[0],
            WaitAgentEvent::TaskStarted { .. }
        ));
    }

    #[tokio::test]
    async fn send_message_next_possible_does_not_wait_for_active_target_injection() {
        let host = LocalSubagentHost::default();
        seed_started_worker(&host);
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let context = test_context_with_host(Arc::new(BlockingInjectManager {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }));

        let response = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            host.send_message(
                &context,
                SendMessageRequest {
                    target: "/root/worker".to_string(),
                    message: "status?".to_string(),
                    delivery: DeliveryMode::NextPossible,
                },
            ),
        )
        .await
        .expect("send_message should not wait for injection")
        .expect("send response");

        assert_eq!(response.status, "delivering");
        tokio::time::timeout(std::time::Duration::from_millis(50), started.notified())
            .await
            .expect("injection attempted in background");
        release.notify_waiters();
    }

    #[tokio::test]
    async fn send_message_wakes_idle_subagent_with_message_turn() {
        let host = LocalSubagentHost::default();
        {
            let mut state = host.state_lock().expect("state");
            let mut tree = AgentTree::default();
            tree.agents
                .insert("/root".to_string(), test_agent("root", None, None));
            tree.agents.insert(
                "/root/worker".to_string(),
                test_agent("worker-session", Some("/root"), None),
            );
            state.trees.insert("root".to_string(), tree);
            state.session_agents.insert(
                "root".to_string(),
                AgentLocator {
                    root_session_id: "root".to_string(),
                    path: "/root".to_string(),
                },
            );
        }
        let manager = Arc::new(RecordingStartManager::default());
        let context = test_context_with_host(manager.clone());

        let response = host
            .send_message(
                &context,
                SendMessageRequest {
                    target: "/root/worker".to_string(),
                    message: "new note".to_string(),
                    delivery: DeliveryMode::NextPossible,
                },
            )
            .await
            .expect("send response");

        assert_eq!(response.status, "started");
        let response = host
            .list_agents(&context, ListAgentsRequest { path_prefix: None })
            .await
            .expect("list response");
        let worker = response
            .agents
            .iter()
            .find(|agent| agent.target == "/root/worker")
            .expect("worker");
        assert_eq!(worker.agent_state, "running");
        assert_eq!(worker.current_task, None);
        assert_eq!(worker.queued_messages, 0);

        let inputs = manager.inputs.lock().expect("inputs");
        assert_eq!(inputs.len(), 1);
        let InputItem::Text { text } = &inputs[0].items[0] else {
            panic!("expected text input");
        };
        assert!(text.contains("## Message from /root"));
        assert!(text.contains("new note"));
    }

    #[tokio::test]
    async fn followup_task_next_turn_queues_without_waking_idle_agent() {
        let host = LocalSubagentHost::default();
        {
            let mut state = host.state_lock().expect("state");
            let mut tree = AgentTree::default();
            tree.agents
                .insert("/root".to_string(), test_agent("root", None, None));
            tree.agents.insert(
                "/root/worker".to_string(),
                test_agent("worker-session", Some("/root"), None),
            );
            state.trees.insert("root".to_string(), tree);
            state.session_agents.insert(
                "root".to_string(),
                AgentLocator {
                    root_session_id: "root".to_string(),
                    path: "/root".to_string(),
                },
            );
        }

        let response = host
            .followup_task(
                &test_context(),
                FollowupTaskRequest {
                    target: "/root/worker".to_string(),
                    task: "later".to_string(),
                    turn_input: TurnInput {
                        items: vec![InputItem::Text {
                            text: "later".to_string(),
                        }],
                        image_blobs: HashMap::new(),
                        user_input: None,
                        mode: None,
                        mode_turn_options: None,
                    },
                    delivery: DeliveryMode::NextTurn,
                },
            )
            .await
            .expect("followup response");

        assert_eq!(response.status, "queued");
        let listed = host
            .list_agents(&test_context(), ListAgentsRequest { path_prefix: None })
            .await
            .expect("list response");
        let worker = listed
            .agents
            .iter()
            .find(|agent| agent.target == "/root/worker")
            .expect("worker");
        assert_eq!(worker.agent_state, "idle");
        assert_eq!(worker.queued_tasks, 1);
    }

    #[tokio::test]
    async fn list_agents_separates_agent_state_from_last_task_state() {
        let host = LocalSubagentHost::default();
        {
            let mut state = host.state_lock().expect("state");
            let mut tree = AgentTree::default();
            tree.agents
                .insert("/root".to_string(), test_agent("root", None, None));
            let mut worker = test_agent("worker-session", Some("/root"), None);
            worker.last_task_state = Some("completed".to_string());
            tree.agents.insert("/root/worker".to_string(), worker);
            state.trees.insert("root".to_string(), tree);
            state.session_agents.insert(
                "root".to_string(),
                AgentLocator {
                    root_session_id: "root".to_string(),
                    path: "/root".to_string(),
                },
            );
        }

        let response = host
            .list_agents(&test_context(), ListAgentsRequest { path_prefix: None })
            .await
            .expect("list response");
        let worker = response
            .agents
            .iter()
            .find(|agent| agent.target == "/root/worker")
            .expect("worker agent");

        assert_eq!(worker.agent_state, "idle");
        assert_eq!(worker.current_task_state, None);
        assert_eq!(worker.last_task_state.as_deref(), Some("completed"));
        assert_eq!(worker.task_id, "subagent:/root/worker");
    }

    /// Regression: `followup_task` on an idle agent used to leave the
    /// prior `TaskCompleted` in the per-tree event queue, so the next
    /// `wait_agent` drained that stale event and returned immediately
    /// instead of waiting for the follow-up. Queueing a new
    /// `TaskStarted` for a target must purge any stale terminal events
    /// for that target.
    #[test]
    fn queue_event_task_started_purges_stale_completion_for_same_target() {
        let mut tree = AgentTree::default();
        tree.events.push_back(test_completed_event("/root/worker"));
        tree.events.push_back(WaitAgentEvent::AgentClosed {
            target: "/root/worker".to_string(),
        });
        // Events for a different agent must NOT be purged.
        tree.events.push_back(test_completed_event("/root/other"));

        LocalSubagentHost::queue_event(
            &mut tree,
            WaitAgentEvent::TaskStarted {
                target: "/root/worker".to_string(),
                task: "follow-up".to_string(),
                session_id: "worker-session".to_string(),
                capability: "low".to_string(),
                model: "mock-model".to_string(),
                model_variant: None,
            },
        );

        let remaining: Vec<_> = tree
            .events
            .iter()
            .map(|event| match event {
                WaitAgentEvent::TaskCompleted { target, .. } => format!("completed:{target}"),
                WaitAgentEvent::AgentClosed { target } => format!("closed:{target}"),
                WaitAgentEvent::TaskStarted { target, .. } => format!("started:{target}"),
                WaitAgentEvent::Message { from, to, .. } => format!("message:{from}->{to}"),
            })
            .collect();

        assert_eq!(
            remaining,
            vec![
                "completed:/root/other".to_string(),
                "started:/root/worker".to_string(),
            ],
            "stale terminal events for /root/worker should be purged; \
             events for other agents must survive",
        );
    }
}
