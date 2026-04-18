use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use lash::{
    AssembledTurn, InputItem, ManagedRunState, ManagedTaskKind, ManagedTaskSpec,
    SessionCreateRequest, SessionManager, TurnInput, TurnStatus,
};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Low,
    Medium,
    High,
}

impl Capability {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionAgentInfo {
    pub path: String,
    pub capability: Option<Capability>,
}

#[derive(Clone, Debug)]
pub struct SpawnAgentRequest {
    pub task_name: String,
    pub task: String,
    pub capability: Capability,
    pub create_request: SessionCreateRequest,
    pub turn_input: TurnInput,
}

#[derive(Clone, Debug)]
pub struct SendMessageRequest {
    pub target: String,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct FollowupTaskRequest {
    pub target: String,
    pub task: String,
    pub turn_input: TurnInput,
    pub interrupt: bool,
}

#[derive(Clone, Debug)]
pub struct WaitAgentRequest {
    pub targets: Vec<String>,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct CloseAgentRequest {
    pub target: String,
}

#[derive(Clone, Debug)]
pub struct ListAgentsRequest {
    pub path_prefix: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpawnAgentResponse {
    pub task_name: String,
    pub path: String,
    pub session_id: String,
    pub status: String,
    pub capability: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    /// Populated when the requested `task_name` was normalized (e.g.
    /// hyphens/spaces converted to underscores, uppercase lowered).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_name_note: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SendMessageResponse {
    pub from: String,
    pub to: String,
    pub queued: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FollowupTaskResponse {
    pub target: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentResponse {
    pub timed_out: bool,
    pub events: Vec<WaitAgentEvent>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CloseAgentResponse {
    pub closed: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ListAgentsResponse {
    pub agents: Vec<AgentSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentSummary {
    pub path: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_task: Option<String>,
    pub queued_tasks: usize,
    pub inbox_messages: usize,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TaskToolCallSummary {
    pub tool: String,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaitAgentSessionSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub task: String,
    pub iterations: usize,
    pub tool_calls: usize,
    pub tool_call_details: Vec<TaskToolCallSummary>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub token_usage: Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WaitAgentEvent {
    TaskStarted {
        path: String,
        task: String,
        session_id: String,
        capability: String,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        model_variant: Option<String>,
    },
    Message {
        from: String,
        to: String,
        message: String,
    },
    TaskCompleted {
        path: String,
        task: String,
        status: String,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        session: WaitAgentSessionSummary,
    },
    AgentClosed {
        path: String,
    },
}

#[async_trait]
pub trait SubagentHost: Send + Sync {
    fn session_info(&self, session_id: &str) -> Option<SessionAgentInfo>;

    async fn spawn_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: SpawnAgentRequest,
    ) -> Result<SpawnAgentResponse, String>;

    async fn send_message(
        &self,
        context: &lash::ToolExecutionContext,
        request: SendMessageRequest,
    ) -> Result<SendMessageResponse, String>;

    async fn followup_task(
        &self,
        context: &lash::ToolExecutionContext,
        request: FollowupTaskRequest,
    ) -> Result<FollowupTaskResponse, String>;

    async fn wait_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: WaitAgentRequest,
    ) -> Result<WaitAgentResponse, String>;

    async fn close_agent(
        &self,
        context: &lash::ToolExecutionContext,
        request: CloseAgentRequest,
    ) -> Result<CloseAgentResponse, String>;

    async fn list_agents(
        &self,
        context: &lash::ToolExecutionContext,
        request: ListAgentsRequest,
    ) -> Result<ListAgentsResponse, String>;
}

#[derive(Clone, Default)]
pub struct LocalSubagentHost {
    state: Arc<StdMutex<HostState>>,
}

#[derive(Default)]
struct HostState {
    trees: HashMap<String, AgentTree>,
    session_agents: HashMap<String, AgentLocator>,
}

struct AgentTree {
    agents: BTreeMap<String, AgentRecord>,
    events: VecDeque<WaitAgentEvent>,
    notify: Arc<tokio::sync::Notify>,
}

impl Default for AgentTree {
    fn default() -> Self {
        Self {
            agents: BTreeMap::new(),
            events: VecDeque::new(),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

#[derive(Clone)]
struct AgentLocator {
    root_session_id: String,
    path: String,
}

#[derive(Clone)]
struct InboxMessage {
    from: String,
    message: String,
}

struct AgentRecord {
    session_id: String,
    parent_session_id: Option<String>,
    parent_path: Option<String>,
    capability: Option<Capability>,
    model: String,
    model_variant: Option<String>,
    active_task: Option<ActiveTask>,
    queued_tasks: VecDeque<QueuedTask>,
    inbox: VecDeque<InboxMessage>,
    closing: bool,
    last_status: String,
    /// Session that spawned this agent; used to complete the agent's entry
    /// in that session's background task registry when the agent exits.
    owner_session_id: String,
}

struct ActiveTask {
    task: String,
    turn_id: String,
}

struct QueuedTask {
    task: String,
    turn_input: TurnInput,
}

impl LocalSubagentHost {
    fn state_lock(&self) -> Result<std::sync::MutexGuard<'_, HostState>, String> {
        self.state
            .lock()
            .map_err(|_| "subagent host state poisoned".to_string())
    }

    fn prepare_task_launch(
        &self,
        root_session_id: &str,
        path: &str,
    ) -> Result<Option<(String, String, TurnInput, Arc<tokio::sync::Notify>)>, String> {
        let mut state = self.state_lock()?;
        let tree = match state.trees.get_mut(root_session_id) {
            Some(tree) => tree,
            None => return Ok(None),
        };
        let agent = match tree.agents.get_mut(path) {
            Some(agent) => agent,
            None => return Ok(None),
        };
        if agent.closing || agent.active_task.is_some() {
            return Ok(None);
        }
        let Some(mut queued) = agent.queued_tasks.pop_front() else {
            agent.last_status = "idle".to_string();
            return Ok(None);
        };
        if !agent.inbox.is_empty() {
            let inbox = agent.inbox.drain(..).collect::<Vec<_>>();
            queued.turn_input = merge_inbox_into_turn_input(queued.turn_input, &inbox);
        }
        Ok(Some((
            agent.session_id.clone(),
            queued.task,
            queued.turn_input,
            tree.notify.clone(),
        )))
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
                active_task: None,
                queued_tasks: VecDeque::new(),
                inbox: VecDeque::new(),
                closing: false,
                last_status: "idle".to_string(),
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

    fn queue_event(tree: &mut AgentTree, event: WaitAgentEvent) {
        tree.events.push_back(event);
        tree.notify.notify_waiters();
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
            return Ok(normalize_absolute_path(rest)?);
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
                    agent.queued_tasks.clear();
                    agent.inbox.clear();
                    if let Some(active) = &agent.active_task {
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
                    Self::queue_event(tree, WaitAgentEvent::AgentClosed { path: path.clone() });
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

    async fn start_next_task(
        &self,
        manager: Arc<dyn SessionManager>,
        root_session_id: String,
        path: String,
    ) -> Result<(), String> {
        let Some((session_id, task, turn_input, notify)) =
            self.prepare_task_launch(&root_session_id, &path)?
        else {
            return Ok(());
        };
        let turn = manager
            .start_turn_stream(&session_id, turn_input)
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
                agent.active_task = Some(ActiveTask {
                    task: task.clone(),
                    turn_id: turn_id.clone(),
                });
                agent.last_status = "running".to_string();
                WaitAgentEvent::TaskStarted {
                    path: path.clone(),
                    task: task.clone(),
                    session_id: session_id.clone(),
                    capability: agent
                        .capability
                        .map(|capability| capability.as_str().to_string())
                        .unwrap_or_else(|| "root".to_string()),
                    model: agent.model.clone(),
                    model_variant: agent.model_variant.clone(),
                }
            };
            Self::queue_event(tree, event);
        }
        notify.notify_waiters();

        let host = self.clone();
        tokio::spawn(async move {
            let outcome = manager.await_turn(&turn_id).await;
            host.finish_task(root_session_id, path, outcome, manager)
                .await;
        });
        Ok(())
    }

    async fn finish_task(
        &self,
        root_session_id: String,
        path: String,
        outcome: Result<AssembledTurn, lash::PluginError>,
        manager: Arc<dyn SessionManager>,
    ) {
        let mut close_session_id = None;
        let mut start_next = false;
        if let Ok(mut state) = self.state_lock() {
            if let Some(tree) = state.trees.get_mut(&root_session_id)
                && tree.agents.contains_key(&path)
            {
                let (event, closing_session_id, queued_more) = {
                    let agent = tree
                        .agents
                        .get_mut(&path)
                        .expect("checked contains_key above");
                    let completed_task = agent.active_task.take().map(|active| active.task);
                    let event = match &outcome {
                        Ok(turn) => {
                            let task = completed_task.unwrap_or_else(|| "task".to_string());
                            let status = turn_status_label(&turn.status);
                            agent.last_status = status.to_string();
                            let session = build_session_summary(agent, &task, turn);
                            WaitAgentEvent::TaskCompleted {
                                path: path.clone(),
                                task,
                                status: status.to_string(),
                                result: task_result_value(turn),
                                error: None,
                                session,
                            }
                        }
                        Err(err) => {
                            let task = completed_task.unwrap_or_else(|| "task".to_string());
                            agent.last_status = "failed".to_string();
                            WaitAgentEvent::TaskCompleted {
                                path: path.clone(),
                                task: task.clone(),
                                status: "failed".to_string(),
                                result: Value::Null,
                                error: Some(err.to_string()),
                                session: WaitAgentSessionSummary {
                                    id: agent.session_id.clone(),
                                    parent_session_id: agent.parent_session_id.clone(),
                                    task,
                                    iterations: 0,
                                    tool_calls: 0,
                                    tool_call_details: Vec::new(),
                                    model: agent.model.clone(),
                                    model_variant: agent.model_variant.clone(),
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
                    };
                    let closing_session_id = if agent.closing {
                        Some(agent.session_id.clone())
                    } else {
                        None
                    };
                    (event, closing_session_id, !agent.queued_tasks.is_empty())
                };

                Self::queue_event(tree, event);
                close_session_id = closing_session_id;
                start_next = close_session_id.is_none() && queued_more;
            }
        }

        if let Some(session_id) = close_session_id {
            let _ = manager.close_session(&session_id).await;
            let owner_session_id = {
                if let Ok(mut state) = self.state_lock() {
                    let owner =
                        Self::owner_session_id_for_path(&state, &root_session_id, &path);
                    if let Some(tree) = state.trees.get_mut(&root_session_id) {
                        Self::queue_event(
                            tree,
                            WaitAgentEvent::AgentClosed { path: path.clone() },
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
                let _ = handle.block_on(host.start_next_task(manager, root_session_id, path));
            });
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
            capability: agent.capability,
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
                    capability: Some(request.capability),
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
                    active_task: None,
                    queued_tasks: VecDeque::new(),
                    inbox: VecDeque::new(),
                    closing: false,
                    last_status: "spawning".to_string(),
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
            agent.queued_tasks.push_back(QueuedTask {
                task: request.task.clone(),
                turn_input: request.turn_input,
            });
            state.session_agents.insert(
                session.session_id.clone(),
                AgentLocator {
                    root_session_id: root_session_id.clone(),
                    path: path.clone(),
                },
            );
        }

        if let Err(err) = self
            .start_next_task(
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
            path,
            session_id: session.session_id,
            status: "running".to_string(),
            capability: request.capability.as_str().to_string(),
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
        let (from, to) = {
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
            target.inbox.push_back(InboxMessage {
                from: locator.path.clone(),
                message: request.message.clone(),
            });
            Self::queue_event(
                tree,
                WaitAgentEvent::Message {
                    from: locator.path.clone(),
                    to: to.clone(),
                    message: request.message.clone(),
                },
            );
            (locator.path, to)
        };

        Ok(SendMessageResponse {
            from,
            to,
            queued: true,
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
            let active_turn = agent.active_task.as_ref().map(|task| task.turn_id.clone());
            let should_start = active_turn.is_none();
            let disposition = if active_turn.is_some() && request.interrupt {
                agent.queued_tasks.push_front(QueuedTask {
                    task: request.task.clone(),
                    turn_input: request.turn_input,
                });
                "queued_after_interrupt"
            } else if active_turn.is_some() {
                agent.queued_tasks.push_back(QueuedTask {
                    task: request.task.clone(),
                    turn_input: request.turn_input,
                });
                "queued"
            } else {
                agent.queued_tasks.push_back(QueuedTask {
                    task: request.task.clone(),
                    turn_input: request.turn_input,
                });
                "started"
            };
            (
                locator.root_session_id,
                target_path,
                disposition.to_string(),
                active_turn.filter(|_| request.interrupt),
                should_start,
            )
        };

        if let Some(turn_id) = turn_to_cancel {
            let _ = context.host.cancel_turn(&turn_id).await;
        }
        if should_start {
            self.start_next_task(
                Arc::clone(&context.host),
                root_session_id,
                target_path.clone(),
            )
            .await?;
        }

        Ok(FollowupTaskResponse {
            target: target_path,
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

        {
            let mut state = self.state_lock()?;
            let tree = state
                .trees
                .get_mut(&root_session_id)
                .ok_or_else(|| "subagent tree missing".to_string())?;
            let events = Self::drain_matching_events(tree, &current_path, &targets);
            if !events.is_empty() {
                return Ok(WaitAgentResponse {
                    timed_out: false,
                    events,
                });
            }
        }

        let timeout_ms = request.timeout_ms.unwrap_or(30_000);
        if timeout_ms == 0 {
            return Ok(WaitAgentResponse {
                timed_out: true,
                events: Vec::new(),
            });
        }

        let timed_out = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            notify.notified(),
        )
        .await
        .is_err();

        let mut state = self.state_lock()?;
        let tree = state
            .trees
            .get_mut(&root_session_id)
            .ok_or_else(|| "subagent tree missing".to_string())?;
        let events = Self::drain_matching_events(tree, &current_path, &targets);
        Ok(WaitAgentResponse { timed_out, events })
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
                path: path.clone(),
                session_id: agent.session_id.clone(),
                parent_path: agent.parent_path.clone(),
                capability: agent
                    .capability
                    .map(|capability| capability.as_str().to_string()),
                status: agent.last_status.clone(),
                active_task: agent.active_task.as_ref().map(|task| task.task.clone()),
                queued_tasks: agent.queued_tasks.len(),
                inbox_messages: agent.inbox.len(),
                model: agent.model.clone(),
                model_variant: agent.model_variant.clone(),
            })
            .collect();
        Ok(ListAgentsResponse { agents })
    }
}

fn event_matches(event: &WaitAgentEvent, current_path: &str, targets: &[String]) -> bool {
    if targets.is_empty() {
        return match event {
            WaitAgentEvent::TaskStarted { path, .. }
            | WaitAgentEvent::TaskCompleted { path, .. }
            | WaitAgentEvent::AgentClosed { path } => is_same_or_descendant(path, current_path),
            WaitAgentEvent::Message { from, to, .. } => {
                to == current_path || is_same_or_descendant(from, current_path)
            }
        };
    }

    targets.iter().any(|target| match event {
        WaitAgentEvent::TaskStarted { path, .. }
        | WaitAgentEvent::TaskCompleted { path, .. }
        | WaitAgentEvent::AgentClosed { path } => path == target,
        WaitAgentEvent::Message { from, to, .. } => from == target || to == target,
    })
}

fn build_session_summary(
    agent: &AgentRecord,
    task: &str,
    turn: &AssembledTurn,
) -> WaitAgentSessionSummary {
    WaitAgentSessionSummary {
        id: agent.session_id.clone(),
        parent_session_id: agent.parent_session_id.clone(),
        task: task.to_string(),
        iterations: turn.state.iteration,
        tool_calls: turn.state.projected_tool_calls().len(),
        tool_call_details: turn
            .state
            .projected_tool_calls()
            .iter()
            .map(|tool_call| TaskToolCallSummary {
                tool: tool_call.tool.clone(),
                success: tool_call.success,
                duration_ms: tool_call.duration_ms,
            })
            .collect(),
        model: turn.state.policy.model.clone(),
        model_variant: turn.state.policy.model_variant.clone(),
        token_usage: json!({
            "input_tokens": turn.token_usage.input_tokens,
            "output_tokens": turn.token_usage.output_tokens,
            "cached_input_tokens": turn.token_usage.cached_input_tokens,
            "reasoning_tokens": turn.token_usage.reasoning_tokens,
            "total_tokens": turn.token_usage.total(),
        }),
    }
}

fn task_result_value(turn: &AssembledTurn) -> Value {
    if let Some(value) = &turn.typed_finish {
        return value.clone();
    }
    if !turn.assistant_output.safe_text.trim().is_empty() {
        return json!(turn.assistant_output.safe_text.trim().to_string());
    }
    json!(turn.assistant_output.raw_text.trim().to_string())
}

fn turn_status_label(status: &TurnStatus) -> &'static str {
    match status {
        TurnStatus::Completed => "completed",
        TurnStatus::Interrupted => "interrupted",
        TurnStatus::Failed => "failed",
    }
}

fn merge_inbox_into_turn_input(mut turn_input: TurnInput, inbox: &[InboxMessage]) -> TurnInput {
    if inbox.is_empty() {
        return turn_input;
    }
    let mailbox_text = inbox
        .iter()
        .map(|message| format!("- from {}: {}", message.from, message.message))
        .collect::<Vec<_>>()
        .join("\n");
    let section = format!("## Inbox\n\n{mailbox_text}");
    let mut merged = false;
    for item in &mut turn_input.items {
        if let InputItem::Text { text } = item {
            if !text.trim().is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&section);
            merged = true;
            break;
        }
    }
    if !merged {
        turn_input
            .items
            .insert(0, InputItem::Text { text: section });
    }
    turn_input
}

fn normalize_relative_path(value: &str) -> Result<String, String> {
    let segments = value
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(validate_segment)
        .collect::<Result<Vec<_>, _>>()?;
    if segments.is_empty() {
        return Err("path must not be empty".to_string());
    }
    Ok(segments.join("/"))
}

fn normalize_absolute_path(value: &str) -> Result<String, String> {
    Ok(format!("/{}", normalize_relative_path(value)?))
}

fn validate_segment(segment: &str) -> Result<String, String> {
    let mut out = String::with_capacity(segment.len());
    let mut prev_was_sep = false;
    for ch in segment.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_was_sep = false;
        } else if !out.is_empty() && !prev_was_sep {
            out.push('_');
            prev_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        Err(format!(
            "task name segment `{segment}` has no usable characters — use letters or digits"
        ))
    } else {
        Ok(out)
    }
}

fn join_path(parent: &str, relative: &str) -> String {
    if parent == "/root" {
        format!("/root/{relative}")
    } else {
        format!("{parent}/{relative}")
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_relative_path, validate_segment};

    #[test]
    fn slugifies_mixed_case_and_hyphens() {
        assert_eq!(validate_segment("Task-Lifecycle Test").unwrap(), "task_lifecycle_test");
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
}

fn is_same_or_descendant(path: &str, prefix: &str) -> bool {
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

pub fn truncate_snapshot_to_recent_turns(
    mut snapshot: lash::SessionSnapshot,
    turns: usize,
) -> lash::SessionSnapshot {
    if turns == 0 {
        return snapshot;
    }

    let messages = snapshot.project_messages();
    let user_turn_starts = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(message.role, lash::MessageRole::User))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let Some(&start) = user_turn_starts.get(user_turn_starts.len().saturating_sub(turns)) else {
        return snapshot;
    };
    let kept_messages = messages[start..].to_vec();
    let referenced = kept_messages
        .iter()
        .flat_map(|message| message.parts.iter())
        .filter_map(|part| part.tool_call_id.clone())
        .collect::<HashSet<_>>();
    let kept_tool_calls = snapshot
        .project_tool_calls()
        .into_iter()
        .filter(|tool_call| {
            tool_call
                .call_id
                .as_ref()
                .is_some_and(|call_id| referenced.contains(call_id))
        })
        .collect::<Vec<_>>();
    snapshot.replace_projection(&kept_messages, &kept_tool_calls);
    snapshot
}
