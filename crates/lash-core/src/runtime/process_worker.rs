use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::effect::ProcessRunner;
use super::session_manager::RuntimeSessionServices;
use super::{EmbeddedRuntimeBuilder, RUNTIME_TURN_LEASE_TTL_MS, RuntimeHostConfig};
use crate::{
    InMemorySessionStore, LashRuntime, PluginError, PluginFactory, PluginHost, PluginStack,
    ProcessAwaitOutput, ProcessExecutionContext, ProcessInput, ProcessLease,
    ProcessLeaseCompletion, ProcessRecord, ProcessRegistration, ProcessRegistry,
    SessionStoreFactory,
};

/// Deployment-local configuration for rebuilding durable process executions.
///
/// Process rows intentionally carry only portable process input and provenance.
/// Workers provide the host profile, plugins, providers, stores, secrets, and
/// host capabilities for the deployment that owns those rows.
#[derive(Clone)]
pub struct DurableProcessWorkerConfig {
    pub plugin_host: Arc<PluginHost>,
    pub runtime_host: RuntimeHostConfig,
    pub session_policy: crate::SessionPolicy,
    pub session_store_factory: Arc<dyn SessionStoreFactory>,
    pub process_registry: Arc<dyn ProcessRegistry>,
    pub trigger_store: Arc<dyn crate::TriggerStore>,
    #[doc(hidden)]
    pub turn_phase_probe_slot: crate::runtime::RuntimeTurnPhaseProbeSlot,
    /// Residency for sessions the worker rebuilds to run a process. Defaults to
    /// [`Residency::KeepAll`]; a host running [`Residency::ActivePathOnly`] wires
    /// it here so the worker's rebuilt sessions trim to the active path too,
    /// instead of silently diverging from the live runtime by keeping the full
    /// graph resident.
    pub residency: crate::Residency,
}

impl DurableProcessWorkerConfig {
    pub fn new(
        plugin_host: Arc<PluginHost>,
        runtime_host: RuntimeHostConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self {
            plugin_host,
            runtime_host,
            session_policy: crate::SessionPolicy::default(),
            session_store_factory,
            process_registry,
            trigger_store: Arc::new(crate::InMemoryTriggerStore::default()),
            turn_phase_probe_slot: crate::runtime::RuntimeTurnPhaseProbeSlot::default(),
            residency: crate::Residency::default(),
        }
    }

    pub fn with_trigger_store(mut self, store: Arc<dyn crate::TriggerStore>) -> Self {
        self.trigger_store = store;
        self
    }

    pub fn with_session_policy(mut self, policy: crate::SessionPolicy) -> Self {
        self.session_policy = policy;
        self
    }

    pub fn with_residency(mut self, residency: crate::Residency) -> Self {
        self.residency = residency;
        self
    }

    #[doc(hidden)]
    pub fn with_turn_phase_probe_slot(
        mut self,
        slot: crate::runtime::RuntimeTurnPhaseProbeSlot,
    ) -> Self {
        self.turn_phase_probe_slot = slot;
        self
    }

    pub fn from_plugin_factories(
        plugin_factories: impl IntoIterator<Item = Arc<dyn PluginFactory>>,
        runtime_host: RuntimeHostConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self::new(
            Arc::new(PluginHost::new(plugin_factories.into_iter().collect())),
            runtime_host,
            session_store_factory,
            process_registry,
        )
    }

    pub fn from_plugin_stack(
        plugin_stack: PluginStack,
        runtime_host: RuntimeHostConfig,
        session_store_factory: Arc<dyn SessionStoreFactory>,
        process_registry: Arc<dyn ProcessRegistry>,
    ) -> Self {
        Self::from_plugin_factories(
            plugin_stack.into_factories(),
            runtime_host,
            session_store_factory,
            process_registry,
        )
    }
}

/// Reconstructable background-process worker.
#[derive(Clone)]
pub struct DurableProcessWorker {
    config: Arc<DurableProcessWorkerConfig>,
}

/// Why a recovery run did not produce a terminal outcome under the lease.
enum RecoverFailure {
    /// The lease was lost mid-run (another owner reclaimed an expired lease).
    /// The losing worker must not write a terminal outcome — the new owner is
    /// now the single writer.
    LeaseLost(PluginError),
    /// The process could not be run (rebuild/store-facet failure). The lease is
    /// still held, so this worker terminalizes the row.
    Run(PluginError),
}

impl DurableProcessWorker {
    pub fn new(config: DurableProcessWorkerConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub fn from_shared_config(config: Arc<DurableProcessWorkerConfig>) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DurableProcessWorkerConfig {
        &self.config
    }

    pub async fn run_process(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
        cancellation: CancellationToken,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        let scoped_effect_controller = self
            .config
            .runtime_host
            .control
            .effect_host
            .scoped_static(crate::ExecutionScope::process(registration.id.clone()))
            .map_err(|err| PluginError::Session(err.to_string()))?
            .ok_or_else(|| {
                PluginError::Session(
                    "process worker effect host must provide a static process scope".to_string(),
                )
            })?;
        self.run_process_with_scoped_effect_controller(
            registration,
            execution_context,
            scoped_effect_controller,
            cancellation,
        )
        .await
    }

    pub async fn run_process_with_scoped_effect_controller(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
        scoped_effect_controller: crate::ScopedEffectController<'_>,
        cancellation: CancellationToken,
    ) -> Result<ProcessAwaitOutput, PluginError> {
        self.ensure_stable_process_id(&registration)?;
        self.ensure_host_profile_matches(&registration)?;
        self.ensure_durable_store_facets()?;
        if let ProcessInput::External { metadata } = registration.input.as_ref() {
            return Ok(ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata.clone() }),
                control: None,
            });
        }
        let mut runtime = self.runtime_for_registration(&registration).await?;
        let probe_scope = registration.wake_target.as_ref().or_else(|| {
            if let crate::ProcessOriginator::Session { scope } = &registration.provenance.originator
            {
                Some(scope)
            } else {
                None
            }
        });
        if let Some(probe) =
            probe_scope.and_then(|scope| self.config.turn_phase_probe_slot.get_for_scope(scope))
        {
            runtime.set_turn_phase_probe(probe);
        }
        let manager = RuntimeSessionServices::new(&runtime, true, None).map_err(|err| {
            PluginError::Session(format!(
                "failed to build runtime env for process `{}`: {err}",
                registration.id
            ))
        })?;
        Ok(manager
            .run_process(
                registration,
                execution_context,
                Arc::clone(&self.config.process_registry),
                scoped_effect_controller,
                cancellation,
            )
            .await)
    }

    /// Sweep the registry for non-terminal processes and re-execute the ones
    /// this worker can claim, driving each to a terminal state.
    ///
    /// This is the crash-recovery counterpart to a worker that ran a process
    /// from a live turn: a trigger/trigger-started process whose worker
    /// died mid-flight is left non-terminal in the registry, and a subsequent
    /// worker reopening that registry must finish it. The sweep:
    ///
    /// 1. lists every non-terminal process ([`ProcessRegistry::list_non_terminal`]);
    /// 2. claims the durable single-owner [`ProcessLease`] over each — a process
    ///    already leased live by *another* owner is skipped (it is being run by
    ///    that owner right now), so a non-terminal process is re-run by exactly
    ///    one owner (lease fencing);
    /// 3. runs the claimed process on this worker's wired controller, renewing
    ///    the lease across the long-running execution so a healthy recovery is
    ///    not swept out from under itself;
    /// 4. writes the terminal outcome and releases the lease.
    ///
    /// Idempotent by `process_id`: terminal processes are never in the worklist,
    /// and a process that became terminal between the list and the claim is
    /// detected after claiming and skipped, so re-running a recovery sweep does
    /// not double-execute completed work.
    pub async fn drive_pending_processes(&self) -> Result<(), PluginError> {
        let records = self.config.process_registry.list_non_terminal().await?;
        for record in records {
            // Run each claimed process on its OWN lease-fenced task. A sequential
            // drive that awaited each process to terminal would deadlock a process
            // that blocks awaiting a nested child (`start child` then `await`, or a
            // subagent fan-out): the one drive task would park inside the parent's
            // await and never claim the child. Spawning frees the loop so a
            // subsequent drive (poke or poll) claims and runs the child, and the
            // per-process `ProcessLease` fences concurrent owners — so spawning a
            // task per pending row on every drive is idempotent (a row already
            // running is skipped on claim conflict) and one failing row never
            // aborts the rest of the sweep.
            let worker = self.clone();
            tokio::spawn(async move { worker.recover_process(record).await });
        }
        Ok(())
    }

    async fn recover_process(&self, record: ProcessRecord) {
        let owner_id = format!("process-recovery-{}", uuid::Uuid::new_v4());
        let process_id = record.id.clone();
        // Skip if held live by another owner: a claim conflict means a worker is
        // already running this process, so re-running here would violate the
        // single-owner contract. Treat any claim failure as "leased elsewhere".
        let Ok(lease) = self
            .config
            .process_registry
            .claim_process_lease(&process_id, &owner_id, RUNTIME_TURN_LEASE_TTL_MS)
            .await
        else {
            return;
        };
        // The process may have reached a terminal state between the list and the
        // claim. Idempotent by process_id: do not re-execute a finished process.
        if self
            .config
            .process_registry
            .get_process(&process_id)
            .await
            .is_some_and(|current| current.is_terminal())
        {
            self.release_or_log(&lease).await;
            return;
        }
        let registration = ProcessRegistration {
            id: record.id,
            input: record.input,
            event_types: record.event_types,
            provenance: record.provenance.clone(),
            env_ref: record.env_ref.clone(),
            wake_target: record.wake_target.clone(),
        };
        let execution_context = ProcessExecutionContext::default();
        match self
            .run_process_with_lease_renewal(registration, execution_context, lease.clone())
            .await
        {
            // Ran to a terminal outcome (success or a process-level failure) while
            // holding the lease: this owner is the single writer of the terminal.
            Ok(output) => self.complete_and_release(&lease, &process_id, output).await,
            // The lease was lost mid-run — another owner reclaimed the expired
            // lease and is now running this process. Do NOT write a terminal
            // outcome or release the lease: that would race the new owner and
            // could record a succeeded process as Failed. Leave the row to the
            // lease holder; it will finish (or another sweep retries it).
            Err(RecoverFailure::LeaseLost(err)) => {
                tracing::warn!(
                    process_id = %process_id,
                    error = %err,
                    "process recovery lost its lease mid-run; deferring to the new owner",
                );
            }
            // The process could not be run at all (rebuild/store-facet failure):
            // terminalize as a recovery failure so the row leaves the worklist.
            Err(RecoverFailure::Run(err)) => {
                let output = ProcessAwaitOutput::Failure {
                    class: crate::ToolFailureClass::Execution,
                    code: "process_recovery_failed".to_string(),
                    message: err.to_string(),
                    raw: None,
                    control: None,
                };
                self.complete_and_release(&lease, &process_id, output).await;
            }
        }
    }

    /// Write a recovered process's terminal outcome (the running lease owner is
    /// the single writer) and then release the lease, logging either failure
    /// rather than aborting — the lease's TTL is the backstop.
    async fn complete_and_release(
        &self,
        lease: &ProcessLease,
        process_id: &str,
        output: ProcessAwaitOutput,
    ) {
        // Fence the terminal write: re-confirm the lease immediately before
        // writing. `renew_process_lease` is rejected (by owner/lease_token/
        // fencing_token) if another owner has reclaimed an expired lease, and on
        // success extends the window so the back-to-back write lands inside the
        // owned interval. A worker that stalled past its TTL therefore cannot
        // overwrite the new owner's outcome — it defers instead.
        let fenced = match self
            .config
            .process_registry
            .renew_process_lease(lease, RUNTIME_TURN_LEASE_TTL_MS)
            .await
        {
            Ok(renewed) => renewed,
            Err(err) => {
                tracing::warn!(
                    process_id = %process_id,
                    error = %err,
                    "lost process lease before terminal write; deferring to the new owner",
                );
                return;
            }
        };
        if let Err(err) = self
            .config
            .process_registry
            .complete_process(process_id, output)
            .await
        {
            tracing::warn!(
                process_id = %process_id,
                error = %err,
                "failed to write recovered process terminal outcome",
            );
        }
        self.release_or_log(&fenced).await;
    }

    async fn release_or_log(&self, lease: &ProcessLease) {
        if let Err(err) = self.release_process_lease(lease).await {
            tracing::warn!(
                process_id = %lease.process_id,
                error = %err,
                "failed to release recovered process lease",
            );
        }
    }

    /// Run a recovered process while renewing its lease across the execution,
    /// mirroring the turn-lease renewal that keeps a long-running effect's lease
    /// from expiring under the live owner.
    async fn run_process_with_lease_renewal(
        &self,
        registration: ProcessRegistration,
        execution_context: ProcessExecutionContext,
        mut lease: ProcessLease,
    ) -> Result<ProcessAwaitOutput, RecoverFailure> {
        let process_id = registration.id.clone();
        let cancellation = CancellationToken::new();
        let cancel_watcher = {
            let registry = Arc::clone(&self.config.process_registry);
            let process_id = process_id.clone();
            let cancellation = cancellation.clone();
            tokio::spawn(async move {
                match registry
                    .wait_event_after(&process_id, "process.cancel_requested", 0)
                    .await
                {
                    Ok(_) => cancellation.cancel(),
                    Err(err) => tracing::warn!(
                        process_id = %process_id,
                        error = %err,
                        "process cancel watcher stopped before observing cancellation",
                    ),
                }
            })
        };
        let pending = self.run_process(registration, execution_context, cancellation.clone());
        tokio::pin!(pending);
        loop {
            tokio::select! {
                outcome = &mut pending => {
                    cancel_watcher.abort();
                    return outcome.map_err(RecoverFailure::Run);
                }
                _ = tokio::time::sleep(process_lease_renew_interval()) => {
                    match self
                        .config
                        .process_registry
                        .renew_process_lease(&lease, RUNTIME_TURN_LEASE_TTL_MS)
                        .await
                    {
                        Ok(renewed) => lease = renewed,
                        Err(err) => {
                            cancellation.cancel();
                            cancel_watcher.abort();
                            return Err(RecoverFailure::LeaseLost(err));
                        }
                    }
                }
            }
        }
    }

    async fn release_process_lease(&self, lease: &ProcessLease) -> Result<(), PluginError> {
        self.config
            .process_registry
            .complete_process_lease(&ProcessLeaseCompletion::from_lease(lease))
            .await
    }

    pub async fn request_process_cancel(
        &self,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<(), PluginError> {
        self.config
            .process_registry
            .append_event(
                process_id,
                crate::ProcessEventAppendRequest::cancel_requested(process_id, reason),
            )
            .await
            .map(|_| ())
    }

    async fn runtime_for_registration(
        &self,
        registration: &ProcessRegistration,
    ) -> Result<LashRuntime, PluginError> {
        match registration.input.as_ref() {
            ProcessInput::SessionTurn { create_request, .. } => {
                self.runtime_for_session_turn(registration, create_request.as_ref())
                    .await
            }
            ProcessInput::ToolCall { .. } | ProcessInput::LashlangProcess { .. } => {
                self.runtime_for_process_env(registration).await
            }
            ProcessInput::External { .. } => unreachable!("external processes short-circuit"),
        }
    }

    async fn runtime_for_session_turn(
        &self,
        registration: &ProcessRegistration,
        create_request: &crate::SessionCreateRequest,
    ) -> Result<LashRuntime, PluginError> {
        let mut policy = create_request
            .policy
            .clone()
            .unwrap_or_else(|| self.config.session_policy.clone());
        if policy.recorded_provider_id().is_empty() {
            policy.provider_id = self.config.session_policy.provider_id.clone();
        }
        self.build_ephemeral_runtime(
            format!("process-session-turn:{}", registration.id),
            policy,
            create_request.plugin_options.clone(),
            "session turn request",
        )
        .await
    }

    async fn runtime_for_process_env(
        &self,
        registration: &ProcessRegistration,
    ) -> Result<LashRuntime, PluginError> {
        let Some(env_ref) = registration.env_ref.as_ref() else {
            return Err(PluginError::Session(format!(
                "process `{}` is missing a captured execution env",
                registration.id
            )));
        };
        let env = crate::load_process_execution_env(
            self.config
                .runtime_host
                .durability
                .lashlang_artifact_store
                .as_ref(),
            env_ref,
        )
        .await?;
        self.build_ephemeral_runtime(
            format!("process-env:{}", registration.id),
            env.policy,
            env.plugin_options,
            env_ref.as_str(),
        )
        .await
    }

    async fn build_ephemeral_runtime(
        &self,
        session_id: String,
        policy: crate::SessionPolicy,
        plugin_options: crate::PluginOptions,
        source_label: &str,
    ) -> Result<LashRuntime, PluginError> {
        let store = Arc::new(InMemorySessionStore::default());
        EmbeddedRuntimeBuilder::new()
            .with_session_id(session_id.to_string())
            .with_plugin_host(self.config.plugin_host.as_ref().clone())
            .with_runtime_host(self.config.runtime_host.clone())
            .with_policy(policy)
            .with_plugin_options(plugin_options)
            .with_session_store_factory(Arc::clone(&self.config.session_store_factory))
            .with_trigger_store(Arc::clone(&self.config.trigger_store))
            .with_process_registry(Arc::clone(&self.config.process_registry))
            .with_residency(self.config.residency)
            .with_store(store)
            .build()
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to build process worker runtime for {source_label}: {err}"
                ))
            })
    }

    /// Enforce the durable-first wiring invariant at the worker process-run
    /// boundary: when the worker was wired with a durable effect host, every
    /// store it will execute against must also be durable. A durable host
    /// running against any ephemeral store fails loudly here rather than
    /// silently re-executing a process against non-durable state.
    ///
    /// Inline controllers (the default tier) impose no requirement, so
    /// inline/in-memory workers pass unchanged.
    fn ensure_durable_store_facets(&self) -> Result<(), PluginError> {
        if self
            .config
            .runtime_host
            .control
            .effect_host
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Ok(());
        }
        let require = |facet: crate::DurableStoreFacet| {
            PluginError::Session(crate::RuntimeError::durable_store_required(facet).to_string())
        };
        if self
            .config
            .runtime_host
            .durability
            .attachment_store
            .persistence()
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(require(crate::DurableStoreFacet::AttachmentStore));
        }
        if self
            .config
            .runtime_host
            .durability
            .lashlang_artifact_store
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(require(crate::DurableStoreFacet::ArtifactStore));
        }
        if self.config.session_store_factory.durability_tier() != crate::DurabilityTier::Durable {
            return Err(require(crate::DurableStoreFacet::SessionStore));
        }
        if self.config.process_registry.durability_tier() != crate::DurabilityTier::Durable {
            return Err(require(crate::DurableStoreFacet::ProcessRegistry));
        }
        if self.config.trigger_store.durability_tier() != crate::DurabilityTier::Durable {
            return Err(require(crate::DurableStoreFacet::TriggerStore));
        }
        Ok(())
    }

    /// Enforce the stable-process-id invariant at every (re-)execution: process
    /// execution identity is the persisted `process_id`, so a retry — a Restate
    /// `run` re-invocation (keyed `LashProcessWorkflow/{process_id}`) or a
    /// recovery sweep re-running a non-terminal row — must present that stable
    /// id. An empty/fresh id has lost its idempotency anchor and is rejected
    /// loudly here, mirroring how `ExecutionScope` rejects an
    /// empty turn id at the durable-effect boundary.
    fn ensure_stable_process_id(
        &self,
        registration: &ProcessRegistration,
    ) -> Result<(), PluginError> {
        if registration.id.trim().is_empty() {
            return Err(PluginError::Session(
                crate::RuntimeError::missing_process_execution_id().to_string(),
            ));
        }
        Ok(())
    }

    fn ensure_host_profile_matches(
        &self,
        registration: &ProcessRegistration,
    ) -> Result<(), PluginError> {
        let actual = registration.provenance.host_profile_id.as_str();
        let expected = self.config.runtime_host.profile.host_profile_id.as_str();
        if actual.is_empty() || actual == expected {
            return Ok(());
        }
        Err(PluginError::Session(format!(
            "process `{}` was created for host profile `{actual}` but this worker is `{expected}`",
            registration.id
        )))
    }
}

fn process_lease_renew_interval() -> Duration {
    Duration::from_millis(process_lease_renew_interval_ms())
}

#[cfg(test)]
fn process_lease_renew_interval_ms() -> u64 {
    25
}

#[cfg(not(test))]
fn process_lease_renew_interval_ms() -> u64 {
    30_000
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::{
        AttachmentStore, AttachmentStoreError, AttachmentStorePersistence, DurabilityTier,
        DurableStoreFacet, InMemoryAttachmentStore, LashlangArtifactStore, ProcessInput,
        ProcessRegistration, RuntimeEffectController, RuntimeError, StoredAttachment, TriggerStore,
    };
    use lash_sansio::{AttachmentCreateMeta, AttachmentId, AttachmentRef};

    /// Effect controller that reports the durable tier; the worker boundary
    /// only reads the tier, so the effect path is never exercised here.
    #[derive(Default)]
    struct DurableController;

    #[async_trait::async_trait]
    impl RuntimeEffectController for DurableController {
        fn durability_tier(&self) -> DurabilityTier {
            DurabilityTier::Durable
        }

        async fn execute_effect(
            &self,
            _envelope: crate::RuntimeEffectEnvelope,
            _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
            unreachable!("worker boundary rejects before executing any effect")
        }
    }

    /// Attachment store reporting a durable tier over in-memory storage.
    #[derive(Default)]
    struct DurableAttachmentStore {
        inner: InMemoryAttachmentStore,
    }

    #[async_trait::async_trait]
    impl AttachmentStore for DurableAttachmentStore {
        fn persistence(&self) -> AttachmentStorePersistence {
            AttachmentStorePersistence::Durable
        }

        async fn put(
            &self,
            bytes: Vec<u8>,
            meta: AttachmentCreateMeta,
        ) -> Result<AttachmentRef, AttachmentStoreError> {
            self.inner.put(bytes, meta).await
        }

        async fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
            self.inner.get(id).await
        }
    }

    /// Lashlang artifact store reporting a durable tier over in-memory storage.
    #[derive(Default)]
    struct DurableArtifactStore {
        inner: lashlang::InMemoryLashlangArtifactStore,
    }

    #[async_trait::async_trait]
    impl LashlangArtifactStore for DurableArtifactStore {
        fn durability_tier(&self) -> DurabilityTier {
            DurabilityTier::Durable
        }

        async fn put_module_artifact(
            &self,
            artifact: &lashlang::ModuleArtifact,
        ) -> Result<(), lashlang::ArtifactStoreError> {
            self.inner.put_module_artifact(artifact).await
        }

        async fn get_module_artifact(
            &self,
            module_ref: &lashlang::ModuleRef,
        ) -> Result<Option<Arc<lashlang::ModuleArtifact>>, lashlang::ArtifactStoreError> {
            self.inner.get_module_artifact(module_ref).await
        }

        async fn put_artifact_bytes(
            &self,
            artifact_ref: &str,
            descriptor: &str,
            bytes: &[u8],
        ) -> Result<(), lashlang::ArtifactStoreError> {
            self.inner
                .put_artifact_bytes(artifact_ref, descriptor, bytes)
                .await
        }

        async fn get_artifact_bytes(
            &self,
            artifact_ref: &str,
        ) -> Result<Option<Vec<u8>>, lashlang::ArtifactStoreError> {
            self.inner.get_artifact_bytes(artifact_ref).await
        }
    }

    /// Session store factory whose declared tier is configurable; it never has
    /// to create a store because the worker boundary rejects first.
    struct TierSessionStoreFactory {
        tier: DurabilityTier,
    }

    #[async_trait::async_trait]
    impl SessionStoreFactory for TierSessionStoreFactory {
        fn durability_tier(&self) -> DurabilityTier {
            self.tier
        }

        async fn create_store(
            &self,
            _request: &crate::SessionStoreCreateRequest,
        ) -> Result<Arc<dyn crate::RuntimePersistence>, String> {
            unreachable!("worker boundary rejects before creating a session store")
        }

        async fn delete_session(&self, _session_id: &str) -> Result<(), String> {
            Ok(())
        }
    }

    struct TierTriggerStore {
        tier: DurabilityTier,
        inner: crate::InMemoryTriggerStore,
    }

    impl TierTriggerStore {
        fn new(tier: DurabilityTier) -> Self {
            Self {
                tier,
                inner: crate::InMemoryTriggerStore::default(),
            }
        }
    }

    #[async_trait::async_trait]
    impl TriggerStore for TierTriggerStore {
        fn durability_tier(&self) -> DurabilityTier {
            self.tier
        }

        async fn register_subscription(
            &self,
            draft: crate::TriggerSubscriptionDraft,
        ) -> Result<crate::TriggerSubscriptionRecord, PluginError> {
            self.inner.register_subscription(draft).await
        }

        async fn list_subscriptions(
            &self,
            filter: crate::TriggerSubscriptionFilter,
        ) -> Result<Vec<crate::TriggerSubscriptionRecord>, PluginError> {
            self.inner.list_subscriptions(filter).await
        }

        async fn cancel_subscription(
            &self,
            session_id: &str,
            handle: &str,
        ) -> Result<bool, PluginError> {
            self.inner.cancel_subscription(session_id, handle).await
        }

        async fn delete_session_subscriptions(
            &self,
            session_id: &str,
        ) -> Result<usize, PluginError> {
            self.inner.delete_session_subscriptions(session_id).await
        }

        async fn record_occurrence(
            &self,
            request: crate::TriggerOccurrenceRequest,
        ) -> Result<crate::TriggerOccurrenceRecord, PluginError> {
            self.inner.record_occurrence(request).await
        }

        async fn reserve_matching_deliveries(
            &self,
            occurrence_id: &str,
        ) -> Result<Vec<crate::TriggerDeliveryReservation>, PluginError> {
            self.inner.reserve_matching_deliveries(occurrence_id).await
        }
    }

    /// Build a worker whose controller is durable but whose stores can be set
    /// per-facet to durable/ephemeral, so each facet's loud rejection can be
    /// exercised independently.
    fn worker(
        attachment: Arc<dyn AttachmentStore>,
        artifact: Arc<dyn LashlangArtifactStore>,
        session_store_tier: DurabilityTier,
    ) -> DurableProcessWorker {
        worker_with_store_tiers(
            attachment,
            artifact,
            session_store_tier,
            DurabilityTier::Durable,
            DurabilityTier::Durable,
        )
    }

    fn worker_with_store_tiers(
        attachment: Arc<dyn AttachmentStore>,
        artifact: Arc<dyn LashlangArtifactStore>,
        session_store_tier: DurabilityTier,
        process_registry_tier: DurabilityTier,
        trigger_store_tier: DurabilityTier,
    ) -> DurableProcessWorker {
        let mut runtime_host = RuntimeHostConfig::in_memory();
        runtime_host.control.effect_host =
            Arc::new(crate::InlineEffectHost::new(Arc::new(DurableController)));
        runtime_host.durability.attachment_store = attachment;
        runtime_host.durability.lashlang_artifact_store = artifact;
        let plugin_host = Arc::new(crate::PluginHost::new(Vec::new()));
        let factory: Arc<dyn SessionStoreFactory> = Arc::new(TierSessionStoreFactory {
            tier: session_store_tier,
        });
        let registry: Arc<dyn ProcessRegistry> = Arc::new(
            crate::TestLocalProcessRegistry::default().with_durability_tier(process_registry_tier),
        );
        let trigger_store: Arc<dyn TriggerStore> =
            Arc::new(TierTriggerStore::new(trigger_store_tier));
        DurableProcessWorker::new(
            DurableProcessWorkerConfig::new(plugin_host, runtime_host, factory, registry)
                .with_trigger_store(trigger_store),
        )
    }

    fn external_registration() -> ProcessRegistration {
        ProcessRegistration::new(
            "worker-boundary-process",
            ProcessInput::External {
                metadata: serde_json::json!({}),
            },
            crate::ProcessProvenance::host("default"),
        )
    }

    async fn run(worker: &DurableProcessWorker) -> Result<ProcessAwaitOutput, PluginError> {
        worker
            .run_process(
                external_registration(),
                ProcessExecutionContext::default(),
                CancellationToken::new(),
            )
            .await
    }

    fn assert_facet(err: PluginError, facet: DurableStoreFacet) {
        let PluginError::Session(message) = err else {
            panic!("expected PluginError::Session, got {err:?}");
        };
        let expected = RuntimeError::durable_store_required(facet).to_string();
        assert_eq!(message, expected, "worker must reject the {facet:?} facet");
    }

    #[tokio::test]
    async fn durable_worker_rejects_ephemeral_attachment_store() {
        let worker = worker(
            Arc::new(InMemoryAttachmentStore::new()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Durable,
        );
        let err = run(&worker)
            .await
            .expect_err("ephemeral attachment store must be rejected at the worker boundary");
        assert_facet(err, DurableStoreFacet::AttachmentStore);
    }

    #[tokio::test]
    async fn durable_worker_rejects_ephemeral_artifact_store() {
        let worker = worker(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(lashlang::InMemoryLashlangArtifactStore::new()),
            DurabilityTier::Durable,
        );
        let err = run(&worker)
            .await
            .expect_err("ephemeral artifact store must be rejected at the worker boundary");
        assert_facet(err, DurableStoreFacet::ArtifactStore);
    }

    #[tokio::test]
    async fn durable_worker_rejects_ephemeral_session_store_factory() {
        let worker = worker(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Inline,
        );
        let err = run(&worker)
            .await
            .expect_err("ephemeral session store factory must be rejected at the worker boundary");
        assert_facet(err, DurableStoreFacet::SessionStore);
    }

    #[tokio::test]
    async fn durable_worker_rejects_ephemeral_process_registry() {
        let worker = worker_with_store_tiers(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Durable,
            DurabilityTier::Inline,
            DurabilityTier::Durable,
        );
        let err = run(&worker)
            .await
            .expect_err("ephemeral process registry must be rejected at the worker boundary");
        assert_facet(err, DurableStoreFacet::ProcessRegistry);
    }

    #[tokio::test]
    async fn durable_worker_rejects_ephemeral_trigger_store() {
        let worker = worker_with_store_tiers(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Durable,
            DurabilityTier::Durable,
            DurabilityTier::Inline,
        );
        let err = run(&worker)
            .await
            .expect_err("ephemeral trigger store must be rejected at the worker boundary");
        assert_facet(err, DurableStoreFacet::TriggerStore);
    }

    #[tokio::test]
    async fn durable_worker_with_all_durable_stores_passes_store_facet_check() {
        // Positive control: a durable worker wired against fully durable stores
        // clears the store-facet guard and proceeds to run the (External)
        // process.
        let worker = worker(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Durable,
        );
        let output = run(&worker)
            .await
            .expect("all-durable worker should pass the store-facet guard");
        assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
    }

    #[tokio::test]
    async fn inline_worker_passes_store_facet_check_with_ephemeral_stores() {
        // Inline controllers impose no requirement, so an in-memory worker runs
        // unchanged — the durable-first guard must not regress inline hosts.
        let runtime_host = RuntimeHostConfig::in_memory();
        let plugin_host = Arc::new(crate::PluginHost::new(Vec::new()));
        let factory: Arc<dyn SessionStoreFactory> = Arc::new(TierSessionStoreFactory {
            tier: DurabilityTier::Inline,
        });
        let registry: Arc<dyn ProcessRegistry> =
            Arc::new(crate::TestLocalProcessRegistry::default());
        let worker = DurableProcessWorker::new(DurableProcessWorkerConfig::new(
            plugin_host,
            runtime_host,
            factory,
            registry,
        ));
        let output = run(&worker)
            .await
            .expect("inline worker should pass the store-facet guard");
        assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
    }
}
