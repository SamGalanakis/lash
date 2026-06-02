use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::effect::ProcessRunner;
use super::session_manager::RuntimeSessionServices;
use super::{EmbeddedRuntimeBuilder, RUNTIME_TURN_LEASE_TTL_MS, RuntimeHostConfig};
use crate::{
    LashRuntime, PluginError, PluginFactory, PluginHost, PluginStack, ProcessAwaitOutput,
    ProcessExecutionContext, ProcessInput, ProcessLease, ProcessLeaseCompletion, ProcessRecord,
    ProcessRegistration, ProcessRegistry, SessionStoreCreateRequest, SessionStoreFactory,
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
            residency: crate::Residency::default(),
        }
    }

    pub fn with_session_policy(mut self, policy: crate::SessionPolicy) -> Self {
        self.session_policy = policy;
        self
    }

    pub fn with_residency(mut self, residency: crate::Residency) -> Self {
        self.residency = residency;
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
    /// The process could not be run (rebuild/substrate failure). The lease is
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
        self.ensure_stable_process_id(&registration)?;
        self.ensure_host_profile_matches(&registration)?;
        self.ensure_durable_substrate()?;
        if let ProcessInput::External { metadata } = registration.input.as_ref() {
            return Ok(ProcessAwaitOutput::Success {
                value: serde_json::json!({ "metadata": metadata.clone() }),
                control: None,
            });
        }
        let session_id = registration.provenance.owner_scope.session_id.as_str();
        if session_id.is_empty() {
            return Err(PluginError::Session(format!(
                "process `{}` is missing a structured owner scope",
                registration.id
            )));
        }
        let runtime = self.rebuild_runtime(session_id).await?;
        let manager = RuntimeSessionServices::new(&runtime, true, None, None).map_err(|err| {
            PluginError::Session(format!(
                "failed to rebuild runtime session `{session_id}` for process `{}`: {err}",
                registration.id
            ))
        })?;
        Ok(manager
            .run_process(
                registration,
                execution_context,
                Arc::clone(&self.config.process_registry),
                cancellation,
            )
            .await)
    }

    /// Sweep the registry for non-terminal processes and re-execute the ones
    /// this worker can claim, driving each to a terminal state.
    ///
    /// This is the crash-recovery counterpart to a worker that ran a process
    /// from a live turn: a trigger/host-event-started process whose worker
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
    ///    not swept out from under itself (mirrors the turn-lease renewal in
    ///    `renew_runtime_turn_lease_for_effect`);
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
        };
        // Wakes route to the creator scope; on recovery the owner scope persisted
        // in provenance is that creator scope, so it is the wake target.
        let execution_context = ProcessExecutionContext::default()
            .with_wake_target_scope(record.provenance.owner_scope);
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
            // The process could not be run at all (rebuild/substrate failure):
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

    async fn rebuild_runtime(&self, session_id: &str) -> Result<LashRuntime, PluginError> {
        let store = self
            .config
            .session_store_factory
            .create_store(&SessionStoreCreateRequest {
                session_id: session_id.to_string(),
                relation: crate::SessionRelation::Root,
                policy: self.config.session_policy.clone(),
            })
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to open session store for process worker session `{session_id}`: {err}"
                ))
            })?;
        EmbeddedRuntimeBuilder::new()
            .with_session_id(session_id.to_string())
            .with_plugin_host(self.config.plugin_host.as_ref().clone())
            .with_runtime_host(self.config.runtime_host.clone())
            .with_policy(self.config.session_policy.clone())
            .with_session_store_factory(Arc::clone(&self.config.session_store_factory))
            .with_process_registry(Arc::clone(&self.config.process_registry))
            .with_residency(self.config.residency)
            .with_store(store)
            .build()
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to rebuild process worker runtime for session `{session_id}`: {err}"
                ))
            })
    }

    /// Enforce the durable-first wiring invariant at the worker process-run
    /// boundary: when the worker was wired with a durable effect controller,
    /// every store it will execute against must also be durable. A durable
    /// controller running against any ephemeral store fails loudly here rather
    /// than silently re-executing a process on a non-durable substrate.
    ///
    /// Inline controllers (the default tier) impose no requirement, so
    /// inline/in-memory workers pass unchanged.
    fn ensure_durable_substrate(&self) -> Result<(), PluginError> {
        if self
            .config
            .runtime_host
            .control
            .effect_controller
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Ok(());
        }
        let require = |facet: crate::DurableSubstrateFacet| {
            PluginError::Session(crate::RuntimeError::durable_substrate_required(facet).to_string())
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
            return Err(require(crate::DurableSubstrateFacet::AttachmentStore));
        }
        if self
            .config
            .runtime_host
            .durability
            .lashlang_artifact_store
            .durability_tier()
            != crate::DurabilityTier::Durable
        {
            return Err(require(crate::DurableSubstrateFacet::ArtifactStore));
        }
        if self.config.session_store_factory.durability_tier() != crate::DurabilityTier::Durable {
            return Err(require(crate::DurableSubstrateFacet::SessionStore));
        }
        Ok(())
    }

    /// Enforce the stable-process-id invariant at every (re-)execution: process
    /// execution identity is the persisted `process_id`, so a retry — a Restate
    /// `run` re-invocation (keyed `LashProcessWorkflow/{process_id}`) or a
    /// recovery sweep re-running a non-terminal row — must present that stable
    /// id. An empty/fresh id has lost its idempotency anchor and is rejected
    /// loudly here, mirroring how `DurableTurnScope::new` rejects an
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
        DurableSubstrateFacet, InMemoryAttachmentStore, LashlangArtifactStore, ProcessInput,
        ProcessRegistration, RuntimeEffectController, RuntimeError, StoredAttachment,
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

    impl AttachmentStore for DurableAttachmentStore {
        fn persistence(&self) -> AttachmentStorePersistence {
            AttachmentStorePersistence::Durable
        }

        fn put(
            &self,
            bytes: Vec<u8>,
            meta: AttachmentCreateMeta,
        ) -> Result<AttachmentRef, AttachmentStoreError> {
            self.inner.put(bytes, meta)
        }

        fn get(&self, id: &AttachmentId) -> Result<StoredAttachment, AttachmentStoreError> {
            self.inner.get(id)
        }
    }

    /// Lashlang artifact store reporting a durable tier over in-memory storage.
    #[derive(Default)]
    struct DurableArtifactStore {
        inner: lashlang::InMemoryLashlangArtifactStore,
    }

    impl LashlangArtifactStore for DurableArtifactStore {
        fn durability_tier(&self) -> DurabilityTier {
            DurabilityTier::Durable
        }

        fn put_module_artifact(
            &self,
            artifact: &lashlang::ModuleArtifact,
        ) -> Result<(), lashlang::ArtifactStoreError> {
            self.inner.put_module_artifact(artifact)
        }

        fn get_module_artifact(
            &self,
            module_ref: &lashlang::ModuleRef,
        ) -> Result<Option<lashlang::ModuleArtifact>, lashlang::ArtifactStoreError> {
            self.inner.get_module_artifact(module_ref)
        }
    }

    /// Session store factory whose declared tier is configurable; it never has
    /// to create a store because the worker boundary rejects first.
    struct TierSessionStoreFactory {
        tier: DurabilityTier,
    }

    impl SessionStoreFactory for TierSessionStoreFactory {
        fn durability_tier(&self) -> DurabilityTier {
            self.tier
        }

        fn create_store(
            &self,
            _request: &crate::SessionStoreCreateRequest,
        ) -> Result<Arc<dyn crate::RuntimePersistence>, String> {
            unreachable!("worker boundary rejects before creating a session store")
        }

        fn delete_session(&self, _session_id: &str) -> Result<(), String> {
            Ok(())
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
        let mut runtime_host = RuntimeHostConfig::in_memory();
        runtime_host.control.effect_controller = Arc::new(DurableController);
        runtime_host.durability.attachment_store = attachment;
        runtime_host.durability.lashlang_artifact_store = artifact;
        let plugin_host = Arc::new(crate::PluginHost::new(Vec::new()));
        let factory: Arc<dyn SessionStoreFactory> = Arc::new(TierSessionStoreFactory {
            tier: session_store_tier,
        });
        let registry: Arc<dyn ProcessRegistry> =
            Arc::new(crate::TestLocalProcessRegistry::default());
        DurableProcessWorker::new(DurableProcessWorkerConfig::new(
            plugin_host,
            runtime_host,
            factory,
            registry,
        ))
    }

    fn external_registration() -> ProcessRegistration {
        ProcessRegistration::new(
            "worker-boundary-process",
            ProcessInput::External {
                metadata: serde_json::json!({}),
            },
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

    fn assert_facet(err: PluginError, facet: DurableSubstrateFacet) {
        let PluginError::Session(message) = err else {
            panic!("expected PluginError::Session, got {err:?}");
        };
        let expected = RuntimeError::durable_substrate_required(facet).to_string();
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
        assert_facet(err, DurableSubstrateFacet::AttachmentStore);
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
        assert_facet(err, DurableSubstrateFacet::ArtifactStore);
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
        assert_facet(err, DurableSubstrateFacet::SessionStore);
    }

    #[tokio::test]
    async fn durable_worker_with_all_durable_stores_passes_substrate_check() {
        // Positive control: a durable worker wired against fully durable stores
        // clears the substrate guard and proceeds to run the (External) process.
        let worker = worker(
            Arc::new(DurableAttachmentStore::default()),
            Arc::new(DurableArtifactStore::default()),
            DurabilityTier::Durable,
        );
        let output = run(&worker)
            .await
            .expect("all-durable worker should pass the substrate guard");
        assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
    }

    #[tokio::test]
    async fn inline_worker_passes_substrate_check_with_ephemeral_stores() {
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
            .expect("inline worker should pass the substrate guard");
        assert!(matches!(output, ProcessAwaitOutput::Success { .. }));
    }
}
