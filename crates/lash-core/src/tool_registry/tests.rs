#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolDefinition;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockTool;
    struct MixedEnabledTool;
    struct ExternalMockSource;
    struct ExactResolvingSource {
        manifest_resolutions: Arc<AtomicUsize>,
        contract_resolutions: Arc<AtomicUsize>,
        executions: Arc<AtomicUsize>,
    }
    struct NamedExactSource {
        id: &'static str,
    }
    struct DynamicToolProvider {
        names: Arc<std::sync::Mutex<Vec<String>>>,
    }

    fn test_tool(
        name: &str,
        description: &str,
        availability: crate::ToolAvailabilityConfig,
    ) -> ToolDefinition {
        ToolDefinition::raw_with_id(
            format!("tool:{name}"),
            name,
            description,
            ToolDefinition::default_input_schema(),
            json!({ "type": "string" }),
        )
        .with_availability(availability)
    }

    fn manifests(definitions: Vec<ToolDefinition>) -> Vec<ToolManifest> {
        definitions
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn contract_from(definitions: Vec<ToolDefinition>, name: &str) -> Option<Arc<ToolContract>> {
        definitions
            .into_iter()
            .find(|tool| tool.name() == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    fn dynamic_definition(name: &str) -> ToolDefinition {
        test_tool(name, "dynamic", crate::ToolAvailabilityConfig::callable())
    }

    fn test_tool_context() -> crate::ToolContext<'static> {
        crate::ToolContext::builder(
            "registry-test".to_string(),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::testing::MockSessionManager::default()),
            Arc::new(crate::UnavailableProcessService),
            Arc::new(crate::DefaultProcessCancelAbility),
            crate::runtime::RuntimeEffectControllerHandle::shared(Arc::new(
                crate::InlineRuntimeEffectController,
            )),
            Arc::new(crate::InMemoryAttachmentStore::new()),
            crate::DirectCompletionClient::unavailable(
                "direct completions are unavailable in this test context",
            ),
        )
        .build()
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockTool {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            manifests(vec![test_tool(
                "mock_tool",
                "mock",
                crate::ToolAvailabilityConfig::callable(),
            )])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![test_tool(
                    "mock_tool",
                    "mock",
                    crate::ToolAvailabilityConfig::callable(),
                )],
                name,
            )
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for MixedEnabledTool {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            manifests(vec![
                test_tool(
                    "enabled_tool",
                    "enabled",
                    crate::ToolAvailabilityConfig::callable(),
                ),
                test_tool(
                    "disabled_tool",
                    "disabled",
                    crate::ToolAvailabilityConfig::off(),
                ),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![
                    test_tool(
                        "enabled_tool",
                        "enabled",
                        crate::ToolAvailabilityConfig::callable(),
                    ),
                    test_tool(
                        "disabled_tool",
                        "disabled",
                        crate::ToolAvailabilityConfig::off(),
                    ),
                ],
                name,
            )
        }

        async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(serde_json::json!("ok"))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for ExternalMockSource {
        fn id(&self) -> &str {
            "external"
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            manifests(vec![ToolDefinition::raw_with_id(
                "tool:mcp__demo__search",
                "mcp__demo__search",
                "search",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
                json!({ "type": "object", "additionalProperties": true }),
            )])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            contract_from(
                vec![ToolDefinition::raw_with_id(
                    "tool:mcp__demo__search",
                    "mcp__demo__search",
                    "search",
                    json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }),
                    json!({ "type": "object", "additionalProperties": true }),
                )],
                name,
            )
        }

        async fn execute(
            &self,
            tool: &str,
            args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!({
                "tool": tool,
                "args": args
            }))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for ExactResolvingSource {
        fn id(&self) -> &str {
            "exact"
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
            self.manifest_resolutions.fetch_add(1, Ordering::SeqCst);
            (name == "host_only").then(|| {
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest()
            })
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            self.contract_resolutions.fetch_add(1, Ordering::SeqCst);
            contract_from(
                vec![test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )],
                name,
            )
        }

        async fn execute(
            &self,
            tool: &str,
            _args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            self.executions.fetch_add(1, Ordering::SeqCst);
            ToolResult::ok(json!(tool))
        }
    }

    #[async_trait::async_trait]
    impl ToolSourceExecutor for NamedExactSource {
        fn id(&self) -> &str {
            self.id
        }

        fn advertised_tools(&self) -> Vec<ToolManifest> {
            Vec::new()
        }

        fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
            (name == "host_only").then(|| {
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest()
            })
        }

        fn resolve_contract(&self, _name: &str) -> Option<Arc<ToolContract>> {
            None
        }

        async fn execute(
            &self,
            tool: &str,
            _args: &serde_json::Value,
            _context: &ToolContext<'_>,
            _progress: Option<&ProgressSender>,
        ) -> ToolResult {
            ToolResult::ok(json!(tool))
        }
    }

    #[async_trait::async_trait]
    impl ToolProvider for DynamicToolProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            self.names
                .lock()
                .expect("dynamic tool names lock")
                .iter()
                .map(|name| dynamic_definition(name).manifest())
                .collect()
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
            self.names
                .lock()
                .expect("dynamic tool names lock")
                .iter()
                .any(|tool_name| tool_name == name)
                .then(|| Arc::new(dynamic_definition(name).contract()))
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            ToolResult::ok(json!(call.name))
        }
    }

    #[test]
    fn registry_preserves_initial_availability_state() {
        let registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("registry");
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot
                .get("enabled_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Callable
        );
        assert_eq!(
            snapshot
                .get("disabled_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Off
        );
    }

    #[test]
    fn exported_tool_state_is_source_free() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .add_tool_provider(Arc::new(MixedEnabledTool))
            .expect("live provider registered");

        let value = serde_json::to_value(registry.export_state()).expect("serialized tool state");
        let serialized = value.to_string();

        assert!(!serialized.contains("source_id"));
        assert!(!serialized.contains(PLUGIN_SOURCE_ID));
        assert!(!serialized.contains("live:"));
    }

    #[test]
    fn apply_state_rebinds_source_free_snapshot_to_current_sources() {
        let source_registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("source registry");
        let snapshot = source_registry.export_state();

        let target_registry =
            ToolRegistry::from_tool_provider(Arc::new(MixedEnabledTool)).expect("target registry");
        let next_generation = target_registry
            .apply_state(snapshot.with_generation(target_registry.generation()))
            .expect("state rebound");

        assert_eq!(next_generation, target_registry.generation());
        assert!(target_registry.resolve_contract("enabled_tool").is_some());
    }

    #[test]
    fn apply_state_rejects_tools_not_advertised_by_source() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        let snapshot = registry.export_state();
        let generation = snapshot.generation();
        let mut tools = snapshot.entries().clone();
        tools.insert(
            "missing".to_string(),
            ToolStateEntry::new(
                test_tool(
                    "missing",
                    "missing",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest(),
            ),
        );
        let snapshot = ToolState::new(generation, tools);
        assert!(matches!(
            registry.apply_state(snapshot),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[test]
    fn apply_state_rejects_snapshot_when_provider_is_absent() {
        let source_registry =
            ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        source_registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        let snapshot = source_registry.export_state();

        let target_registry =
            ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target registry");
        let err = target_registry
            .apply_state(snapshot.with_generation(target_registry.generation()))
            .expect_err("missing provider should fail");

        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn apply_state_rejects_ambiguous_current_source_binding() {
        let registry = ToolRegistry::empty();
        registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source a registered");
        registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-b" }))
            .expect("source b registered");

        let mut tools = BTreeMap::new();
        tools.insert(
            "host_only".to_string(),
            ToolStateEntry::new(
                test_tool(
                    "host_only",
                    "host-only",
                    crate::ToolAvailabilityConfig::callable(),
                )
                .manifest(),
            ),
        );

        let err = registry
            .apply_state(ToolState::new(registry.generation(), tools))
            .expect_err("ambiguous source binding should fail");

        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn advertised_manifest_resolves_without_exact_host_lookup() {
        let manifest_resolutions = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExactResolvingSource {
                manifest_resolutions: Arc::clone(&manifest_resolutions),
                contract_resolutions: Arc::new(AtomicUsize::new(0)),
                executions: Arc::new(AtomicUsize::new(0)),
            }))
            .expect("source registered");

        assert_eq!(
            registry
                .resolve_manifest("mock_tool")
                .map(|manifest| manifest.name),
            Some("mock_tool".to_string())
        );
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn refresh_sources_re_reads_group_provider_manifests() {
        let names = Arc::new(std::sync::Mutex::new(vec!["dynamic_one".to_string()]));
        let provider: Arc<dyn ToolProvider> = Arc::new(DynamicToolProvider {
            names: Arc::clone(&names),
        });
        let registry = ToolRegistry::from_tool_providers(vec![provider]).expect("registry");

        let tool_names = || {
            registry
                .tool_manifests()
                .into_iter()
                .map(|manifest| manifest.name)
                .collect::<BTreeSet<_>>()
        };

        assert!(tool_names().contains("dynamic_one"));
        assert!(!tool_names().contains("dynamic_two"));

        names
            .lock()
            .expect("dynamic tool names lock")
            .push("dynamic_two".to_string());
        registry.refresh_sources().expect("refresh sources");
        let refreshed = tool_names();
        assert!(refreshed.contains("dynamic_one"));
        assert!(refreshed.contains("dynamic_two"));

        names
            .lock()
            .expect("dynamic tool names lock")
            .retain(|name| name != "dynamic_one");
        registry.refresh_sources().expect("refresh sources");
        let refreshed = tool_names();
        assert!(!refreshed.contains("dynamic_one"));
        assert!(refreshed.contains("dynamic_two"));
    }

    #[tokio::test]
    async fn unknown_manifest_exact_resolves_and_routes_to_owner() {
        let manifest_resolutions = Arc::new(AtomicUsize::new(0));
        let contract_resolutions = Arc::new(AtomicUsize::new(0));
        let executions = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExactResolvingSource {
                manifest_resolutions: Arc::clone(&manifest_resolutions),
                contract_resolutions: Arc::clone(&contract_resolutions),
                executions: Arc::clone(&executions),
            }))
            .expect("source registered");

        assert_eq!(
            registry
                .resolve_manifest("host_only")
                .map(|manifest| manifest.name),
            Some("host_only".to_string())
        );
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 1);

        let contract = registry.resolve_contract("host_only");
        assert!(contract.is_some());
        assert_eq!(manifest_resolutions.load(Ordering::SeqCst), 1);
        assert_eq!(contract_resolutions.load(Ordering::SeqCst), 1);

        let context = test_tool_context();
        let args = json!({});
        let result = registry
            .execute(crate::ToolCall {
                name: "host_only",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success());
        assert_eq!(result.value_for_projection(), json!("host_only"));
        assert_eq!(executions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unknown_manifest_without_host_resolver_is_unavailable() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");

        assert!(registry.resolve_manifest("missing").is_none());
        assert!(registry.resolve_contract("missing").is_none());
    }

    #[tokio::test]
    async fn upsert_source_registers_and_executes_external_tools() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");

        let defs = registry.tool_manifests();
        assert!(defs.iter().any(|def| def.name == "mcp__demo__search"));

        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = registry
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success());
        assert_eq!(
            result.value_for_projection()["tool"],
            json!("mcp__demo__search")
        );
        assert_eq!(
            result.value_for_projection()["args"]["query"],
            json!("hello")
        );
    }

    #[test]
    fn upsert_source_preserves_availability_override_on_refresh() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        let mut snapshot = registry.export_state();
        snapshot
            .set_availability("mcp__demo__search", Some(crate::ToolAvailability::Off))
            .unwrap();
        registry.apply_state(snapshot).unwrap();
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source refreshed");
        let snapshot = registry.export_state();
        assert_eq!(
            snapshot
                .get("mcp__demo__search")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Off
        );
    }

    #[test]
    fn restore_state_adopts_generation_at_or_above_three() {
        // Cold rebuild ratchet: a session whose tool catalog advanced to
        // generation >= 3 restores onto a fresh base-1 registry. `restore_state`
        // adopts the snapshot's generation verbatim; `apply_state` (a gen-matched
        // delta) rejects it. This is the exact divergence the durable worker /
        // session resume rebuild relies on `restore_state` to absorb.
        let source = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        let snapshot = source.export_state().with_generation(3);

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target registry");
        assert_eq!(
            target.generation(),
            1,
            "a fresh registry starts at generation 1"
        );
        let restored = target
            .restore_state(snapshot.clone())
            .expect("restore adopts the snapshot generation");
        assert_eq!(
            restored.generation, 3,
            "restore returns the adopted generation"
        );
        assert!(
            restored.orphaned.is_empty(),
            "all tools resolve, so nothing orphans"
        );
        assert_eq!(
            target.generation(),
            3,
            "restore adopts gen 3 onto a base-1 registry without bumping"
        );
        // A re-export round-trips at the same generation (idempotent).
        assert_eq!(target.export_state().generation(), 3);

        // apply_state on the same high-generation snapshot is rejected — proving
        // the rebuild would have failed without restore_state.
        let fresh = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("fresh registry");
        assert!(
            matches!(
                fresh.apply_state(snapshot),
                Err(ReconfigureError::GenerationMismatch {
                    expected: 3,
                    actual: 1
                })
            ),
            "apply_state must reject a gen-3 snapshot on a base-1 registry"
        );
    }

    /// Build a snapshot whose `mcp__demo__search` entry only resolves while
    /// `ExternalMockSource` is registered — restoring it elsewhere orphans it.
    fn snapshot_with_external_tool() -> ToolState {
        let source = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("source registry");
        source
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        source.export_state()
    }

    #[tokio::test]
    async fn restore_orphans_unresolved_tools_instead_of_failing() {
        let mut snapshot = snapshot_with_external_tool();
        snapshot
            .set_availability(
                "mcp__demo__search",
                Some(crate::ToolAvailability::Showcased),
            )
            .expect("override set");

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        let report = target
            .restore_state(snapshot)
            .expect("restore tolerates the missing source");
        assert_eq!(report.orphaned, vec!["mcp__demo__search".to_string()]);

        // The orphan surfaces as Off without mutating the stored manifest.
        let view = target
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == "mcp__demo__search")
            .expect("orphan stays in the surface listing");
        assert_eq!(
            view.effective_availability(),
            crate::ToolAvailability::Off,
            "orphans are forced Off in the view"
        );
        let exported = target.export_state();
        let exported_view = exported
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == "mcp__demo__search")
            .expect("orphan is visible in exported tool state");
        assert_eq!(
            exported_view.effective_availability(),
            crate::ToolAvailability::Off,
            "exported ToolState exposes the same forced-Off orphan view"
        );
        let entry = exported.get("mcp__demo__search").expect("orphan exported");
        assert!(entry.is_orphaned());
        assert_eq!(
            entry.manifest().effective_availability(),
            crate::ToolAvailability::Off,
            "entry manifest is also the public forced-Off view"
        );
        assert_eq!(
            entry.stored_manifest().availability_override,
            Some(crate::ToolAvailability::Showcased),
            "the persisted override survives orphaning"
        );

        // Execution fails loudly with a precise error.
        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = target
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(!result.is_success());
        assert!(
            format!("{result:?}").contains("unavailable"),
            "orphan execution error names the condition: {result:?}"
        );

        // Bound tools are unaffected.
        assert!(target.resolve_contract("mock_tool").is_some());
    }

    #[tokio::test]
    async fn orphan_rebinds_when_source_is_upserted_again() {
        let mut snapshot = snapshot_with_external_tool();
        snapshot
            .set_availability(
                "mcp__demo__search",
                Some(crate::ToolAvailability::Showcased),
            )
            .expect("override set");
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target.restore_state(snapshot).expect("restore");
        let orphaned_generation = target.generation();

        target
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("the returning source must not conflict with its own orphan");
        assert!(
            target.generation() > orphaned_generation,
            "rebinding bumps the generation"
        );

        let exported = target.export_state();
        let entry = exported.get("mcp__demo__search").expect("entry kept");
        assert!(
            !entry.is_orphaned(),
            "the orphan rebound to the live source"
        );
        assert_eq!(
            entry.manifest().availability_override,
            Some(crate::ToolAvailability::Showcased),
            "rebinding preserves the persisted override"
        );

        let context = test_tool_context();
        let args = json!({ "query": "hello" });
        let result = target
            .execute(crate::ToolCall {
                name: "mcp__demo__search",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.is_success(), "rebound tool executes: {result:?}");
    }

    #[tokio::test]
    async fn orphan_rebinds_lazily_via_resolve_manifest() {
        // `NamedExactSource` advertises nothing, so reconcile-on-upsert cannot
        // rebind; only the lazy `resolve_manifest` path can.
        let source_registry = ToolRegistry::empty();
        source_registry
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source registered");
        assert!(source_registry.resolve_manifest("host_only").is_some());
        let snapshot = source_registry.export_state();

        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        let report = target.restore_state(snapshot).expect("restore");
        assert_eq!(report.orphaned, vec!["host_only".to_string()]);

        target
            .upsert_source(Arc::new(NamedExactSource { id: "exact-a" }))
            .expect("source returns");
        let manifest = target
            .resolve_manifest("host_only")
            .expect("resolves after the source returned");
        assert_eq!(
            manifest.effective_availability(),
            crate::ToolAvailability::Callable,
            "lazy rebind drops the forced-Off orphan view"
        );
        assert!(
            !target
                .export_state()
                .get("host_only")
                .expect("entry kept")
                .is_orphaned()
        );
    }

    #[test]
    fn restore_still_fails_when_name_resolves_with_different_id() {
        struct ReplacedSearchTool;
        #[async_trait::async_trait]
        impl ToolProvider for ReplacedSearchTool {
            fn tool_manifests(&self) -> Vec<ToolManifest> {
                manifests(vec![ToolDefinition::raw_with_id(
                    "tool:replaced",
                    "mcp__demo__search",
                    "a different implementation under the same name",
                    ToolDefinition::default_input_schema(),
                    json!({}),
                )])
            }
            fn resolve_contract(&self, _name: &str) -> Option<Arc<ToolContract>> {
                None
            }
            async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
                ToolResult::ok(json!("ok"))
            }
        }

        let snapshot = snapshot_with_external_tool();
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .add_tool_provider(Arc::new(ReplacedSearchTool))
            .expect("replacement registered");
        let err = target
            .restore_state(snapshot)
            .expect_err("same name with a different id is a real conflict");
        assert!(matches!(err, ReconfigureError::Validation(_)));
    }

    #[test]
    fn apply_state_round_trips_while_orphans_exist() {
        // `export_state` → edit → `apply_state` must work with an orphan in
        // the snapshot: the exported orphan flag exempts it from strictness.
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .restore_state(snapshot_with_external_tool())
            .expect("restore");

        let mut edited = target.export_state();
        edited
            .set_availability("mock_tool", Some(crate::ToolAvailability::Searchable))
            .expect("edit bound tool");
        target
            .apply_state(edited)
            .expect("apply accepts the snapshot it exported");
        let exported = target.export_state();
        assert!(exported.get("mcp__demo__search").unwrap().is_orphaned());
        assert_eq!(
            exported
                .get("mock_tool")
                .unwrap()
                .manifest()
                .effective_availability(),
            crate::ToolAvailability::Searchable
        );

        // But a snapshot that does NOT mark the tool orphaned still fails —
        // strictness is preserved for entries that were bound at export.
        let strict = snapshot_with_external_tool().with_generation(target.generation());
        assert!(matches!(
            target.apply_state(strict),
            Err(ReconfigureError::Validation(_))
        ));
    }

    #[test]
    fn orphan_flag_serializes_and_legacy_snapshots_deserialize_as_bound() {
        let target = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("target");
        target
            .restore_state(snapshot_with_external_tool())
            .expect("restore");
        let value = serde_json::to_value(target.export_state()).expect("serializes");
        assert_eq!(value["tools"]["mcp__demo__search"]["orphaned"], json!(true));
        assert!(
            value["tools"]["mock_tool"].get("orphaned").is_none(),
            "bound entries omit the flag, keeping old and new snapshots byte-compatible"
        );

        let legacy: ToolStateEntry = serde_json::from_value(json!({
            "manifest": value["tools"]["mock_tool"]["manifest"]
        }))
        .expect("legacy entry without the flag deserializes");
        assert!(!legacy.is_orphaned());
    }

    #[test]
    fn remove_source_removes_all_source_tools() {
        let registry = ToolRegistry::from_tool_provider(Arc::new(MockTool)).expect("registry");
        registry
            .upsert_source(Arc::new(ExternalMockSource))
            .expect("source registered");
        registry
            .remove_source_id("external")
            .expect("source removed");
        let defs = registry.tool_manifests();
        assert!(!defs.iter().any(|def| def.name == "mcp__demo__search"));
    }

    #[test]
    fn project_tool_catalog_keeps_searchable_tools_with_catalog_metadata() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            let tool = crate::ToolDefinition::raw_with_id(
                format!("tool:{name}"),
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            );
            match name {
                "read_file" => {
                    tool.with_lashlang_binding(crate::LashlangToolBinding::new(["files"], "read"))
                }
                "search_tools" => {
                    tool.with_lashlang_binding(crate::LashlangToolBinding::new(["tools"], "search"))
                }
                _ => tool,
            }
        }
        let catalog = project_tool_catalog([
            crate::ToolCatalogEntry {
                manifest: dummy_tool("read_file").manifest(),
                availability: crate::ToolAvailability::Showcased,
            },
            crate::ToolCatalogEntry {
                manifest: dummy_tool("search_tools").manifest(),
                availability: crate::ToolAvailability::Callable,
            },
        ]);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0]["name"], serde_json::json!("read_file"));
        assert_eq!(
            catalog[0]["contract"]["signature"],
            serde_json::json!("await files.read({})?")
        );
        assert_eq!(catalog[0]["showcased"], serde_json::json!(true));
        assert_eq!(catalog[1]["callable"], serde_json::json!(true));
    }

    #[test]
    fn project_tool_catalog_preserves_dynamic_output_contracts() {
        fn dummy_tool(name: &str) -> crate::ToolDefinition {
            crate::ToolDefinition::raw_with_id(
                format!("tool:{name}"),
                name,
                format!("desc for {name}"),
                crate::ToolDefinition::default_input_schema(),
                serde_json::json!({}),
            )
            .with_lashlang_binding(crate::LashlangToolBinding::new(["llm"], "query"))
        }
        let catalog = project_tool_catalog([crate::ToolCatalogEntry {
            manifest: dummy_tool("llm_query")
                .with_output_from_input_schema(
                    "output",
                    Some(serde_json::json!({ "type": "string" })),
                )
                .manifest(),
            availability: crate::ToolAvailability::Searchable,
        }]);

        assert_eq!(
            catalog[0]["contract"]["signature"],
            serde_json::json!("await llm.query<T = str>({})?")
        );
        assert_eq!(catalog[0]["contract"]["returns"], serde_json::json!("T"));
    }
}

pub(crate) fn project_tool_catalog<I>(entries: I) -> Vec<serde_json::Value>
where
    I: IntoIterator<Item = crate::ToolCatalogEntry>,
{
    entries
        .into_iter()
        .filter(|entry| entry.availability.is_searchable())
        .map(|entry| {
            let manifest = entry.manifest;
            let availability = entry.availability;
            let lashlang_binding = manifest.lashlang_binding.executable_for(&manifest.name);
            let call = lashlang_binding.call_path();
            let mut projected = serde_json::json!({
                "id": manifest.id,
                "name": manifest.name,
                "module_path": lashlang_binding.module_path,
                "operation": lashlang_binding.operation,
                "authority_type": lashlang_binding.authority_type,
                "call": call,
                "description": manifest.description,
                "aliases": lashlang_binding.aliases,
                "availability": availability,
                "callable": availability.is_callable(),
                "showcased": availability.is_showcased(),
                "searchable": availability.is_searchable(),
                "activation": manifest.activation,
            });
            if let Some(contract) = manifest.compact_contract {
                projected
                    .as_object_mut()
                    .expect("projected tool catalog entry is an object")
                    .insert("contract".to_string(), serde_json::json!(contract));
            }
            projected
        })
        .collect()
}
