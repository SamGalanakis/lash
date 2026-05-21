use super::*;

#[test]
fn plugin_host_rejects_observational_memory_without_supporting_plugin() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::testing::FakeStandardContextApproachPluginFactory::rolling_history(),
    )]);
    let result = host.build_session(
        "root",
        ExecutionMode::standard(),
        Some(crate::StandardContextApproach::ObservationalMemory(
            crate::ObservationalMemoryConfig::default(),
        )),
        None,
    );
    let err = match result {
        Ok(_) => panic!("OM should require supporting plugin"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains(
            "standard context approach `ObservationalMemory` requires a supporting plugin factory"
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn plugin_host_rejects_standard_context_for_rlm_sessions() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::testing::FakeStandardContextApproachPluginFactory::rolling_history(),
    )]);
    let result = host.build_session(
        "root",
        ExecutionMode::new("rlm"),
        Some(crate::StandardContextApproach::default()),
        None,
    );
    let err = match result {
        Ok(_) => panic!("RLM sessions should not accept a standard context approach"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("standard context approach only applies to standard execution mode"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn runtime_requires_explicit_max_context_tokens() {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let result = LashRuntime::from_embedded_state(
        SessionPolicy {
            execution_mode: ExecutionMode::standard(),
            provider: mock_provider(Vec::new()).into_handle(),
            model: "mock-model".to_string(),
            max_context_tokens: None,
            ..SessionPolicy::default()
        },
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        RuntimeSessionState::default(),
    )
    .await;
    match result {
        Err(SessionError::Protocol(message)) => {
            assert!(message.contains("max_context_tokens"));
        }
        Err(other) => panic!("unexpected session error: {other}"),
        Ok(_) => panic!("runtime should reject implicit model metadata"),
    }
}
