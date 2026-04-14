#![cfg(feature = "sqlite-store")]

use std::sync::Arc;

use lash::{
    ExecutionMode, LashRuntime, Message, MessageRole, Part, PartKind, PersistedSessionConfig,
    PersistedTurnState, PruneState, RuntimeStore, SessionGraph, SessionHead, Store, TokenUsage,
};

fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

#[tokio::test]
async fn embedded_runtime_builder_loads_state_from_store() {
    let store = Arc::new(Store::memory().expect("store"));
    let checkpoint_ref = store
        .put_checkpoint(&lash::HydratedSessionCheckpoint {
            turn_state: PersistedTurnState {
                iteration: 3,
                token_usage: TokenUsage {
                    input_tokens: 20,
                    output_tokens: 5,
                    cached_input_tokens: 2,
                    reasoning_tokens: 1,
                },
                last_prompt_usage: None,
            },
            dynamic_state_ref: None,
            dynamic_state: None,
            plugin_snapshot_ref: None,
            plugin_snapshot_revision: None,
            plugin_snapshot: None,
        })
        .checkpoint_ref;
    store.save_session_head(SessionHead {
        session_id: "stored-session".to_string(),
        graph: SessionGraph::from_projection(
            &[text_message("u0", MessageRole::User, "stored question")],
            &[],
        ),
        config: PersistedSessionConfig {
            provider_id: "openai-compatible".into(),
            configured_model: "gpt-5.4-mini".into(),
            context_window: 200_000,
            execution_mode: ExecutionMode::Standard,
            context_approach: lash::ContextApproach::default(),
            model_variant: None,
        },
        checkpoint_ref: Some(checkpoint_ref),
        token_ledger: Vec::new(),
    });

    let runtime = LashRuntime::builder()
        .with_store(store.clone() as Arc<dyn RuntimeStore>)
        .build()
        .await
        .expect("runtime");

    let state = runtime.export_state();
    assert_eq!(state.projected_messages().len(), 1);
    assert_eq!(
        state.projected_messages()[0].parts[0].content,
        "stored question"
    );
    assert_eq!(state.iteration, 3);
    assert_eq!(state.token_usage.input_tokens, 20);
    assert_eq!(state.policy.model, "gpt-5.4-mini");
    assert_eq!(state.session_id, "stored-session");
}

#[tokio::test]
async fn embedded_runtime_builder_rejects_store_bound_to_different_session_id() {
    let store = Arc::new(Store::memory().expect("store"));
    store.save_session_head(SessionHead {
        session_id: "alpha".to_string(),
        graph: SessionGraph::default(),
        config: PersistedSessionConfig {
            provider_id: "openai-compatible".into(),
            configured_model: "gpt-5.4-mini".into(),
            context_window: 200_000,
            execution_mode: ExecutionMode::Standard,
            context_approach: lash::ContextApproach::default(),
            model_variant: None,
        },
        checkpoint_ref: None,
        token_ledger: Vec::new(),
    });

    let err = match LashRuntime::builder()
        .with_store(store as Arc<dyn RuntimeStore>)
        .with_session_id("beta")
        .build()
        .await
    {
        Ok(_) => panic!("mismatched store session should fail"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("bound to session `alpha`"));
}
