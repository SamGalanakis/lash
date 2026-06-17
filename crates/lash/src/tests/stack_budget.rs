use super::*;

#[test]
fn stack_budget_rlm_lashlang_process_turn() -> Result<()> {
    run_async_test_on_stack_budget("stack-budget-rlm-lashlang-process-turn", || async {
        let core = explicit_ephemeral_facets(RlmCore::builder())
            .provider(queued_text_provider(vec![
                r#"```lashlang
process child(tools: Tools, value: str) {
  lookup = await tools.app_lookup({})?
  finish { value: value, ok: lookup.ok }
}

left = start child(tools: tools, value: "left")
right = start child(tools: tools, value: "right")
joined = await { left: left, right: right }
submit {
  left: joined.left.value,
  right: joined.right.value,
  ok: joined.left.ok && joined.right.ok
}
```"#,
            ]))
            .model(mock_model_spec())
            .tools(Arc::new(AppTools))
            .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
            .process_registry(Arc::new(TestLocalProcessRegistry::default()))
            .build()?;
        let session = core.session("stack-budget-rlm-lashlang").open().await?;
        let events = RecordingEvents::default();

        let turn = session
            .turn(TurnInput::text("run stack budget process fanout"))
            .stream_to(&events)
            .await?;
        session.processes().await_all().await?;

        assert_eq!(
            turn.submitted_value(),
            Some(&serde_json::json!({
                "left": {
                    "ok": true,
                    "value": "left",
                },
                "right": {
                    "ok": true,
                    "value": "right",
                },
                "ok": true,
            }))
        );
        assert!(
            events
                .snapshot()
                .await
                .iter()
                .any(|activity| matches!(activity.event, TurnEvent::SubmittedValue { .. }))
        );
        Ok(())
    })
}
