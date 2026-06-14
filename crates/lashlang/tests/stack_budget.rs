use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ExecutionOutcome,
    LashlangAbilities, LashlangHostEnvironment, Record, State, Value, compile_linked, execute,
    parse,
};
use std::sync::Arc;

const STACK_BUDGET_BYTES: usize = 2 * 1024 * 1024;

#[test]
fn stack_budget_lashlang_parse_link_compile_execute_process_fanout() {
    run_on_stack_budget("stack-budget-lashlang-process-fanout", || {
        let test = Box::pin(async {
            let program = parse(
                r#"
process child(value: str) {
  finish { value: value, lookup: "lookup:" + value }
}

left = start child(value: "left")
right = start child(value: "right")
joined = await { left: left, right: right }
sleep for "0ms"
submit {
  left: joined.left.lookup,
  right: joined.right.lookup,
  final: "stack-budget"
}
"#,
            )
            .expect("program parses");
            let surface = LashlangHostEnvironment::new(
                lashlang::LashlangHostCatalog::new(),
                LashlangAbilities::all(),
            );
            let linked = lashlang::LinkedModule::link(program, surface).expect("program links");
            let compiled = compile_linked(&linked);
            let mut state = State::new();

            let outcome = execute(&compiled, &mut state, &StackBudgetHost)
                .await
                .expect("program executes");
            let ExecutionOutcome::Finished(value) = outcome else {
                panic!("expected submitted value");
            };

            assert_eq!(
                serde_json::to_value(&value).expect("value json"),
                serde_json::json!({
                    "left": {
                        "ok": true,
                        "value": "lookup:left",
                    },
                    "right": {
                        "ok": true,
                        "value": "lookup:right",
                    },
                    "final": "stack-budget",
                })
            );
        });

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(test)
    });
}

fn run_on_stack_budget(name: &str, test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(STACK_BUDGET_BYTES)
        .spawn(test)
        .expect("spawn stack-budget thread")
        .join()
        .expect("stack-budget thread");
}

struct StackBudgetHost;

impl ExecutionHost for StackBudgetHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::StartProcess(start) => {
                let Value::String(value) = start
                    .args
                    .get("value")
                    .cloned()
                    .unwrap_or(Value::String("unknown".into()))
                else {
                    return Err(ExecutionHostError::new("expected string value"));
                };
                let mut record = Record::new();
                record.insert("value".to_string(), Value::String(value.clone()));
                record.insert(
                    "lookup".to_string(),
                    Value::String(format!("lookup:{value}").into()),
                );
                Ok(AbilityResult::Value(Value::Record(Arc::new(record))))
            }
            AbilityOp::Await(value) => Ok(AbilityResult::Value(value)),
            AbilityOp::Sleep(_) => Ok(AbilityResult::Value(Value::Null)),
            AbilityOp::Submit(value) | AbilityOp::Finish(value) | AbilityOp::Fail(value) => {
                Ok(AbilityResult::Value(value))
            }
            _ => Err(ExecutionHostError::new(
                "unsupported stack-budget host ability",
            )),
        }
    }
}
