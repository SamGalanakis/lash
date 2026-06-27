use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use lashlang::{
    AbilityOp, AbilityResult, ExecutionHost, ExecutionHostError, ExecutionOutcome, Record,
    Snapshot, State, Value, canonical_program_ir, canonical_program_source, parse,
};
use proptest::prelude::*;

#[derive(Default)]
struct DeterministicHost;

impl ExecutionHost for DeterministicHost {
    async fn perform(&self, op: AbilityOp) -> Result<AbilityResult, ExecutionHostError> {
        match op {
            AbilityOp::ResourceOperation(operation) => match operation.operation.as_str() {
                "echo" => Ok(AbilityResult::Value(
                    operation
                        .args
                        .first()
                        .and_then(Value::as_record)
                        .and_then(|record| record.get("value"))
                        .cloned()
                        .unwrap_or(Value::Null),
                )),
                "fail" => Err(ExecutionHostError::new("fail")),
                _ => Err(ExecutionHostError::new(format!(
                    "unknown module operation: {}",
                    operation.operation
                ))),
            },
            AbilityOp::Finish(value) | AbilityOp::Fail(value) => Ok(AbilityResult::Value(value)),
            _ => Err(ExecutionHostError::new("unsupported host ability")),
        }
    }
}

fn run_execute(
    source: &str,
    state: &mut State,
    host: &DeterministicHost,
) -> Result<ExecutionOutcome, ExecuteError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(execute(source, state, host))
}

fn finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("expected `finish`"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] lashlang::ParseError),
    #[error(transparent)]
    Runtime(#[from] lashlang::RuntimeError),
}

async fn execute<H: ExecutionHost>(
    source: &str,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, ExecuteError> {
    let compiled = if source.contains("tools.") {
        let program = parse(source)?;
        let linked =
            lashlang::LinkedModule::link(program, property_host_environment()).map_err(|err| {
                ExecuteError::Runtime(lashlang::RuntimeError::ValueError {
                    message: err.to_string(),
                })
            })?;
        lashlang::compile_linked(&linked)
    } else {
        lashlang::compile(source)?
    };
    lashlang::execute(&compiled, state, host)
        .await
        .map_err(ExecuteError::Runtime)
}

fn property_host_environment() -> lashlang::LashlangHostEnvironment {
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources.add_module_operation(
        ["tools"],
        "Tools",
        "echo",
        "echo",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    resources.add_module_operation(
        ["tools"],
        "Tools",
        "fail",
        "fail",
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    lashlang::LashlangHostEnvironment::new(resources, lashlang::LashlangAbilities::all())
}

#[derive(Clone, Debug)]
enum GenValue {
    Null,
    Bool(bool),
    Number(i32),
    String(String),
    List(Vec<GenValue>),
    Record(Vec<(String, GenValue)>),
}

fn lashlang_string_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            proptest::char::range(' ', '~'),
            Just('\n'),
            Just('\r'),
            Just('\t'),
        ],
        0..20,
    )
    .prop_map(|chars: Vec<char>| chars.into_iter().collect())
}

impl GenValue {
    fn to_value(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(*value as f64),
            Self::String(value) => Value::String(value.clone().into()),
            Self::List(values) => {
                Value::List(values.iter().map(Self::to_value).collect::<Vec<_>>().into())
            }
            Self::Record(entries) => Value::Record(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect::<lashlang::Record>()
                    .into(),
            ),
        }
    }

    fn to_source(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => value.to_string(),
            Self::String(value) => encode_string(value),
            Self::List(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(Self::to_source)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Record(entries) => format!(
                "{{{}}}",
                entries
                    .iter()
                    .map(|(key, value)| format!("{key}: {}", value.to_source()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

fn ident_strategy() -> impl Strategy<Value = String> {
    "[a-z_][a-z0-9_]{0,10}".prop_filter("reserved lashlang keyword", |ident| {
        !matches!(
            ident.as_str(),
            "if" | "else"
                | "for"
                | "in"
                | "finish"
                | "submit"
                | "print"
                | "call"
                | "true"
                | "false"
                | "null"
                | "and"
                | "or"
                | "not"
        )
    })
}

fn gen_value_strategy() -> impl Strategy<Value = GenValue> {
    let leaf = prop_oneof![
        Just(GenValue::Null),
        any::<bool>().prop_map(GenValue::Bool),
        (-10_000i32..=10_000i32).prop_map(GenValue::Number),
        lashlang_string_strategy().prop_map(GenValue::String),
    ];

    leaf.prop_recursive(4, 64, 8, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(GenValue::List),
            prop::collection::vec((ident_strategy(), inner), 0..4).prop_map(GenValue::Record),
        ]
    })
}

fn encode_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn globals_strategy() -> impl Strategy<Value = HashMap<String, GenValue>> {
    prop::collection::hash_map(ident_strategy(), gen_value_strategy(), 0..6)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_local_rejects: 10_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn parse_never_panics_on_arbitrary_input(source in ".*") {
        let result = catch_unwind(AssertUnwindSafe(|| parse(&source)));
        prop_assert!(result.is_ok(), "parse panicked for input: {source:?}");
    }

    #[test]
    fn execute_never_panics_on_arbitrary_input(source in ".*") {
        let host = DeterministicHost;
        let mut state = State::new();
        let result = catch_unwind(AssertUnwindSafe(|| run_execute(&source, &mut state, &host)));
        prop_assert!(result.is_ok(), "execute panicked for input: {source:?}");
    }

    #[test]
    fn generated_value_programs_round_trip_through_parser_and_runtime(
        ident in ident_strategy(),
        value in gen_value_strategy()
    ) {
        let expected = value.to_value();
        let source = format!("{ident} = {}\nfinish {ident}\n", value.to_source());
        let host = DeterministicHost;
        let mut state = State::new();

        let actual = finished(
            run_execute(&source, &mut state, &host)
                .expect("generated value program should execute")
        );

        prop_assert_eq!(actual, expected.clone());
        let globals = state.globals();
        prop_assert_eq!(globals.get(&ident), Some(&expected));
    }

    #[test]
    fn generated_value_programs_round_trip_through_canonical_source(
        ident in ident_strategy(),
        value in gen_value_strategy()
    ) {
        let source = format!("{ident} = {}\nfinish {ident}\n", value.to_source());
        let program = parse(&source).expect("generated source should parse");
        let rendered = canonical_program_source(&program).expect("canonical source should render");
        let reparsed = parse(&rendered).expect("canonical source should parse");

        prop_assert_eq!(canonical_program_ir(reparsed), canonical_program_ir(program));
    }

    #[test]
    fn snapshot_round_trip_preserves_state(
        globals in globals_strategy()
    ) {
        let state = State::from_snapshot(Snapshot {
            globals: globals
                .iter()
                .map(|(key, value)| (key.clone(), value.to_value()))
                .collect(),
        });

        let encoded = serde_json::to_vec(&state.snapshot()).expect("snapshot encode");
        let decoded: Snapshot = serde_json::from_slice(&encoded).expect("snapshot decode");
        let restored = State::from_snapshot(decoded);

        prop_assert_eq!(restored.globals(), state.globals());
    }

    #[test]
    fn execution_from_restored_snapshot_matches_fresh_state(
        globals in globals_strategy(),
        value in gen_value_strategy()
    ) {
        let base_globals: Record = globals
            .iter()
            .map(|(key, value)| (key.clone(), value.to_value()))
            .collect();
        let source = format!("result = {}\nfinish result\n", value.to_source());
        let host = DeterministicHost;

        let mut fresh = State::from_snapshot(Snapshot {
            globals: base_globals.clone(),
        });
        let mut restored = State::from_snapshot(Snapshot {
            globals: base_globals,
        });
        let blob = serde_json::to_vec(&restored.snapshot()).expect("snapshot encode");
        let snapshot: Snapshot = serde_json::from_slice(&blob).expect("snapshot decode");
        restored = State::from_snapshot(snapshot);

        let fresh_value = finished(run_execute(&source, &mut fresh, &host).expect("fresh execution"));
        let restored_value = finished(
            run_execute(&source, &mut restored, &host).expect("restored execution")
        );

        prop_assert_eq!(fresh_value, restored_value);
        prop_assert_eq!(fresh.globals(), restored.globals());
    }

    #[test]
    fn tool_result_contract_is_stable_for_generated_values(
        value in gen_value_strategy()
    ) {
        let source = format!(
            "r = await tools.echo({{ value: {} }})\nfinish r\n",
            value.to_source()
        );
        let host = DeterministicHost;
        let mut state = State::new();

        let result = finished(run_execute(&source, &mut state, &host).expect("tool call should succeed"));
        let record = result.as_record().expect("tool result should be a record");

        prop_assert_eq!(record.get("ok"), Some(&Value::Bool(true)));
        prop_assert_eq!(record.get("value"), Some(&value.to_value()));
    }

    #[test]
    fn ternary_selects_generated_branch_without_evaluating_the_other_side(
        condition in any::<bool>(),
        yes in gen_value_strategy(),
        no in gen_value_strategy()
    ) {
        let expected = if condition { yes.to_value() } else { no.to_value() };
        let source = format!(
            "result = {} ? {} : {}\nfinish result\n",
            if condition { "true" } else { "false" },
            yes.to_source(),
            no.to_source()
        );
        let host = DeterministicHost;
        let mut state = State::new();

        let actual = finished(run_execute(&source, &mut state, &host).expect("ternary execution"));

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn generated_type_literal_always_produces_valid_json_schema(
        ty in gen_type_strategy(6)
    ) {
        let source = format!("x = {}\nfinish x\n", ty.to_source());
        let host = DeterministicHost;
        let mut state = State::new();
        let outcome = run_execute(&source, &mut state, &host);
        let value = finished(outcome.expect("Type literal should execute"));
        let inner = lashlang::unwrap_type_value(&value).expect("wrapped type");
        let schema = inner.as_record().expect("schema record");
        // Every generated type is an object at the top level.
        prop_assert_eq!(&schema["type"], &Value::String("object".into()));
        // `required` always exists as a list (possibly empty if everything is optional).
        prop_assert!(matches!(&schema["required"], Value::List(_)));
        prop_assert_eq!(&schema["additionalProperties"], &Value::Bool(false));
    }
}

// ------------------------------------------------------------------
//  Generator for arbitrary Type literals. Used only by property tests.
// ------------------------------------------------------------------

#[derive(Clone, Debug)]
enum GenType {
    Scalar(&'static str),
    Enum(Vec<String>),
    List(Box<GenType>),
    Object(Vec<(String, GenType, bool)>),
}

impl GenType {
    fn to_source(&self) -> String {
        // Only Object is a valid top-level Type literal; others appear as
        // field types. Caller must wrap scalars/lists/enums in a field.
        match self {
            Self::Scalar(name) => (*name).to_string(),
            Self::Enum(values) => {
                let rendered: Vec<String> = values.iter().map(|v| encode_string(v)).collect();
                format!("enum[{}]", rendered.join(", "))
            }
            Self::List(inner) => format!("list[{}]", inner.to_source()),
            Self::Object(fields) => {
                let rendered: Vec<String> = fields
                    .iter()
                    .map(|(name, ty, optional)| {
                        let opt = if *optional { "?" } else { "" };
                        format!("{name}: {}{opt}", ty.to_source())
                    })
                    .collect();
                format!("Type {{ {} }}", rendered.join(", "))
            }
        }
    }
}

fn gen_field_name() -> impl Strategy<Value = String> {
    // Lowercase ASCII identifiers to avoid collision with keywords.
    "[a-z][a-z0-9_]{0,6}".prop_map(|s| {
        // Avoid reserved identifiers that could shadow keywords.
        const RESERVED: &[&str] = &[
            "if", "else", "for", "in", "start", "await", "cancel", "finish", "submit", "print",
            "call", "and", "or", "not", "true", "false", "null",
        ];
        if RESERVED.iter().any(|r| *r == s) {
            format!("{s}_")
        } else {
            s
        }
    })
}

fn gen_enum_value() -> impl Strategy<Value = String> {
    "[a-z]{1,5}".prop_map(|s| s)
}

fn gen_scalar_name() -> impl Strategy<Value = GenType> {
    prop_oneof![
        Just(GenType::Scalar("str")),
        Just(GenType::Scalar("int")),
        Just(GenType::Scalar("float")),
        Just(GenType::Scalar("bool")),
        Just(GenType::Scalar("dict")),
        Just(GenType::Scalar("any")),
    ]
}

fn gen_type_strategy(max_depth: u32) -> impl Strategy<Value = GenType> {
    gen_type_expr(max_depth).prop_flat_map(|inner| {
        // Wrap in an Object if the inner isn't already one — the top-level
        // Type literal must always be an Object in our grammar.
        match inner {
            GenType::Object(fields) => Just(GenType::Object(fields)).boxed(),
            other => (gen_field_name(), Just(other))
                .prop_map(|(name, ty)| GenType::Object(vec![(name, ty, false)]))
                .boxed(),
        }
    })
}

fn gen_type_expr(_max_depth: u32) -> BoxedStrategy<GenType> {
    let leaf = prop_oneof![
        gen_scalar_name(),
        prop::collection::vec(gen_enum_value(), 1..4).prop_map(|values| {
            // Deduplicate to keep JSON-Schema enums valid.
            let mut seen = std::collections::HashSet::new();
            let unique: Vec<String> = values
                .into_iter()
                .filter(|v| seen.insert(v.clone()))
                .collect();
            GenType::Enum(unique)
        }),
    ];
    leaf.prop_recursive(3, 32, 4, |inner| {
        prop_oneof![
            inner.clone().prop_map(|ty| GenType::List(Box::new(ty))),
            prop::collection::vec((gen_field_name(), inner, any::<bool>()), 1..4).prop_map(
                |fields| {
                    let mut seen = std::collections::HashSet::new();
                    let unique: Vec<(String, GenType, bool)> = fields
                        .into_iter()
                        .filter(|(name, _, _)| seen.insert(name.clone()))
                        .collect();
                    GenType::Object(unique)
                }
            ),
        ]
    })
    .boxed()
}
