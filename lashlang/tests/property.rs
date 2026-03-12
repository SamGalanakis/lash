use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use lashlang::{Record, Snapshot, State, ToolHost, ToolHostError, Value, execute, parse};
use proptest::prelude::*;

#[derive(Default)]
struct DeterministicHost;

impl ToolHost for DeterministicHost {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
        match name {
            "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
            "fail" => Err(ToolHostError::new("fail")),
            _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
        }
    }
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
            Self::String(value) => Value::String(value.clone()),
            Self::List(values) => Value::List(values.iter().map(Self::to_value).collect()),
            Self::Record(entries) => Value::Record(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect(),
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
    ("[a-z_][a-z0-9_]{0,10}").prop_map(|s| s)
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
        let result = catch_unwind(AssertUnwindSafe(|| execute(&source, &mut state, &host)));
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

        let actual = execute(&source, &mut state, &host)
            .expect("generated value program should execute");

        prop_assert_eq!(actual, expected.clone());
        prop_assert_eq!(state.globals().get(&ident), Some(&expected));
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

        let fresh_value = execute(&source, &mut fresh, &host).expect("fresh execution");
        let restored_value = execute(&source, &mut restored, &host).expect("restored execution");

        prop_assert_eq!(fresh_value, restored_value);
        prop_assert_eq!(fresh.globals(), restored.globals());
    }

    #[test]
    fn tool_result_contract_is_stable_for_generated_values(
        value in gen_value_strategy()
    ) {
        let source = format!("r = call echo {{ value: {} }}\nfinish r\n", value.to_source());
        let host = DeterministicHost;
        let mut state = State::new();

        let result = execute(&source, &mut state, &host).expect("tool call should succeed");
        let record = result.as_record().expect("tool result should be a record");

        prop_assert_eq!(record.get("ok"), Some(&Value::Bool(true)));
        prop_assert_eq!(record.get("value"), Some(&value.to_value()));
    }
}
