use std::sync::Arc;

use super::{
    ProjectedValue, Record, SnapshotJson, Value, image_from_json_map, record_with_capacity,
};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const PROJECTED_SNAPSHOT_TAG: &str = "__lashlang_snapshot_projected__";
const PROJECTED_SNAPSHOT_NAME: &str = "name";
const PROJECTED_SNAPSHOT_TYPE_NAME: &str = "type_name";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub(super) globals: Record,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn globals(&self) -> &Record {
        &self.globals
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            globals: self.globals.clone(),
        }
    }

    pub fn from_snapshot(snapshot: Snapshot) -> Self {
        Self {
            globals: snapshot.globals,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub globals: Record,
}

impl Serialize for Snapshot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Snapshot", 1)?;
        state.serialize_field(
            "globals",
            &SnapshotJson(&Value::Record(Arc::new(self.globals.clone()))),
        )?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Snapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SnapshotWire {
            globals: serde_json::Map<String, serde_json::Value>,
        }

        let wire = SnapshotWire::deserialize(deserializer)?;
        Ok(Self {
            globals: snapshot_record_from_json(wire.globals),
        })
    }
}

fn snapshot_record_from_json(map: serde_json::Map<String, serde_json::Value>) -> Record {
    let mut record = record_with_capacity(map.len());
    for (key, value) in map {
        record.insert(key, snapshot_value_from_json(value));
    }
    record
}

fn snapshot_value_from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(value) => Value::Number(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Value::String(value.into()),
        serde_json::Value::Array(values) => Value::List(
            values
                .into_iter()
                .map(snapshot_value_from_json)
                .collect::<Vec<_>>()
                .into(),
        ),
        serde_json::Value::Object(map) => {
            if let Some((name, type_name)) = projected_snapshot_marker(&map) {
                return Value::Projected(ProjectedValue::unavailable_after_restore(
                    name, type_name,
                ));
            }
            image_from_json_map(&map)
                .map(Value::Image)
                .unwrap_or_else(|| Value::Record(Arc::new(snapshot_record_from_json(map))))
        }
    }
}

fn projected_snapshot_marker(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<(String, String)> {
    if map.len() != 3 {
        return None;
    }
    if !map.get(PROJECTED_SNAPSHOT_TAG)?.as_bool()? {
        return None;
    }
    let name = map.get(PROJECTED_SNAPSHOT_NAME)?.as_str()?.to_string();
    let type_name = map.get(PROJECTED_SNAPSHOT_TYPE_NAME)?.as_str()?.to_string();
    Some((name, type_name))
}
