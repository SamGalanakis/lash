//! JSON serialization for `Value`. Bidirectional bridge between the
//! lashlang value tree and `serde_json::Value`. Image attachments encode
//! as `{"type": "image", "id": ..., "label": ..., "size": ..., "width":
//! ..., "height": ...}` and round-trip via `image_to_json` /
//! `image_from_json_map`. Projected values serialize with the canonical
//! `{"__projected__": <inner>}` wrapper (defined in `lash-rlm-types`).

use std::sync::Arc;

use super::*;
use super::{ImageValue, ProjectedFuture, Value};

pub(crate) fn json_number(value: f64) -> Option<serde_json::Number> {
    if !value.is_finite() {
        return None;
    }
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return Some(serde_json::Number::from(value as i64));
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return Some(serde_json::Number::from(value as u64));
        }
    }
    serde_json::Number::from_f64(value)
}

pub(crate) fn to_json_async<'a>(value: &'a Value) -> ProjectedFuture<'a, serde_json::Value> {
    Box::pin(async move {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(value) => serde_json::Value::Bool(*value),
            Value::Number(value) => json_number(*value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(value) => serde_json::Value::String(value.to_string()),
            Value::Image(image) => image_to_json(image),
            Value::List(values) => {
                let mut out = Vec::with_capacity(values.len());
                for value in values.iter() {
                    out.push(to_json_async(value).await);
                }
                serde_json::Value::Array(out)
            }
            Value::Record(record) => {
                let mut object = serde_json::Map::with_capacity(record.len());
                for (key, value) in record.iter() {
                    object.insert(key.to_string(), to_json_async(value).await);
                }
                serde_json::Value::Object(object)
            }
            Value::Projected(value) => to_json_async(&value.materialize_async().await).await,
        }
    })
}

pub(crate) fn to_json(value: &Value) -> serde_json::Value {
    if value_contains_projected(value) {
        futures_executor::block_on(to_json_async(value))
    } else {
        to_json_direct(value)
    }
}

pub(crate) fn to_json_blocking(value: &Value) -> serde_json::Value {
    to_json(value)
}

pub(crate) fn to_json_direct(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Number(value) => json_number(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(value) => serde_json::Value::String(value.to_string()),
        Value::Image(image) => image_to_json(image),
        Value::List(values) => {
            serde_json::Value::Array(values.iter().map(to_json_direct).collect())
        }
        Value::Record(record) => {
            let mut object = serde_json::Map::with_capacity(record.len());
            for (key, value) in record.iter() {
                object.insert(key.to_string(), to_json_direct(value));
            }
            serde_json::Value::Object(object)
        }
        Value::Projected(_) => unreachable!("projected values require async json conversion"),
    }
}

pub(crate) fn image_to_json(image: &ImageValue) -> serde_json::Value {
    let mut object = serde_json::Map::with_capacity(7);
    object.insert(
        "type".to_string(),
        serde_json::Value::String("image".to_string()),
    );
    object.insert(
        "id".to_string(),
        serde_json::Value::String(image.id.clone()),
    );
    object.insert(
        "label".to_string(),
        serde_json::Value::String(image.label.clone()),
    );
    object.insert(
        "size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(image.size)),
    );
    object.insert(
        "width".to_string(),
        image
            .width
            .map(|width| serde_json::Value::Number(serde_json::Number::from(width)))
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "height".to_string(),
        image
            .height
            .map(|height| serde_json::Value::Number(serde_json::Number::from(height)))
            .unwrap_or(serde_json::Value::Null),
    );
    serde_json::Value::Object(object)
}

pub fn from_json(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(value),
        serde_json::Value::Number(value) => Value::Number(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Value::String(value.into()),
        serde_json::Value::Array(values) => {
            Value::List(values.into_iter().map(from_json).collect::<Vec<_>>().into())
        }
        serde_json::Value::Object(map) => image_from_json_map(&map)
            .map(Value::Image)
            .unwrap_or_else(|| {
                Value::Record(Arc::new(
                    map.into_iter()
                        .map(|(key, value)| (key, from_json(value)))
                        .collect(),
                ))
            }),
    }
}

pub(crate) fn image_from_json_map(map: &serde_json::Map<String, serde_json::Value>) -> Option<ImageValue> {
    if map.get("type")?.as_str()? != "image" {
        return None;
    }
    Some(ImageValue {
        id: map.get("id")?.as_str()?.to_string(),
        label: map.get("label")?.as_str()?.to_string(),
        size: map.get("size")?.as_u64()?,
        width: optional_u32_field(map.get("width")?)?,
        height: optional_u32_field(map.get("height")?)?,
    })
}

pub(crate) fn optional_u32_field(value: &serde_json::Value) -> Option<Option<u32>> {
    match value {
        serde_json::Value::Null => Some(None),
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Some),
        _ => None,
    }
}
