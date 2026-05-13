//! Value types: the dynamically-typed `Value` enum, its projection wrapper,
//! the `ImageValue` attachment descriptor, and the public projection traits
//! (`ProjectedHostValue`, `ProjectedRead`, `ProjectedFuture`).
//!
//! The `Value` enum is the universal currency of the lashlang runtime: every
//! load, every binary op, every host-tool argument, every JSON round-trip
//! flows through it. `ProjectedValue` wraps host-side bindings the runtime
//! can read but should not own; field/index access on a projected source
//! propagates the wrapper so downstream consumers can tell that this came
//! from a projected binding.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use compact_str::CompactString;
use rustc_hash::FxHashMap;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::record::{Symbol, intern_symbol, symbol_name};
use super::{
    Name, Record, RuntimeError, execute_contains_direct, from_json, read_field_ref_direct,
    read_index_ref_direct, stringify_value_async, to_json_blocking, value_len, value_type_name,
    write_number,
};

/// Marker key that wraps a Type literal at its outermost level so a host-side
/// consumer can tell a Type value apart from a plain record. The inner value
/// is the JSON-Schema representation of the type.
pub const LASH_TYPE_KEY: &str = "$lash_type";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageValue {
    pub id: String,
    pub label: String,
    pub size: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl ImageValue {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        size: u64,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            size,
            width,
            height,
        }
    }
}

impl Serialize for ImageValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(6))?;
        map.serialize_entry("type", "image")?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("label", &self.label)?;
        map.serialize_entry("size", &self.size)?;
        map.serialize_entry("width", &self.width)?;
        map.serialize_entry("height", &self.height)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for ImageValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ImageDescriptor {
            #[serde(rename = "type")]
            kind: String,
            id: String,
            label: String,
            size: u64,
            #[serde(default)]
            width: Option<u32>,
            #[serde(default)]
            height: Option<u32>,
        }

        let descriptor = ImageDescriptor::deserialize(deserializer)?;
        if descriptor.kind != "image" {
            return Err(serde::de::Error::custom("expected image descriptor"));
        }
        Ok(Self {
            id: descriptor.id,
            label: descriptor.label,
            size: descriptor.size,
            width: descriptor.width,
            height: descriptor.height,
        })
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(CompactString),
    Image(ImageValue),
    List(Arc<[Value]>),
    Record(Arc<Record>),
    Projected(ProjectedValue),
}

impl Value {
    pub fn as_record(&self) -> Option<&Record> {
        match self {
            Self::Record(record) => Some(record.as_ref()),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Number(left), Self::Number(right)) => left == right,
            (Self::String(left), Self::String(right)) => left == right,
            (Self::Image(left), Self::Image(right)) => left == right,
            (Self::List(left), Self::List(right)) => left == right,
            (Self::Record(left), Self::Record(right)) => left == right,
            (Self::Projected(left), Self::Projected(right)) => left == right,
            (Self::Projected(left), right) => left.materialize() == *right,
            (left, Self::Projected(right)) => *left == right.materialize(),
            _ => false,
        }
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        to_json_blocking(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        serde_json::Value::deserialize(deserializer).map(from_json)
    }
}

#[derive(Clone, Default)]
pub struct ProjectedBindings {
    bindings: FxHashMap<Symbol, ProjectedValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedBindingError {
    name: String,
}

impl ProjectedBindingError {
    pub fn duplicate(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Display for ProjectedBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "projected binding `{}` is already bound", self.name)
    }
}

impl std::error::Error for ProjectedBindingError {}

impl ProjectedBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: ProjectedValue) {
        let name = name.into();
        self.try_insert(name, value)
            .expect("projected binding should not be inserted twice");
    }

    pub fn try_insert(
        &mut self,
        name: impl Into<String>,
        value: ProjectedValue,
    ) -> Result<(), ProjectedBindingError> {
        let name = name.into();
        let symbol = intern_symbol(&name);
        if self.bindings.contains_key(&symbol) {
            return Err(ProjectedBindingError::duplicate(name));
        }
        self.bindings.insert(intern_symbol(&name), value);
        Ok(())
    }

    pub(crate) fn get_symbol(&self, symbol: Symbol) -> Option<ProjectedValue> {
        self.bindings.get(&symbol).cloned()
    }

    pub fn get(&self, name: &str) -> Option<ProjectedValue> {
        self.bindings.get(&intern_symbol(name)).cloned()
    }

    pub fn names(&self) -> impl Iterator<Item = String> + '_ {
        self.bindings
            .keys()
            .map(|symbol| symbol_name(*symbol).to_string())
    }
}

#[derive(Clone)]
pub struct ProjectedValue {
    name: Arc<str>,
    kind: ProjectedKind,
}

#[derive(Clone)]
enum ProjectedKind {
    Scalar(Arc<Value>),
    Custom(Arc<dyn ProjectedHostValue>),
}

pub type ProjectedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Clone, Debug, PartialEq)]
pub enum ProjectedRead {
    Missing,
    Value(Value),
}

pub trait ProjectedHostValue: Send + Sync {
    fn type_name(&self) -> &'static str;

    fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
        Box::pin(async { None })
    }

    fn is_empty(&self) -> ProjectedFuture<'_, bool> {
        Box::pin(async { self.len().await.unwrap_or(0) == 0 })
    }

    fn get_index(&self, _index: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn get_field(&self, _field: Arc<str>) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn contains(&self, _needle: Value) -> ProjectedFuture<'_, bool> {
        Box::pin(async { false })
    }

    fn keys(&self) -> ProjectedFuture<'_, Vec<String>> {
        Box::pin(async { Vec::new() })
    }

    fn values(&self) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn starts_with(&self, _prefix: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn ends_with(&self, _suffix: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn split(&self, _needle: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn join(&self, _sep: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn trim(&self) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn slice(
        &self,
        _start: Option<isize>,
        _end: Option<isize>,
    ) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn push(&self, _item: Value) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn to_number(&self) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn json_parse(&self) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async { ProjectedRead::Missing })
    }

    fn render(&self) -> ProjectedFuture<'_, String> {
        Box::pin(async { format!("<{}>", self.type_name()) })
    }

    fn materialize(&self) -> ProjectedFuture<'_, Value>;
}

impl ProjectedValue {
    pub fn scalar(name: impl Into<Arc<str>>, value: Value) -> Self {
        Self {
            name: name.into(),
            kind: ProjectedKind::Scalar(Arc::new(value)),
        }
    }

    pub fn custom(name: impl Into<Arc<str>>, value: Arc<dyn ProjectedHostValue>) -> Self {
        Self {
            name: name.into(),
            kind: ProjectedKind::Custom(value),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Wrap a derived value as a `ProjectedValue` carrying a path-extended name
    /// (e.g. `parent.field`). Pass-through if the inner value is already a
    /// `Value::Projected` so we never double-wrap. Used by field/index access on
    /// projected sources to keep "this came from a projected source" alive
    /// across path expressions; non-path operations (binary ops, builtins,
    /// formatters) auto-strip via their existing materialise-and-evaluate code
    /// paths and so naturally lose the wrapper.
    pub fn propagate_field(parent_name: &str, field: &str, inner: Value) -> Value {
        match inner {
            Value::Projected(_) => inner,
            other => Value::Projected(ProjectedValue::scalar(
                Arc::<str>::from(format!("{parent_name}.{field}")),
                other,
            )),
        }
    }

    pub fn propagate_index(parent_name: &str, index: &Value, inner: Value) -> Value {
        match inner {
            Value::Projected(_) => inner,
            other => {
                let suffix = match index {
                    Value::String(s) => format!("[{s:?}]"),
                    Value::Number(n) => format!("[{n}]"),
                    other => format!("[{other}]"),
                };
                Value::Projected(ProjectedValue::scalar(
                    Arc::<str>::from(format!("{parent_name}{suffix}")),
                    other,
                ))
            }
        }
    }

    pub(crate) async fn len(&self) -> usize {
        match &self.kind {
            ProjectedKind::Scalar(value) => value_len(value).unwrap_or(0),
            ProjectedKind::Custom(value) => value.len().await.unwrap_or(0),
        }
    }

    pub(crate) async fn is_empty(&self) -> bool {
        match &self.kind {
            ProjectedKind::Scalar(value) => value_len(value).unwrap_or(0) == 0,
            ProjectedKind::Custom(value) => value.is_empty().await,
        }
    }

    pub(crate) async fn get_index(&self, index: &Value) -> Result<Value, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => read_index_ref_direct(value, index),
            ProjectedKind::Custom(value) => match value.get_index(index.clone()).await {
                ProjectedRead::Missing => Ok(Value::Null),
                ProjectedRead::Value(value) => Ok(value),
            },
        }
    }

    pub(crate) async fn get_field(&self, field: &Name) -> Result<Value, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => read_field_ref_direct(value, field),
            ProjectedKind::Custom(value) => match value.get_field(field.text.clone()).await {
                ProjectedRead::Missing => Ok(Value::Null),
                ProjectedRead::Value(value) => Ok(value),
            },
        }
    }

    pub(crate) async fn contains(&self, needle: &Value) -> Result<bool, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => execute_contains_direct(value, needle),
            ProjectedKind::Custom(value) => Ok(value.contains(needle.clone()).await),
        }
    }

    pub(crate) async fn keys(&self) -> Vec<String> {
        match &self.kind {
            ProjectedKind::Scalar(value) => match value.as_ref() {
                Value::Record(record) => record.keys().map(ToString::to_string).collect(),
                _ => Vec::new(),
            },
            ProjectedKind::Custom(value) => value.keys().await,
        }
    }

    pub(crate) async fn values(&self) -> Option<Value> {
        match &self.kind {
            ProjectedKind::Scalar(value) => match value.as_ref() {
                Value::Record(record) => Some(Value::List(
                    record.values().cloned().collect::<Vec<_>>().into(),
                )),
                Value::Null => Some(Value::List(Vec::new().into())),
                _ => None,
            },
            ProjectedKind::Custom(value) => match value.values().await {
                ProjectedRead::Value(value) => Some(value),
                ProjectedRead::Missing => None,
            },
        }
    }

    pub(crate) async fn starts_with(&self, prefix: Value) -> Option<Value> {
        self.custom_read_or_missing(|value| value.starts_with(prefix))
            .await
    }

    pub(crate) async fn ends_with(&self, suffix: Value) -> Option<Value> {
        self.custom_read_or_missing(|value| value.ends_with(suffix))
            .await
    }

    pub(crate) async fn split(&self, needle: Value) -> Option<Value> {
        self.custom_read_or_missing(|value| value.split(needle)).await
    }

    pub(crate) async fn join(&self, sep: Value) -> Option<Value> {
        self.custom_read_or_missing(|value| value.join(sep)).await
    }

    pub(crate) async fn trim(&self) -> Option<Value> {
        self.custom_read_or_missing(ProjectedHostValue::trim).await
    }

    pub(crate) async fn slice(&self, start: Option<isize>, end: Option<isize>) -> Option<Value> {
        self.custom_read_or_missing(|value| value.slice(start, end))
            .await
    }

    pub(crate) async fn push(&self, item: Value) -> Option<Value> {
        self.custom_read_or_missing(|value| value.push(item)).await
    }

    pub(crate) async fn to_number(&self) -> Option<Value> {
        self.custom_read_or_missing(ProjectedHostValue::to_number)
            .await
    }

    pub(crate) async fn json_parse(&self) -> Option<Value> {
        self.custom_read_or_missing(ProjectedHostValue::json_parse)
            .await
    }

    async fn custom_read_or_missing<'a>(
        &'a self,
        op: impl FnOnce(&'a dyn ProjectedHostValue) -> ProjectedFuture<'a, ProjectedRead>,
    ) -> Option<Value> {
        match &self.kind {
            ProjectedKind::Scalar(_) => None,
            ProjectedKind::Custom(value) => match op(value.as_ref()).await {
                ProjectedRead::Value(value) => Some(value),
                ProjectedRead::Missing => None,
            },
        }
    }

    pub async fn render(&self) -> String {
        match &self.kind {
            ProjectedKind::Scalar(value) => stringify_value_async(value).await.unwrap_or_default(),
            ProjectedKind::Custom(value) => value.render().await,
        }
    }

    pub async fn materialize_async(&self) -> Value {
        match &self.kind {
            ProjectedKind::Scalar(value) => (**value).clone(),
            ProjectedKind::Custom(value) => value.materialize().await,
        }
    }

    pub fn materialize(&self) -> Value {
        futures_executor::block_on(self.materialize_async())
    }
}

impl fmt::Debug for ProjectedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectedValue")
            .field("name", &self.name)
            .field("kind", &self.value_type_name())
            .finish()
    }
}

impl PartialEq for ProjectedValue {
    fn eq(&self, other: &Self) -> bool {
        self.materialize() == other.materialize()
    }
}

impl ProjectedValue {
    pub(crate) fn value_type_name(&self) -> &'static str {
        match &self.kind {
            ProjectedKind::Scalar(value) => value_type_name(value),
            ProjectedKind::Custom(value) => value.type_name(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Number(value) => write_number(f, *value),
            Self::String(value) => write!(f, "{value}"),
            Self::Image(_) | Self::List(_) | Self::Record(_) | Self::Projected(_) => {
                write!(
                    f,
                    "{}",
                    serde_json::to_string(&to_json_blocking(self)).unwrap_or_default()
                )
            }
        }
    }
}
