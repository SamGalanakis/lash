use crate::ast::{
    AssignPathStep, AssignTarget, BinaryOp, CallExpr, Expr, ParallelBranches, Program, Stmt,
    TypeExpr, UnaryOp,
};
use crate::lexer::Span;
use compact_str::CompactString;
use futures_util::{FutureExt as _, future::join_all};
use rustc_hash::FxHashMap;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use thiserror::Error;

mod cache;
mod host;
mod record;
mod schema;
mod state;

pub use cache::{CompiledProgramCache, CompiledProgramCacheStats};
pub use host::{ToolHost, ToolHostCall, ToolHostError};
pub use record::Record;
use record::{Symbol, intern_symbol, lookup_symbol, record_with_capacity, symbol_name};
use schema::{
    CompiledSchema, compile_schema_value, execute_compiled_validate, execute_validate_builtin,
};
pub use state::{Snapshot, State};

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

impl ProjectedBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: ProjectedValue) {
        let name = name.into();
        self.bindings.insert(intern_symbol(&name), value);
    }

    fn get_symbol(&self, symbol: Symbol) -> Option<ProjectedValue> {
        self.bindings.get(&symbol).cloned()
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

    async fn len(&self) -> usize {
        match &self.kind {
            ProjectedKind::Scalar(value) => value_len(value).unwrap_or(0),
            ProjectedKind::Custom(value) => value.len().await.unwrap_or(0),
        }
    }

    async fn is_empty(&self) -> bool {
        match &self.kind {
            ProjectedKind::Scalar(value) => value_len(value).unwrap_or(0) == 0,
            ProjectedKind::Custom(value) => value.is_empty().await,
        }
    }

    async fn get_index(&self, index: &Value) -> Result<Value, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => read_index_ref_direct(value, index),
            ProjectedKind::Custom(value) => match value.get_index(index.clone()).await {
                ProjectedRead::Missing => Ok(Value::Null),
                ProjectedRead::Value(value) => Ok(value),
            },
        }
    }

    async fn get_field(&self, field: &Name) -> Result<Value, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => read_field_ref_direct(value, field),
            ProjectedKind::Custom(value) => match value.get_field(field.text.clone()).await {
                ProjectedRead::Missing => Ok(Value::Null),
                ProjectedRead::Value(value) => Ok(value),
            },
        }
    }

    async fn contains(&self, needle: &Value) -> Result<bool, RuntimeError> {
        match &self.kind {
            ProjectedKind::Scalar(value) => execute_contains_direct(value, needle),
            ProjectedKind::Custom(value) => Ok(value.contains(needle.clone()).await),
        }
    }

    async fn keys(&self) -> Vec<String> {
        match &self.kind {
            ProjectedKind::Scalar(value) => match value.as_ref() {
                Value::Record(record) => record.keys().map(ToString::to_string).collect(),
                _ => Vec::new(),
            },
            ProjectedKind::Custom(value) => value.keys().await,
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

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("unknown name `{name}`")]
    UndefinedVariable { name: String },
    #[error("`for` expects a list")]
    NonListIteration,
    #[error("`finish` can't be used inside `parallel`")]
    FinishInsideParallel,
    #[error("`parallel` assigns `{name}` more than once")]
    ParallelConflict { name: String },
    #[error("unknown builtin `{name}`")]
    UnknownBuiltin { name: String },
    #[error("{message}")]
    TypeError { message: String },
    #[error("{message}")]
    ValueError { message: String },
}

#[derive(Debug, Error, PartialEq)]
#[error("{error}")]
pub struct RuntimeFailure {
    pub error: RuntimeError,
    pub span: Option<Span>,
}

#[derive(Default)]
pub struct ExecutionScratch {
    stack: Vec<Value>,
    iter_stack: Vec<IterState>,
    slot_values: Vec<Option<Value>>,
}

impl ExecutionScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn compile_source(source: &str) -> Result<CompiledProgram, crate::parser::ParseError> {
    crate::parse(source).map(|program| compile_program(&program))
}

pub fn prewarm() {
    for name in [
        "ok",
        "value",
        "error",
        "__handle__",
        "handle",
        LASH_TYPE_KEY,
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "id",
        "label",
        "size",
        "width",
        "height",
    ] {
        intern_symbol(name);
    }
}

const COOPERATIVE_YIELD_INSTRUCTION_BUDGET: usize = 1024;

pub async fn execute_program<H: ToolHost>(
    program: &Program,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let compiled = compile_program(program);
    execute_compiled(&compiled, state, host).await
}

#[derive(Clone)]
pub struct CompiledProgram {
    chunk: Chunk,
    compile_stats: CompileStats,
}

impl CompiledProgram {
    pub fn compile_stats(&self) -> &CompileStats {
        &self.compile_stats
    }
}

pub fn compile_program(program: &Program) -> CompiledProgram {
    let (chunk, compile_stats) = Compiler::compile_program(program);
    CompiledProgram {
        chunk,
        compile_stats,
    }
}

pub async fn execute_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    execute_compiled_with_projected_bindings(program, state, host, &ProjectedBindings::default())
        .await
}

pub async fn execute_compiled_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    let result = vm.run().await;
    state.globals = vm.into_globals();
    result
}

pub async fn execute_compiled_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<ExecutionOutcome, RuntimeError> {
    execute_compiled_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeError> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    let result = vm.run().await;
    state.globals = vm.recycle_into_globals(scratch);
    result
}

pub async fn execute_compiled_traced<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    execute_compiled_traced_with_projected_bindings(
        program,
        state,
        host,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_traced_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    let result = vm.run_traced().await;
    state.globals = vm.into_globals();
    result
}

pub async fn execute_compiled_traced_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    execute_compiled_traced_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn execute_compiled_traced_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<ExecutionOutcome, RuntimeFailure> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    let result = vm.run_traced().await;
    state.globals = vm.recycle_into_globals(scratch);
    result
}

pub async fn profile_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    profile_compiled_with_projected_bindings(program, state, host, &ProjectedBindings::default())
        .await
}

pub async fn profile_compiled_with_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    projected: &ProjectedBindings,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
            projected,
        ),
        host,
        false,
    );
    vm.enable_profile();
    let result = vm.run().await;
    let mut profile = vm.take_profile();
    state.globals = vm.into_globals();
    profile.compile_stats = program.compile_stats;
    result.map(|outcome| (outcome, profile))
}

pub async fn profile_compiled_with_scratch<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    profile_compiled_with_scratch_and_projected_bindings(
        program,
        state,
        host,
        scratch,
        &ProjectedBindings::default(),
    )
    .await
}

pub async fn profile_compiled_with_scratch_and_projected_bindings<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
    scratch: &mut ExecutionScratch,
    projected: &ProjectedBindings,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let slots = SlotState::from_globals_with_scratch(
        std::mem::take(&mut state.globals),
        &program.chunk.slot_names,
        scratch,
        projected,
    );
    let mut vm = Vm::new_with_scratch(&program.chunk, slots, host, false, scratch);
    vm.enable_profile();
    let result = vm.run().await;
    let mut profile = vm.take_profile();
    state.globals = vm.recycle_into_globals(scratch);
    profile.compile_stats = program.compile_stats;
    result.map(|outcome| (outcome, profile))
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionOutcome {
    Continued,
    Finished(Value),
}

#[derive(Clone, Debug, Default)]
pub struct ProfileReport {
    instruction_stats: Vec<ProfileStat>,
    builtin_stats: Vec<ProfileStat>,
    compile_stats: CompileStats,
}

impl ProfileReport {
    pub fn instruction_stats(&self) -> &[ProfileStat] {
        &self.instruction_stats
    }

    pub fn builtin_stats(&self) -> &[ProfileStat] {
        &self.builtin_stats
    }

    pub fn compile_stats(&self) -> &CompileStats {
        &self.compile_stats
    }

    pub fn merge(&mut self, other: &Self) {
        merge_stats(&mut self.instruction_stats, &other.instruction_stats);
        merge_stats(&mut self.builtin_stats, &other.builtin_stats);
        self.compile_stats.merge(&other.compile_stats);
    }
}

/// Compile-time statistics captured when a program is compiled. Independent
/// of run-time profiling — these counts reflect the shape of the compiled
/// program itself (how many Type literals it contains, how many got
/// const-folded, etc.). Runtime cost of `Type` evaluation appears in the
/// instruction profile under `build_type_ref` / `build_record` / etc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    pub type_literals_total: u64,
    pub type_literals_const_folded: u64,
    pub type_literals_dynamic: u64,
    pub type_ref_sites: u64,
}

impl CompileStats {
    pub fn merge(&mut self, other: &Self) {
        self.type_literals_total += other.type_literals_total;
        self.type_literals_const_folded += other.type_literals_const_folded;
        self.type_literals_dynamic += other.type_literals_dynamic;
        self.type_ref_sites += other.type_ref_sites;
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfileStat {
    pub name: &'static str,
    pub count: u64,
    pub total_ns: u128,
}

impl ProfileStat {
    pub fn avg_ns(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_ns / self.count as u128
        }
    }
}

#[derive(Clone)]
struct Chunk {
    code: Vec<Instruction>,
    spans: Vec<Option<Span>>,
    constants: Vec<Value>,
    names: Vec<Name>,
    slot_names: Vec<Name>,
    key_lists: Vec<Box<[usize]>>,
    format_templates: Vec<CompiledFormatTemplate>,
    compiled_schemas: Vec<CompiledSchema>,
    parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    named_parallel_call_sets: Vec<Box<[NamedParallelCallBranch]>>,
    pure_parallel_sets: Vec<Box<[PureExpr]>>,
    pure_named_parallel_sets: Vec<Box<[(usize, PureExpr)]>>,
    branch_sets: Vec<Box<[Chunk]>>,
    named_branch_sets: Vec<Box<[NamedBranchChunk]>>,
    assign_paths: Vec<CompiledAssignPath>,
}

#[derive(Clone)]
struct Name {
    symbol: Symbol,
    text: Arc<str>,
}

#[derive(Clone)]
struct CompiledFormatTemplate {
    parts: Box<[CompiledFormatPart]>,
    argc: usize,
    min_capacity: usize,
    error: Option<String>,
}

#[derive(Clone)]
enum CompiledFormatPart {
    Literal(Arc<str>),
    Arg(usize),
}

#[derive(Clone)]
struct CompiledAssignPath {
    steps: Box<[CompiledAssignPathStep]>,
    dynamic_index_count: usize,
}

#[derive(Clone, Copy)]
enum CompiledAssignPathStep {
    Field(usize),
    Index,
}

struct ResultWrapperNames {
    ok: Name,
    value: Name,
    error: Name,
}

fn transient_name(name: &str) -> Name {
    let symbol = intern_symbol(name);
    Name {
        symbol,
        text: symbol_name(symbol),
    }
}

fn result_wrapper_names() -> &'static ResultWrapperNames {
    static NAMES: OnceLock<ResultWrapperNames> = OnceLock::new();
    NAMES.get_or_init(|| ResultWrapperNames {
        ok: transient_name("ok"),
        value: transient_name("value"),
        error: transient_name("error"),
    })
}

#[derive(Clone, Copy)]
enum Instruction {
    PushConst(usize),
    PushNull,
    PushBool(bool),
    PushNumber(f64),
    LoadName(usize),
    StoreName(usize),
    StoreConst {
        slot: usize,
        constant: usize,
    },
    BuildList(usize),
    BuildRecord(usize),
    LoadField {
        slot: usize,
        field: usize,
    },
    LoadFieldUnwrap {
        slot: usize,
        field: usize,
    },
    Field(usize),
    Index,
    PathAssign {
        slot: usize,
        path: usize,
    },
    ResultUnwrap,
    Unary(UnaryOp),
    Binary(BinaryOp),
    ToBool,
    Jump(usize),
    JumpIfFalse(usize),
    JumpIfCompareFalse {
        op: BinaryOp,
        target: usize,
    },
    JumpIfSlotNumberCompareFalse {
        slot: usize,
        op: BinaryOp,
        right: f64,
        target: usize,
    },
    JumpIfSlotNumberBinaryCompareFalse {
        slot: usize,
        binary_op: BinaryOp,
        binary_right: f64,
        compare_op: BinaryOp,
        compare_right: f64,
        target: usize,
    },
    JumpIfTrue(usize),
    CallTool {
        name: usize,
        keys: usize,
    },
    CallToolUnwrap {
        name: usize,
        keys: usize,
    },
    StartCallTool {
        name: usize,
        keys: usize,
    },
    AwaitHandle,
    AwaitHandleUnwrap,
    CancelHandle,
    CallBuiltin {
        builtin: Builtin,
        argc: usize,
    },
    Len,
    Join,
    Validate,
    ValidateCompiled(usize),
    Push,
    Range {
        argc: usize,
    },
    FormatCompiled(usize),
    AddAssign(usize),
    AddAssignNumber {
        slot: usize,
        right: f64,
    },
    AddAssignIndexNumber {
        slot: usize,
        right: f64,
    },
    AppendAssign(usize),
    Print,
    Submit,
    Pop,
    BeginIter(usize),
    BeginRangeIter {
        binding: usize,
        argc: usize,
    },
    IterNext {
        jump_to: usize,
    },
    EndIter,
    ParallelCalls(usize),
    ParallelCallsValue(usize),
    ParallelNamedCallsValue(usize),
    PureParallelValue(usize),
    PureParallelNamedValue(usize),
    Parallel(usize),
    ParallelValue(usize),
    ParallelNamed(usize),
    ParallelNamedValue(usize),
    ResolveTypeRef(usize),
    WrapTypeLiteral,
}

#[derive(Clone, Copy)]
enum Builtin {
    Len,
    Empty,
    Keys,
    Values,
    Contains,
    StartsWith,
    EndsWith,
    Split,
    Join,
    Trim,
    Slice,
    ToString,
    ToInt,
    ToFloat,
    JsonParse,
    Format,
    Validate,
    Range,
    Push,
    Unknown(usize),
}

impl Instruction {
    fn profile_tag(self) -> InstructionProfileTag {
        match self {
            Instruction::PushConst(_)
            | Instruction::PushNull
            | Instruction::PushBool(_)
            | Instruction::PushNumber(_) => InstructionProfileTag::PushConst,
            Instruction::LoadName(_) => InstructionProfileTag::LoadName,
            Instruction::StoreName(_)
            | Instruction::StoreConst { .. }
            | Instruction::PathAssign { .. } => InstructionProfileTag::StoreName,
            Instruction::BuildList(_) => InstructionProfileTag::BuildList,
            Instruction::BuildRecord(_) => InstructionProfileTag::BuildRecord,
            Instruction::LoadField { .. } | Instruction::Field(_) => InstructionProfileTag::Field,
            Instruction::Index => InstructionProfileTag::Index,
            Instruction::ResultUnwrap | Instruction::LoadFieldUnwrap { .. } => {
                InstructionProfileTag::ResultUnwrap
            }
            Instruction::Unary(_) => InstructionProfileTag::Unary,
            Instruction::Binary(_) => InstructionProfileTag::Binary,
            Instruction::ToBool => InstructionProfileTag::ToBool,
            Instruction::Jump(_) => InstructionProfileTag::Jump,
            Instruction::JumpIfFalse(_)
            | Instruction::JumpIfCompareFalse { .. }
            | Instruction::JumpIfSlotNumberCompareFalse { .. }
            | Instruction::JumpIfSlotNumberBinaryCompareFalse { .. } => {
                InstructionProfileTag::JumpIfFalse
            }
            Instruction::JumpIfTrue(_) => InstructionProfileTag::JumpIfTrue,
            Instruction::CallTool { .. } | Instruction::CallToolUnwrap { .. } => {
                InstructionProfileTag::CallTool
            }
            Instruction::StartCallTool { .. } => InstructionProfileTag::StartCallTool,
            Instruction::AwaitHandle | Instruction::AwaitHandleUnwrap => {
                InstructionProfileTag::AwaitHandle
            }
            Instruction::CancelHandle => InstructionProfileTag::CancelHandle,
            Instruction::CallBuiltin { .. }
            | Instruction::Len
            | Instruction::Join
            | Instruction::Validate
            | Instruction::ValidateCompiled(_)
            | Instruction::Push
            | Instruction::Range { .. }
            | Instruction::FormatCompiled(_) => InstructionProfileTag::CallBuiltin,
            Instruction::AddAssign(_)
            | Instruction::AddAssignNumber { .. }
            | Instruction::AddAssignIndexNumber { .. } => InstructionProfileTag::AddAssign,
            Instruction::AppendAssign(_) => InstructionProfileTag::AppendAssign,
            Instruction::Print => InstructionProfileTag::Print,
            Instruction::Submit => InstructionProfileTag::Submit,
            Instruction::Pop => InstructionProfileTag::Pop,
            Instruction::BeginIter(_) | Instruction::BeginRangeIter { .. } => {
                InstructionProfileTag::BeginIter
            }
            Instruction::IterNext { .. } => InstructionProfileTag::IterNext,
            Instruction::EndIter => InstructionProfileTag::EndIter,
            Instruction::ParallelCalls(_) => InstructionProfileTag::Parallel,
            Instruction::ParallelCallsValue(_) => InstructionProfileTag::Parallel,
            Instruction::ParallelNamedCallsValue(_) => InstructionProfileTag::Parallel,
            Instruction::PureParallelValue(_) => InstructionProfileTag::Parallel,
            Instruction::PureParallelNamedValue(_) => InstructionProfileTag::Parallel,
            Instruction::Parallel(_) => InstructionProfileTag::Parallel,
            Instruction::ParallelValue(_) => InstructionProfileTag::Parallel,
            Instruction::ParallelNamed(_) => InstructionProfileTag::Parallel,
            Instruction::ParallelNamedValue(_) => InstructionProfileTag::Parallel,
            Instruction::ResolveTypeRef(_) => InstructionProfileTag::ResolveTypeRef,
            Instruction::WrapTypeLiteral => InstructionProfileTag::WrapTypeLiteral,
        }
    }
}

impl Builtin {
    fn profile_tag(self) -> BuiltinProfileTag {
        match self {
            Builtin::Len => BuiltinProfileTag::Len,
            Builtin::Empty => BuiltinProfileTag::Empty,
            Builtin::Keys => BuiltinProfileTag::Keys,
            Builtin::Values => BuiltinProfileTag::Values,
            Builtin::Contains => BuiltinProfileTag::Contains,
            Builtin::StartsWith => BuiltinProfileTag::StartsWith,
            Builtin::EndsWith => BuiltinProfileTag::EndsWith,
            Builtin::Split => BuiltinProfileTag::Split,
            Builtin::Join => BuiltinProfileTag::Join,
            Builtin::Trim => BuiltinProfileTag::Trim,
            Builtin::Slice => BuiltinProfileTag::Slice,
            Builtin::ToString => BuiltinProfileTag::ToString,
            Builtin::ToInt => BuiltinProfileTag::ToInt,
            Builtin::ToFloat => BuiltinProfileTag::ToFloat,
            Builtin::JsonParse => BuiltinProfileTag::JsonParse,
            Builtin::Format => BuiltinProfileTag::Format,
            Builtin::Validate => BuiltinProfileTag::Validate,
            Builtin::Range => BuiltinProfileTag::Range,
            Builtin::Push => BuiltinProfileTag::Push,
            Builtin::Unknown(_) => BuiltinProfileTag::Unknown,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum InstructionProfileTag {
    PushConst,
    LoadName,
    StoreName,
    BuildList,
    BuildRecord,
    Field,
    Index,
    ResultUnwrap,
    Unary,
    Binary,
    ToBool,
    Jump,
    JumpIfFalse,
    JumpIfTrue,
    CallTool,
    StartCallTool,
    AwaitHandle,
    CancelHandle,
    CallBuiltin,
    AddAssign,
    AppendAssign,
    Print,
    Submit,
    Pop,
    BeginIter,
    IterNext,
    EndIter,
    Parallel,
    ResolveTypeRef,
    WrapTypeLiteral,
}

const INSTRUCTION_PROFILE_COUNT: usize = InstructionProfileTag::WrapTypeLiteral as usize + 1;

#[derive(Clone, Copy)]
#[repr(usize)]
enum BuiltinProfileTag {
    Len,
    Empty,
    Keys,
    Values,
    Contains,
    StartsWith,
    EndsWith,
    Split,
    Join,
    Trim,
    Slice,
    ToString,
    ToInt,
    ToFloat,
    JsonParse,
    Format,
    Validate,
    Range,
    Push,
    Unknown,
}

const BUILTIN_PROFILE_COUNT: usize = BuiltinProfileTag::Unknown as usize + 1;

#[derive(Default)]
struct ProfileAccumulator {
    instruction_counts: [u64; INSTRUCTION_PROFILE_COUNT],
    instruction_times: [u128; INSTRUCTION_PROFILE_COUNT],
    builtin_counts: [u64; BUILTIN_PROFILE_COUNT],
    builtin_times: [u128; BUILTIN_PROFILE_COUNT],
}

impl ProfileAccumulator {
    fn finish(self) -> ProfileReport {
        ProfileReport {
            instruction_stats: build_stats(
                &INSTRUCTION_PROFILE_NAMES,
                &self.instruction_counts,
                &self.instruction_times,
            ),
            builtin_stats: build_stats(
                &BUILTIN_PROFILE_NAMES,
                &self.builtin_counts,
                &self.builtin_times,
            ),
            compile_stats: CompileStats::default(),
        }
    }
}

const INSTRUCTION_PROFILE_NAMES: [&str; INSTRUCTION_PROFILE_COUNT] = [
    "push_const",
    "load_name",
    "store_name",
    "build_list",
    "build_record",
    "field",
    "index",
    "result_unwrap",
    "unary",
    "binary",
    "to_bool",
    "jump",
    "jump_if_false",
    "jump_if_true",
    "call_tool",
    "start_call_tool",
    "await_handle",
    "cancel_handle",
    "call_builtin",
    "add_assign",
    "append_assign",
    "print",
    "submit",
    "pop",
    "begin_iter",
    "iter_next",
    "end_iter",
    "parallel",
    "resolve_type_ref",
    "wrap_type_literal",
];

const BUILTIN_PROFILE_NAMES: [&str; BUILTIN_PROFILE_COUNT] = [
    "len",
    "empty",
    "keys",
    "values",
    "contains",
    "starts_with",
    "ends_with",
    "split",
    "join",
    "trim",
    "slice",
    "to_string",
    "to_int",
    "to_float",
    "json_parse",
    "format",
    "validate",
    "range",
    "push",
    "unknown",
];

fn build_stats<const N: usize>(
    names: &[&'static str; N],
    counts: &[u64; N],
    times: &[u128; N],
) -> Vec<ProfileStat> {
    let mut stats = names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            let count = counts[index];
            (count > 0).then_some(ProfileStat {
                name,
                count,
                total_ns: times[index],
            })
        })
        .collect::<Vec<_>>();
    stats.sort_by(|a, b| {
        b.total_ns
            .cmp(&a.total_ns)
            .then_with(|| b.count.cmp(&a.count))
    });
    stats
}

fn merge_stats(target: &mut Vec<ProfileStat>, source: &[ProfileStat]) {
    for stat in source {
        if let Some(existing) = target.iter_mut().find(|entry| entry.name == stat.name) {
            existing.count += stat.count;
            existing.total_ns += stat.total_ns;
        } else {
            target.push(stat.clone());
        }
    }
    target.sort_by(|a, b| {
        b.total_ns
            .cmp(&a.total_ns)
            .then_with(|| b.count.cmp(&a.count))
    });
}

struct Compiler {
    code: Vec<Instruction>,
    spans: Vec<Option<Span>>,
    constants: Vec<Value>,
    names: Vec<Name>,
    name_lookup: FxHashMap<Symbol, usize>,
    slots: Rc<RefCell<SlotTable>>,
    key_lists: Vec<Box<[usize]>>,
    format_templates: Vec<CompiledFormatTemplate>,
    compiled_schemas: Vec<CompiledSchema>,
    parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    named_parallel_call_sets: Vec<Box<[NamedParallelCallBranch]>>,
    pure_parallel_sets: Vec<Box<[PureExpr]>>,
    pure_named_parallel_sets: Vec<Box<[(usize, PureExpr)]>>,
    branch_sets: Vec<Box<[Chunk]>>,
    named_branch_sets: Vec<Box<[NamedBranchChunk]>>,
    assign_paths: Vec<CompiledAssignPath>,
    compile_stats: Rc<RefCell<CompileStats>>,
    const_slots: Vec<Option<Value>>,
    loop_contexts: Vec<LoopContext>,
}

struct LoopContext {
    continue_target: usize,
    break_jumps: SmallVec<[usize; 4]>,
}

#[derive(Default)]
struct SlotTable {
    names: Vec<Name>,
    lookup: FxHashMap<Symbol, usize>,
}

#[derive(Clone)]
struct ParallelCallBranch {
    slot: usize,
    name: usize,
    args: PureExpr,
}

#[derive(Clone)]
struct NamedParallelCallBranch {
    output_name: usize,
    name: usize,
    args: PureExpr,
}

#[derive(Clone)]
struct NamedBranchChunk {
    name: usize,
    chunk: Chunk,
}

#[derive(Clone)]
enum PureExpr {
    Const(Value),
    Slot(usize),
    List(Box<[PureExpr]>),
    Record(Box<[(usize, PureExpr)]>),
    Builtin {
        builtin: Builtin,
        args: Box<[PureExpr]>,
    },
    Format {
        template: CompiledFormatTemplate,
        args: Box<[PureExpr]>,
    },
    ResultUnwrap(Box<PureExpr>),
    Field {
        target: Box<PureExpr>,
        field: usize,
    },
    Index {
        target: Box<PureExpr>,
        index: Box<PureExpr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<PureExpr>,
    },
    Conditional {
        condition: Box<PureExpr>,
        then_expr: Box<PureExpr>,
        else_expr: Box<PureExpr>,
    },
    Binary {
        left: Box<PureExpr>,
        op: BinaryOp,
        right: Box<PureExpr>,
    },
}

impl Compiler {
    fn compile_program(program: &Program) -> (Chunk, CompileStats) {
        let stats = Rc::new(RefCell::new(CompileStats::default()));
        let mut compiler =
            Self::with_slots_and_stats(Rc::new(RefCell::new(SlotTable::default())), stats.clone());
        compiler.compile_program_block(program);
        let chunk = compiler.finish();
        let compile_stats = *stats.borrow();
        (chunk, compile_stats)
    }

    fn with_slots_and_stats(
        slots: Rc<RefCell<SlotTable>>,
        compile_stats: Rc<RefCell<CompileStats>>,
    ) -> Self {
        Self {
            code: Vec::new(),
            spans: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            name_lookup: FxHashMap::default(),
            slots,
            key_lists: Vec::new(),
            format_templates: Vec::new(),
            compiled_schemas: Vec::new(),
            parallel_call_sets: Vec::new(),
            named_parallel_call_sets: Vec::new(),
            pure_parallel_sets: Vec::new(),
            pure_named_parallel_sets: Vec::new(),
            branch_sets: Vec::new(),
            named_branch_sets: Vec::new(),
            assign_paths: Vec::new(),
            compile_stats,
            const_slots: Vec::new(),
            loop_contexts: Vec::new(),
        }
    }

    fn finish(self) -> Chunk {
        let slot_names = self.slots.borrow().names.clone();
        let mut spans = self.spans;
        spans.resize(self.code.len(), None);
        Chunk {
            code: self.code,
            spans,
            constants: self.constants,
            names: self.names,
            slot_names,
            key_lists: self.key_lists,
            format_templates: self.format_templates,
            compiled_schemas: self.compiled_schemas,
            parallel_call_sets: self.parallel_call_sets,
            named_parallel_call_sets: self.named_parallel_call_sets,
            pure_parallel_sets: self.pure_parallel_sets,
            pure_named_parallel_sets: self.pure_named_parallel_sets,
            branch_sets: self.branch_sets,
            named_branch_sets: self.named_branch_sets,
            assign_paths: self.assign_paths,
        }
    }

    fn push_const(&mut self, value: Value) -> usize {
        let index = self.constants.len();
        self.constants.push(value);
        index
    }

    fn emit_push_value(&mut self, value: Value) {
        match value {
            Value::Null => self.code.push(Instruction::PushNull),
            Value::Bool(value) => self.code.push(Instruction::PushBool(value)),
            Value::Number(value) => self.code.push(Instruction::PushNumber(value)),
            value => {
                let index = self.push_const(value);
                self.code.push(Instruction::PushConst(index));
            }
        }
    }

    fn push_name(&mut self, name: &str) -> usize {
        let symbol = intern_symbol(name);
        if let Some(index) = self.name_lookup.get(&symbol) {
            return *index;
        }

        let index = self.names.len();
        self.names.push(Name {
            symbol,
            text: symbol_name(symbol),
        });
        self.name_lookup.insert(symbol, index);
        index
    }

    fn push_slot(&mut self, name: &str) -> usize {
        let symbol = intern_symbol(name);
        let mut slots = self.slots.borrow_mut();
        if let Some(index) = slots.lookup.get(&symbol) {
            let index = *index;
            drop(slots);
            self.ensure_const_slot(index);
            return index;
        }
        let index = slots.names.len();
        slots.names.push(Name {
            symbol,
            text: symbol_name(symbol),
        });
        slots.lookup.insert(symbol, index);
        drop(slots);
        self.ensure_const_slot(index);
        index
    }

    fn push_key_list<'a>(&mut self, keys: impl Iterator<Item = &'a str>) -> usize {
        let index = self.key_lists.len();
        let keys = keys
            .map(|key| self.push_name(key))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.key_lists.push(keys);
        index
    }

    fn push_assign_path(&mut self, steps: &[AssignPathStep]) -> usize {
        let index = self.assign_paths.len();
        let mut dynamic_index_count = 0;
        let steps = steps
            .iter()
            .map(|step| match step {
                AssignPathStep::Field(field) => {
                    CompiledAssignPathStep::Field(self.push_name(field))
                }
                AssignPathStep::Index(_) => {
                    dynamic_index_count += 1;
                    CompiledAssignPathStep::Index
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.assign_paths.push(CompiledAssignPath {
            steps,
            dynamic_index_count,
        });
        index
    }

    fn push_format_template(&mut self, template: &str, argc: usize) -> usize {
        let index = self.format_templates.len();
        self.format_templates
            .push(compile_format_template(template, argc));
        index
    }

    fn push_compiled_schema(&mut self, schema: &Value) -> usize {
        let index = self.compiled_schemas.len();
        self.compiled_schemas.push(compile_schema_value(schema));
        index
    }

    fn push_branch_set(&mut self, branches: Vec<Chunk>) -> usize {
        let index = self.branch_sets.len();
        self.branch_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_named_branch_set(&mut self, branches: Vec<NamedBranchChunk>) -> usize {
        let index = self.named_branch_sets.len();
        self.named_branch_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_parallel_call_set(&mut self, branches: Vec<ParallelCallBranch>) -> usize {
        let index = self.parallel_call_sets.len();
        self.parallel_call_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_named_parallel_call_set(&mut self, branches: Vec<NamedParallelCallBranch>) -> usize {
        let index = self.named_parallel_call_sets.len();
        self.named_parallel_call_sets
            .push(branches.into_boxed_slice());
        index
    }

    fn push_pure_parallel_set(&mut self, branches: Vec<PureExpr>) -> usize {
        let index = self.pure_parallel_sets.len();
        self.pure_parallel_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_pure_named_parallel_set(&mut self, branches: Vec<(usize, PureExpr)>) -> usize {
        let index = self.pure_named_parallel_sets.len();
        self.pure_named_parallel_sets
            .push(branches.into_boxed_slice());
        index
    }

    fn ensure_const_slot(&mut self, slot: usize) {
        if self.const_slots.len() <= slot {
            self.const_slots.resize(slot + 1, None);
        }
    }

    fn set_const_slot(&mut self, slot: usize, value: Option<Value>) {
        self.ensure_const_slot(slot);
        self.const_slots[slot] = value;
    }

    fn clear_const_slots(&mut self) {
        self.const_slots.fill(None);
    }

    fn const_for_slot(&self, slot: usize) -> Option<Value> {
        self.const_slots.get(slot).cloned().flatten()
    }

    fn const_for_name(&self, name: &str) -> Option<Value> {
        let symbol = lookup_symbol(name)?;
        let slots = self.slots.borrow();
        let slot = *slots.lookup.get(&symbol)?;
        drop(slots);
        self.const_for_slot(slot)
    }

    fn resolve_builtin(&mut self, name: &str) -> Builtin {
        match name {
            "len" => Builtin::Len,
            "empty" => Builtin::Empty,
            "keys" => Builtin::Keys,
            "values" => Builtin::Values,
            "contains" => Builtin::Contains,
            "starts_with" => Builtin::StartsWith,
            "ends_with" => Builtin::EndsWith,
            "split" => Builtin::Split,
            "join" => Builtin::Join,
            "trim" => Builtin::Trim,
            "slice" => Builtin::Slice,
            "to_string" => Builtin::ToString,
            "to_int" => Builtin::ToInt,
            "to_float" => Builtin::ToFloat,
            "json_parse" => Builtin::JsonParse,
            "format" => Builtin::Format,
            "validate" => Builtin::Validate,
            "range" => Builtin::Range,
            "push" => Builtin::Push,
            _ => Builtin::Unknown(self.push_name(name)),
        }
    }

    fn compile_program_block(&mut self, program: &Program) {
        for (index, statement) in program.statements.iter().enumerate() {
            let span = program.statement_spans.get(index).copied();
            self.compile_stmt_with_span(statement, span);
        }
    }

    fn compile_block(&mut self, statements: &[Stmt]) {
        for statement in statements {
            self.compile_stmt_with_span(statement, None);
        }
    }

    fn compile_stmt_with_span(&mut self, statement: &Stmt, span: Option<Span>) {
        let start = self.code.len();
        self.compile_stmt(statement);
        if self.spans.len() < self.code.len() {
            self.spans.resize(self.code.len(), None);
        }
        for entry in &mut self.spans[start..self.code.len()] {
            *entry = span;
        }
    }

    fn compile_stmt(&mut self, statement: &Stmt) {
        match statement {
            Stmt::Assign { target, expr } if target.is_simple() => {
                let name = &target.root;
                let slot = self.push_slot(name);
                let has_type_literal = contains_type_literal(expr);
                let const_value = if let Expr::TypeLiteral(ty) = expr {
                    fold_type(ty).map(wrap_type_schema_value)
                } else if has_type_literal {
                    None
                } else {
                    self.fold_compile_time_expr(expr)
                };
                if let Expr::Binary {
                    left,
                    op: BinaryOp::Add,
                    right,
                } = expr
                    && matches!(left.as_ref(), Expr::Variable(var) if var == name)
                {
                    if let Expr::List(items) = right.as_ref()
                        && items.len() == 1
                    {
                        self.compile_expr(&items[0]);
                        self.code.push(Instruction::AppendAssign(slot));
                        self.set_const_slot(slot, None);
                        return;
                    }
                    if let Some(Value::Number(right)) = self.fold_compile_time_expr(right) {
                        self.code.push(Instruction::AddAssignNumber { slot, right });
                        self.set_const_slot(slot, None);
                        return;
                    }
                    self.compile_expr(right);
                    self.code.push(Instruction::AddAssign(slot));
                    self.set_const_slot(slot, None);
                    return;
                }

                if let Some(value) = const_value.clone()
                    && !has_type_literal
                {
                    let constant = self.push_const(value);
                    self.code.push(Instruction::StoreConst { slot, constant });
                    self.set_const_slot(slot, const_value);
                    return;
                }

                self.compile_expr(expr);
                self.code.push(Instruction::StoreName(slot));
                self.set_const_slot(slot, const_value);
            }
            Stmt::Assign { target, expr } => {
                let slot = self.push_slot(&target.root);
                if let [AssignPathStep::Index(index)] = target.steps.as_slice()
                    && is_pure_expr(index)
                    && let Expr::Binary {
                        left,
                        op: BinaryOp::Add,
                        right,
                    } = expr
                    && let Expr::Index {
                        target: left_target,
                        index: left_index,
                    } = left.as_ref()
                    && matches!(left_target.as_ref(), Expr::Variable(name) if name == target.root)
                    && left_index.as_ref() == index
                    && let Some(Value::Number(right)) = self.fold_compile_time_expr(right)
                {
                    self.compile_expr(index);
                    self.code
                        .push(Instruction::AddAssignIndexNumber { slot, right });
                    self.set_const_slot(slot, None);
                    return;
                }
                for step in &target.steps {
                    if let AssignPathStep::Index(index) = step {
                        self.compile_expr(index);
                    }
                }
                self.compile_expr(expr);
                let path = self.push_assign_path(&target.steps);
                self.code.push(Instruction::PathAssign { slot, path });
                self.set_const_slot(slot, None);
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Pop);
            }
            Stmt::Call(call) => {
                self.compile_call_expr(call);
                self.code.push(Instruction::Pop);
            }
            Stmt::Cancel(handle) => {
                self.compile_expr(handle);
                self.code.push(Instruction::CancelHandle);
            }
            Stmt::Print(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Print);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                let jump_to_else = self.compile_condition_jump_if_false(condition);
                self.compile_block(then_block);
                if else_block.is_empty() {
                    self.patch_jump(jump_to_else, self.code.len());
                } else {
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_else, self.code.len());
                    self.compile_block(else_block);
                    self.patch_jump(jump_to_end, self.code.len());
                }
                self.clear_const_slots();
            }
            Stmt::For {
                binding,
                iterable,
                body,
            } => {
                let binding = self.push_slot(binding);
                if let Expr::BuiltinCall { name, args } = iterable
                    && name.as_str() == "range"
                {
                    for arg in args {
                        self.compile_expr(arg);
                    }
                    self.clear_const_slots();
                    self.set_const_slot(binding, None);
                    self.code.push(Instruction::BeginRangeIter {
                        binding,
                        argc: args.len(),
                    });
                    self.compile_for_loop_body(body);
                    return;
                }

                self.compile_expr(iterable);
                self.clear_const_slots();
                self.set_const_slot(binding, None);
                self.code.push(Instruction::BeginIter(binding));
                self.compile_for_loop_body(body);
            }
            Stmt::Break => {
                let jump = self.emit_jump();
                self.loop_contexts
                    .last_mut()
                    .expect("parser rejects `break` outside loops")
                    .break_jumps
                    .push(jump);
                self.clear_const_slots();
            }
            Stmt::Continue => {
                let continue_target = self
                    .loop_contexts
                    .last()
                    .expect("parser rejects `continue` outside loops")
                    .continue_target;
                self.code.push(Instruction::Jump(continue_target));
                self.clear_const_slots();
            }
            Stmt::Parallel { branches } => {
                self.compile_parallel(branches, false);
                self.clear_const_slots();
            }
            Stmt::Submit(expr) => {
                if let Some(expr) = expr {
                    self.compile_expr(expr);
                } else {
                    self.compile_expr(&Expr::Null);
                }
                self.code.push(Instruction::Submit);
            }
        }
    }

    fn compile_for_loop_body(&mut self, body: &[Stmt]) {
        let loop_start = self.code.len();
        let iter_next = self.code.len();
        self.code.push(Instruction::IterNext {
            jump_to: usize::MAX,
        });
        self.loop_contexts.push(LoopContext {
            continue_target: loop_start,
            break_jumps: SmallVec::new(),
        });
        self.compile_block(body);
        let loop_context = self
            .loop_contexts
            .pop()
            .expect("loop context should exist while compiling `for`");
        self.code.push(Instruction::Jump(loop_start));
        let loop_end = self.code.len();
        self.code.push(Instruction::EndIter);
        self.patch_jump(iter_next, loop_end);
        for break_jump in loop_context.break_jumps {
            self.patch_jump(break_jump, loop_end);
        }
        self.clear_const_slots();
    }

    fn compile_parallel_calls(&mut self, branches: &[Stmt]) -> Option<Vec<ParallelCallBranch>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Assign { target, expr } = branch else {
                return None;
            };
            if !target.is_simple() {
                return None;
            }
            let Expr::ToolCall(call) = expr else {
                return None;
            };
            if call.args.iter().any(|(_, expr)| !is_pure_expr(expr)) {
                return None;
            }

            let slot = self.push_slot(&target.root);
            let args = PureExpr::Record(
                call.args
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()
                    .ok()?
                    .into_boxed_slice(),
            );
            let name = self.push_name(&call.name);
            compiled.push(ParallelCallBranch { slot, name, args });
        }
        Some(compiled)
    }

    fn compile_pure_parallel_exprs(&mut self, branches: &[Stmt]) -> Option<Vec<PureExpr>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Expr(expr) = branch else {
                return None;
            };
            compiled.push(self.compile_pure_expr(expr).ok()?);
        }
        Some(compiled)
    }

    fn compile_named_parallel_calls(
        &mut self,
        branches: &[crate::ast::NamedParallelBranch],
    ) -> Option<Vec<NamedParallelCallBranch>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let call = match &branch.stmt {
                Stmt::Call(call) | Stmt::Expr(Expr::ToolCall(call)) => call,
                _ => return None,
            };
            if call.args.iter().any(|(_, expr)| !is_pure_expr(expr)) {
                return None;
            }
            let args = PureExpr::Record(
                call.args
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()
                    .ok()?
                    .into_boxed_slice(),
            );
            compiled.push(NamedParallelCallBranch {
                output_name: self.push_name(&branch.name),
                name: self.push_name(&call.name),
                args,
            });
        }
        Some(compiled)
    }

    fn compile_pure_named_parallel_exprs(
        &mut self,
        branches: &[crate::ast::NamedParallelBranch],
    ) -> Option<Vec<(usize, PureExpr)>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Expr(expr) = &branch.stmt else {
                return None;
            };
            let expr = self.compile_pure_expr(expr).ok()?;
            compiled.push((self.push_name(&branch.name), expr));
        }
        Some(compiled)
    }

    fn compile_pure_expr(&mut self, expr: &Expr) -> Result<PureExpr, RuntimeError> {
        match expr {
            Expr::Null => Ok(PureExpr::Const(Value::Null)),
            Expr::Bool(value) => Ok(PureExpr::Const(Value::Bool(*value))),
            Expr::Number(value) => Ok(PureExpr::Const(Value::Number(*value))),
            Expr::String(value) => Ok(PureExpr::Const(Value::String(value.clone()))),
            Expr::Variable(name) => Ok(PureExpr::Slot(self.push_slot(name))),
            Expr::List(items) => Ok(PureExpr::List(
                items
                    .iter()
                    .map(|item| self.compile_pure_expr(item))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            )),
            Expr::Record(entries) => Ok(PureExpr::Record(
                entries
                    .iter()
                    .map(|(key, expr)| Ok((self.push_name(key), self.compile_pure_expr(expr)?)))
                    .collect::<Result<Vec<_>, RuntimeError>>()?
                    .into_boxed_slice(),
            )),
            Expr::ToolCall(_) => Err(RuntimeError::ValueError {
                message: "tool calls are not allowed in pure expressions".to_string(),
            }),
            Expr::StartToolCall(_) => Err(RuntimeError::ValueError {
                message: "async tool starts are not allowed in pure expressions".to_string(),
            }),
            Expr::Parallel { .. } => Err(RuntimeError::ValueError {
                message: "`parallel` is not allowed in pure expressions".to_string(),
            }),
            Expr::Await(_) => Err(RuntimeError::ValueError {
                message: "`await` is not allowed in pure expressions".to_string(),
            }),
            Expr::ResultUnwrap(expr) => Ok(PureExpr::ResultUnwrap(Box::new(
                self.compile_pure_expr(expr)?,
            ))),
            Expr::BuiltinCall { name, args } => {
                if name == "format"
                    && let Some((Expr::String(template), value_args)) = args.split_first()
                {
                    return Ok(PureExpr::Format {
                        template: compile_format_template(template, value_args.len()),
                        args: value_args
                            .iter()
                            .map(|arg| self.compile_pure_expr(arg))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_boxed_slice(),
                    });
                }

                Ok(PureExpr::Builtin {
                    builtin: self.resolve_builtin(name),
                    args: args
                        .iter()
                        .map(|arg| self.compile_pure_expr(arg))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                })
            }
            Expr::Field { target, field } => Ok(PureExpr::Field {
                target: Box::new(self.compile_pure_expr(target)?),
                field: self.push_name(field),
            }),
            Expr::Index { target, index } => Ok(PureExpr::Index {
                target: Box::new(self.compile_pure_expr(target)?),
                index: Box::new(self.compile_pure_expr(index)?),
            }),
            Expr::Unary { op, expr } => Ok(PureExpr::Unary {
                op: *op,
                expr: Box::new(self.compile_pure_expr(expr)?),
            }),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => Ok(PureExpr::Conditional {
                condition: Box::new(self.compile_pure_expr(condition)?),
                then_expr: Box::new(self.compile_pure_expr(then_expr)?),
                else_expr: Box::new(self.compile_pure_expr(else_expr)?),
            }),
            Expr::Binary { left, op, right } => Ok(PureExpr::Binary {
                left: Box::new(self.compile_pure_expr(left)?),
                op: *op,
                right: Box::new(self.compile_pure_expr(right)?),
            }),
            Expr::TypeLiteral(ty) => {
                let schema = fold_type(ty).ok_or_else(|| RuntimeError::ValueError {
                    message: "Type literals with `Ref` are not allowed in pure expressions"
                        .to_string(),
                })?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                Ok(PureExpr::Const(Value::Record(Arc::new(wrapper))))
            }
        }
    }

    fn fold_compile_time_expr(&self, expr: &Expr) -> Option<Value> {
        match expr {
            Expr::Null => Some(Value::Null),
            Expr::Bool(value) => Some(Value::Bool(*value)),
            Expr::Number(value) => Some(Value::Number(*value)),
            Expr::String(value) => Some(Value::String(value.clone())),
            Expr::Variable(name) => self.const_for_name(name),
            Expr::List(items) => Some(Value::List(
                items
                    .iter()
                    .map(|item| self.fold_compile_time_expr(item))
                    .collect::<Option<Vec<_>>>()?
                    .into(),
            )),
            Expr::Record(entries) => {
                let mut record = record_with_capacity(entries.len());
                for (key, value) in entries {
                    record.insert(key.to_string(), self.fold_compile_time_expr(value)?);
                }
                Some(Value::Record(Arc::new(record)))
            }
            Expr::BuiltinCall { name, args } => {
                let values = args
                    .iter()
                    .map(|arg| self.fold_compile_time_expr(arg))
                    .collect::<Option<Vec<_>>>()?;
                let builtin = match name.as_str() {
                    "len" => Builtin::Len,
                    "empty" => Builtin::Empty,
                    "keys" => Builtin::Keys,
                    "values" => Builtin::Values,
                    "contains" => Builtin::Contains,
                    "starts_with" => Builtin::StartsWith,
                    "ends_with" => Builtin::EndsWith,
                    "split" => Builtin::Split,
                    "join" => Builtin::Join,
                    "trim" => Builtin::Trim,
                    "slice" => Builtin::Slice,
                    "to_string" => Builtin::ToString,
                    "to_int" => Builtin::ToInt,
                    "to_float" => Builtin::ToFloat,
                    "json_parse" => Builtin::JsonParse,
                    "format" => Builtin::Format,
                    "validate" => Builtin::Validate,
                    "range" => Builtin::Range,
                    "push" => Builtin::Push,
                    _ => return None,
                };
                match builtin {
                    Builtin::Range => execute_range_builtin(&values).ok(),
                    _ => {
                        let _ = values;
                        None
                    }
                }
            }
            Expr::Field { target, field } => {
                let target = self.fold_compile_time_expr(target)?;
                read_field_direct(target, &transient_name(field)).ok()
            }
            Expr::Index { target, index } => {
                let target = self.fold_compile_time_expr(target)?;
                let index = self.fold_compile_time_expr(index)?;
                read_index_direct(target, index).ok()
            }
            Expr::Unary { op, expr } => {
                let value = self.fold_compile_time_expr(expr)?;
                match op {
                    UnaryOp::Negate => Some(Value::Number(-as_number(&value).ok()?)),
                    UnaryOp::Not => Some(Value::Bool(!is_truthy(&value))),
                }
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                if is_truthy(&self.fold_compile_time_expr(condition)?) {
                    self.fold_compile_time_expr(then_expr)
                } else {
                    self.fold_compile_time_expr(else_expr)
                }
            }
            Expr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    let left = self.fold_compile_time_expr(left)?;
                    if !is_truthy(&left) {
                        Some(Value::Bool(false))
                    } else {
                        Some(Value::Bool(is_truthy(&self.fold_compile_time_expr(right)?)))
                    }
                }
                BinaryOp::Or => {
                    let left = self.fold_compile_time_expr(left)?;
                    if is_truthy(&left) {
                        Some(Value::Bool(true))
                    } else {
                        Some(Value::Bool(is_truthy(&self.fold_compile_time_expr(right)?)))
                    }
                }
                _ => {
                    let left = self.fold_compile_time_expr(left)?;
                    let right = self.fold_compile_time_expr(right)?;
                    eval_binary_values(left, *op, right).ok()
                }
            },
            Expr::TypeLiteral(ty) => {
                let schema = fold_type(ty)?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                Some(Value::Record(Arc::new(wrapper)))
            }
            Expr::ToolCall(_)
            | Expr::StartToolCall(_)
            | Expr::Parallel { .. }
            | Expr::Await(_)
            | Expr::ResultUnwrap(_) => None,
        }
    }

    fn emit_builtin_call(&mut self, name: &str, args: &[Expr]) {
        if name == "format"
            && let Some((Expr::String(template), value_args)) = args.split_first()
        {
            for arg in value_args {
                self.compile_expr(arg);
            }
            let template = self.push_format_template(template, value_args.len());
            self.code.push(Instruction::FormatCompiled(template));
            return;
        }

        match (name, args.len()) {
            ("len", 1) => {
                self.compile_expr(&args[0]);
                self.code.push(Instruction::Len);
            }
            ("join", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Join);
            }
            ("validate", 2) => {
                if let Some(schema_wrapper) = self.fold_compile_time_expr(&args[1])
                    && let Some(schema) = unwrap_type_value(&schema_wrapper).cloned()
                {
                    self.compile_expr(&args[0]);
                    let schema = self.push_compiled_schema(&schema);
                    self.code.push(Instruction::ValidateCompiled(schema));
                    return;
                }

                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Validate);
            }
            ("push", 2) => {
                self.compile_expr(&args[0]);
                self.compile_expr(&args[1]);
                self.code.push(Instruction::Push);
            }
            ("range", 1 | 2) => {
                for arg in args {
                    self.compile_expr(arg);
                }
                self.code.push(Instruction::Range { argc: args.len() });
            }
            _ => {
                for arg in args {
                    self.compile_expr(arg);
                }
                let builtin = self.resolve_builtin(name);
                self.code.push(Instruction::CallBuiltin {
                    builtin,
                    argc: args.len(),
                });
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) {
        if !contains_type_literal(expr)
            && let Some(value) = self.fold_compile_time_expr(expr)
        {
            self.emit_push_value(value);
            return;
        }

        match expr {
            Expr::Null => {
                self.code.push(Instruction::PushNull);
            }
            Expr::Bool(value) => {
                self.code.push(Instruction::PushBool(*value));
            }
            Expr::Number(value) => {
                self.code.push(Instruction::PushNumber(*value));
            }
            Expr::String(value) => {
                let value = self.push_const(Value::String(value.clone()));
                self.code.push(Instruction::PushConst(value));
            }
            Expr::Variable(name) => {
                let name = self.push_slot(name);
                if let Some(value) = self.const_for_slot(name) {
                    self.emit_push_value(value);
                } else {
                    self.code.push(Instruction::LoadName(name));
                }
            }
            Expr::List(items) => {
                for item in items {
                    self.compile_expr(item);
                }
                self.code.push(Instruction::BuildList(items.len()));
            }
            Expr::Record(entries) => {
                for (_, value) in entries {
                    self.compile_expr(value);
                }
                let keys = self.push_key_list(entries.iter().map(|(key, _)| key.as_str()));
                self.code.push(Instruction::BuildRecord(keys));
            }
            Expr::ToolCall(call) => self.compile_call_expr(call),
            Expr::StartToolCall(call) => self.compile_start_call_expr(call),
            Expr::Parallel { branches } => self.compile_parallel(branches, true),
            Expr::Await(handle) => {
                self.compile_expr(handle);
                self.code.push(Instruction::AwaitHandle);
            }
            Expr::ResultUnwrap(expr) => {
                if let Expr::ToolCall(call) = expr.as_ref() {
                    self.compile_call_unwrap_expr(call);
                } else if let Expr::Await(handle) = expr.as_ref() {
                    self.compile_expr(handle);
                    self.code.push(Instruction::AwaitHandleUnwrap);
                } else if let Expr::Field { target, field } = expr.as_ref()
                    && let Expr::Variable(name) = target.as_ref()
                {
                    let slot = self.push_slot(name);
                    let field = self.push_name(field);
                    self.code.push(Instruction::LoadFieldUnwrap { slot, field });
                } else {
                    self.compile_expr(expr);
                    self.code.push(Instruction::ResultUnwrap);
                }
            }
            Expr::BuiltinCall { name, args } => {
                self.emit_builtin_call(name, args);
            }
            Expr::Field { target, field } => {
                if let Expr::Variable(name) = target.as_ref() {
                    let slot = self.push_slot(name);
                    let field = self.push_name(field);
                    self.code.push(Instruction::LoadField { slot, field });
                    return;
                }
                self.compile_expr(target);
                let field = self.push_name(field);
                self.code.push(Instruction::Field(field));
            }
            Expr::Index { target, index } => {
                self.compile_expr(target);
                self.compile_expr(index);
                self.code.push(Instruction::Index);
            }
            Expr::Unary { op, expr } => {
                self.compile_expr(expr);
                self.code.push(Instruction::Unary(*op));
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let jump_to_else = self.compile_condition_jump_if_false(condition);
                self.compile_expr(then_expr);
                let jump_to_end = self.emit_jump();
                self.patch_jump(jump_to_else, self.code.len());
                self.compile_expr(else_expr);
                self.patch_jump(jump_to_end, self.code.len());
            }
            Expr::TypeLiteral(ty) => self.compile_type_literal(ty),
            Expr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    self.compile_expr(left);
                    let jump_to_false = self.emit_jump_if_false();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_false, self.code.len());
                    self.code.push(Instruction::PushBool(false));
                    self.patch_jump(jump_to_end, self.code.len());
                }
                BinaryOp::Or => {
                    self.compile_expr(left);
                    let jump_to_true = self.emit_jump_if_true();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_true, self.code.len());
                    self.code.push(Instruction::PushBool(true));
                    self.patch_jump(jump_to_end, self.code.len());
                }
                _ => {
                    self.compile_expr(left);
                    self.compile_expr(right);
                    self.code.push(Instruction::Binary(*op));
                }
            },
        }
    }

    fn compile_call_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::CallTool { name, keys });
    }

    fn compile_call_unwrap_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::CallToolUnwrap { name, keys });
    }

    fn compile_start_call_expr(&mut self, call: &CallExpr) {
        for (_, expr) in &call.args {
            self.compile_expr(expr);
        }
        let keys = self.push_key_list(call.args.iter().map(|(name, _)| name.as_str()));
        let name = self.push_name(&call.name);
        self.code.push(Instruction::StartCallTool { name, keys });
    }

    fn compile_parallel(&mut self, branches: &ParallelBranches, want_value: bool) {
        match branches {
            ParallelBranches::Positional(branches) => {
                if let Some(branches) = self.compile_parallel_calls(branches) {
                    let branches = self.push_parallel_call_set(branches);
                    self.code.push(if want_value {
                        Instruction::ParallelCallsValue(branches)
                    } else {
                        Instruction::ParallelCalls(branches)
                    });
                    return;
                }

                if want_value && let Some(branches) = self.compile_pure_parallel_exprs(branches) {
                    let branches = self.push_pure_parallel_set(branches);
                    self.code.push(Instruction::PureParallelValue(branches));
                    return;
                }

                let branches = branches
                    .iter()
                    .map(|branch| {
                        let mut compiler = Self::with_slots_and_stats(
                            self.slots.clone(),
                            self.compile_stats.clone(),
                        );
                        compiler.compile_stmt(branch);
                        compiler.finish()
                    })
                    .collect::<Vec<_>>();
                let branches = self.push_branch_set(branches);
                self.code.push(if want_value {
                    Instruction::ParallelValue(branches)
                } else {
                    Instruction::Parallel(branches)
                });
            }
            ParallelBranches::Named(branches) => {
                if want_value && let Some(branches) = self.compile_named_parallel_calls(branches) {
                    let branches = self.push_named_parallel_call_set(branches);
                    self.code
                        .push(Instruction::ParallelNamedCallsValue(branches));
                    return;
                }

                if want_value
                    && let Some(branches) = self.compile_pure_named_parallel_exprs(branches)
                {
                    let branches = self.push_pure_named_parallel_set(branches);
                    self.code
                        .push(Instruction::PureParallelNamedValue(branches));
                    return;
                }

                let branches = branches
                    .iter()
                    .map(|branch| {
                        let mut compiler = Self::with_slots_and_stats(
                            self.slots.clone(),
                            self.compile_stats.clone(),
                        );
                        compiler.compile_stmt(&branch.stmt);
                        NamedBranchChunk {
                            name: self.push_name(&branch.name),
                            chunk: compiler.finish(),
                        }
                    })
                    .collect::<Vec<_>>();
                let branches = self.push_named_branch_set(branches);
                self.code.push(if want_value {
                    Instruction::ParallelNamedValue(branches)
                } else {
                    Instruction::ParallelNamed(branches)
                });
            }
        }
    }

    fn emit_jump_if_false(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::JumpIfFalse(usize::MAX));
        index
    }

    fn compile_condition_jump_if_false(&mut self, condition: &Expr) -> usize {
        if !contains_type_literal(condition)
            && let Some(value) = self.fold_compile_time_expr(condition)
        {
            self.emit_push_value(value);
            return self.emit_jump_if_false();
        }

        if let Expr::Binary { left, op, right } = condition
            && is_comparison_binary_op(*op)
        {
            if let (
                Expr::Binary {
                    left: inner_left,
                    op: binary_op,
                    right: inner_right,
                },
                Some(Value::Number(compare_right)),
            ) = (left.as_ref(), self.fold_compile_time_expr(right))
                && is_numeric_binary_op(*binary_op)
                && let (Expr::Variable(name), Some(Value::Number(binary_right))) = (
                    inner_left.as_ref(),
                    self.fold_compile_time_expr(inner_right),
                )
            {
                let slot = self.push_slot(name);
                let index = self.code.len();
                self.code
                    .push(Instruction::JumpIfSlotNumberBinaryCompareFalse {
                        slot,
                        binary_op: *binary_op,
                        binary_right,
                        compare_op: *op,
                        compare_right,
                        target: usize::MAX,
                    });
                return index;
            }

            if let (Expr::Variable(name), Some(Value::Number(right))) =
                (left.as_ref(), self.fold_compile_time_expr(right))
            {
                let slot = self.push_slot(name);
                let index = self.code.len();
                self.code.push(Instruction::JumpIfSlotNumberCompareFalse {
                    slot,
                    op: *op,
                    right,
                    target: usize::MAX,
                });
                return index;
            }
            self.compile_expr(left);
            self.compile_expr(right);
            let index = self.code.len();
            self.code.push(Instruction::JumpIfCompareFalse {
                op: *op,
                target: usize::MAX,
            });
            return index;
        }

        self.compile_expr(condition);
        self.emit_jump_if_false()
    }

    fn emit_jump_if_true(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::JumpIfTrue(usize::MAX));
        index
    }

    fn emit_jump(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::Jump(usize::MAX));
        index
    }

    fn compile_type_literal(&mut self, ty: &TypeExpr) {
        self.compile_stats.borrow_mut().type_literals_total += 1;

        if let Some(schema) = fold_type(ty) {
            let idx = self.push_const(wrap_type_schema_value(schema));
            self.code.push(Instruction::PushConst(idx));
            self.compile_stats.borrow_mut().type_literals_const_folded += 1;
            return;
        }

        self.compile_type_expr(ty);
        self.code.push(Instruction::WrapTypeLiteral);
        self.compile_stats.borrow_mut().type_literals_dynamic += 1;
    }

    fn compile_type_expr(&mut self, ty: &TypeExpr) {
        if let Some(value) = fold_type(ty) {
            let idx = self.push_const(value);
            self.code.push(Instruction::PushConst(idx));
            return;
        }

        match ty {
            TypeExpr::Ref(name) => {
                let slot = self.push_slot(name);
                self.code.push(Instruction::ResolveTypeRef(slot));
                self.compile_stats.borrow_mut().type_ref_sites += 1;
            }
            TypeExpr::List(inner) => {
                let kind_idx = self.push_const(Value::String("array".into()));
                self.code.push(Instruction::PushConst(kind_idx));
                self.compile_type_expr(inner);
                let keys = self.push_key_list(["type", "items"].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Object(fields) => {
                let kind_idx = self.push_const(Value::String("object".into()));
                self.code.push(Instruction::PushConst(kind_idx));

                for field in fields {
                    self.compile_type_expr(&field.ty);
                }
                let prop_keys = self.push_key_list(fields.iter().map(|f| f.name.as_str()));
                self.code.push(Instruction::BuildRecord(prop_keys));

                let required: Vec<&str> = fields
                    .iter()
                    .filter(|f| !f.optional)
                    .map(|f| f.name.as_str())
                    .collect();
                for name in &required {
                    let idx = self.push_const(Value::String((*name).into()));
                    self.code.push(Instruction::PushConst(idx));
                }
                self.code.push(Instruction::BuildList(required.len()));

                self.code.push(Instruction::PushBool(false));

                let obj_keys = self.push_key_list(
                    ["type", "properties", "required", "additionalProperties"].into_iter(),
                );
                self.code.push(Instruction::BuildRecord(obj_keys));
            }
            TypeExpr::Union(variants) => {
                // Union reaches this arm only when at least one variant
                // contains a `Ref` that couldn't const-fold. Compile
                // each variant and pack them into an `anyOf` list.
                for variant in variants {
                    self.compile_type_expr(variant);
                }
                self.code.push(Instruction::BuildList(variants.len()));
                let keys = self.push_key_list(["anyOf"].into_iter());
                self.code.push(Instruction::BuildRecord(keys));
            }
            TypeExpr::Any
            | TypeExpr::Str
            | TypeExpr::Int
            | TypeExpr::Float
            | TypeExpr::Bool
            | TypeExpr::Dict
            | TypeExpr::Null
            | TypeExpr::Enum(_) => {
                unreachable!("scalar/enum types must const-fold")
            }
        }
    }

    fn patch_jump(&mut self, index: usize, target: usize) {
        match &mut self.code[index] {
            Instruction::Jump(slot)
            | Instruction::JumpIfFalse(slot)
            | Instruction::JumpIfCompareFalse { target: slot, .. }
            | Instruction::JumpIfSlotNumberCompareFalse { target: slot, .. }
            | Instruction::JumpIfSlotNumberBinaryCompareFalse { target: slot, .. }
            | Instruction::JumpIfTrue(slot)
            | Instruction::IterNext { jump_to: slot } => *slot = target,
            _ => unreachable!("patched non-jump instruction"),
        }
    }
}

#[derive(Clone)]
struct SlotState {
    values: Vec<Option<Value>>,
    projected: Vec<bool>,
    extras: Record,
}

impl SlotState {
    fn from_globals(
        mut globals: Record,
        slot_names: &[Name],
        projected_bindings: &ProjectedBindings,
    ) -> Self {
        let mut values = Vec::with_capacity(slot_names.len());
        let mut projected = Vec::with_capacity(slot_names.len());
        for name in slot_names {
            if let Some(value) = projected_bindings.get_symbol(name.symbol) {
                globals.remove_symbol(name.symbol);
                values.push(Some(Value::Projected(value)));
                projected.push(true);
            } else {
                values.push(globals.remove_symbol(name.symbol));
                projected.push(false);
            }
        }
        Self {
            values,
            projected,
            extras: globals,
        }
    }

    fn from_globals_with_scratch(
        mut globals: Record,
        slot_names: &[Name],
        scratch: &mut ExecutionScratch,
        projected_bindings: &ProjectedBindings,
    ) -> Self {
        let mut values = std::mem::take(&mut scratch.slot_values);
        values.clear();
        if values.capacity() < slot_names.len() {
            values.reserve(slot_names.len() - values.capacity());
        }
        let mut projected = Vec::with_capacity(slot_names.len());
        for name in slot_names {
            if let Some(value) = projected_bindings.get_symbol(name.symbol) {
                globals.remove_symbol(name.symbol);
                values.push(Some(Value::Projected(value)));
                projected.push(true);
            } else {
                values.push(globals.remove_symbol(name.symbol));
                projected.push(false);
            }
        }
        Self {
            values,
            projected,
            extras: globals,
        }
    }

    fn get(&self, slot: usize) -> Option<&Value> {
        self.values.get(slot).and_then(Option::as_ref)
    }

    fn get_mut(&mut self, slot: usize) -> Option<&mut Value> {
        self.values.get_mut(slot).and_then(Option::as_mut)
    }

    fn assign(
        &mut self,
        slot: usize,
        value: Value,
        slot_names: &[Name],
    ) -> Result<(), RuntimeError> {
        self.ensure_assignable(slot, slot_names)?;
        self.values[slot] = Some(materialize_value(value));
        Ok(())
    }

    fn ensure_assignable(&self, slot: usize, slot_names: &[Name]) -> Result<(), RuntimeError> {
        if self.projected.get(slot).copied().unwrap_or(false) {
            return Err(RuntimeError::TypeError {
                message: format!(
                    "`{}` is a read-only projected binding",
                    slot_names[slot].text
                ),
            });
        }
        Ok(())
    }

    fn capture_temporary(&self, slot: usize) -> LoopRestore {
        LoopRestore {
            previous: self.values[slot].clone(),
        }
    }

    fn restore_temporary(&mut self, slot: usize, restore: LoopRestore) {
        self.values[slot] = restore.previous;
    }

    fn into_globals(self, slot_names: &[Name]) -> Record {
        let mut extras = self.extras;
        for ((name, value), projected) in slot_names.iter().zip(self.values).zip(self.projected) {
            if projected {
                extras.remove_symbol(name.symbol);
                continue;
            }
            match value {
                Some(value) => {
                    extras.insert_symbolized(
                        name.symbol,
                        name.text.clone(),
                        materialize_value(value),
                    );
                }
                None => {
                    extras.remove_symbol(name.symbol);
                }
            }
        }
        extras
    }

    fn recycle_into_globals(
        self,
        slot_names: &[Name],
        slot_values: &mut Vec<Option<Value>>,
    ) -> Record {
        let mut extras = self.extras;
        let mut values = self.values;
        for ((name, value), projected) in
            slot_names.iter().zip(values.iter_mut()).zip(self.projected)
        {
            if projected {
                extras.remove_symbol(name.symbol);
                continue;
            }
            match value.take() {
                Some(value) => {
                    extras.insert_symbolized(
                        name.symbol,
                        name.text.clone(),
                        materialize_value(value),
                    );
                }
                None => {
                    extras.remove_symbol(name.symbol);
                }
            }
        }
        values.clear();
        *slot_values = values;
        extras
    }
}

struct Vm<'a, H> {
    chunk: &'a Chunk,
    ip: usize,
    stack: Vec<Value>,
    last_value: Option<Value>,
    slots: SlotState,
    host: &'a H,
    in_parallel_branch: bool,
    assigned: Option<Vec<bool>>,
    iter_stack: Vec<IterState>,
    profile: Option<ProfileAccumulator>,
}

enum VmStep {
    Continue,
    Finish(Value),
    Effect(VmEffect),
}

#[derive(Clone, Copy)]
enum VmEffect {
    CallTool { name: usize, keys: usize },
    CallToolUnwrap { name: usize, keys: usize },
    StartCallTool { name: usize, keys: usize },
    AwaitHandle,
    AwaitHandleUnwrap,
    CancelHandle,
    Print,
    ParallelCalls(usize),
    ParallelCallsValue(usize),
    ParallelNamedCallsValue(usize),
    Parallel(usize),
    ParallelValue(usize),
    ParallelNamed(usize),
    ParallelNamedValue(usize),
}

struct VmTrap {
    error: RuntimeError,
    instruction_ip: usize,
}

impl<'a, H: ToolHost> Vm<'a, H> {
    fn new(chunk: &'a Chunk, slots: SlotState, host: &'a H, in_parallel_branch: bool) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: Vec::new(),
            last_value: None,
            slots,
            host,
            in_parallel_branch,
            assigned: in_parallel_branch.then(|| vec![false; chunk.slot_names.len()]),
            iter_stack: Vec::new(),
            profile: None,
        }
    }

    fn new_with_scratch(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
        scratch: &mut ExecutionScratch,
    ) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: std::mem::take(&mut scratch.stack),
            last_value: None,
            slots,
            host,
            in_parallel_branch,
            assigned: in_parallel_branch.then(|| vec![false; chunk.slot_names.len()]),
            iter_stack: std::mem::take(&mut scratch.iter_stack),
            profile: None,
        }
    }

    fn enable_profile(&mut self) {
        self.profile = Some(ProfileAccumulator::default());
    }

    async fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        let result = self.run_loop().await.map_err(|trap| trap.error);
        self.unwind_iterators();
        result
    }

    async fn run_traced(&mut self) -> Result<ExecutionOutcome, RuntimeFailure> {
        let result = self.run_loop().await.map_err(|trap| RuntimeFailure {
            error: trap.error,
            span: self.chunk.spans.get(trap.instruction_ip).copied().flatten(),
        });
        self.unwind_iterators();
        result
    }

    async fn run_loop(&mut self) -> Result<ExecutionOutcome, VmTrap> {
        let mut budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
        while let Some(instruction) = self.chunk.code.get(self.ip).copied() {
            let instruction_ip = self.ip;
            self.ip += 1;
            let profile = self
                .profile
                .as_ref()
                .map(|_| (instruction.profile_tag(), Instant::now()));
            let result = match self.step_instruction(instruction).await {
                Ok(VmStep::Continue) => Ok(None),
                Ok(VmStep::Finish(value)) => Ok(Some(ExecutionOutcome::Finished(value))),
                Ok(VmStep::Effect(effect)) => self.resolve_effect(effect).await.map(|()| None),
                Err(error) => Err(error),
            };
            if let Some((tag, start)) = profile {
                self.record_instruction_profile(tag, start.elapsed().as_nanos());
            }
            match result {
                Ok(Some(outcome)) => return Ok(outcome),
                Ok(None) => {}
                Err(error) => {
                    return Err(VmTrap {
                        error,
                        instruction_ip,
                    });
                }
            }
            budget -= 1;
            if budget == 0 {
                self.host.yield_now().await;
                budget = COOPERATIVE_YIELD_INSTRUCTION_BUDGET;
            }
        }
        Ok(ExecutionOutcome::Continued)
    }

    #[inline(always)]
    async fn step_instruction(&mut self, instruction: Instruction) -> Result<VmStep, RuntimeError> {
        match instruction {
            Instruction::PushConst(index) => {
                self.stack.push(self.chunk.constants[index].clone());
            }
            Instruction::PushNull => {
                self.stack.push(Value::Null);
            }
            Instruction::PushBool(value) => {
                self.stack.push(Value::Bool(value));
            }
            Instruction::PushNumber(value) => {
                self.stack.push(Value::Number(value));
            }
            Instruction::LoadName(name) => {
                let value = self.load_slot(name)?.clone();
                self.stack.push(value);
            }
            Instruction::StoreName(name) => {
                let value = self.pop_stack()?;
                self.slots
                    .assign(name, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(name);
                self.last_value = Some(value);
            }
            Instruction::StoreConst { slot, constant } => {
                let value = self.chunk.constants[constant].clone();
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::LoadField { slot, field } => {
                let value = self.load_slot(slot)?;
                let field = &self.chunk.names[field];
                let value = match value {
                    Value::Projected(projected) => projected.get_field(field).await?,
                    value => read_field_ref_direct(value, field)?,
                };
                self.stack.push(value);
            }
            Instruction::LoadFieldUnwrap { slot, field } => {
                let value = self.load_slot(slot)?;
                let field = &self.chunk.names[field];
                let value = match value {
                    Value::Projected(projected) => projected.get_field(field).await?,
                    value => read_field_ref_direct(value, field)?,
                };
                self.stack.push(unwrap_tool_result(value)?);
            }
            Instruction::BuildList(len) => {
                let values = self.pop_n(len)?;
                self.stack.push(Value::List(values.into()));
            }
            Instruction::BuildRecord(keys) => {
                let record = self.drain_record_from_stack(keys)?;
                self.stack.push(Value::Record(Arc::new(record)));
            }
            Instruction::Field(field) => {
                let target = self.pop_stack()?;
                let field = &self.chunk.names[field];
                let value = match target {
                    Value::Projected(projected) => projected.get_field(field).await?,
                    target => read_field_direct(target, field)?,
                };
                self.stack.push(value);
            }
            Instruction::Index => {
                let index = self.pop_stack()?;
                let target = self.pop_stack()?;
                let value = match target {
                    Value::Projected(projected) => projected.get_index(&index).await?,
                    target => read_index_direct(target, index)?,
                };
                self.stack.push(value);
            }
            Instruction::PathAssign { slot, path } => {
                let value = self.pop_stack()?;
                let last_value = value.clone();
                let path = &self.chunk.assign_paths[path];
                let index_start = self.stack_drain_start(path.dynamic_index_count)?;
                let indexes = &self.stack[index_start..];
                let root_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: root_name.text.to_string(),
                        })?;
                assign_path(root, path, indexes, value, &self.chunk.names)?;
                self.stack.truncate(index_start);
                self.record_assignment(slot);
                self.last_value = Some(last_value);
            }
            Instruction::ResultUnwrap => {
                let value = self.pop_stack()?;
                self.stack.push(unwrap_tool_result(value)?);
            }
            Instruction::Unary(op) => {
                let value = self.pop_stack()?;
                let value = match op {
                    UnaryOp::Negate => Value::Number(-as_number(&value)?),
                    UnaryOp::Not => Value::Bool(match &value {
                        Value::Projected(_) => !is_truthy_async(&value).await,
                        _ => !is_truthy(&value),
                    }),
                };
                self.stack.push(value);
            }
            Instruction::Binary(op) => {
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        eval_number_binary_values(left, op, right)
                    }
                    (left, right) => {
                        let has_projected = matches!(left, Value::Projected(_))
                            || matches!(right, Value::Projected(_));
                        if has_projected {
                            eval_binary_values_async(left, op, right).await?
                        } else {
                            eval_binary_values(left, op, right)?
                        }
                    }
                };
                self.stack.push(value);
            }
            Instruction::ToBool => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                self.stack.push(Value::Bool(truthy));
            }
            Instruction::Jump(target) => self.ip = target,
            Instruction::JumpIfFalse(target) => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfCompareFalse { op, target } => {
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                if !eval_compare_values_async(left, op, right).await? {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberCompareFalse {
                slot,
                op,
                right,
                target,
            } => {
                let value = self.load_slot(slot)?;
                let truthy = match value {
                    Value::Number(left) => eval_number_compare_values(*left, op, right),
                    value => {
                        eval_compare_values_async(value.clone(), op, Value::Number(right)).await?
                    }
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfSlotNumberBinaryCompareFalse {
                slot,
                binary_op,
                binary_right,
                compare_op,
                compare_right,
                target,
            } => {
                let value = self.load_slot(slot)?;
                let truthy = match value {
                    Value::Number(left) => {
                        let value =
                            eval_number_numeric_binary_value(*left, binary_op, binary_right);
                        eval_number_compare_values(value, compare_op, compare_right)
                    }
                    value => {
                        let value = eval_binary_values_async(
                            value.clone(),
                            binary_op,
                            Value::Number(binary_right),
                        )
                        .await?;
                        eval_compare_values_async(value, compare_op, Value::Number(compare_right))
                            .await?
                    }
                };
                if !truthy {
                    self.ip = target;
                }
            }
            Instruction::JumpIfTrue(target) => {
                let value = self.pop_stack()?;
                let truthy = match &value {
                    Value::Projected(_) => is_truthy_async(&value).await,
                    _ => is_truthy(&value),
                };
                if truthy {
                    self.ip = target;
                }
            }
            Instruction::CallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallTool { name, keys }));
            }
            Instruction::CallToolUnwrap { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::CallToolUnwrap { name, keys }));
            }
            Instruction::StartCallTool { name, keys } => {
                return Ok(VmStep::Effect(VmEffect::StartCallTool { name, keys }));
            }
            Instruction::AwaitHandle => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandle));
            }
            Instruction::AwaitHandleUnwrap => {
                return Ok(VmStep::Effect(VmEffect::AwaitHandleUnwrap));
            }
            Instruction::CancelHandle => {
                return Ok(VmStep::Effect(VmEffect::CancelHandle));
            }
            Instruction::CallBuiltin { builtin, argc } => {
                let values = self.stack_tail(argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_builtin(builtin, &self.chunk.names, values).await?;
                if let Some(start) = start {
                    self.record_builtin_profile(builtin, start.elapsed().as_nanos());
                }
                self.stack.truncate(self.stack.len() - argc);
                self.stack.push(value);
            }
            Instruction::Len => {
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = match &value {
                    Value::Projected(_) => execute_len_builtin(&value).await?,
                    _ => execute_len_direct(&value)?,
                };
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Len, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Join => {
                let sep = self.pop_stack()?;
                let items = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_join_builtin(&items, &sep)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Join, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Validate => {
                let schema = self.pop_stack()?;
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_validate_builtin(value, &schema)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Validate, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::ValidateCompiled(schema) => {
                let value = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_compiled_validate(value, &self.chunk.compiled_schemas[schema])?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Validate, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Push => {
                let item = self.pop_stack()?;
                let list = self.pop_stack()?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_push_builtin(&list, item)?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Push, start.elapsed().as_nanos());
                }
                self.stack.push(value);
            }
            Instruction::Range { argc } => {
                let start_index = self.stack_drain_start(argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_range_builtin(&self.stack[start_index..])?;
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Range, start.elapsed().as_nanos());
                }
                self.stack.truncate(start_index);
                self.stack.push(value);
            }
            Instruction::FormatCompiled(template) => {
                let template = &self.chunk.format_templates[template];
                let values = self.stack_tail(template.argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = if values
                    .iter()
                    .any(|value| matches!(value, Value::Projected(_)))
                {
                    execute_compiled_format(template, values).await?
                } else {
                    execute_compiled_format_direct(template, values)?
                };
                let value = Value::String(value.into());
                if let Some(start) = start {
                    self.record_builtin_profile(Builtin::Format, start.elapsed().as_nanos());
                }
                self.stack.truncate(self.stack.len() - template.argc);
                self.stack.push(value);
            }
            Instruction::AddAssign(slot) => {
                let right = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let value = {
                    let left = self.slots.get_mut(slot).ok_or_else(|| {
                        RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        }
                    })?;
                    match (left, right) {
                        (Value::Number(left), Value::Number(right)) => {
                            *left += right;
                            Value::Number(*left)
                        }
                        (left, right) => {
                            let value = add_values(left.clone(), right)?;
                            *left = value.clone();
                            value
                        }
                    }
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignNumber { slot, right } => {
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let value = {
                    let left = self.slots.get_mut(slot).ok_or_else(|| {
                        RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        }
                    })?;
                    match left {
                        Value::Number(left) => {
                            *left += right;
                            Value::Number(*left)
                        }
                        left => {
                            let value = add_values(left.clone(), Value::Number(right))?;
                            *left = value.clone();
                            value
                        }
                    }
                };
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AddAssignIndexNumber { slot, right } => {
                let index = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let root =
                    self.slots
                        .get_mut(slot)
                        .ok_or_else(|| RuntimeError::UndefinedVariable {
                            name: slot_name.text.to_string(),
                        })?;
                let value = add_assign_index_number(root, &index, right)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::AppendAssign(slot) => {
                let item = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                self.slots.ensure_assignable(slot, &self.chunk.slot_names)?;
                let current = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.text.to_string(),
                    }
                })?;
                let value = match current {
                    Value::List(items) => {
                        let mut values = Vec::with_capacity(items.len() + 1);
                        values.extend(items.iter().cloned());
                        values.push(item);
                        Value::List(values.into())
                    }
                    other => add_values(other, Value::List(vec![item].into()))?,
                };
                self.slots
                    .assign(slot, value.clone(), &self.chunk.slot_names)?;
                self.record_assignment(slot);
                self.last_value = Some(value);
            }
            Instruction::Print => {
                return Ok(VmStep::Effect(VmEffect::Print));
            }
            Instruction::Submit => {
                if self.in_parallel_branch {
                    return Err(RuntimeError::FinishInsideParallel);
                }
                return Ok(VmStep::Finish(self.pop_stack()?));
            }
            Instruction::Pop => {
                self.last_value = Some(self.pop_stack()?);
            }
            Instruction::BeginIter(binding) => {
                let iterable = self.pop_stack()?;
                let values = iterable_values(iterable).await?;
                self.iter_stack.push(IterState {
                    cursor: IterCursor::List { values, index: 0 },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::BeginRangeIter { binding, argc } => {
                let start_index = self.stack_drain_start(argc)?;
                let (start, end) = range_bounds(&self.stack[start_index..])?;
                self.stack.truncate(start_index);
                self.iter_stack.push(IterState {
                    cursor: IterCursor::Range { next: start, end },
                    binding,
                    restore: self.slots.capture_temporary(binding),
                });
            }
            Instruction::IterNext { jump_to } => {
                let Some(iter_state) = self.iter_stack.last_mut() else {
                    return Err(RuntimeError::ValueError {
                        message: "missing loop state".to_string(),
                    });
                };
                let Some(value) = iter_state.cursor.next_value() else {
                    self.ip = jump_to;
                    return Ok(VmStep::Continue);
                };
                self.slots
                    .assign(iter_state.binding, value, &self.chunk.slot_names)?;
            }
            Instruction::EndIter => {
                if let Some(iter_state) = self.iter_stack.pop() {
                    self.slots
                        .restore_temporary(iter_state.binding, iter_state.restore);
                }
            }
            Instruction::ParallelCalls(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelCalls(branches)));
            }
            Instruction::ParallelCallsValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelCallsValue(branches)));
            }
            Instruction::ParallelNamedCallsValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamedCallsValue(branches)));
            }
            Instruction::PureParallelValue(branches) => {
                let value = self.exec_pure_parallel_value(branches)?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            Instruction::PureParallelNamedValue(branches) => {
                let value = self.exec_pure_parallel_named_value(branches)?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            Instruction::Parallel(branches) => {
                return Ok(VmStep::Effect(VmEffect::Parallel(branches)));
            }
            Instruction::ParallelValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelValue(branches)));
            }
            Instruction::ParallelNamed(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamed(branches)));
            }
            Instruction::ParallelNamedValue(branches) => {
                return Ok(VmStep::Effect(VmEffect::ParallelNamedValue(branches)));
            }
            Instruction::ResolveTypeRef(slot) => {
                let slot_name = &self.chunk.slot_names[slot];
                let value = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.text.to_string(),
                    }
                })?;
                let schema =
                    unwrap_type_value(&value)
                        .cloned()
                        .ok_or_else(|| RuntimeError::TypeError {
                            message: format!(
                                "`{}` is not a Type value (missing `{LASH_TYPE_KEY}`)",
                                slot_name.text
                            ),
                        })?;
                self.stack.push(schema);
            }
            Instruction::WrapTypeLiteral => {
                let schema = self.pop_stack()?;
                let mut wrapper = record_with_capacity(1);
                wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
                self.stack.push(Value::Record(Arc::new(wrapper)));
            }
        }
        Ok(VmStep::Continue)
    }

    async fn resolve_effect(&mut self, effect: VmEffect) -> Result<(), RuntimeError> {
        match effect {
            VmEffect::CallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let result = match self
                    .host
                    .call(self.chunk.names[name].text.to_string(), args)
                    .await
                {
                    Ok(value) => success(value),
                    Err(error) => error_value(error.to_string()),
                };
                self.stack.push(result);
            }
            VmEffect::CallToolUnwrap { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .call(self.chunk.names[name].text.to_string(), args)
                    .await
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed tool result: {error}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::StartCallTool { name, keys } => {
                let args = self.drain_record_from_stack(keys)?;
                let value = self
                    .host
                    .start_call(self.chunk.names[name].text.to_string(), args)
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("async start failed: {err}"),
                    })?;
                self.stack.push(value);
            }
            VmEffect::AwaitHandle => {
                let handle = self.pop_stack()?;
                let result = self.await_value(handle).await;
                self.stack.push(result);
            }
            VmEffect::AwaitHandleUnwrap => {
                let handle = self.pop_stack()?;
                let result = self.await_value_unwrap(handle).await?;
                self.stack.push(result);
            }
            VmEffect::CancelHandle => {
                let handle = self.pop_stack()?;
                let value = self.host.cancel_handle(handle).await.map_err(|err| {
                    RuntimeError::ValueError {
                        message: format!("cancel failed: {err}"),
                    }
                })?;
                self.last_value = Some(value);
            }
            VmEffect::Print => {
                let value = self.pop_stack()?;
                let host_value = match &value {
                    Value::Projected(projected) => Value::String(projected.render().await.into()),
                    _ => value.clone(),
                };
                self.host
                    .print(host_value)
                    .await
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("print failed: {err}"),
                    })?;
                self.last_value = Some(value);
            }
            VmEffect::ParallelCalls(branches) => {
                self.exec_parallel_calls(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelCallsValue(branches) => {
                let value = self.exec_parallel_calls_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ParallelNamedCallsValue(branches) => {
                let value = self.exec_parallel_named_calls_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::Parallel(branches) => {
                self.exec_parallel(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelValue(branches) => {
                let value = self.exec_parallel_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
            VmEffect::ParallelNamed(branches) => {
                self.exec_parallel_named(branches).await?;
                self.last_value = Some(Value::Null);
            }
            VmEffect::ParallelNamedValue(branches) => {
                let value = self.exec_parallel_named_value(branches).await?;
                self.last_value = Some(value.clone());
                self.stack.push(value);
            }
        }
        Ok(())
    }

    async fn exec_parallel(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }

        let base_slots = self.slots.clone();
        let results = join_all(
            branches
                .iter()
                .map(|branch| Self::run_branch(branch, base_slots.clone(), self.host, true)),
        )
        .await;

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    async fn exec_parallel_value(&mut self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(Value::List(Vec::<Value>::new().into()));
        }

        let base_slots = self.slots.clone();
        let results = join_all(
            branches
                .iter()
                .map(|branch| Self::run_branch(branch, base_slots.clone(), self.host, true)),
        )
        .await;

        let mut outputs = Vec::with_capacity(results.len());
        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            let result = result?;
            outputs.push(result.output.clone());
            self.merge_branch_result(result, &mut merged_names)?;
        }
        Ok(Value::List(outputs.into()))
    }

    async fn exec_parallel_named(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.named_branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }

        let base_slots = self.slots.clone();
        let results =
            join_all(branches.iter().map(|branch| {
                Self::run_branch(&branch.chunk, base_slots.clone(), self.host, true)
            }))
            .await;

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    async fn exec_parallel_named_value(
        &mut self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.named_branch_sets[branches_index];
        let base_slots = self.slots.clone();
        let results = join_all(branches.iter().map(|branch| async {
            let result = Self::run_branch(&branch.chunk, base_slots.clone(), self.host, true).await;
            (branch.name, result)
        }))
        .await;

        let mut record = record_with_capacity(results.len());
        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for (name, result) in results {
            let result = result?;
            let name_entry = &self.chunk.names[name];
            record.insert_symbolized(
                name_entry.symbol,
                name_entry.text.clone(),
                result.output.clone(),
            );
            self.merge_branch_result(result, &mut merged_names)?;
        }
        Ok(Value::Record(Arc::new(record)))
    }

    fn exec_pure_parallel_value(&self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.pure_parallel_sets[branches_index];
        Ok(Value::List(
            branches
                .iter()
                .map(|expr| {
                    eval_pure_expr(expr, &self.slots, &self.chunk.names, &self.chunk.slot_names)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ))
    }

    fn exec_pure_parallel_named_value(&self, branches_index: usize) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.pure_named_parallel_sets[branches_index];
        let mut record = record_with_capacity(branches.len());
        for (name, expr) in branches.iter() {
            let name_entry = &self.chunk.names[*name];
            let value =
                eval_pure_expr(expr, &self.slots, &self.chunk.names, &self.chunk.slot_names)?;
            record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), value);
        }
        Ok(Value::Record(Arc::new(record)))
    }

    async fn exec_parallel_named_calls_value(
        &self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.named_parallel_call_sets[branches_index];
        match branches.len() {
            0 => return Ok(Value::Record(Arc::new(Record::default()))),
            1 => {
                let call = self.prepare_named_parallel_call(&branches[0])?;
                let result = Self::run_prepared_named_call(self.chunk, call, self.host).await?;
                let mut record = record_with_capacity(1);
                self.insert_named_parallel_call_result(&mut record, result);
                return Ok(Value::Record(Arc::new(record)));
            }
            2 => {
                let left_call = self.prepare_named_parallel_call(&branches[0])?;
                let right_call = self.prepare_named_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                let results =
                    Self::run_prepared_named_calls_batch_2(self.chunk, &calls, self.host).await?;
                let mut record = record_with_capacity(2);
                for result in results {
                    self.insert_named_parallel_call_result(&mut record, result);
                }
                return Ok(Value::Record(Arc::new(record)));
            }
            _ => {}
        }

        let mut calls = Vec::with_capacity(branches.len());
        for branch in branches {
            calls.push(self.prepare_named_parallel_call(branch)?);
        }

        let results = Self::run_prepared_named_calls_batch(self.chunk, &calls, self.host).await?;
        let mut record = record_with_capacity(results.len());
        for result in results {
            self.insert_named_parallel_call_result(&mut record, result);
        }
        Ok(Value::Record(Arc::new(record)))
    }

    async fn exec_parallel_calls(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.parallel_call_sets[branches_index];
        match branches.len() {
            0 => Ok(()),
            1 => {
                let call = self.prepare_parallel_call(&branches[0])?;
                let result = Self::run_prepared_call(self.chunk, call, self.host).await?;
                self.slots
                    .assign(result.slot, result.output, &self.chunk.slot_names)?;
                self.record_assignment(result.slot);
                Ok(())
            }
            2 => {
                let left_call = self.prepare_parallel_call(&branches[0])?;
                let right_call = self.prepare_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                if calls[0].slot == calls[1].slot {
                    return Err(RuntimeError::ParallelConflict {
                        name: self.chunk.slot_names[calls[0].slot].text.to_string(),
                    });
                }
                let results =
                    Self::run_prepared_calls_batch_2(self.chunk, &calls, self.host).await?;
                for result in results {
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(())
            }
            _ => {
                let mut calls = Vec::with_capacity(branches.len());
                for branch in branches {
                    calls.push(self.prepare_parallel_call(branch)?);
                }
                self.ensure_distinct_parallel_call_slots(&calls)?;
                let results = Self::run_prepared_calls_batch(self.chunk, &calls, self.host).await?;
                for result in results {
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(())
            }
        }
    }

    async fn exec_parallel_calls_value(
        &mut self,
        branches_index: usize,
    ) -> Result<Value, RuntimeError> {
        let branches = &self.chunk.parallel_call_sets[branches_index];
        match branches.len() {
            0 => Ok(Value::List(Vec::<Value>::new().into())),
            1 => {
                let call = self.prepare_parallel_call(&branches[0])?;
                let result = Self::run_prepared_call(self.chunk, call, self.host).await?;
                let output = result.output.clone();
                self.slots
                    .assign(result.slot, result.output, &self.chunk.slot_names)?;
                self.record_assignment(result.slot);
                Ok(Value::List(vec![output].into()))
            }
            2 => {
                let left_call = self.prepare_parallel_call(&branches[0])?;
                let right_call = self.prepare_parallel_call(&branches[1])?;
                let calls = [left_call, right_call];
                if calls[0].slot == calls[1].slot {
                    return Err(RuntimeError::ParallelConflict {
                        name: self.chunk.slot_names[calls[0].slot].text.to_string(),
                    });
                }
                let results =
                    Self::run_prepared_calls_batch_2(self.chunk, &calls, self.host).await?;
                let mut outputs = Vec::with_capacity(2);
                for result in results {
                    outputs.push(result.output.clone());
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(Value::List(outputs.into()))
            }
            _ => {
                let mut calls = Vec::with_capacity(branches.len());
                for branch in branches {
                    calls.push(self.prepare_parallel_call(branch)?);
                }
                self.ensure_distinct_parallel_call_slots(&calls)?;
                let results = Self::run_prepared_calls_batch(self.chunk, &calls, self.host).await?;
                let mut outputs = Vec::with_capacity(results.len());
                for result in results {
                    outputs.push(result.output.clone());
                    self.slots
                        .assign(result.slot, result.output, &self.chunk.slot_names)?;
                    self.record_assignment(result.slot);
                }
                Ok(Value::List(outputs.into()))
            }
        }
    }

    fn prepare_parallel_call(
        &self,
        branch: &ParallelCallBranch,
    ) -> Result<PreparedParallelCall, RuntimeError> {
        let value = eval_pure_expr(
            &branch.args,
            &self.slots,
            &self.chunk.names,
            &self.chunk.slot_names,
        )?;
        let Value::Record(args) = value else {
            return Err(RuntimeError::TypeError {
                message: "parallel call args must compile to a record".to_string(),
            });
        };
        Ok(PreparedParallelCall {
            slot: branch.slot,
            name: branch.name,
            args: Arc::unwrap_or_clone(args),
        })
    }

    fn ensure_distinct_parallel_call_slots(
        &self,
        calls: &[PreparedParallelCall],
    ) -> Result<(), RuntimeError> {
        for (index, call) in calls.iter().enumerate() {
            if calls[..index]
                .iter()
                .any(|previous| previous.slot == call.slot)
            {
                return Err(RuntimeError::ParallelConflict {
                    name: self.chunk.slot_names[call.slot].text.to_string(),
                });
            }
        }
        Ok(())
    }

    fn prepare_named_parallel_call(
        &self,
        branch: &NamedParallelCallBranch,
    ) -> Result<PreparedNamedParallelCall, RuntimeError> {
        let value = eval_pure_expr(
            &branch.args,
            &self.slots,
            &self.chunk.names,
            &self.chunk.slot_names,
        )?;
        let Value::Record(args) = value else {
            return Err(RuntimeError::TypeError {
                message: "parallel call args must compile to a record".to_string(),
            });
        };
        Ok(PreparedNamedParallelCall {
            output_name: branch.output_name,
            name: branch.name,
            args: Arc::unwrap_or_clone(args),
        })
    }

    fn insert_named_parallel_call_result(
        &self,
        record: &mut Record,
        result: NamedParallelCallResult,
    ) {
        let name_entry = &self.chunk.names[result.output_name];
        record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), result.output);
    }

    async fn run_host_batch<const N: usize>(
        host_calls: Vec<ToolHostCall>,
        host: &'a H,
    ) -> Result<HostBatchResults<N>, RuntimeError>
    where
        [HostBatchItemResult; N]: smallvec::Array<Item = HostBatchItemResult>,
    {
        let run = std::panic::AssertUnwindSafe(host.call_batch(host_calls))
            .catch_unwind()
            .await;
        match run {
            Ok(results) => Ok(results.into_iter().collect()),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    async fn run_prepared_calls_batch(
        chunk: &'a Chunk,
        calls: &[PreparedParallelCall],
        host: &'a H,
    ) -> Result<SmallVec<[ParallelCallResult; 4]>, RuntimeError> {
        let host_calls = calls
            .iter()
            .map(|call| ToolHostCall {
                name: chunk.names[call.name].text.to_string(),
                args: call.args.clone(),
            })
            .collect::<Vec<_>>();
        let results = Self::run_host_batch::<4>(host_calls, host).await?;
        Self::finish_prepared_calls_batch(calls, results)
    }

    async fn run_prepared_calls_batch_2(
        chunk: &'a Chunk,
        calls: &[PreparedParallelCall; 2],
        host: &'a H,
    ) -> Result<[ParallelCallResult; 2], RuntimeError> {
        let host_calls = vec![
            ToolHostCall {
                name: chunk.names[calls[0].name].text.to_string(),
                args: calls[0].args.clone(),
            },
            ToolHostCall {
                name: chunk.names[calls[1].name].text.to_string(),
                args: calls[1].args.clone(),
            },
        ];
        let results = Self::run_host_batch::<2>(host_calls, host).await?;
        Self::finish_prepared_calls_batch_2(calls, results)
    }

    fn finish_prepared_call_result(
        call: &PreparedParallelCall,
        result: Result<Value, ToolHostError>,
    ) -> ParallelCallResult {
        ParallelCallResult {
            slot: call.slot,
            output: match result {
                Ok(value) => success(value),
                Err(error) => error_value(error.to_string()),
            },
        }
    }

    fn finish_host_batch_pair(
        results: HostBatchResults<2>,
    ) -> Result<[HostBatchItemResult; 2], RuntimeError> {
        if results.len() != 2 {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        let mut results = results.into_iter();
        Ok([
            results.next().expect("length checked"),
            results.next().expect("length checked"),
        ])
    }

    fn finish_prepared_calls_batch_2(
        calls: &[PreparedParallelCall; 2],
        results: HostBatchResults<2>,
    ) -> Result<[ParallelCallResult; 2], RuntimeError> {
        let [left, right] = Self::finish_host_batch_pair(results)?;
        Ok([
            Self::finish_prepared_call_result(&calls[0], left),
            Self::finish_prepared_call_result(&calls[1], right),
        ])
    }

    fn finish_prepared_calls_batch(
        calls: &[PreparedParallelCall],
        results: HostBatchResults<4>,
    ) -> Result<SmallVec<[ParallelCallResult; 4]>, RuntimeError> {
        if results.len() != calls.len() {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        Ok(calls
            .iter()
            .zip(results)
            .map(|(call, result)| Self::finish_prepared_call_result(call, result))
            .collect())
    }

    async fn run_prepared_named_calls_batch(
        chunk: &'a Chunk,
        calls: &[PreparedNamedParallelCall],
        host: &'a H,
    ) -> Result<SmallVec<[NamedParallelCallResult; 4]>, RuntimeError> {
        let host_calls = calls
            .iter()
            .map(|call| ToolHostCall {
                name: chunk.names[call.name].text.to_string(),
                args: call.args.clone(),
            })
            .collect::<Vec<_>>();
        let results = Self::run_host_batch::<4>(host_calls, host).await?;
        Self::finish_prepared_named_calls_batch(calls, results)
    }

    async fn run_prepared_named_calls_batch_2(
        chunk: &'a Chunk,
        calls: &[PreparedNamedParallelCall; 2],
        host: &'a H,
    ) -> Result<[NamedParallelCallResult; 2], RuntimeError> {
        let host_calls = vec![
            ToolHostCall {
                name: chunk.names[calls[0].name].text.to_string(),
                args: calls[0].args.clone(),
            },
            ToolHostCall {
                name: chunk.names[calls[1].name].text.to_string(),
                args: calls[1].args.clone(),
            },
        ];
        let results = Self::run_host_batch::<2>(host_calls, host).await?;
        Self::finish_prepared_named_calls_batch_2(calls, results)
    }

    fn finish_prepared_named_call_result(
        call: &PreparedNamedParallelCall,
        result: Result<Value, ToolHostError>,
    ) -> NamedParallelCallResult {
        NamedParallelCallResult {
            output_name: call.output_name,
            output: match result {
                Ok(value) => success(value),
                Err(error) => error_value(error.to_string()),
            },
        }
    }

    fn finish_prepared_named_calls_batch_2(
        calls: &[PreparedNamedParallelCall; 2],
        results: HostBatchResults<2>,
    ) -> Result<[NamedParallelCallResult; 2], RuntimeError> {
        let [left, right] = Self::finish_host_batch_pair(results)?;
        Ok([
            Self::finish_prepared_named_call_result(&calls[0], left),
            Self::finish_prepared_named_call_result(&calls[1], right),
        ])
    }

    fn finish_prepared_named_calls_batch(
        calls: &[PreparedNamedParallelCall],
        results: HostBatchResults<4>,
    ) -> Result<SmallVec<[NamedParallelCallResult; 4]>, RuntimeError> {
        if results.len() != calls.len() {
            return Err(RuntimeError::ValueError {
                message: "parallel call batch returned the wrong number of results".to_string(),
            });
        }
        Ok(calls
            .iter()
            .zip(results)
            .map(|(call, result)| Self::finish_prepared_named_call_result(call, result))
            .collect())
    }

    async fn run_branch(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
    ) -> Result<BranchResult, RuntimeError> {
        let mut vm = Self::new(chunk, slots, host, in_parallel_branch);
        let run = std::panic::AssertUnwindSafe(vm.run()).catch_unwind().await;
        match run {
            Ok(Ok(ExecutionOutcome::Continued)) => Ok(vm.into_branch_result()),
            Ok(Ok(ExecutionOutcome::Finished(_))) => Err(RuntimeError::FinishInsideParallel),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    async fn run_prepared_call(
        chunk: &'a Chunk,
        call: PreparedParallelCall,
        host: &'a H,
    ) -> Result<ParallelCallResult, RuntimeError> {
        let slot = call.slot;
        let name = chunk.names[call.name].text.to_string();
        let run = std::panic::AssertUnwindSafe(host.call(name, call.args))
            .catch_unwind()
            .await;
        match run {
            Ok(result) => match result {
                Ok(value) => Ok(success(value)),
                Err(error) => Ok(error_value(error.to_string())),
            },
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
        .map(|value| ParallelCallResult {
            slot,
            output: value,
        })
    }

    async fn run_prepared_named_call(
        chunk: &'a Chunk,
        call: PreparedNamedParallelCall,
        host: &'a H,
    ) -> Result<NamedParallelCallResult, RuntimeError> {
        let output_name = call.output_name;
        let name = chunk.names[call.name].text.to_string();
        let run = std::panic::AssertUnwindSafe(host.call(name, call.args))
            .catch_unwind()
            .await;
        match run {
            Ok(result) => match result {
                Ok(value) => Ok(success(value)),
                Err(error) => Ok(error_value(error.to_string())),
            },
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
        .map(|output| NamedParallelCallResult {
            output_name,
            output,
        })
    }

    fn merge_branch_result(
        &mut self,
        result: BranchResult,
        merged_names: &mut [bool],
    ) -> Result<(), RuntimeError> {
        for (slot, value) in result.values {
            if std::mem::replace(&mut merged_names[slot], true) {
                return Err(RuntimeError::ParallelConflict {
                    name: self.chunk.slot_names[slot].text.to_string(),
                });
            }
            self.slots.assign(slot, value, &self.chunk.slot_names)?;
            self.record_assignment(slot);
        }
        Ok(())
    }

    fn pop_stack(&mut self) -> Result<Value, RuntimeError> {
        self.stack.pop().ok_or_else(|| RuntimeError::ValueError {
            message: "vm stack underflow".to_string(),
        })
    }

    fn load_slot(&self, slot: usize) -> Result<&Value, RuntimeError> {
        self.slots
            .get(slot)
            .ok_or_else(|| RuntimeError::UndefinedVariable {
                name: self.chunk.slot_names[slot].text.to_string(),
            })
    }

    fn drain_record_from_stack(&mut self, keys: usize) -> Result<Record, RuntimeError> {
        let key_indices = &self.chunk.key_lists[keys];
        let start = self.stack_drain_start(key_indices.len())?;
        let mut record = record_with_capacity(key_indices.len());
        for (key, value) in key_indices.iter().zip(self.stack.drain(start..)) {
            let name_entry = &self.chunk.names[*key];
            record.insert_symbolized(name_entry.symbol, name_entry.text.clone(), value);
        }
        Ok(record)
    }

    fn await_value(
        &self,
        handle: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send + '_>> {
        Box::pin(async move {
            match handle {
                Value::List(handles) => {
                    let mut values = Vec::with_capacity(handles.len());
                    for handle in handles.iter().cloned() {
                        values.push(self.await_value(handle).await);
                    }
                    Value::List(values.into())
                }
                Value::Record(handles) if is_async_handle_record(&handles) => {
                    match self.host.await_handle(Value::Record(handles)).await {
                        Ok(value) => success(value),
                        Err(error) => error_value(error.to_string()),
                    }
                }
                Value::Record(handles) => {
                    let mut record = record_with_capacity(handles.len());
                    for entry in handles.entries.iter() {
                        record.insert_symbolized(
                            entry.symbol,
                            entry.name.clone(),
                            self.await_value(entry.value.clone()).await,
                        );
                    }
                    Value::Record(Arc::new(record))
                }
                handle => match self.host.await_handle(handle).await {
                    Ok(value) => success(value),
                    Err(error) => error_value(error.to_string()),
                },
            }
        })
    }

    async fn await_value_unwrap(&self, handle: Value) -> Result<Value, RuntimeError> {
        match handle {
            Value::Record(handles) if is_async_handle_record(&handles) => self
                .host
                .await_handle(Value::Record(handles))
                .await
                .map_err(|error| RuntimeError::ValueError {
                    message: format!("`?` unwrapped failed tool result: {error}"),
                }),
            Value::List(_) | Value::Record(_) => unwrap_tool_result(self.await_value(handle).await),
            handle => {
                self.host
                    .await_handle(handle)
                    .await
                    .map_err(|error| RuntimeError::ValueError {
                        message: format!("`?` unwrapped failed tool result: {error}"),
                    })
            }
        }
    }

    fn pop_n(&mut self, len: usize) -> Result<Vec<Value>, RuntimeError> {
        if self.stack.len() < len {
            return Err(RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            });
        }
        let start = self.stack.len() - len;
        Ok(self.stack.split_off(start))
    }

    fn stack_tail(&self, len: usize) -> Result<&[Value], RuntimeError> {
        if self.stack.len() < len {
            return Err(RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            });
        }
        Ok(&self.stack[self.stack.len() - len..])
    }

    fn stack_drain_start(&self, len: usize) -> Result<usize, RuntimeError> {
        self.stack
            .len()
            .checked_sub(len)
            .ok_or_else(|| RuntimeError::ValueError {
                message: "vm stack underflow".to_string(),
            })
    }

    fn unwind_iterators(&mut self) {
        while let Some(iter_state) = self.iter_stack.pop() {
            self.slots
                .restore_temporary(iter_state.binding, iter_state.restore);
        }
    }

    fn record_assignment(&mut self, slot: usize) {
        if let Some(assigned) = &mut self.assigned {
            assigned[slot] = true;
        }
    }

    fn into_branch_result(self) -> BranchResult {
        let assigned = self.assigned.unwrap_or_default();
        let mut values = Vec::with_capacity(assigned.iter().filter(|assigned| **assigned).count());
        for (slot, assigned) in assigned.into_iter().enumerate() {
            if assigned && let Some(value) = self.slots.get(slot).cloned() {
                values.push((slot, value));
            }
        }
        BranchResult {
            values,
            output: self.last_value.unwrap_or(Value::Null),
        }
    }

    fn into_globals(self) -> Record {
        self.slots.into_globals(&self.chunk.slot_names)
    }

    fn recycle_into_globals(mut self, scratch: &mut ExecutionScratch) -> Record {
        self.stack.clear();
        self.iter_stack.clear();
        scratch.stack = std::mem::take(&mut self.stack);
        scratch.iter_stack = std::mem::take(&mut self.iter_stack);
        self.slots
            .recycle_into_globals(&self.chunk.slot_names, &mut scratch.slot_values)
    }

    fn record_instruction_profile(&mut self, tag: InstructionProfileTag, elapsed_ns: u128) {
        let Some(profile) = &mut self.profile else {
            return;
        };
        let index = tag as usize;
        profile.instruction_counts[index] += 1;
        profile.instruction_times[index] += elapsed_ns;
    }

    fn record_builtin_profile(&mut self, builtin: Builtin, elapsed_ns: u128) {
        let Some(profile) = &mut self.profile else {
            return;
        };
        let index = builtin.profile_tag() as usize;
        profile.builtin_counts[index] += 1;
        profile.builtin_times[index] += elapsed_ns;
    }

    fn take_profile(&mut self) -> ProfileReport {
        let Some(profile) = self.profile.take() else {
            return ProfileReport::default();
        };
        profile.finish()
    }
}

struct BranchResult {
    values: Vec<(usize, Value)>,
    output: Value,
}

struct ParallelCallResult {
    slot: usize,
    output: Value,
}

struct NamedParallelCallResult {
    output_name: usize,
    output: Value,
}

struct PreparedParallelCall {
    slot: usize,
    name: usize,
    args: Record,
}

struct PreparedNamedParallelCall {
    output_name: usize,
    name: usize,
    args: Record,
}

type HostBatchItemResult = Result<Value, ToolHostError>;
type HostBatchResults<const N: usize> = SmallVec<[HostBatchItemResult; N]>;

struct IterState {
    cursor: IterCursor,
    binding: usize,
    restore: LoopRestore,
}

enum IterCursor {
    List { values: Arc<[Value]>, index: usize },
    Range { next: i64, end: i64 },
}

impl IterCursor {
    fn next_value(&mut self) -> Option<Value> {
        match self {
            Self::List { values, index } => {
                let value = values.get(*index)?.clone();
                *index += 1;
                Some(value)
            }
            Self::Range { next, end } => {
                if *next >= *end {
                    return None;
                }
                let value = *next;
                *next += 1;
                Some(Value::Number(value as f64))
            }
        }
    }
}

struct LoopRestore {
    previous: Option<Value>,
}

fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Null | Expr::Bool(_) | Expr::Number(_) | Expr::String(_) | Expr::Variable(_) => true,
        Expr::List(items) => items.iter().all(is_pure_expr),
        Expr::Record(entries) => entries.iter().all(|(_, value)| is_pure_expr(value)),
        Expr::ToolCall(_) => false,
        Expr::StartToolCall(_) => false,
        Expr::Parallel { .. } => false,
        Expr::Await(_) => false,
        Expr::ResultUnwrap(expr) => is_pure_expr(expr),
        Expr::BuiltinCall { args, .. } => args.iter().all(is_pure_expr),
        Expr::Field { target, .. } => is_pure_expr(target),
        Expr::Index { target, index } => is_pure_expr(target) && is_pure_expr(index),
        Expr::Unary { expr, .. } => is_pure_expr(expr),
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => is_pure_expr(condition) && is_pure_expr(then_expr) && is_pure_expr(else_expr),
        Expr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        Expr::TypeLiteral(ty) => fold_type(ty).is_some(),
    }
}

fn contains_type_literal(expr: &Expr) -> bool {
    match expr {
        Expr::TypeLiteral(_) => true,
        Expr::List(items) => items.iter().any(contains_type_literal),
        Expr::Record(entries) => entries
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::ToolCall(call) | Expr::StartToolCall(call) => call
            .args
            .iter()
            .any(|(_, value)| contains_type_literal(value)),
        Expr::Parallel { branches } => match branches {
            ParallelBranches::Positional(statements) => {
                statements.iter().any(stmt_contains_type_literal)
            }
            ParallelBranches::Named(branches) => branches
                .iter()
                .any(|branch| stmt_contains_type_literal(&branch.stmt)),
        },
        Expr::Await(expr) | Expr::ResultUnwrap(expr) | Expr::Unary { expr, .. } => {
            contains_type_literal(expr)
        }
        Expr::BuiltinCall { args, .. } => args.iter().any(contains_type_literal),
        Expr::Field { target, .. } => contains_type_literal(target),
        Expr::Index { target, index } => {
            contains_type_literal(target) || contains_type_literal(index)
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            contains_type_literal(condition)
                || contains_type_literal(then_expr)
                || contains_type_literal(else_expr)
        }
        Expr::Binary { left, right, .. } => {
            contains_type_literal(left) || contains_type_literal(right)
        }
        Expr::Null | Expr::Bool(_) | Expr::Number(_) | Expr::String(_) | Expr::Variable(_) => false,
    }
}

fn stmt_contains_type_literal(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Assign { target, expr } => {
            assign_target_contains_type_literal(target) || contains_type_literal(expr)
        }
        Stmt::Expr(expr) | Stmt::Cancel(expr) | Stmt::Print(expr) => contains_type_literal(expr),
        Stmt::Call(call) => call
            .args
            .iter()
            .any(|(_, expr)| contains_type_literal(expr)),
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            contains_type_literal(condition)
                || then_block.iter().any(stmt_contains_type_literal)
                || else_block.iter().any(stmt_contains_type_literal)
        }
        Stmt::For { iterable, body, .. } => {
            contains_type_literal(iterable) || body.iter().any(stmt_contains_type_literal)
        }
        Stmt::Break | Stmt::Continue => false,
        Stmt::Parallel { branches } => match branches {
            ParallelBranches::Positional(statements) => {
                statements.iter().any(stmt_contains_type_literal)
            }
            ParallelBranches::Named(branches) => branches
                .iter()
                .any(|branch| stmt_contains_type_literal(&branch.stmt)),
        },
        Stmt::Submit(expr) => expr.as_ref().is_some_and(contains_type_literal),
    }
}

fn assign_target_contains_type_literal(target: &AssignTarget) -> bool {
    target.steps.iter().any(|step| match step {
        AssignPathStep::Field(_) => false,
        AssignPathStep::Index(expr) => contains_type_literal(expr),
    })
}

/// Best-effort compile-time construction of a JSON-Schema Value for a
/// [`TypeExpr`]. Returns `None` when the expression contains a [`TypeExpr::Ref`]
/// (or a nested composite that contains one) — those must be resolved at
/// runtime via [`Instruction::ResolveTypeRef`].
fn fold_type(ty: &TypeExpr) -> Option<Value> {
    match ty {
        TypeExpr::Any => Some(interned_scalar_schema(ScalarSchemaKind::Any)),
        TypeExpr::Str => Some(interned_scalar_schema(ScalarSchemaKind::Str)),
        TypeExpr::Int => Some(interned_scalar_schema(ScalarSchemaKind::Int)),
        TypeExpr::Float => Some(interned_scalar_schema(ScalarSchemaKind::Float)),
        TypeExpr::Bool => Some(interned_scalar_schema(ScalarSchemaKind::Bool)),
        TypeExpr::Dict => Some(interned_scalar_schema(ScalarSchemaKind::Dict)),
        TypeExpr::Null => Some(interned_scalar_schema(ScalarSchemaKind::Null)),
        TypeExpr::Enum(values) => {
            let mut rec = record_with_capacity(2);
            rec.insert("type".into(), Value::String("string".into()));
            let items: Vec<Value> = values.iter().map(|v| Value::String(v.clone())).collect();
            rec.insert("enum".into(), Value::List(items.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::List(inner) => {
            let inner_value = fold_type(inner)?;
            let mut rec = record_with_capacity(2);
            rec.insert("type".into(), Value::String("array".into()));
            rec.insert("items".into(), inner_value);
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Object(fields) => {
            let mut properties = record_with_capacity(fields.len());
            for field in fields {
                properties.insert(field.name.to_string(), fold_type(&field.ty)?);
            }
            let required: Vec<Value> = fields
                .iter()
                .filter(|f| !f.optional)
                .map(|f| Value::String(f.name.clone()))
                .collect();
            let mut rec = record_with_capacity(4);
            rec.insert("type".into(), Value::String("object".into()));
            rec.insert("properties".into(), Value::Record(Arc::new(properties)));
            rec.insert("required".into(), Value::List(required.into()));
            rec.insert("additionalProperties".into(), Value::Bool(false));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Union(variants) => {
            let folded: Option<Vec<Value>> = variants.iter().map(fold_type).collect();
            let folded = folded?;
            let mut rec = record_with_capacity(1);
            rec.insert("anyOf".into(), Value::List(folded.into()));
            Some(Value::Record(Arc::new(rec)))
        }
        TypeExpr::Ref(_) => None,
    }
}

fn wrap_type_schema_value(schema: Value) -> Value {
    let mut wrapper = record_with_capacity(1);
    wrapper.insert(LASH_TYPE_KEY.to_string(), schema);
    Value::Record(Arc::new(wrapper))
}

#[derive(Clone, Copy)]
enum ScalarSchemaKind {
    Any,
    Str,
    Int,
    Float,
    Bool,
    Dict,
    Null,
}

/// Returns an `Arc`-shared schema for a scalar. All sites referencing `str`
/// point at the same `Arc<Record>`, so emitting a Type literal with N string
/// fields allocates one record, not N.
fn interned_scalar_schema(kind: ScalarSchemaKind) -> Value {
    static CACHE: OnceLock<[Value; 7]> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let build = |ty: &str| {
            let mut rec = record_with_capacity(1);
            rec.insert("type".into(), Value::String(ty.into()));
            Value::Record(Arc::new(rec))
        };
        [
            Value::Record(Arc::new(record_with_capacity(0))), // Any == {}
            build("string"),
            build("integer"),
            build("number"),
            build("boolean"),
            build("object"),
            build("null"),
        ]
    });
    cache[kind as usize].clone()
}

/// Unwrap a `Value::Record` that carries the `$lash_type` marker back into the
/// inner JSON-Schema value. Returns `None` when the value is not a wrapped
/// Type literal.
pub fn unwrap_type_value(value: &Value) -> Option<&Value> {
    let record = value.as_record()?;
    if record.len() != 1 {
        return None;
    }
    record.get(LASH_TYPE_KEY)
}

fn eval_pure_expr(
    expr: &PureExpr,
    slots: &SlotState,
    names: &[Name],
    slot_names: &[Name],
) -> Result<Value, RuntimeError> {
    match expr {
        PureExpr::Const(value) => Ok(value.clone()),
        PureExpr::Slot(slot) => {
            slots
                .get(*slot)
                .cloned()
                .ok_or_else(|| RuntimeError::UndefinedVariable {
                    name: slot_names[*slot].text.to_string(),
                })
        }
        PureExpr::List(items) => Ok(Value::List(
            items
                .iter()
                .map(|item| eval_pure_expr(item, slots, names, slot_names))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        )),
        PureExpr::Record(entries) => {
            let mut record = record_with_capacity(entries.len());
            for (key, value) in entries.iter() {
                let name_entry = &names[*key];
                record.insert_symbolized(
                    name_entry.symbol,
                    name_entry.text.clone(),
                    eval_pure_expr(value, slots, names, slot_names)?,
                );
            }
            Ok(Value::Record(Arc::new(record)))
        }
        PureExpr::Builtin { builtin, args } => {
            let mut values = SmallVec::<[Value; 8]>::with_capacity(args.len());
            for arg in args.iter() {
                values.push(eval_pure_expr(arg, slots, names, slot_names)?);
            }
            execute_builtin_blocking(*builtin, names, &values)
        }
        PureExpr::Format { template, args } => {
            let mut values = SmallVec::<[Value; 8]>::with_capacity(args.len());
            for arg in args.iter() {
                values.push(eval_pure_expr(arg, slots, names, slot_names)?);
            }
            Ok(Value::String(
                execute_compiled_format_blocking(template, &values)?.into(),
            ))
        }
        PureExpr::ResultUnwrap(expr) => {
            let value = eval_pure_expr(expr, slots, names, slot_names)?;
            unwrap_tool_result(value)
        }
        PureExpr::Field { target, field } => {
            let value = eval_pure_expr(target, slots, names, slot_names)?;
            read_field_blocking(value, &names[*field])
        }
        PureExpr::Index { target, index } => {
            let target = eval_pure_expr(target, slots, names, slot_names)?;
            let index = eval_pure_expr(index, slots, names, slot_names)?;
            read_index_blocking(target, index)
        }
        PureExpr::Unary { op, expr } => {
            let value = eval_pure_expr(expr, slots, names, slot_names)?;
            match op {
                UnaryOp::Negate => Ok(Value::Number(-as_number(&value)?)),
                UnaryOp::Not => Ok(Value::Bool(!is_truthy(&value))),
            }
        }
        PureExpr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            if is_truthy(&eval_pure_expr(condition, slots, names, slot_names)?) {
                eval_pure_expr(then_expr, slots, names, slot_names)
            } else {
                eval_pure_expr(else_expr, slots, names, slot_names)
            }
        }
        PureExpr::Binary { left, op, right } => match op {
            BinaryOp::And => Ok(Value::Bool(
                is_truthy(&eval_pure_expr(left, slots, names, slot_names)?)
                    && is_truthy(&eval_pure_expr(right, slots, names, slot_names)?),
            )),
            BinaryOp::Or => Ok(Value::Bool(
                is_truthy(&eval_pure_expr(left, slots, names, slot_names)?)
                    || is_truthy(&eval_pure_expr(right, slots, names, slot_names)?),
            )),
            _ => {
                let left = eval_pure_expr(left, slots, names, slot_names)?;
                let right = eval_pure_expr(right, slots, names, slot_names)?;
                eval_binary_values(left, *op, right)
            }
        },
    }
}

fn expect_arg_count(name: &str, values: &[Value], expected: usize) -> Result<(), RuntimeError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(RuntimeError::TypeError {
            message: format!("`{name}` takes {expected} arg(s), got {}", values.len()),
        })
    }
}

async fn execute_builtin(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    match builtin {
        Builtin::Len => {
            expect_arg_count("len", values, 1)?;
            execute_len_builtin(&values[0]).await
        }
        Builtin::Empty => {
            expect_arg_count("empty", values, 1)?;
            match &values[0] {
                Value::String(value) => Ok(Value::Bool(value.is_empty())),
                Value::List(values) => Ok(Value::Bool(values.is_empty())),
                Value::Record(record) => Ok(Value::Bool(record.is_empty())),
                Value::Projected(value) => Ok(Value::Bool(value.is_empty().await)),
                Value::Null => Ok(Value::Bool(true)),
                _ => Err(RuntimeError::TypeError {
                    message: "`empty` requires a string, list, record, or null".to_string(),
                }),
            }
        }
        Builtin::Keys => {
            expect_arg_count("keys", values, 1)?;
            match &values[0] {
                Value::Record(record) => Ok(Value::List(
                    record
                        .keys()
                        .map(|key| Value::String(key.to_string().into()))
                        .collect::<Vec<_>>()
                        .into(),
                )),
                Value::Projected(value) => Ok(Value::List(
                    value
                        .keys()
                        .await
                        .into_iter()
                        .map(|key| Value::String(key.into()))
                        .collect::<Vec<_>>()
                        .into(),
                )),
                Value::Null => Ok(Value::List(Vec::new().into())),
                _ => Err(RuntimeError::TypeError {
                    message: "`keys` requires a record or null".to_string(),
                }),
            }
        }
        Builtin::Values => {
            expect_arg_count("values", values, 1)?;
            match &values[0] {
                Value::Record(record) => Ok(Value::List(
                    record.values().cloned().collect::<Vec<_>>().into(),
                )),
                Value::Null => Ok(Value::List(Vec::new().into())),
                _ => Err(RuntimeError::TypeError {
                    message: "`values` requires a record or null".to_string(),
                }),
            }
        }
        Builtin::Contains => {
            expect_arg_count("contains", values, 2)?;
            execute_contains_builtin(&values[0], &values[1]).await
        }
        Builtin::StartsWith => {
            expect_arg_count("starts_with", values, 2)?;
            let value = coerce_string(&values[0])?;
            let prefix = coerce_string(&values[1])?;
            Ok(Value::Bool(value.starts_with(prefix.as_ref())))
        }
        Builtin::EndsWith => {
            expect_arg_count("ends_with", values, 2)?;
            let value = coerce_string(&values[0])?;
            let suffix = coerce_string(&values[1])?;
            Ok(Value::Bool(value.ends_with(suffix.as_ref())))
        }
        Builtin::Split => {
            expect_arg_count("split", values, 2)?;
            let value = coerce_string(&values[0])?;
            let needle = coerce_string(&values[1])?;
            Ok(Value::List(
                value
                    .split(needle.as_ref())
                    .map(|part| Value::String(part.to_string().into()))
                    .collect::<Vec<_>>()
                    .into(),
            ))
        }
        Builtin::Join => {
            expect_arg_count("join", values, 2)?;
            execute_join_builtin(&values[0], &values[1])
        }
        Builtin::Trim => {
            expect_arg_count("trim", values, 1)?;
            Ok(Value::String(
                coerce_string(&values[0])?.trim().to_string().into(),
            ))
        }
        Builtin::Slice => {
            expect_arg_count("slice", values, 3)?;
            let start = as_slice_bound(&values[1])?;
            let end = as_slice_bound(&values[2])?;
            match &values[0] {
                Value::String(value) => Ok(Value::String(slice_string(value, start, end).into())),
                Value::List(items) => {
                    let Some((start, end)) = clamp_slice_bounds(start, end, items.len()) else {
                        return Ok(Value::List(Vec::new().into()));
                    };
                    Ok(Value::List(items[start..end].to_vec().into()))
                }
                _ => Err(RuntimeError::TypeError {
                    message: "`slice` requires a string or list".to_string(),
                }),
            }
        }
        Builtin::ToString => {
            expect_arg_count("to_string", values, 1)?;
            let value = if value_contains_projected(&values[0]) {
                stringify_value_async(&values[0]).await?
            } else {
                stringify_value_direct(&values[0])?
            };
            Ok(Value::String(value.into()))
        }
        Builtin::ToInt => {
            expect_arg_count("to_int", values, 1)?;
            Ok(Value::Number(as_number(&values[0])?.trunc()))
        }
        Builtin::ToFloat => {
            expect_arg_count("to_float", values, 1)?;
            Ok(Value::Number(as_number(&values[0])?))
        }
        Builtin::JsonParse => {
            expect_arg_count("json_parse", values, 1)?;
            let parsed: serde_json::Value = serde_json::from_str(&coerce_string(&values[0])?)
                .map_err(|err| RuntimeError::ValueError {
                    message: format!("invalid json: {err}"),
                })?;
            Ok(from_json(parsed))
        }
        Builtin::Format => {
            if values.is_empty() {
                return Err(RuntimeError::TypeError {
                    message: "`format` requires at least a template string".to_string(),
                });
            }
            let template = match &values[0] {
                Value::String(value) => value.as_str(),
                other => {
                    return Err(RuntimeError::TypeError {
                        message: format!(
                            "`format` template must be a string, got {}",
                            value_type_name(other)
                        ),
                    });
                }
            };
            Ok(Value::String(
                apply_format_async(template, &values[1..]).await?.into(),
            ))
        }
        Builtin::Validate => {
            expect_arg_count("validate", values, 2)?;
            execute_validate_builtin(values[0].clone(), &values[1])
        }
        Builtin::Range => execute_range_builtin(values),
        Builtin::Push => {
            expect_arg_count("push", values, 2)?;
            execute_push_builtin(&values[0], values[1].clone())
        }
        Builtin::Unknown(index) => Err(RuntimeError::UnknownBuiltin {
            name: names[index].text.to_string(),
        }),
    }
}

fn execute_builtin_blocking(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    futures_executor::block_on(execute_builtin(builtin, names, values))
}

async fn execute_len_builtin(value: &Value) -> Result<Value, RuntimeError> {
    if let Value::Projected(value) = value {
        return Ok(Value::Number(value.len().await as f64));
    }
    execute_len_direct(value)
}

fn execute_len_direct(value: &Value) -> Result<Value, RuntimeError> {
    value_len(value)
        .map(|len| Value::Number(len as f64))
        .ok_or_else(|| RuntimeError::TypeError {
            message: "`len` requires a string, list, record, or null; use `.size` for images"
                .to_string(),
        })
}

async fn execute_contains_builtin(haystack: &Value, needle: &Value) -> Result<Value, RuntimeError> {
    if !matches!(haystack, Value::Projected(_)) {
        return execute_contains_direct(haystack, needle).map(Value::Bool);
    }
    match (haystack, needle) {
        (Value::Projected(value), needle) => Ok(Value::Bool(value.contains(needle).await?)),
        (Value::Null, _) => Ok(Value::Bool(false)),
        _ => Err(RuntimeError::TypeError {
            message:
                "`contains` requires a string/string, list/value, record/key, or null/value pair"
                    .to_string(),
        }),
    }
}

fn execute_contains_direct(haystack: &Value, needle: &Value) -> Result<bool, RuntimeError> {
    match (haystack, needle) {
        (Value::String(haystack), needle) => Ok(haystack.contains(coerce_string(needle)?.as_ref())),
        (Value::List(items), needle) => Ok(items.contains(needle)),
        (Value::Record(record), needle) => {
            Ok(record.get(coerce_string(needle)?.as_ref()).is_some())
        }
        (Value::Null, _) => Ok(false),
        _ => Err(RuntimeError::TypeError {
            message:
                "`contains` requires a string/string, list/value, record/key, or null/value pair"
                    .to_string(),
        }),
    }
}

fn value_len(value: &Value) -> Option<usize> {
    match value {
        Value::String(value) => Some(value.chars().count()),
        Value::List(values) => Some(values.len()),
        Value::Record(record) => Some(record.len()),
        Value::Null => Some(0),
        _ => None,
    }
}

async fn iterable_values(value: Value) -> Result<Arc<[Value]>, RuntimeError> {
    match value {
        Value::List(values) => Ok(values),
        Value::Projected(value) => match value.materialize_async().await {
            Value::List(values) => Ok(values),
            _ => Err(RuntimeError::NonListIteration),
        },
        _ => Err(RuntimeError::NonListIteration),
    }
}

fn execute_join_builtin(items: &Value, sep: &Value) -> Result<Value, RuntimeError> {
    let Value::List(items) = items else {
        return Err(RuntimeError::TypeError {
            message: "`join` requires a list as the first argument".to_string(),
        });
    };
    let sep = coerce_string(sep)?;
    let mut joined = String::new();
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            joined.push_str(sep.as_ref());
        }
        joined.push_str(coerce_string(item)?.as_ref());
    }
    Ok(Value::String(joined.into()))
}

fn execute_range_builtin(values: &[Value]) -> Result<Value, RuntimeError> {
    let (start, end) = range_bounds(values)?;
    build_range(start, end)
}

fn range_bounds(values: &[Value]) -> Result<(i64, i64), RuntimeError> {
    let (start, end) = match values {
        [end] => (0, as_range_bound(end)?),
        [start, end] => (as_range_bound(start)?, as_range_bound(end)?),
        _ => {
            return Err(RuntimeError::TypeError {
                message: format!("`range` takes 1 or 2 arg(s), got {}", values.len()),
            });
        }
    };
    validate_range_len(start, end)?;
    Ok((start, end))
}

fn execute_push_builtin(list: &Value, item: Value) -> Result<Value, RuntimeError> {
    let Value::List(items) = list else {
        return Err(RuntimeError::TypeError {
            message: "`push` requires a list as the first argument".to_string(),
        });
    };
    let mut values = Vec::with_capacity(items.len() + 1);
    values.extend(items.iter().cloned());
    values.push(item);
    Ok(Value::List(values.into()))
}

fn as_range_bound(value: &Value) -> Result<i64, RuntimeError> {
    let Value::Number(number) = value else {
        return Err(RuntimeError::TypeError {
            message: format!(
                "`range` bounds must be finite integers, got {}",
                value_type_name(value)
            ),
        });
    };
    if !number.is_finite()
        || number.fract() != 0.0
        || *number < i64::MIN as f64
        || *number > i64::MAX as f64
    {
        return Err(RuntimeError::TypeError {
            message: "`range` bounds must be finite integers".to_string(),
        });
    }
    Ok(*number as i64)
}

fn build_range(start: i64, end: i64) -> Result<Value, RuntimeError> {
    if start >= end {
        return Ok(Value::List(Vec::new().into()));
    }
    Ok(Value::List(
        (start..end)
            .map(|value| Value::Number(value as f64))
            .collect::<Vec<_>>()
            .into(),
    ))
}

fn validate_range_len(start: i64, end: i64) -> Result<(), RuntimeError> {
    const MAX_RANGE_ITEMS: i64 = 1_000_000;
    if start >= end {
        return Ok(());
    }
    let len = end as i128 - start as i128;
    if len > MAX_RANGE_ITEMS as i128 {
        return Err(RuntimeError::ValueError {
            message: format!("`range` would create more than {MAX_RANGE_ITEMS} items"),
        });
    }
    Ok(())
}

fn read_field_ref_direct(value: &Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Record(record) => Ok(record
            .get_symbol(field.symbol)
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Image(image) => read_image_field(image, field),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't read `.{}` from {}",
                field.text,
                value_type_name(value)
            ),
        }),
    }
}

fn unwrap_tool_result(value: Value) -> Result<Value, RuntimeError> {
    let Value::Record(record) = value else {
        return Err(RuntimeError::TypeError {
            message: format!(
                "`?` expected a tool result wrapper, got {}",
                value_type_name(&value)
            ),
        });
    };

    let result_names = result_wrapper_names();
    match record.get_symbol(result_names.ok.symbol) {
        Some(Value::Bool(true)) => record
            .get_symbol(result_names.value.symbol)
            .cloned()
            .ok_or_else(|| RuntimeError::TypeError {
                message: "`?` found a successful tool result wrapper missing `value`".to_string(),
            }),
        Some(Value::Bool(false)) => {
            let message = record
                .get_symbol(result_names.error.symbol)
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown error".to_string());
            Err(RuntimeError::ValueError {
                message: format!("`?` unwrapped failed tool result: {message}"),
            })
        }
        _ => Err(RuntimeError::TypeError {
            message: "`?` expected a tool result wrapper with boolean `ok`".to_string(),
        }),
    }
}

fn is_async_handle_record(record: &Record) -> bool {
    record.get("__handle__").is_some() || record.get("handle").is_some()
}

async fn read_field(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Projected(value) => value.get_field(field).await,
        other => read_field_direct(other, field),
    }
}

fn read_field_direct(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Record(record) => Ok(record
            .get_symbol(field.symbol)
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Image(image) => read_image_field(&image, field),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't read `.{}` from {}",
                field.text,
                value_type_name(&value)
            ),
        }),
    }
}

fn read_field_blocking(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_field(value, field))
}

fn read_image_field(image: &ImageValue, field: &Name) -> Result<Value, RuntimeError> {
    match field.text.as_ref() {
        "id" => Ok(Value::String(image.id.clone().into())),
        "label" => Ok(Value::String(image.label.clone().into())),
        "size" => Ok(Value::Number(image.size as f64)),
        "width" => Ok(image
            .width
            .map(|width| Value::Number(width as f64))
            .unwrap_or(Value::Null)),
        "height" => Ok(image
            .height
            .map(|height| Value::Number(height as f64))
            .unwrap_or(Value::Null)),
        _ => Ok(Value::Null),
    }
}

async fn read_index(target: Value, index: Value) -> Result<Value, RuntimeError> {
    match target {
        Value::Projected(value) => value.get_index(&index).await,
        other => read_index_direct(other, index),
    }
}

fn read_index_direct(target: Value, index: Value) -> Result<Value, RuntimeError> {
    read_index_ref_direct(&target, &index)
}

fn read_index_ref_direct(target: &Value, index: &Value) -> Result<Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_index(index, values.len())?;
            Ok(idx
                .and_then(|idx| values.get(idx).cloned())
                .unwrap_or(Value::Null))
        }
        Value::String(value) => {
            let idx = resolve_index(index, value.chars().count())?;
            Ok(idx
                .and_then(|idx| value.chars().nth(idx))
                .map(|ch| Value::String(ch.to_string().into()))
                .unwrap_or(Value::Null))
        }
        Value::Record(record) => Ok(record
            .get(coerce_string(index)?.as_ref())
            .cloned()
            .unwrap_or(Value::Null)),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't index {}", value_type_name(target)),
        }),
    }
}

fn read_index_blocking(target: Value, index: Value) -> Result<Value, RuntimeError> {
    futures_executor::block_on(read_index(target, index))
}

fn assign_path(
    root: &mut Value,
    path: &CompiledAssignPath,
    indexes: &[Value],
    value: Value,
    names: &[Name],
) -> Result<(), RuntimeError> {
    let mut index_cursor = 0;
    assign_path_steps(root, &path.steps, indexes, &mut index_cursor, value, names)
}

fn assign_path_steps(
    target: &mut Value,
    steps: &[CompiledAssignPathStep],
    indexes: &[Value],
    index_cursor: &mut usize,
    value: Value,
    names: &[Name],
) -> Result<(), RuntimeError> {
    let Some((step, rest)) = steps.split_first() else {
        *target = value;
        return Ok(());
    };

    match *step {
        CompiledAssignPathStep::Field(field) if rest.is_empty() => {
            assign_record_field(target, &names[field], value)
        }
        CompiledAssignPathStep::Field(field) => {
            let child = descend_record_field(target, &names[field])?;
            assign_path_steps(child, rest, indexes, index_cursor, value, names)
        }
        CompiledAssignPathStep::Index if rest.is_empty() => {
            let index = next_assign_index(indexes, index_cursor)?;
            assign_index(target, index, value)
        }
        CompiledAssignPathStep::Index => {
            let index = next_assign_index(indexes, index_cursor)?;
            let child = descend_index(target, index)?;
            assign_path_steps(child, rest, indexes, index_cursor, value, names)
        }
    }
}

fn next_assign_index<'a>(
    indexes: &'a [Value],
    index_cursor: &mut usize,
) -> Result<&'a Value, RuntimeError> {
    let index = indexes
        .get(*index_cursor)
        .ok_or_else(|| RuntimeError::ValueError {
            message: "missing assignment index".to_string(),
        })?;
    *index_cursor += 1;
    Ok(index)
}

fn assign_record_field(target: &mut Value, field: &Name, value: Value) -> Result<(), RuntimeError> {
    match target {
        Value::Record(record) => {
            Arc::make_mut(record).insert_symbolized(field.symbol, field.text.clone(), value);
            Ok(())
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't assign `.{}` on {}",
                field.text,
                value_type_name(target)
            ),
        }),
    }
}

fn descend_record_field<'a>(
    target: &'a mut Value,
    field: &Name,
) -> Result<&'a mut Value, RuntimeError> {
    match target {
        Value::Record(record) => Arc::make_mut(record)
            .get_symbol_mut(field.symbol)
            .ok_or_else(|| RuntimeError::ValueError {
                message: format!("can't assign through missing field `.{}`", field.text),
            }),
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign through image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!(
                "can't assign through `.{}` on {}",
                field.text,
                value_type_name(target)
            ),
        }),
    }
}

fn assign_index(target: &mut Value, index: &Value, value: Value) -> Result<(), RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            Arc::make_mut(values)[idx] = value;
            Ok(())
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            Arc::make_mut(record).insert_str(key.as_ref(), value);
            Ok(())
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign index on {}", value_type_name(target)),
        }),
    }
}

fn add_assign_index_number(
    target: &mut Value,
    index: &Value,
    right: f64,
) -> Result<Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            add_assign_value_number(&mut Arc::make_mut(values)[idx], right)
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            let record = Arc::make_mut(record);
            if let Some(value) = record.get_mut(key.as_ref()) {
                add_assign_value_number(value, right)
            } else {
                let value = Value::Number(right);
                record.insert_str(key.as_ref(), value.clone());
                Ok(value)
            }
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign index on {}", value_type_name(target)),
        }),
    }
}

fn add_assign_value_number(value: &mut Value, right: f64) -> Result<Value, RuntimeError> {
    match value {
        Value::Number(left) => {
            *left += right;
            Ok(Value::Number(*left))
        }
        left => {
            let value = add_values(left.clone(), Value::Number(right))?;
            *left = value.clone();
            Ok(value)
        }
    }
}

fn descend_index<'a>(target: &'a mut Value, index: &Value) -> Result<&'a mut Value, RuntimeError> {
    match target {
        Value::List(values) => {
            let idx = resolve_existing_list_assignment_index(index, values.len())?;
            Ok(&mut Arc::make_mut(values)[idx])
        }
        Value::Record(record) => {
            let key = coerce_string(index)?;
            let record = Arc::make_mut(record);
            if let Some(value) = record.get_mut(key.as_ref()) {
                Ok(value)
            } else {
                Err(RuntimeError::ValueError {
                    message: format!("can't assign through missing key `{}`", key.as_ref()),
                })
            }
        }
        Value::Image(_) => Err(RuntimeError::TypeError {
            message: "can't assign through image fields; images are immutable".to_string(),
        }),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't assign through index on {}", value_type_name(target)),
        }),
    }
}

fn eval_binary_values(left: Value, op: BinaryOp, right: Value) -> Result<Value, RuntimeError> {
    match op {
        BinaryOp::Add => add_values(left, right),
        BinaryOp::Subtract => numeric_binary_values(left, right, |a, b| a - b),
        BinaryOp::Multiply => numeric_binary_values(left, right, |a, b| a * b),
        BinaryOp::Divide => numeric_binary_values(left, right, |a, b| a / b),
        BinaryOp::Modulo => numeric_binary_values(left, right, |a, b| a % b),
        BinaryOp::Equal => Ok(Value::Bool(left == right)),
        BinaryOp::NotEqual => Ok(Value::Bool(left != right)),
        BinaryOp::Less => compare_ordered(left, right, |a, b| a < b, |a, b| a < b),
        BinaryOp::LessEqual => compare_ordered(left, right, |a, b| a <= b, |a, b| a <= b),
        BinaryOp::Greater => compare_ordered(left, right, |a, b| a > b, |a, b| a > b),
        BinaryOp::GreaterEqual => compare_ordered(left, right, |a, b| a >= b, |a, b| a >= b),
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are compiled with jumps"),
    }
}

fn eval_number_binary_values(left: f64, op: BinaryOp, right: f64) -> Value {
    match op {
        BinaryOp::Add
        | BinaryOp::Subtract
        | BinaryOp::Multiply
        | BinaryOp::Divide
        | BinaryOp::Modulo => Value::Number(eval_number_numeric_binary_value(left, op, right)),
        BinaryOp::Equal => Value::Bool(left == right),
        BinaryOp::NotEqual => Value::Bool(left != right),
        BinaryOp::Less => Value::Bool(left < right),
        BinaryOp::LessEqual => Value::Bool(left <= right),
        BinaryOp::Greater => Value::Bool(left > right),
        BinaryOp::GreaterEqual => Value::Bool(left >= right),
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are compiled with jumps"),
    }
}

fn eval_number_numeric_binary_value(left: f64, op: BinaryOp, right: f64) -> f64 {
    match op {
        BinaryOp::Add => left + right,
        BinaryOp::Subtract => left - right,
        BinaryOp::Multiply => left * right,
        BinaryOp::Divide => left / right,
        BinaryOp::Modulo => left % right,
        _ => unreachable!("non-numeric op in fused numeric branch"),
    }
}

fn eval_number_compare_values(left: f64, op: BinaryOp, right: f64) -> bool {
    match op {
        BinaryOp::Equal => left == right,
        BinaryOp::NotEqual => left != right,
        BinaryOp::Less => left < right,
        BinaryOp::LessEqual => left <= right,
        BinaryOp::Greater => left > right,
        BinaryOp::GreaterEqual => left >= right,
        _ => unreachable!("non-comparison op in fused slot branch"),
    }
}

fn is_comparison_binary_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
    )
}

fn is_numeric_binary_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo
    )
}

fn eval_compare_values(left: Value, op: BinaryOp, right: Value) -> Result<bool, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(eval_number_compare_values(left, op, right))
        }
        (left, right) => match op {
            BinaryOp::Equal => Ok(left == right),
            BinaryOp::NotEqual => Ok(left != right),
            BinaryOp::Less => {
                compare_ordered(left, right, |a, b| a < b, |a, b| a < b).map(expect_bool_value)
            }
            BinaryOp::LessEqual => {
                compare_ordered(left, right, |a, b| a <= b, |a, b| a <= b).map(expect_bool_value)
            }
            BinaryOp::Greater => {
                compare_ordered(left, right, |a, b| a > b, |a, b| a > b).map(expect_bool_value)
            }
            BinaryOp::GreaterEqual => {
                compare_ordered(left, right, |a, b| a >= b, |a, b| a >= b).map(expect_bool_value)
            }
            _ => unreachable!("non-comparison op in fused branch"),
        },
    }
}

async fn eval_compare_values_async(
    left: Value,
    op: BinaryOp,
    right: Value,
) -> Result<bool, RuntimeError> {
    let has_projected = matches!(left, Value::Projected(_)) || matches!(right, Value::Projected(_));
    if has_projected {
        eval_compare_values(
            materialize_projected_async(left).await,
            op,
            materialize_projected_async(right).await,
        )
    } else {
        eval_compare_values(left, op, right)
    }
}

fn expect_bool_value(value: Value) -> bool {
    match value {
        Value::Bool(value) => value,
        _ => unreachable!("comparison produced non-bool value"),
    }
}

async fn eval_binary_values_async(
    left: Value,
    op: BinaryOp,
    right: Value,
) -> Result<Value, RuntimeError> {
    eval_binary_values(
        materialize_projected_async(left).await,
        op,
        materialize_projected_async(right).await,
    )
}

async fn materialize_projected_async(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize_async().await,
        other => other,
    }
}

fn numeric_binary_values(
    left: Value,
    right: Value,
    op: impl FnOnce(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Number(op(left, right))),
        (left, right) => Ok(Value::Number(op(as_number(&left)?, as_number(&right)?))),
    }
}

fn as_number(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Number(value) => Ok(*value),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Null => Ok(0.0),
        Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Ok(0.0);
            }
            value.parse::<f64>().map_err(|_| RuntimeError::TypeError {
                message: "expected a number".to_string(),
            })
        }
        _ => Err(RuntimeError::TypeError {
            message: format!("expected a number, got {}", value_type_name(value)),
        }),
    }
}

fn coerce_string(value: &Value) -> Result<Cow<'_, str>, RuntimeError> {
    match value {
        Value::String(value) => Ok(Cow::Borrowed(value)),
        Value::Null => Ok(Cow::Borrowed("null")),
        Value::Bool(value) => Ok(Cow::Owned(value.to_string())),
        Value::Number(value) => Ok(Cow::Owned(value.to_string())),
        Value::Image(_) | Value::List(_) | Value::Record(_) | Value::Projected(_) => {
            Err(RuntimeError::TypeError {
                message: format!("expected text, got {}", value_type_name(value)),
            })
        }
    }
}

fn as_offset(value: &Value) -> Result<isize, RuntimeError> {
    let number = as_number(value)?;
    if !number.is_finite() || number.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "index must be an integer".to_string(),
        });
    }
    Ok(number as isize)
}

fn as_slice_bound(value: &Value) -> Result<Option<isize>, RuntimeError> {
    match value {
        Value::Null => Ok(None),
        other => as_offset(other).map(Some),
    }
}

fn compare_numbers(
    left: Value,
    right: Value,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(Value::Bool(cmp(left, right))),
        (left, right) => Ok(Value::Bool(cmp(as_number(&left)?, as_number(&right)?))),
    }
}

fn compare_ordered(
    left: Value,
    right: Value,
    number_cmp: impl FnOnce(f64, f64) -> bool,
    string_cmp: impl FnOnce(&str, &str) -> bool,
) -> Result<Value, RuntimeError> {
    match (&left, &right) {
        (Value::String(left), Value::String(right)) => Ok(Value::Bool(string_cmp(left, right))),
        _ => compare_numbers(left, right, number_cmp),
    }
}

fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
        (Value::String(mut a), other) => {
            a.push_str(&stringify_value_blocking(&other)?);
            Ok(Value::String(a))
        }
        (other, Value::String(b)) => {
            let mut text = stringify_value_blocking(&other)?;
            text.push_str(&b);
            Ok(Value::String(text.into()))
        }
        (Value::List(a), Value::List(b)) => {
            let mut values = Vec::with_capacity(a.len() + b.len());
            values.extend(a.iter().cloned());
            values.extend(b.iter().cloned());
            Ok(Value::List(values.into()))
        }
        (left, right) => Ok(Value::Number(as_number(&left)? + as_number(&right)?)),
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => *value != 0.0 && !value.is_nan(),
        Value::String(value) => !value.is_empty(),
        Value::Image(_) | Value::List(_) | Value::Record(_) => true,
        Value::Projected(value) => !futures_executor::block_on(value.is_empty()),
    }
}

async fn is_truthy_async(value: &Value) -> bool {
    match value {
        Value::Projected(value) => !value.is_empty().await,
        other => is_truthy(other),
    }
}

fn success(value: Value) -> Value {
    let result_names = result_wrapper_names();
    let mut record = record_with_capacity(2);
    record.insert_symbolized(
        result_names.ok.symbol,
        result_names.ok.text.clone(),
        Value::Bool(true),
    );
    record.insert_symbolized(
        result_names.value.symbol,
        result_names.value.text.clone(),
        value,
    );
    Value::Record(Arc::new(record))
}

fn error_value(message: String) -> Value {
    let result_names = result_wrapper_names();
    let mut record = record_with_capacity(2);
    record.insert_symbolized(
        result_names.ok.symbol,
        result_names.ok.text.clone(),
        Value::Bool(false),
    );
    record.insert_symbolized(
        result_names.error.symbol,
        result_names.error.text.clone(),
        Value::String(message.into()),
    );
    Value::Record(Arc::new(record))
}

async fn stringify_value_async(value: &Value) -> Result<String, RuntimeError> {
    let mut output = String::new();
    append_stringified_value_async(&mut output, value).await?;
    Ok(output)
}

fn stringify_value(value: &Value) -> Result<String, RuntimeError> {
    if value_contains_projected(value) {
        futures_executor::block_on(stringify_value_async(value))
    } else {
        stringify_value_direct(value)
    }
}

fn stringify_value_blocking(value: &Value) -> Result<String, RuntimeError> {
    stringify_value(value)
}

fn stringify_value_direct(value: &Value) -> Result<String, RuntimeError> {
    let mut output = String::new();
    append_stringified_value_direct(&mut output, value)?;
    Ok(output)
}

fn append_stringified_value_direct(output: &mut String, value: &Value) -> Result<(), RuntimeError> {
    match value {
        Value::String(value) => output.push_str(value),
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            write_number(output, *value).expect("string writes should not fail")
        }
        Value::Projected(_) => unreachable!("projected values require async stringification"),
        Value::Image(_) | Value::List(_) | Value::Record(_) => output.push_str(
            &serde_json::to_string(&to_json_direct(value))
                .expect("value json serialization should succeed"),
        ),
    }
    Ok(())
}

fn append_stringified_value_async<'a>(
    output: &'a mut String,
    value: &'a Value,
) -> ProjectedFuture<'a, Result<(), RuntimeError>> {
    Box::pin(async move {
        match value {
            Value::String(value) => output.push_str(value),
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => {
                write_number(output, *value).expect("string writes should not fail")
            }
            Value::Projected(value) => output.push_str(&value.render().await),
            Value::Image(_) | Value::List(_) | Value::Record(_) => output.push_str(
                &serde_json::to_string(&to_json_async(value).await)
                    .expect("value json serialization should succeed"),
            ),
        }
        Ok(())
    })
}

#[cfg(test)]
fn append_stringified_value(output: &mut String, value: &Value) -> Result<(), RuntimeError> {
    futures_executor::block_on(append_stringified_value_async(output, value))
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Image(_) => "image",
        Value::List(_) => "list",
        Value::Record(_) => "record",
        Value::Projected(value) => value.value_type_name(),
    }
}

fn value_contains_projected(value: &Value) -> bool {
    match value {
        Value::Projected(_) => true,
        Value::List(values) => values.iter().any(value_contains_projected),
        Value::Record(record) => record.values().any(value_contains_projected),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Image(_) => {
            false
        }
    }
}

fn materialize_value(value: Value) -> Value {
    match value {
        Value::Projected(projected) => projected.materialize(),
        other => other,
    }
}

fn write_number(output: &mut impl fmt::Write, value: f64) -> fmt::Result {
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return write!(output, "{}", value as i64);
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return write!(output, "{}", value as u64);
        }
    }
    write!(output, "{value}")
}

fn resolve_index(index: &Value, len: usize) -> Result<Option<usize>, RuntimeError> {
    let index = as_offset(index)?;
    let len = len as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

fn resolve_existing_list_assignment_index(
    index: &Value,
    len: usize,
) -> Result<usize, RuntimeError> {
    let Value::Number(index) = index else {
        return Err(RuntimeError::TypeError {
            message: "list assignment index must be an integer".to_string(),
        });
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "list assignment index must be an integer".to_string(),
        });
    }
    let index = *index as isize;
    let len = len as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Err(RuntimeError::ValueError {
            message: "list assignment index out of bounds".to_string(),
        });
    }
    Ok(normalized as usize)
}

async fn apply_format_async(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    let mut output = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut last_literal = 0;
    let mut uses_sequential = false;
    let mut uses_indexed = false;
    let mut next_sequential = 0usize;
    let mut used_indexed_bits = 0u64;
    let mut used_indexed_args: Option<Vec<bool>> = None;
    while index < bytes.len() {
        match bytes[index] {
            b'{' if bytes.get(index + 1) == Some(&b'{') => {
                output.push_str(&template[last_literal..index]);
                output.push('{');
                index += 2;
                last_literal = index;
                continue;
            }
            b'{' => {
                output.push_str(&template[last_literal..index]);
                let mut cursor = index + 1;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }

                let (slot, slot_text) = if cursor >= bytes.len() {
                    return Err(RuntimeError::ValueError {
                        message: "unmatched `{` in format string".to_string(),
                    });
                } else if bytes[cursor] != b'}' {
                    return Err(RuntimeError::ValueError {
                        message: "invalid format placeholder".to_string(),
                    });
                } else if cursor == index + 1 {
                    if uses_indexed {
                        return Err(RuntimeError::ValueError {
                            message: "can't mix `{}` and indexed format placeholders".to_string(),
                        });
                    }
                    uses_sequential = true;
                    let slot = next_sequential;
                    next_sequential += 1;
                    (slot, None)
                } else {
                    if uses_sequential {
                        return Err(RuntimeError::ValueError {
                            message: "can't mix `{}` and indexed format placeholders".to_string(),
                        });
                    }
                    uses_indexed = true;
                    let digits = &template[index + 1..cursor];
                    let slot = digits
                        .parse::<usize>()
                        .map_err(|_| RuntimeError::ValueError {
                            message: format!("bad format slot `{digits}`"),
                        })?;
                    if slot < args.len() {
                        if args.len() <= u64::BITS as usize {
                            used_indexed_bits |= 1u64 << slot;
                        } else {
                            let used_args =
                                used_indexed_args.get_or_insert_with(|| vec![false; args.len()]);
                            used_args[slot] = true;
                        }
                    }
                    (slot, Some(digits))
                };

                let value = args.get(slot).ok_or_else(|| RuntimeError::ValueError {
                    message: match slot_text {
                        Some(slot_text) => format!("format slot `{slot_text}` is out of range"),
                        None => "format slot `{}` is out of range".to_string(),
                    },
                })?;
                append_stringified_value_async(&mut output, value).await?;
                index = cursor + 1;
                last_literal = index;
                continue;
            }
            b'}' if bytes.get(index + 1) == Some(&b'}') => {
                output.push_str(&template[last_literal..index]);
                output.push('}');
                index += 2;
                last_literal = index;
                continue;
            }
            b'}' => {
                return Err(RuntimeError::ValueError {
                    message: "unmatched `}` in format string".to_string(),
                });
            }
            _ => {}
        }
        index += 1;
    }
    output.push_str(&template[last_literal..]);
    let unused_index = if uses_sequential {
        (next_sequential < args.len()).then_some(next_sequential)
    } else if uses_indexed {
        if args.len() <= u64::BITS as usize {
            (0..args.len()).find(|slot| used_indexed_bits & (1u64 << slot) == 0)
        } else {
            used_indexed_args
                .as_ref()
                .and_then(|used_args| used_args.iter().position(|used| !*used))
        }
    } else {
        (!args.is_empty()).then_some(0)
    };
    if let Some(unused_index) = unused_index {
        return Err(RuntimeError::ValueError {
            message: format!("format argument `{unused_index}` is unused"),
        });
    }
    Ok(output)
}

#[cfg(test)]
fn apply_format(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    futures_executor::block_on(apply_format_async(template, args))
}

fn compile_format_template(template: &str, argc: usize) -> CompiledFormatTemplate {
    match parse_format_template(template, argc) {
        Ok(parts) => CompiledFormatTemplate {
            parts: parts.into_boxed_slice(),
            argc,
            min_capacity: template.len(),
            error: None,
        },
        Err(message) => CompiledFormatTemplate {
            parts: Box::new([]),
            argc,
            min_capacity: template.len(),
            error: Some(message),
        },
    }
}

fn parse_format_template(template: &str, argc: usize) -> Result<Vec<CompiledFormatPart>, String> {
    let mut parts = Vec::new();
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut last_literal = 0;
    let mut uses_sequential = false;
    let mut uses_indexed = false;
    let mut next_sequential = 0usize;
    let mut used_indexed_bits = 0u64;
    let mut used_indexed_args: Option<Vec<bool>> = None;

    while index < bytes.len() {
        match bytes[index] {
            b'{' if bytes.get(index + 1) == Some(&b'{') => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                push_format_literal(&mut parts, "{");
                index += 2;
                last_literal = index;
                continue;
            }
            b'{' => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                let mut cursor = index + 1;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }

                let (slot, slot_text) = if cursor >= bytes.len() {
                    return Err("unmatched `{` in format string".to_string());
                } else if bytes[cursor] != b'}' {
                    return Err("invalid format placeholder".to_string());
                } else if cursor == index + 1 {
                    if uses_indexed {
                        return Err("can't mix `{}` and indexed format placeholders".to_string());
                    }
                    uses_sequential = true;
                    let slot = next_sequential;
                    next_sequential += 1;
                    (slot, None)
                } else {
                    if uses_sequential {
                        return Err("can't mix `{}` and indexed format placeholders".to_string());
                    }
                    uses_indexed = true;
                    let digits = &template[index + 1..cursor];
                    let slot = digits
                        .parse::<usize>()
                        .map_err(|_| format!("bad format slot `{digits}`"))?;
                    if slot < argc {
                        if argc <= u64::BITS as usize {
                            used_indexed_bits |= 1u64 << slot;
                        } else {
                            let used_args =
                                used_indexed_args.get_or_insert_with(|| vec![false; argc]);
                            used_args[slot] = true;
                        }
                    }
                    (slot, Some(digits))
                };

                if slot >= argc {
                    return Err(match slot_text {
                        Some(slot_text) => format!("format slot `{slot_text}` is out of range"),
                        None => "format slot `{}` is out of range".to_string(),
                    });
                }
                parts.push(CompiledFormatPart::Arg(slot));
                index = cursor + 1;
                last_literal = index;
                continue;
            }
            b'}' if bytes.get(index + 1) == Some(&b'}') => {
                push_format_literal(&mut parts, &template[last_literal..index]);
                push_format_literal(&mut parts, "}");
                index += 2;
                last_literal = index;
                continue;
            }
            b'}' => {
                return Err("unmatched `}` in format string".to_string());
            }
            _ => {}
        }
        index += 1;
    }

    push_format_literal(&mut parts, &template[last_literal..]);
    let unused_index = if uses_sequential {
        (next_sequential < argc).then_some(next_sequential)
    } else if uses_indexed {
        if argc <= u64::BITS as usize {
            (0..argc).find(|slot| used_indexed_bits & (1u64 << slot) == 0)
        } else {
            used_indexed_args
                .as_ref()
                .and_then(|used_args| used_args.iter().position(|used| !*used))
        }
    } else {
        (argc > 0).then_some(0)
    };
    if let Some(unused_index) = unused_index {
        return Err(format!("format argument `{unused_index}` is unused"));
    }
    Ok(parts)
}

fn push_format_literal(parts: &mut Vec<CompiledFormatPart>, literal: &str) {
    if !literal.is_empty() {
        parts.push(CompiledFormatPart::Literal(Arc::<str>::from(literal)));
    }
}

async fn execute_compiled_format(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if let Some(message) = &template.error {
        return Err(RuntimeError::ValueError {
            message: message.clone(),
        });
    }

    let mut output = String::with_capacity(template.min_capacity);
    for part in template.parts.iter() {
        match part {
            CompiledFormatPart::Literal(literal) => output.push_str(literal),
            CompiledFormatPart::Arg(slot) => {
                let value = args.get(*slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot `{slot}` is out of range"),
                })?;
                append_stringified_value_async(&mut output, value).await?;
            }
        }
    }
    Ok(output)
}

fn execute_compiled_format_blocking(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if args
        .iter()
        .any(|value| matches!(value, Value::Projected(_)))
    {
        futures_executor::block_on(execute_compiled_format(template, args))
    } else {
        execute_compiled_format_direct(template, args)
    }
}

fn execute_compiled_format_direct(
    template: &CompiledFormatTemplate,
    args: &[Value],
) -> Result<String, RuntimeError> {
    if let Some(message) = &template.error {
        return Err(RuntimeError::ValueError {
            message: message.clone(),
        });
    }

    let mut output = String::with_capacity(template.min_capacity);
    for part in template.parts.iter() {
        match part {
            CompiledFormatPart::Literal(literal) => output.push_str(literal),
            CompiledFormatPart::Arg(slot) => {
                let value = args.get(*slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot `{slot}` is out of range"),
                })?;
                append_stringified_value_direct(&mut output, value)?;
            }
        }
    }
    Ok(output)
}

fn clamp_slice_bounds(
    start: Option<isize>,
    end: Option<isize>,
    len: usize,
) -> Option<(usize, usize)> {
    let len = len as isize;
    let start = normalize_slice_bound(start.unwrap_or(0), len);
    let end = normalize_slice_bound(end.unwrap_or(len), len);
    (start < end).then_some((start, end))
}

fn normalize_slice_bound(bound: isize, len: isize) -> usize {
    let normalized = if bound < 0 { len + bound } else { bound };
    normalized.clamp(0, len) as usize
}

fn slice_string(value: &str, start: Option<isize>, end: Option<isize>) -> String {
    let char_count = value.chars().count();
    let Some((start, end)) = clamp_slice_bounds(start, end, char_count) else {
        return String::new();
    };

    let mut indices = value
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(value.len()));
    let Some(byte_start) = indices.nth(start) else {
        return String::new();
    };
    let Some(byte_end) = indices.nth(end - start - 1) else {
        return value[byte_start..].to_string();
    };
    value[byte_start..byte_end].to_string()
}

fn json_number(value: f64) -> Option<serde_json::Number> {
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

fn to_json_async<'a>(value: &'a Value) -> ProjectedFuture<'a, serde_json::Value> {
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

fn to_json(value: &Value) -> serde_json::Value {
    if value_contains_projected(value) {
        futures_executor::block_on(to_json_async(value))
    } else {
        to_json_direct(value)
    }
}

fn to_json_blocking(value: &Value) -> serde_json::Value {
    to_json(value)
}

fn to_json_direct(value: &Value) -> serde_json::Value {
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

fn image_to_json(image: &ImageValue) -> serde_json::Value {
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

fn from_json(value: serde_json::Value) -> Value {
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

fn image_from_json_map(map: &serde_json::Map<String, serde_json::Value>) -> Option<ImageValue> {
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

fn optional_u32_field(value: &serde_json::Value) -> Option<Option<u32>> {
    match value {
        serde_json::Value::Null => Some(None),
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Some),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
