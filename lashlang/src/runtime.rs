use crate::ast::{BinaryOp, CallExpr, Expr, Program, Stmt, UnaryOp};
use compact_str::CompactString;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::ops::Index;
use std::rc::Rc;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;
use thiserror::Error;

const RECORD_INDEX_THRESHOLD: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Symbol(u32);

#[derive(Default)]
struct SymbolTable {
    lookup: FxHashMap<Arc<str>, Symbol>,
    names: Vec<Arc<str>>,
}

fn symbol_table() -> &'static RwLock<SymbolTable> {
    static TABLE: OnceLock<RwLock<SymbolTable>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(SymbolTable::default()))
}

fn lookup_symbol(name: &str) -> Option<Symbol> {
    symbol_table()
        .read()
        .expect("symbol table read lock poisoned")
        .lookup
        .get(name)
        .copied()
}

fn intern_symbol(name: &str) -> Symbol {
    if let Some(symbol) = lookup_symbol(name) {
        return symbol;
    }

    let mut table = symbol_table()
        .write()
        .expect("symbol table write lock poisoned");
    if let Some(symbol) = table.lookup.get(name) {
        return *symbol;
    }

    let symbol = Symbol(table.names.len() as u32);
    let text: Arc<str> = Arc::<str>::from(name);
    table.names.push(text.clone());
    table.lookup.insert(text, symbol);
    symbol
}

fn symbol_name(symbol: Symbol) -> Arc<str> {
    symbol_table()
        .read()
        .expect("symbol table read lock poisoned")
        .names[symbol.0 as usize]
        .clone()
}

#[derive(Clone, Debug, PartialEq)]
struct RecordEntry {
    symbol: Symbol,
    name: Arc<str>,
    value: Value,
}

#[derive(Clone, Debug, Default)]
pub struct Record {
    entries: Vec<RecordEntry>,
    index: Option<FxHashMap<Symbol, usize>>,
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            index: (capacity > RECORD_INDEX_THRESHOLD)
                .then(|| FxHashMap::with_capacity_and_hasher(capacity, Default::default())),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.get_symbol(lookup_symbol(name)?)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Value> {
        let symbol = lookup_symbol(name)?;
        let index = self.position_for(symbol)?;
        Some(&mut self.entries[index].value)
    }

    pub fn remove(&mut self, name: &str) -> Option<Value> {
        let symbol = lookup_symbol(name)?;
        self.remove_symbol(symbol)
    }

    pub fn insert(&mut self, name: String, value: Value) -> Option<Value> {
        let symbol = intern_symbol(&name);
        self.insert_symbolized(symbol, Arc::<str>::from(name), value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|entry| (entry.name.as_ref(), &entry.value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.name.as_ref())
    }

    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.entries.iter().map(|entry| &entry.value)
    }

    fn get_symbol(&self, symbol: Symbol) -> Option<&Value> {
        let index = self.position_for(symbol)?;
        Some(&self.entries[index].value)
    }

    fn insert_symbol(&mut self, symbol: Symbol, value: Value) -> Option<Value> {
        self.insert_symbolized(symbol, symbol_name(symbol), value)
    }

    fn insert_symbolized(&mut self, symbol: Symbol, name: Arc<str>, value: Value) -> Option<Value> {
        if let Some(index) = self.position_for(symbol) {
            return Some(std::mem::replace(&mut self.entries[index].value, value));
        }

        let index = self.entries.len();
        self.entries.push(RecordEntry {
            symbol,
            name,
            value,
        });
        self.reindex_after_insert(index);
        None
    }

    fn remove_symbol(&mut self, symbol: Symbol) -> Option<Value> {
        let index = self.position_for(symbol)?;
        let removed = self.entries.swap_remove(index);
        self.reindex_after_remove(symbol, index);
        Some(removed.value)
    }

    fn position_for(&self, symbol: Symbol) -> Option<usize> {
        if let Some(index) = &self.index {
            return index.get(&symbol).copied();
        }
        self.entries.iter().position(|entry| entry.symbol == symbol)
    }

    fn rebuild_index(&mut self) {
        self.index = (self.entries.len() > RECORD_INDEX_THRESHOLD).then(|| {
            let mut index =
                FxHashMap::with_capacity_and_hasher(self.entries.len(), Default::default());
            for (slot, entry) in self.entries.iter().enumerate() {
                index.insert(entry.symbol, slot);
            }
            index
        });
    }

    fn reindex_after_insert(&mut self, index: usize) {
        if let Some(map) = &mut self.index {
            map.insert(self.entries[index].symbol, index);
            return;
        }
        if self.entries.len() > RECORD_INDEX_THRESHOLD {
            self.rebuild_index();
        }
    }

    fn reindex_after_remove(&mut self, removed: Symbol, index: usize) {
        if self.entries.len() <= RECORD_INDEX_THRESHOLD {
            self.index = None;
            return;
        }

        let Some(map) = &mut self.index else {
            self.rebuild_index();
            return;
        };
        map.remove(&removed);
        if let Some(moved) = self.entries.get(index) {
            map.insert(moved.symbol, index);
        }
    }
}

impl Index<&str> for Record {
    type Output = Value;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("missing record key `{name}`"))
    }
}

impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.entries.iter().all(|entry| {
            other
                .get_symbol(entry.symbol)
                .is_some_and(|value| value == &entry.value)
        })
    }
}

impl FromIterator<(String, Value)> for Record {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut record = Record::with_capacity(lower);
        for (name, value) in iter {
            record.insert(name, value);
        }
        record
    }
}

impl Serialize for Record {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for entry in &self.entries {
            map.serialize_entry(entry.name.as_ref(), &entry.value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Record {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map = FxHashMap::<String, Value>::deserialize(deserializer)?;
        Ok(map.into_iter().collect())
    }
}

fn record_with_capacity(capacity: usize) -> Record {
    Record::with_capacity(capacity)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(CompactString),
    List(Arc<[Value]>),
    Record(Arc<Record>),
}

impl Value {
    pub fn as_record(&self) -> Option<&Record> {
        match self {
            Self::Record(record) => Some(record.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Number(value) => write!(f, "{value}"),
            Self::String(value) => write!(f, "{value}"),
            Self::List(values) => {
                write!(f, "{}", serde_json::to_string(values).unwrap_or_default())
            }
            Self::Record(record) => {
                write!(f, "{}", serde_json::to_string(record).unwrap_or_default())
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct State {
    globals: Record,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub globals: Record,
}

pub trait ToolHost: Sync {
    fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError>;

    fn observe(&self, _value: &Value) -> Result<(), ToolHostError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ToolHostError {
    message: String,
}

impl ToolHostError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
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

pub fn execute_program<H: ToolHost>(
    program: &Program,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let compiled = compile_program(program);
    execute_compiled(&compiled, state, host)
}

#[derive(Clone)]
pub struct CompiledProgram {
    chunk: Chunk,
}

pub fn compile_program(program: &Program) -> CompiledProgram {
    CompiledProgram {
        chunk: Compiler::compile_program(program),
    }
}

pub fn execute_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<ExecutionOutcome, RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
        ),
        host,
        false,
    );
    let result = vm.run();
    state.globals = vm.into_globals();
    result
}

pub fn profile_compiled<H: ToolHost>(
    program: &CompiledProgram,
    state: &mut State,
    host: &H,
) -> Result<(ExecutionOutcome, ProfileReport), RuntimeError> {
    let mut vm = Vm::new(
        &program.chunk,
        SlotState::from_globals(
            std::mem::take(&mut state.globals),
            &program.chunk.slot_names,
        ),
        host,
        false,
    );
    vm.enable_profile();
    let result = vm.run();
    let profile = vm.take_profile();
    state.globals = vm.into_globals();
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
}

impl ProfileReport {
    pub fn instruction_stats(&self) -> &[ProfileStat] {
        &self.instruction_stats
    }

    pub fn builtin_stats(&self) -> &[ProfileStat] {
        &self.builtin_stats
    }

    pub fn merge(&mut self, other: &Self) {
        merge_stats(&mut self.instruction_stats, &other.instruction_stats);
        merge_stats(&mut self.builtin_stats, &other.builtin_stats);
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
    constants: Vec<Value>,
    names: Vec<Name>,
    slot_names: Vec<String>,
    key_lists: Vec<Box<[usize]>>,
    parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    branch_sets: Vec<Box<[Chunk]>>,
}

#[derive(Clone)]
struct Name {
    symbol: Symbol,
    text: Arc<str>,
}

#[derive(Clone, Copy)]
enum Instruction {
    PushConst(usize),
    LoadName(usize),
    StoreName(usize),
    BuildList(usize),
    BuildRecord(usize),
    Field(usize),
    Index,
    Unary(UnaryOp),
    Binary(BinaryOp),
    ToBool,
    Jump(usize),
    JumpIfFalse(usize),
    JumpIfTrue(usize),
    CallTool { name: usize, keys: usize },
    CallBuiltin { builtin: Builtin, argc: usize },
    AddAssign(usize),
    AppendAssign(usize),
    Observe,
    Finish,
    Pop,
    BeginIter(usize),
    IterNext { jump_to: usize },
    EndIter,
    ParallelCalls(usize),
    Parallel(usize),
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
    Unknown(usize),
}

impl Instruction {
    fn profile_tag(self) -> InstructionProfileTag {
        match self {
            Instruction::PushConst(_) => InstructionProfileTag::PushConst,
            Instruction::LoadName(_) => InstructionProfileTag::LoadName,
            Instruction::StoreName(_) => InstructionProfileTag::StoreName,
            Instruction::BuildList(_) => InstructionProfileTag::BuildList,
            Instruction::BuildRecord(_) => InstructionProfileTag::BuildRecord,
            Instruction::Field(_) => InstructionProfileTag::Field,
            Instruction::Index => InstructionProfileTag::Index,
            Instruction::Unary(_) => InstructionProfileTag::Unary,
            Instruction::Binary(_) => InstructionProfileTag::Binary,
            Instruction::ToBool => InstructionProfileTag::ToBool,
            Instruction::Jump(_) => InstructionProfileTag::Jump,
            Instruction::JumpIfFalse(_) => InstructionProfileTag::JumpIfFalse,
            Instruction::JumpIfTrue(_) => InstructionProfileTag::JumpIfTrue,
            Instruction::CallTool { .. } => InstructionProfileTag::CallTool,
            Instruction::CallBuiltin { .. } => InstructionProfileTag::CallBuiltin,
            Instruction::AddAssign(_) => InstructionProfileTag::AddAssign,
            Instruction::AppendAssign(_) => InstructionProfileTag::AppendAssign,
            Instruction::Observe => InstructionProfileTag::Observe,
            Instruction::Finish => InstructionProfileTag::Finish,
            Instruction::Pop => InstructionProfileTag::Pop,
            Instruction::BeginIter(_) => InstructionProfileTag::BeginIter,
            Instruction::IterNext { .. } => InstructionProfileTag::IterNext,
            Instruction::EndIter => InstructionProfileTag::EndIter,
            Instruction::ParallelCalls(_) => InstructionProfileTag::Parallel,
            Instruction::Parallel(_) => InstructionProfileTag::Parallel,
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
    Unary,
    Binary,
    ToBool,
    Jump,
    JumpIfFalse,
    JumpIfTrue,
    CallTool,
    CallBuiltin,
    AddAssign,
    AppendAssign,
    Observe,
    Finish,
    Pop,
    BeginIter,
    IterNext,
    EndIter,
    Parallel,
}

const INSTRUCTION_PROFILE_COUNT: usize = InstructionProfileTag::Parallel as usize + 1;

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
    "unary",
    "binary",
    "to_bool",
    "jump",
    "jump_if_false",
    "jump_if_true",
    "call_tool",
    "call_builtin",
    "add_assign",
    "append_assign",
    "observe",
    "finish",
    "pop",
    "begin_iter",
    "iter_next",
    "end_iter",
    "parallel",
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
    constants: Vec<Value>,
    names: Vec<Name>,
    name_lookup: FxHashMap<Symbol, usize>,
    slots: Rc<RefCell<SlotTable>>,
    key_lists: Vec<Box<[usize]>>,
    parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    branch_sets: Vec<Box<[Chunk]>>,
}

#[derive(Default)]
struct SlotTable {
    names: Vec<String>,
    lookup: FxHashMap<String, usize>,
}

#[derive(Clone)]
struct ParallelCallBranch {
    slot: usize,
    name: usize,
    args: PureExpr,
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
    fn compile_program(program: &Program) -> Chunk {
        let mut compiler = Self::new();
        compiler.compile_block(&program.statements);
        compiler.finish()
    }

    fn new() -> Self {
        Self::with_slots(Rc::new(RefCell::new(SlotTable::default())))
    }

    fn with_slots(slots: Rc<RefCell<SlotTable>>) -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            name_lookup: FxHashMap::default(),
            slots,
            key_lists: Vec::new(),
            parallel_call_sets: Vec::new(),
            branch_sets: Vec::new(),
        }
    }

    fn finish(self) -> Chunk {
        let slot_names = self.slots.borrow().names.clone();
        Chunk {
            code: self.code,
            constants: self.constants,
            names: self.names,
            slot_names,
            key_lists: self.key_lists,
            parallel_call_sets: self.parallel_call_sets,
            branch_sets: self.branch_sets,
        }
    }

    fn push_const(&mut self, value: Value) -> usize {
        let index = self.constants.len();
        self.constants.push(value);
        index
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
        let mut slots = self.slots.borrow_mut();
        if let Some(index) = slots.lookup.get(name) {
            return *index;
        }
        let index = slots.names.len();
        let owned = name.to_string();
        slots.names.push(owned.clone());
        slots.lookup.insert(owned, index);
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

    fn push_branch_set(&mut self, branches: Vec<Chunk>) -> usize {
        let index = self.branch_sets.len();
        self.branch_sets.push(branches.into_boxed_slice());
        index
    }

    fn push_parallel_call_set(&mut self, branches: Vec<ParallelCallBranch>) -> usize {
        let index = self.parallel_call_sets.len();
        self.parallel_call_sets.push(branches.into_boxed_slice());
        index
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
            _ => Builtin::Unknown(self.push_name(name)),
        }
    }

    fn compile_block(&mut self, statements: &[Stmt]) {
        for statement in statements {
            self.compile_stmt(statement);
        }
    }

    fn compile_stmt(&mut self, statement: &Stmt) {
        match statement {
            Stmt::Assign { name, expr } => {
                let slot = self.push_slot(name);
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
                        return;
                    }
                    self.compile_expr(right);
                    self.code.push(Instruction::AddAssign(slot));
                    return;
                }

                self.compile_expr(expr);
                self.code.push(Instruction::StoreName(slot));
            }
            Stmt::Call(call) => {
                self.compile_call_expr(call);
                self.code.push(Instruction::Pop);
            }
            Stmt::Observe(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Observe);
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                self.compile_expr(condition);
                let jump_to_else = self.emit_jump_if_false();
                self.compile_block(then_block);
                if else_block.is_empty() {
                    self.patch_jump(jump_to_else, self.code.len());
                } else {
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_else, self.code.len());
                    self.compile_block(else_block);
                    self.patch_jump(jump_to_end, self.code.len());
                }
            }
            Stmt::For {
                binding,
                iterable,
                body,
            } => {
                self.compile_expr(iterable);
                let binding = self.push_slot(binding);
                self.code.push(Instruction::BeginIter(binding));
                let loop_start = self.code.len();
                let iter_next = self.code.len();
                self.code.push(Instruction::IterNext {
                    jump_to: usize::MAX,
                });
                self.compile_block(body);
                self.code.push(Instruction::Jump(loop_start));
                let loop_end = self.code.len();
                self.code.push(Instruction::EndIter);
                self.patch_jump(iter_next, loop_end);
            }
            Stmt::Parallel { branches } => {
                if let Some(branches) = self.compile_parallel_calls(branches) {
                    let branches = self.push_parallel_call_set(branches);
                    self.code.push(Instruction::ParallelCalls(branches));
                    return;
                }

                let branches = branches
                    .iter()
                    .map(|branch| {
                        let mut compiler = Self::with_slots(self.slots.clone());
                        compiler.compile_stmt(branch);
                        compiler.finish()
                    })
                    .collect::<Vec<_>>();
                let branches = self.push_branch_set(branches);
                self.code.push(Instruction::Parallel(branches));
            }
            Stmt::Finish(expr) => {
                self.compile_expr(expr);
                self.code.push(Instruction::Finish);
            }
        }
    }

    fn compile_parallel_calls(&mut self, branches: &[Stmt]) -> Option<Vec<ParallelCallBranch>> {
        let mut compiled = Vec::with_capacity(branches.len());
        for branch in branches {
            let Stmt::Assign { name, expr } = branch else {
                return None;
            };
            let Expr::ToolCall(call) = expr else {
                return None;
            };
            if call.args.iter().any(|(_, expr)| !is_pure_expr(expr)) {
                return None;
            }

            let slot = self.push_slot(name);
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

    fn compile_pure_expr(&mut self, expr: &Expr) -> Result<PureExpr, RuntimeError> {
        match expr {
            Expr::Null => Ok(PureExpr::Const(Value::Null)),
            Expr::Bool(value) => Ok(PureExpr::Const(Value::Bool(*value))),
            Expr::Number(value) => Ok(PureExpr::Const(Value::Number(*value))),
            Expr::String(value) => Ok(PureExpr::Const(Value::String(value.clone().into()))),
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
            Expr::BuiltinCall { name, args } => Ok(PureExpr::Builtin {
                builtin: self.resolve_builtin(name),
                args: args
                    .iter()
                    .map(|arg| self.compile_pure_expr(arg))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            }),
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
        }
    }

    fn compile_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Null => {
                let value = self.push_const(Value::Null);
                self.code.push(Instruction::PushConst(value));
            }
            Expr::Bool(value) => {
                let value = self.push_const(Value::Bool(*value));
                self.code.push(Instruction::PushConst(value));
            }
            Expr::Number(value) => {
                let value = self.push_const(Value::Number(*value));
                self.code.push(Instruction::PushConst(value));
            }
            Expr::String(value) => {
                let value = self.push_const(Value::String(value.clone().into()));
                self.code.push(Instruction::PushConst(value));
            }
            Expr::Variable(name) => {
                let name = self.push_slot(name);
                self.code.push(Instruction::LoadName(name));
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
            Expr::BuiltinCall { name, args } => {
                for arg in args {
                    self.compile_expr(arg);
                }
                let builtin = self.resolve_builtin(name);
                self.code.push(Instruction::CallBuiltin {
                    builtin,
                    argc: args.len(),
                });
            }
            Expr::Field { target, field } => {
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
                self.compile_expr(condition);
                let jump_to_else = self.emit_jump_if_false();
                self.compile_expr(then_expr);
                let jump_to_end = self.emit_jump();
                self.patch_jump(jump_to_else, self.code.len());
                self.compile_expr(else_expr);
                self.patch_jump(jump_to_end, self.code.len());
            }
            Expr::Binary { left, op, right } => match op {
                BinaryOp::And => {
                    self.compile_expr(left);
                    let jump_to_false = self.emit_jump_if_false();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_false, self.code.len());
                    let value = self.push_const(Value::Bool(false));
                    self.code.push(Instruction::PushConst(value));
                    self.patch_jump(jump_to_end, self.code.len());
                }
                BinaryOp::Or => {
                    self.compile_expr(left);
                    let jump_to_true = self.emit_jump_if_true();
                    self.compile_expr(right);
                    self.code.push(Instruction::ToBool);
                    let jump_to_end = self.emit_jump();
                    self.patch_jump(jump_to_true, self.code.len());
                    let value = self.push_const(Value::Bool(true));
                    self.code.push(Instruction::PushConst(value));
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

    fn emit_jump_if_false(&mut self) -> usize {
        let index = self.code.len();
        self.code.push(Instruction::JumpIfFalse(usize::MAX));
        index
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

    fn patch_jump(&mut self, index: usize, target: usize) {
        match &mut self.code[index] {
            Instruction::Jump(slot)
            | Instruction::JumpIfFalse(slot)
            | Instruction::JumpIfTrue(slot)
            | Instruction::IterNext { jump_to: slot } => *slot = target,
            _ => unreachable!("patched non-jump instruction"),
        }
    }
}

#[derive(Clone)]
struct SlotState {
    values: Vec<Option<Value>>,
    extras: Record,
}

impl SlotState {
    fn from_globals(mut globals: Record, slot_names: &[String]) -> Self {
        let mut values = Vec::with_capacity(slot_names.len());
        for name in slot_names {
            values.push(globals.remove(name));
        }
        Self {
            values,
            extras: globals,
        }
    }

    fn get(&self, slot: usize) -> Option<&Value> {
        self.values.get(slot).and_then(Option::as_ref)
    }

    fn assign(&mut self, slot: usize, value: Value) {
        self.values[slot] = Some(value);
    }

    fn capture_temporary(&self, slot: usize) -> LoopRestore {
        LoopRestore {
            previous: self.values[slot].clone(),
        }
    }

    fn restore_temporary(&mut self, slot: usize, restore: LoopRestore) {
        self.values[slot] = restore.previous;
    }

    fn into_globals(self, slot_names: &[String]) -> Record {
        let mut extras = self.extras;
        for (name, value) in slot_names.iter().zip(self.values) {
            match value {
                Some(value) => {
                    extras.insert(name.clone(), value);
                }
                None => {
                    extras.remove(name);
                }
            }
        }
        extras
    }
}

struct Vm<'a, H> {
    chunk: &'a Chunk,
    ip: usize,
    stack: Vec<Value>,
    slots: SlotState,
    host: &'a H,
    in_parallel_branch: bool,
    assigned: Option<Vec<bool>>,
    iter_stack: Vec<IterState>,
    profile: Option<ProfileAccumulator>,
}

impl<'a, H: ToolHost> Vm<'a, H> {
    fn new(chunk: &'a Chunk, slots: SlotState, host: &'a H, in_parallel_branch: bool) -> Self {
        Self {
            chunk,
            ip: 0,
            stack: Vec::new(),
            slots,
            host,
            in_parallel_branch,
            assigned: in_parallel_branch.then(|| vec![false; chunk.slot_names.len()]),
            iter_stack: Vec::new(),
            profile: None,
        }
    }

    fn enable_profile(&mut self) {
        self.profile = Some(ProfileAccumulator::default());
    }

    fn run(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        let result = self.run_inner();
        self.unwind_iterators();
        result
    }

    fn run_inner(&mut self) -> Result<ExecutionOutcome, RuntimeError> {
        while let Some(instruction) = self.chunk.code.get(self.ip).copied() {
            self.ip += 1;
            let tag = instruction.profile_tag();
            let start = self.profile.as_ref().map(|_| Instant::now());
            let result = self.execute_instruction(instruction);
            if let Some(start) = start {
                self.record_instruction_profile(tag, start.elapsed().as_nanos());
            }
            if let Some(outcome) = result? {
                return Ok(outcome);
            }
        }
        Ok(ExecutionOutcome::Continued)
    }

    fn execute_instruction(
        &mut self,
        instruction: Instruction,
    ) -> Result<Option<ExecutionOutcome>, RuntimeError> {
        match instruction {
            Instruction::PushConst(index) => {
                self.stack.push(self.chunk.constants[index].clone());
            }
            Instruction::LoadName(name) => {
                let slot_name = &self.chunk.slot_names[name];
                let value = self.slots.get(name).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.clone(),
                    }
                })?;
                self.stack.push(value);
            }
            Instruction::StoreName(name) => {
                let value = self.pop_stack()?;
                self.slots.assign(name, value);
                self.record_assignment(name);
            }
            Instruction::BuildList(len) => {
                let values = self.pop_n(len)?;
                self.stack.push(Value::List(values.into()));
            }
            Instruction::BuildRecord(keys) => {
                let key_indices = &self.chunk.key_lists[keys];
                let values = self.stack_tail(key_indices.len())?;
                let mut record = record_with_capacity(key_indices.len());
                for (key, value) in key_indices.iter().zip(values.iter()) {
                    record.insert_symbol(self.chunk.names[*key].symbol, value.clone());
                }
                self.stack.truncate(self.stack.len() - key_indices.len());
                self.stack.push(Value::Record(Arc::new(record)));
            }
            Instruction::Field(field) => {
                let target = self.pop_stack()?;
                self.stack
                    .push(read_field(target, &self.chunk.names[field])?);
            }
            Instruction::Index => {
                let index = self.pop_stack()?;
                let target = self.pop_stack()?;
                self.stack.push(read_index(target, index)?);
            }
            Instruction::Unary(op) => {
                let value = self.pop_stack()?;
                let value = match op {
                    UnaryOp::Negate => Value::Number(-as_number(&value)?),
                    UnaryOp::Not => Value::Bool(!is_truthy(&value)),
                };
                self.stack.push(value);
            }
            Instruction::Binary(op) => {
                let right = self.pop_stack()?;
                let left = self.pop_stack()?;
                self.stack.push(eval_binary_values(left, op, right)?);
            }
            Instruction::ToBool => {
                let value = self.pop_stack()?;
                self.stack.push(Value::Bool(is_truthy(&value)));
            }
            Instruction::Jump(target) => self.ip = target,
            Instruction::JumpIfFalse(target) => {
                let value = self.pop_stack()?;
                if !is_truthy(&value) {
                    self.ip = target;
                }
            }
            Instruction::JumpIfTrue(target) => {
                let value = self.pop_stack()?;
                if is_truthy(&value) {
                    self.ip = target;
                }
            }
            Instruction::CallTool { name, keys } => {
                let key_indices = &self.chunk.key_lists[keys];
                let values = self.stack_tail(key_indices.len())?;
                let mut args = record_with_capacity(key_indices.len());
                for (key, value) in key_indices.iter().zip(values.iter()) {
                    args.insert_symbol(self.chunk.names[*key].symbol, value.clone());
                }
                self.stack.truncate(self.stack.len() - key_indices.len());
                let result = match self.host.call(self.chunk.names[name].text.as_ref(), &args) {
                    Ok(value) => success(value),
                    Err(error) => error_value(error.to_string()),
                };
                self.stack.push(result);
            }
            Instruction::CallBuiltin { builtin, argc } => {
                let values = self.stack_tail(argc)?;
                let start = self.profile.as_ref().map(|_| Instant::now());
                let value = execute_builtin(builtin, &self.chunk.names, values)?;
                if let Some(start) = start {
                    self.record_builtin_profile(builtin, start.elapsed().as_nanos());
                }
                self.stack.truncate(self.stack.len() - argc);
                self.stack.push(value);
            }
            Instruction::AddAssign(slot) => {
                let right = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                let left = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.clone(),
                    }
                })?;
                self.slots.assign(slot, add_values(left, right)?);
                self.record_assignment(slot);
            }
            Instruction::AppendAssign(slot) => {
                let item = self.pop_stack()?;
                let slot_name = &self.chunk.slot_names[slot];
                let current = self.slots.get(slot).cloned().ok_or_else(|| {
                    RuntimeError::UndefinedVariable {
                        name: slot_name.clone(),
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
                self.slots.assign(slot, value);
                self.record_assignment(slot);
            }
            Instruction::Observe => {
                let value = self.pop_stack()?;
                self.host
                    .observe(&value)
                    .map_err(|err| RuntimeError::ValueError {
                        message: format!("observe failed: {err}"),
                    })?;
            }
            Instruction::Finish => {
                if self.in_parallel_branch {
                    return Err(RuntimeError::FinishInsideParallel);
                }
                return Ok(Some(ExecutionOutcome::Finished(self.pop_stack()?)));
            }
            Instruction::Pop => {
                let _ = self.pop_stack()?;
            }
            Instruction::BeginIter(binding) => {
                let iterable = self.pop_stack()?;
                let Value::List(values) = iterable else {
                    return Err(RuntimeError::NonListIteration);
                };
                self.iter_stack.push(IterState {
                    values,
                    index: 0,
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
                if iter_state.index >= iter_state.values.len() {
                    self.ip = jump_to;
                } else {
                    let value = iter_state.values[iter_state.index].clone();
                    iter_state.index += 1;
                    self.slots.assign(iter_state.binding, value);
                }
            }
            Instruction::EndIter => {
                if let Some(iter_state) = self.iter_stack.pop() {
                    self.slots
                        .restore_temporary(iter_state.binding, iter_state.restore);
                }
            }
            Instruction::ParallelCalls(branches) => {
                self.exec_parallel_calls(branches)?;
            }
            Instruction::Parallel(branches) => {
                self.exec_parallel(branches)?;
            }
        }
        Ok(None)
    }

    fn exec_parallel(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.branch_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }
        if branches.len() == 1 {
            let result = Self::run_branch(&branches[0], self.slots.clone(), self.host, true)?;
            self.merge_branch_result(result, &mut vec![false; self.chunk.slot_names.len()])?;
            return Ok(());
        }

        let base_slots = self.slots.clone();
        let results = branches
            .par_iter()
            .map(|branch| Self::run_branch(branch, base_slots.clone(), self.host, true))
            .collect::<Vec<_>>();

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    fn exec_parallel_calls(&mut self, branches_index: usize) -> Result<(), RuntimeError> {
        let branches = &self.chunk.parallel_call_sets[branches_index];
        if branches.is_empty() {
            return Ok(());
        }

        let mut calls = Vec::with_capacity(branches.len());
        for branch in branches {
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
            calls.push(PreparedParallelCall {
                slot: branch.slot,
                name: branch.name,
                args,
            });
        }

        let results = if calls.len() == 1 {
            vec![Self::run_prepared_call(self.chunk, &calls[0], self.host)]
        } else if calls.len() == 2 {
            let (left, right) = rayon::join(
                || Self::run_prepared_call(self.chunk, &calls[0], self.host),
                || Self::run_prepared_call(self.chunk, &calls[1], self.host),
            );
            vec![left, right]
        } else {
            calls
                .par_iter()
                .map(|call| Self::run_prepared_call(self.chunk, call, self.host))
                .collect()
        };

        let mut merged_names = vec![false; self.chunk.slot_names.len()];
        for result in results {
            self.merge_branch_result(result?, &mut merged_names)?;
        }
        Ok(())
    }

    fn run_branch(
        chunk: &'a Chunk,
        slots: SlotState,
        host: &'a H,
        in_parallel_branch: bool,
    ) -> Result<BranchResult, RuntimeError> {
        let mut vm = Self::new(chunk, slots, host, in_parallel_branch);
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vm.run()));
        match run {
            Ok(Ok(ExecutionOutcome::Continued)) => Ok(vm.into_branch_result()),
            Ok(Ok(ExecutionOutcome::Finished(_))) => Err(RuntimeError::FinishInsideParallel),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    fn run_prepared_call(
        chunk: &'a Chunk,
        call: &PreparedParallelCall,
        host: &'a H,
    ) -> Result<BranchResult, RuntimeError> {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match host.call(chunk.names[call.name].text.as_ref(), call.args.as_ref()) {
                Ok(value) => Ok(success(value)),
                Err(error) => Ok(error_value(error.to_string())),
            }
        }));
        match run {
            Ok(Ok(value)) => Ok(BranchResult {
                values: vec![(call.slot, value)],
            }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::ValueError {
                message: "parallel branch panicked".to_string(),
            }),
        }
    }

    fn merge_branch_result(
        &mut self,
        result: BranchResult,
        merged_names: &mut [bool],
    ) -> Result<(), RuntimeError> {
        for (slot, value) in result.values {
            if std::mem::replace(&mut merged_names[slot], true) {
                return Err(RuntimeError::ParallelConflict {
                    name: self.chunk.slot_names[slot].clone(),
                });
            }
            self.slots.assign(slot, value);
            self.record_assignment(slot);
        }
        Ok(())
    }

    fn pop_stack(&mut self) -> Result<Value, RuntimeError> {
        self.stack.pop().ok_or_else(|| RuntimeError::ValueError {
            message: "vm stack underflow".to_string(),
        })
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
        BranchResult { values }
    }

    fn into_globals(self) -> Record {
        self.slots.into_globals(&self.chunk.slot_names)
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
}

struct PreparedParallelCall {
    slot: usize,
    name: usize,
    args: Arc<Record>,
}

struct IterState {
    values: Arc<[Value]>,
    index: usize,
    binding: usize,
    restore: LoopRestore,
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
    }
}

fn eval_pure_expr(
    expr: &PureExpr,
    slots: &SlotState,
    names: &[Name],
    slot_names: &[String],
) -> Result<Value, RuntimeError> {
    match expr {
        PureExpr::Const(value) => Ok(value.clone()),
        PureExpr::Slot(slot) => {
            slots
                .get(*slot)
                .cloned()
                .ok_or_else(|| RuntimeError::UndefinedVariable {
                    name: slot_names[*slot].clone(),
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
                record.insert_symbol(
                    names[*key].symbol,
                    eval_pure_expr(value, slots, names, slot_names)?,
                );
            }
            Ok(Value::Record(Arc::new(record)))
        }
        PureExpr::Builtin { builtin, args } => {
            let values = args
                .iter()
                .map(|arg| eval_pure_expr(arg, slots, names, slot_names))
                .collect::<Result<Vec<_>, _>>()?;
            execute_builtin(*builtin, names, &values)
        }
        PureExpr::Field { target, field } => {
            let value = eval_pure_expr(target, slots, names, slot_names)?;
            read_field(value, &names[*field])
        }
        PureExpr::Index { target, index } => {
            let target = eval_pure_expr(target, slots, names, slot_names)?;
            let index = eval_pure_expr(index, slots, names, slot_names)?;
            read_index(target, index)
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

fn execute_builtin(
    builtin: Builtin,
    names: &[Name],
    values: &[Value],
) -> Result<Value, RuntimeError> {
    match builtin {
        Builtin::Len => {
            expect_arg_count("len", values, 1)?;
            match &values[0] {
                Value::String(value) => Ok(Value::Number(value.chars().count() as f64)),
                Value::List(values) => Ok(Value::Number(values.len() as f64)),
                Value::Record(record) => Ok(Value::Number(record.len() as f64)),
                Value::Null => Ok(Value::Number(0.0)),
                _ => Err(RuntimeError::TypeError {
                    message: "`len` requires a string, list, record, or null".to_string(),
                }),
            }
        }
        Builtin::Empty => {
            expect_arg_count("empty", values, 1)?;
            match &values[0] {
                Value::String(value) => Ok(Value::Bool(value.is_empty())),
                Value::List(values) => Ok(Value::Bool(values.is_empty())),
                Value::Record(record) => Ok(Value::Bool(record.is_empty())),
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
            match (&values[0], &values[1]) {
                (Value::String(haystack), needle) => Ok(Value::Bool(
                    haystack.contains(coerce_string(needle)?.as_ref()),
                )),
                (Value::List(items), needle) => Ok(Value::Bool(items.contains(needle))),
                (Value::Null, _) => Ok(Value::Bool(false)),
                _ => Err(RuntimeError::TypeError {
                    message: "`contains` requires a string/string, list/value, or null/value pair"
                        .to_string(),
                }),
            }
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
            let Value::List(items) = &values[0] else {
                return Err(RuntimeError::TypeError {
                    message: "`join` requires a list as the first argument".to_string(),
                });
            };
            let sep = coerce_string(&values[1])?;
            let mut joined = String::new();
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    joined.push_str(sep.as_ref());
                }
                joined.push_str(coerce_string(item)?.as_ref());
            }
            Ok(Value::String(joined.into()))
        }
        Builtin::Trim => {
            expect_arg_count("trim", values, 1)?;
            Ok(Value::String(
                coerce_string(&values[0])?.trim().to_string().into(),
            ))
        }
        Builtin::Slice => {
            expect_arg_count("slice", values, 3)?;
            let start = as_index(&values[1])?;
            let end = as_index(&values[2])?;
            match &values[0] {
                Value::String(value) => Ok(Value::String(slice_string(value, start, end).into())),
                Value::List(items) => {
                    let Some((start, end)) = clamp_range(start, end, items.len()) else {
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
            Ok(Value::String(stringify_value(&values[0])?.into()))
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
            if values.len() == 1 {
                return Ok(Value::String(stringify_value(&values[0])?.into()));
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
            Ok(Value::String(apply_format(template, &values[1..])?.into()))
        }
        Builtin::Unknown(index) => Err(RuntimeError::UnknownBuiltin {
            name: names[index].text.to_string(),
        }),
    }
}

fn read_field(value: Value, field: &Name) -> Result<Value, RuntimeError> {
    match value {
        Value::Record(record) => Ok(record
            .get_symbol(field.symbol)
            .cloned()
            .unwrap_or(Value::Null)),
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

fn read_index(target: Value, index: Value) -> Result<Value, RuntimeError> {
    let idx = as_index(&index)?;
    match target {
        Value::List(values) => Ok(values.get(idx).cloned().unwrap_or(Value::Null)),
        Value::String(value) => Ok(value
            .chars()
            .nth(idx)
            .map(|ch| Value::String(ch.to_string().into()))
            .unwrap_or(Value::Null)),
        Value::Null => Ok(Value::Null),
        _ => Err(RuntimeError::TypeError {
            message: format!("can't index {}", value_type_name(&target)),
        }),
    }
}

fn eval_binary_values(left: Value, op: BinaryOp, right: Value) -> Result<Value, RuntimeError> {
    match op {
        BinaryOp::Add => add_values(left, right),
        BinaryOp::Subtract => Ok(Value::Number(as_number(&left)? - as_number(&right)?)),
        BinaryOp::Multiply => Ok(Value::Number(as_number(&left)? * as_number(&right)?)),
        BinaryOp::Divide => Ok(Value::Number(as_number(&left)? / as_number(&right)?)),
        BinaryOp::Modulo => Ok(Value::Number(as_number(&left)? % as_number(&right)?)),
        BinaryOp::Equal => Ok(Value::Bool(left == right)),
        BinaryOp::NotEqual => Ok(Value::Bool(left != right)),
        BinaryOp::Less => compare_numbers(left, right, |a, b| a < b),
        BinaryOp::LessEqual => compare_numbers(left, right, |a, b| a <= b),
        BinaryOp::Greater => compare_numbers(left, right, |a, b| a > b),
        BinaryOp::GreaterEqual => compare_numbers(left, right, |a, b| a >= b),
        BinaryOp::And | BinaryOp::Or => unreachable!("logical ops are compiled with jumps"),
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
        Value::List(_) | Value::Record(_) => Err(RuntimeError::TypeError {
            message: format!("expected text, got {}", value_type_name(value)),
        }),
    }
}

fn as_index(value: &Value) -> Result<usize, RuntimeError> {
    let number = as_number(value)?;
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
        return Err(RuntimeError::TypeError {
            message: "index must be a non-negative integer".to_string(),
        });
    }
    Ok(number as usize)
}

fn compare_numbers(
    left: Value,
    right: Value,
    cmp: impl FnOnce(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    Ok(Value::Bool(cmp(as_number(&left)?, as_number(&right)?)))
}

fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
        (Value::String(mut a), other) => {
            a.push_str(&stringify_value(&other)?);
            Ok(Value::String(a))
        }
        (other, Value::String(b)) => {
            let mut text = stringify_value(&other)?;
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
        Value::List(_) | Value::Record(_) => true,
    }
}

fn success(value: Value) -> Value {
    let mut record = record_with_capacity(2);
    record.insert_symbol(intern_symbol("ok"), Value::Bool(true));
    record.insert_symbol(intern_symbol("value"), value);
    Value::Record(Arc::new(record))
}

fn error_value(message: String) -> Value {
    let mut record = record_with_capacity(2);
    record.insert_symbol(intern_symbol("ok"), Value::Bool(false));
    record.insert_symbol(intern_symbol("error"), Value::String(message.into()));
    Value::Record(Arc::new(record))
}

fn stringify_value(value: &Value) -> Result<String, RuntimeError> {
    match value {
        Value::String(value) => Ok(value.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::List(_) | Value::Record(_) => Ok(serde_json::to_string(&to_json(value))
            .expect("value json serialization should succeed")),
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::List(_) => "list",
        Value::Record(_) => "record",
    }
}

fn apply_format(template: &str, args: &[Value]) -> Result<String, RuntimeError> {
    let mut output = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut last_literal = 0;
    while index < bytes.len() {
        if bytes[index] == b'{' {
            let mut cursor = index + 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'}' && cursor > index + 1 {
                output.push_str(&template[last_literal..index]);
                let digits = &template[index + 1..cursor];
                let slot = digits
                    .parse::<usize>()
                    .map_err(|_| RuntimeError::ValueError {
                        message: format!("bad format slot `{digits}`"),
                    })?;
                let value = args.get(slot).ok_or_else(|| RuntimeError::ValueError {
                    message: format!("format slot `{slot}` is out of range"),
                })?;
                output.push_str(&stringify_value(value)?);
                index = cursor + 1;
                last_literal = index;
                continue;
            }
        }
        index += 1;
    }
    output.push_str(&template[last_literal..]);
    Ok(output)
}

fn clamp_range(start: usize, end: usize, len: usize) -> Option<(usize, usize)> {
    let start = start.min(len);
    let end = end.min(len);
    (start < end).then_some((start, end))
}

fn slice_string(value: &str, start: usize, end: usize) -> String {
    if start >= end {
        return String::new();
    }

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

fn to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Number(value) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(value) => serde_json::Value::String(value.to_string()),
        Value::List(values) => serde_json::Value::Array(values.iter().map(to_json).collect()),
        Value::Record(record) => serde_json::Value::Object(
            record
                .iter()
                .map(|(key, value)| (key.to_string(), to_json(value)))
                .collect(),
        ),
    }
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
        serde_json::Value::Object(map) => Value::Record(Arc::new(
            map.into_iter()
                .map(|(key, value)| (key, from_json(value)))
                .collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Host;

    impl ToolHost for Host {
        fn call(&self, name: &str, args: &Record) -> Result<Value, ToolHostError> {
            match name {
                "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
                "err" => Err(ToolHostError::new("boom")),
                "panic" => panic!("boom"),
                _ => Err(ToolHostError::new(format!("unknown tool: {name}"))),
            }
        }
    }

    fn exec(source: &str) -> Result<Value, RuntimeError> {
        let program = crate::parse(source).expect("program should parse");
        let mut state = State::new();
        match execute_program(&program, &mut state, &Host)? {
            ExecutionOutcome::Finished(value) => Ok(value),
            ExecutionOutcome::Continued => panic!("expected `finish` in test program"),
        }
    }

    fn exec_outcome(source: &str) -> Result<ExecutionOutcome, RuntimeError> {
        let program = crate::parse(source).expect("program should parse");
        let mut state = State::new();
        execute_program(&program, &mut state, &Host)
    }

    #[test]
    fn value_helpers_and_display_cover_all_variants() {
        let mut record = Record::default();
        record.insert("k".to_string(), Value::Number(1.0));

        assert_eq!(Value::Null.to_string(), "null");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Number(1.5).to_string(), "1.5");
        assert_eq!(Value::String("x".to_string().into()).to_string(), "x");
        assert_eq!(
            Value::List(vec![Value::Bool(true)].into()).to_string(),
            "[true]"
        );
        assert_eq!(
            Value::Record(record.clone().into()).as_record().unwrap()["k"],
            Value::Number(1.0)
        );
        assert!(Value::String("x".to_string().into()).as_record().is_none());
        assert!(
            Value::Record(record.into())
                .to_string()
                .contains("\"k\":1.0")
        );
    }

    #[test]
    fn continuation_and_undefined_variable_are_reported() {
        let outcome = exec_outcome("x = 1").expect("missing finish should continue");
        assert_eq!(outcome, ExecutionOutcome::Continued);

        let err = exec("finish x").expect_err("undefined variable should fail");
        assert_eq!(
            err,
            RuntimeError::UndefinedVariable {
                name: "x".to_string()
            }
        );
    }

    #[test]
    fn condition_and_iteration_errors_are_reported() {
        let value = exec("if 1 { finish 1 } else { finish 2 }")
            .expect("numeric truthiness should be accepted");
        assert_eq!(value, Value::Number(1.0));

        let value =
            exec("if \"\" { finish 1 } else { finish 2 }").expect("empty string should be falsy");
        assert_eq!(value, Value::Number(2.0));

        let err = exec("for x in 1 { finish x }").expect_err("non-list iteration should fail");
        assert_eq!(err, RuntimeError::NonListIteration);
    }

    #[test]
    fn stmt_call_and_tool_results_cover_success_and_error() {
        exec("call echo { value: 1 } finish 1").expect("statement call should succeed");
        let missing =
            exec("bad = call missing {} finish bad").expect("missing tool should be wrapped");
        assert_eq!(
            missing.as_record().expect("result should be a record")["ok"],
            Value::Bool(false)
        );

        let value =
            exec("ok = call echo { value: 7 } bad = call err {} finish { ok: ok, bad: bad }")
                .expect("tool call program should succeed");
        let record = value.as_record().expect("expected record");
        assert_eq!(record["ok"].as_record().unwrap()["ok"], Value::Bool(true));
        assert_eq!(record["bad"].as_record().unwrap()["ok"], Value::Bool(false));
    }

    #[test]
    fn field_index_unary_and_boolean_paths_are_covered() {
        let value = exec(
            r#"
            rec = { nested: { name: "lash" } }
            xs = ["a", "b"]
            ok = false and missing
            alt = true or missing
            finish [rec.nested.name, xs[1], "abc"[2], -1, not false, !false, ok, alt]
            "#,
        )
        .expect("program should succeed");

        assert_eq!(
            value,
            Value::List(
                vec![
                    Value::String("lash".to_string().into()),
                    Value::String("b".to_string().into()),
                    Value::String("c".to_string().into()),
                    Value::Number(-1.0),
                    Value::Bool(true),
                    Value::Bool(true),
                    Value::Bool(false),
                    Value::Bool(true),
                ]
                .into()
            )
        );

        let value = exec("finish true and false").expect("and path should succeed");
        assert_eq!(value, Value::Bool(false));

        let value = exec("finish false or true").expect("or path should succeed");
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn field_index_and_type_errors_are_covered() {
        let err = exec("n = 1 finish n.name").expect_err("field access should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let value = exec("rec = {} finish rec.name").expect("missing field should yield null");
        assert_eq!(value, Value::Null);

        let err = exec("finish 1[0]").expect_err("bad index target should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let value = exec("finish [1][2]").expect("list oob should yield null");
        assert_eq!(value, Value::Null);

        let value = exec("finish \"a\"[2]").expect("string oob should yield null");
        assert_eq!(value, Value::Null);

        let err = exec("finish [1][1.5]").expect_err("fractional index should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let err = exec("finish [1][-1]").expect_err("negative index should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));

        let value = exec("finish not 1").expect("not should use truthiness");
        assert_eq!(value, Value::Bool(false));

        let value = exec("finish not 0").expect("zero should be falsy");
        assert_eq!(value, Value::Bool(true));

        let value = exec("rec = { ok: false } finish len(rec.value.items)")
            .expect("null chain should work");
        assert_eq!(value, Value::Number(0.0));
    }

    #[test]
    fn arithmetic_and_compare_errors_are_covered() {
        assert_eq!(
            exec("finish 7 - 2").expect("subtract should succeed"),
            Value::Number(5.0)
        );
        assert_eq!(
            exec("finish 3 * 4").expect("multiply should succeed"),
            Value::Number(12.0)
        );
        assert_eq!(
            exec("finish 8 / 2").expect("divide should succeed"),
            Value::Number(4.0)
        );
        assert_eq!(
            exec("finish 8 % 3").expect("modulo should succeed"),
            Value::Number(2.0)
        );
        assert_eq!(
            exec("finish 1 != 2").expect("not equal should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 1 <= 2").expect("less-equal should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 2 > 1").expect("greater should succeed"),
            Value::Bool(true)
        );
        assert_eq!(
            exec("finish 2 >= 1").expect("greater-equal should succeed"),
            Value::Bool(true)
        );

        let value = exec("finish [1,2] + [3]").expect("list concat should succeed");
        assert_eq!(
            value,
            Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)].into())
        );

        let value = exec("finish \"a\" + \"b\"").expect("string add should succeed");
        assert_eq!(value, Value::String("ab".to_string().into()));

        let value = exec("finish \"a\" + 1").expect("string coercion should succeed");
        assert_eq!(value, Value::String("a1".to_string().into()));

        let value = exec("finish 1 + \"b\"").expect("string coercion should succeed");
        assert_eq!(value, Value::String("1b".to_string().into()));

        let value = exec("finish 1 + true").expect("bool should coerce for addition");
        assert_eq!(value, Value::Number(2.0));

        let value = exec("finish null + 2").expect("null should coerce for addition");
        assert_eq!(value, Value::Number(2.0));

        let value = exec("finish \"2\" * 3").expect("numeric strings should coerce");
        assert_eq!(value, Value::Number(6.0));

        let value = exec("finish \"2\" < 10").expect("numeric strings should compare");
        assert_eq!(value, Value::Bool(true));

        let err = exec("finish {} + 1").expect_err("records should still fail arithmetic");
        assert!(matches!(err, RuntimeError::TypeError { .. }));
    }

    #[test]
    fn builtin_success_matrix_is_covered() {
        let value = exec(
            r#"
            rec = { a: 1, b: 2 }
            finish {
              len_s: len("ab"),
              len_l: len([1,2,3]),
              len_r: len(rec),
              len_n: len(null),
              empty_n: empty(null),
              empty_s: empty(""),
              empty_l: empty([]),
              empty_r: empty({}),
              keys_n: keys(null),
              values_n: values(null),
              keys: keys(rec),
              values: values(rec),
              contains_s: contains("abc", "b"),
              contains_num: contains("123", 2),
              contains_l: contains([1,2,3], 2),
              contains_n: contains(null, 2),
              starts: starts_with("lash", "la"),
              starts_num: starts_with(123, 12),
              ends: ends_with("lash", "sh"),
              split: split(101, 0),
              join: join(["a",2,true], "-"),
              trim: trim(101),
              slice_s: slice("abcd", 1, 3),
              slice_back_s: slice("abcd", 3, 1),
              slice_l: slice([1,2,3,4], 1, 3),
              slice_back_l: slice([1,2,3,4], 3, 1),
              to_s: to_string({ a: 1 }),
              to_i_n: to_int(3.9),
              to_i_s: to_int("4"),
              to_i_b: to_int(true),
              to_f_n: to_float(1),
              to_f_s: to_float("2.5"),
              to_f_nl: to_float(null),
              fmt_value: format({ a: 1 }),
              fmt: format("x={0},y={1}", 1, true)
            }
            "#,
        )
        .expect("builtins should succeed");

        let record = value.as_record().expect("expected record");
        assert_eq!(record["len_s"], Value::Number(2.0));
        assert_eq!(record["len_n"], Value::Number(0.0));
        assert_eq!(record["contains_num"], Value::Bool(true));
        assert_eq!(record["contains_l"], Value::Bool(true));
        assert_eq!(record["contains_n"], Value::Bool(false));
        assert_eq!(record["keys_n"], Value::List(Vec::new().into()));
        assert_eq!(record["values_n"], Value::List(Vec::new().into()));
        assert_eq!(record["starts_num"], Value::Bool(true));
        assert_eq!(
            record["split"],
            Value::List(
                vec![
                    Value::String("1".to_string().into()),
                    Value::String("1".to_string().into())
                ]
                .into()
            )
        );
        assert_eq!(record["join"], Value::String("a-2-true".to_string().into()));
        assert_eq!(record["trim"], Value::String("101".to_string().into()));
        assert_eq!(record["slice_s"], Value::String("bc".to_string().into()));
        assert_eq!(record["slice_back_s"], Value::String(String::new().into()));
        assert_eq!(record["slice_back_l"], Value::List(Vec::new().into()));
        assert_eq!(record["to_i_n"], Value::Number(3.0));
        assert_eq!(record["to_i_b"], Value::Number(1.0));
        assert_eq!(record["to_f_s"], Value::Number(2.5));
        assert_eq!(record["to_f_nl"], Value::Number(0.0));
        assert_eq!(
            record["fmt_value"],
            Value::String("{\"a\":1.0}".to_string().into())
        );
        assert_eq!(
            record["fmt"],
            Value::String("x=1,y=true".to_string().into())
        );
    }

    #[test]
    fn builtin_error_matrix_is_covered() {
        let cases = [
            ("finish len(true)", "len"),
            ("finish empty(true)", "empty"),
            ("finish keys([])", "keys"),
            ("finish values([])", "values"),
            ("finish contains(1, 2)", "contains"),
            ("finish starts_with({}, \"a\")", "starts_with"),
            ("finish ends_with({}, \"a\")", "ends_with"),
            ("finish split({}, \",\")", "split"),
            ("finish join(1, \",\")", "join"),
            ("finish trim({})", "trim"),
            ("finish slice(1, 0, 1)", "slice"),
            ("finish to_int({})", "to_int"),
            ("finish to_int(\"x\")", "to_int"),
            ("finish to_float({})", "to_float"),
            ("finish to_float(\"x\")", "to_float"),
            ("finish json_parse(\"{\")", "json_parse"),
            ("finish format()", "format"),
            ("finish format(\"{1}\", \"x\")", "format"),
            ("finish no_such_builtin()", "no_such_builtin"),
        ];

        for (source, _) in cases {
            let err = exec(source).expect_err("builtin should fail");
            assert!(matches!(
                err,
                RuntimeError::TypeError { .. }
                    | RuntimeError::ValueError { .. }
                    | RuntimeError::UnknownBuiltin { .. }
            ));
        }

        let err = exec("finish len()").expect_err("arity error should fail");
        assert!(matches!(err, RuntimeError::TypeError { .. }));
    }

    #[test]
    fn helper_functions_are_covered_directly() {
        assert!(expect_arg_count("x", &[Value::Null], 1).is_ok());
        assert!(expect_arg_count("x", &[], 1).is_err());
        assert_eq!(as_number(&Value::Number(1.0)).expect("number"), 1.0);
        assert_eq!(as_number(&Value::Bool(true)).expect("bool"), 1.0);
        assert_eq!(as_number(&Value::Null).expect("null"), 0.0);
        assert_eq!(
            as_number(&Value::String("2.5".to_string().into())).expect("numeric"),
            2.5
        );
        assert_eq!(
            coerce_string(&Value::String("x".to_string().into())).expect("string"),
            "x"
        );
        assert_eq!(coerce_string(&Value::Bool(true)).expect("bool"), "true");
        assert_eq!(as_index(&Value::Number(2.0)).expect("index"), 2);
        assert!(as_index(&Value::Number(-1.0)).is_err());
        assert_eq!(slice_string("héllo", 1, 4), "éll");
        assert_eq!(slice_string("abc", 3, 1), "");
        assert_eq!(clamp_range(1, 3, 4), Some((1, 3)));
        assert_eq!(clamp_range(3, 1, 4), None);

        assert_eq!(
            compare_numbers(Value::Number(1.0), Value::Number(2.0), |a, b| a < b).expect("compare"),
            Value::Bool(true)
        );
        assert_eq!(
            add_values(Value::Number(1.0), Value::Number(2.0)).expect("add"),
            Value::Number(3.0)
        );
        assert_eq!(
            add_values(Value::String("a".to_string().into()), Value::Bool(true)).expect("concat"),
            Value::String("atrue".to_string().into())
        );
        assert_eq!(
            add_values(Value::Bool(true), Value::Number(2.0)).expect("numeric coercion"),
            Value::Number(3.0)
        );
        assert_eq!(
            success(Value::Number(1.0)).as_record().unwrap()["ok"],
            Value::Bool(true)
        );
        assert_eq!(
            error_value("x".to_string()).as_record().unwrap()["error"],
            Value::String("x".to_string().into())
        );
        assert_eq!(stringify_value(&Value::Null).expect("stringify"), "null");
        assert_eq!(
            apply_format("a{0}b", &[Value::Number(1.0)]).expect("format"),
            "a1b"
        );
        assert_eq!(
            apply_format("{999999999999999999999999999999999999}", &[])
                .expect_err("overflow slot should fail"),
            RuntimeError::ValueError {
                message: "bad format slot `999999999999999999999999999999999999`".to_string()
            }
        );
        assert_eq!(
            apply_format("{x}", &[]).expect("invalid brace pattern should pass through"),
            "{x}"
        );
        assert_eq!(
            value_type_name(&Value::Record(Record::default().into())),
            "record"
        );
        assert_eq!(
            RuntimeError::UndefinedVariable {
                name: "x".to_string()
            }
            .to_string(),
            "unknown name `x`"
        );
        assert_eq!(
            RuntimeError::TypeError {
                message: "can't index record".to_string()
            }
            .to_string(),
            "can't index record"
        );
    }

    #[test]
    fn json_helpers_cover_special_paths() {
        let json = to_json(&Value::Number(f64::NAN));
        assert_eq!(json, serde_json::Value::Null);
        assert_eq!(to_json(&Value::Null), serde_json::Value::Null);
        assert_eq!(to_json(&Value::Bool(true)), serde_json::Value::Bool(true));
        assert_eq!(
            to_json(&Value::String("x".to_string().into())),
            serde_json::Value::String("x".to_string())
        );
        assert_eq!(
            to_json(&Value::List(vec![Value::Number(1.0)].into())),
            serde_json::json!([1.0])
        );
        assert_eq!(
            to_json(&Value::Record({
                let mut record = Record::default();
                record.insert("a".to_string(), Value::Number(1.0));
                record.into()
            })),
            serde_json::json!({"a": 1.0})
        );

        let value = from_json(serde_json::json!({
            "a": [1, true, null, "x"]
        }));
        let record = value.as_record().expect("expected record");
        assert!(matches!(record["a"], Value::List(_)));
    }

    #[test]
    fn false_if_branch_and_finish_inside_loop_are_covered() {
        let value = exec(
            r#"
            if false {
              out = 1
            } else {
              out = 2
            }
            finish out
            "#,
        )
        .expect("else branch should succeed");
        assert_eq!(value, Value::Number(2.0));

        let value = exec(
            r#"
            for x in [1, 2] {
              finish x
            }
            finish 0
            "#,
        )
        .expect("finish inside loop should bubble out");
        assert_eq!(value, Value::Number(1.0));
    }

    #[test]
    fn parallel_branch_panics_are_reported_as_runtime_errors() {
        let err = exec(
            r#"
            parallel {
              crash = call panic {}
            }
            finish 1
            "#,
        )
        .expect_err("parallel panic should be reported");

        assert_eq!(
            err,
            RuntimeError::ValueError {
                message: "parallel branch panicked".to_string()
            }
        );
    }

    #[test]
    fn truthiness_covers_scalar_and_container_values() {
        assert!(!is_truthy(&Value::Null));
        assert!(!is_truthy(&Value::Bool(false)));
        assert!(!is_truthy(&Value::Number(0.0)));
        assert!(!is_truthy(&Value::String(String::new().into())));
        assert!(is_truthy(&Value::Bool(true)));
        assert!(is_truthy(&Value::Number(1.0)));
        assert!(is_truthy(&Value::List(Vec::new().into())));
        assert!(is_truthy(&Value::Record(Record::default().into())));
    }
}
