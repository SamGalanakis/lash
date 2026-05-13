//! Bytecode instruction set + the inert data types that flow from the
//! compiler to the VM: `Chunk`, `Name`, `Instruction`, `Builtin`, the
//! profile-tag enums and accumulator, the format-template / assign-path
//! shapes, and the pure-expression form used by the parallel-call
//! optimization.
//!
//! Everything here is internal to the runtime crate — the visibility is
//! `pub(crate)` because compiler.rs produces these structures and vm.rs
//! consumes them. None of these types are part of the lashlang public API.

use std::sync::{Arc, OnceLock};

use crate::ast::{BinaryOp, UnaryOp};
use crate::lexer::Span;

use super::record::{Symbol, intern_symbol, symbol_name};
use super::schema::CompiledSchema;
use super::{CompileStats, ProfileReport, ProfileStat, Value};

#[derive(Clone)]
pub(crate) struct Chunk {
    pub(crate) code: Vec<Instruction>,
    pub(crate) spans: Vec<Option<Span>>,
    pub(crate) constants: Vec<Value>,
    pub(crate) names: Vec<Name>,
    pub(crate) slot_names: Vec<Name>,
    pub(crate) key_lists: Vec<Box<[usize]>>,
    pub(crate) format_templates: Vec<CompiledFormatTemplate>,
    pub(crate) compiled_schemas: Vec<CompiledSchema>,
    pub(crate) parallel_call_sets: Vec<Box<[ParallelCallBranch]>>,
    pub(crate) named_parallel_call_sets: Vec<Box<[NamedParallelCallBranch]>>,
    pub(crate) pure_parallel_sets: Vec<Box<[PureExpr]>>,
    pub(crate) pure_named_parallel_sets: Vec<Box<[(usize, PureExpr)]>>,
    pub(crate) branch_sets: Vec<Box<[Chunk]>>,
    pub(crate) named_branch_sets: Vec<Box<[NamedBranchChunk]>>,
    pub(crate) assign_paths: Vec<CompiledAssignPath>,
}

#[derive(Clone)]
pub(crate) struct Name {
    pub(crate) symbol: Symbol,
    pub(crate) text: Arc<str>,
}

#[derive(Clone)]
pub(crate) struct CompiledFormatTemplate {
    pub(crate) parts: Box<[CompiledFormatPart]>,
    pub(crate) argc: usize,
    pub(crate) min_capacity: usize,
    pub(crate) error: Option<String>,
}

#[derive(Clone)]
pub(crate) enum CompiledFormatPart {
    Literal(Arc<str>),
    Arg(usize),
}

#[derive(Clone)]
pub(crate) struct CompiledAssignPath {
    pub(crate) steps: Box<[CompiledAssignPathStep]>,
    pub(crate) dynamic_index_count: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum CompiledAssignPathStep {
    Field(usize),
    Index,
}

pub(crate) struct ResultWrapperNames {
    pub(crate) ok: Name,
    pub(crate) value: Name,
    pub(crate) error: Name,
}

pub(crate) fn transient_name(name: &str) -> Name {
    let symbol = intern_symbol(name);
    Name {
        symbol,
        text: symbol_name(symbol),
    }
}

pub(crate) fn result_wrapper_names() -> &'static ResultWrapperNames {
    static NAMES: OnceLock<ResultWrapperNames> = OnceLock::new();
    NAMES.get_or_init(|| ResultWrapperNames {
        ok: transient_name("ok"),
        value: transient_name("value"),
        error: transient_name("error"),
    })
}

#[derive(Clone, Copy)]
pub(crate) enum Instruction {
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
    PushAssign(usize),
    Range {
        argc: usize,
    },
    FormatCompiled(usize),
    AddAssign(usize),
    AddAssignNumber {
        slot: usize,
        right: f64,
    },
    AddAssignSlot {
        slot: usize,
        right: usize,
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
pub(crate) enum Builtin {
    Len,
    Empty,
    Keys,
    Values,
    Contains,
    Find,
    GrepText,
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
    pub(crate) fn profile_tag(self) -> InstructionProfileTag {
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
            | Instruction::PushAssign(_)
            | Instruction::Range { .. }
            | Instruction::FormatCompiled(_) => InstructionProfileTag::CallBuiltin,
            Instruction::AddAssign(_)
            | Instruction::AddAssignNumber { .. }
            | Instruction::AddAssignSlot { .. }
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
    pub(crate) fn profile_tag(self) -> BuiltinProfileTag {
        match self {
            Builtin::Len => BuiltinProfileTag::Len,
            Builtin::Empty => BuiltinProfileTag::Empty,
            Builtin::Keys => BuiltinProfileTag::Keys,
            Builtin::Values => BuiltinProfileTag::Values,
            Builtin::Contains => BuiltinProfileTag::Contains,
            Builtin::Find => BuiltinProfileTag::Find,
            Builtin::GrepText => BuiltinProfileTag::GrepText,
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
pub(crate) enum InstructionProfileTag {
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
pub(crate) enum BuiltinProfileTag {
    Len,
    Empty,
    Keys,
    Values,
    Contains,
    Find,
    GrepText,
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
pub(crate) struct ProfileAccumulator {
    pub(crate) instruction_counts: [u64; INSTRUCTION_PROFILE_COUNT],
    pub(crate) instruction_times: [u128; INSTRUCTION_PROFILE_COUNT],
    pub(crate) builtin_counts: [u64; BUILTIN_PROFILE_COUNT],
    pub(crate) builtin_times: [u128; BUILTIN_PROFILE_COUNT],
}

impl ProfileAccumulator {
    pub(crate) fn finish(self) -> ProfileReport {
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
    "find",
    "grep_text",
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

pub(crate) fn merge_stats(target: &mut Vec<ProfileStat>, source: &[ProfileStat]) {
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

#[derive(Clone)]
pub(crate) struct ParallelCallBranch {
    pub(crate) slot: usize,
    pub(crate) name: usize,
    pub(crate) args: PureExpr,
}

#[derive(Clone)]
pub(crate) struct NamedParallelCallBranch {
    pub(crate) output_name: usize,
    pub(crate) name: usize,
    pub(crate) args: PureExpr,
}

#[derive(Clone)]
pub(crate) struct NamedBranchChunk {
    pub(crate) name: usize,
    pub(crate) chunk: Chunk,
}

#[derive(Clone)]
pub(crate) enum PureExpr {
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
