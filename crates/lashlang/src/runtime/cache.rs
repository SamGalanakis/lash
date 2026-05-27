use crate::ast::Program;
use crate::{
    LASHLANG_COMPILER_VERSION, LASHLANG_VM_ABI_VERSION, ModuleArtifact, ProcessRef,
    RequiredSurfaceRef,
};

use super::entry_points::{compile_module_artifact_process, compile_program_internal};
use super::{CompiledProgram, prewarm};
use rustc_hash::FxHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const DEFAULT_COMPILED_PROGRAM_CACHE_CAPACITY: usize = 64;
const AST_CACHE_VERSION: &str = "lashlang-ast-v2";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompiledProgramCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub capacity: usize,
}

const DEFAULT_COMPILED_PROCESS_CACHE_CAPACITY: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompiledProcessCacheKey {
    pub process_ref: ProcessRef,
    pub required_surface_ref: RequiredSurfaceRef,
    pub compiler_version: &'static str,
    pub vm_abi_version: &'static str,
}

impl CompiledProcessCacheKey {
    pub fn new(process_ref: ProcessRef, required_surface_ref: RequiredSurfaceRef) -> Self {
        Self {
            process_ref,
            required_surface_ref,
            compiler_version: LASHLANG_COMPILER_VERSION,
            vm_abi_version: LASHLANG_VM_ABI_VERSION,
        }
    }
}

pub struct CompiledProcessCache {
    entries: VecDeque<CachedCompiledProcess>,
    hits: u64,
    misses: u64,
    evictions: u64,
    capacity: usize,
}

struct CachedCompiledProcess {
    key: CompiledProcessCacheKey,
    compiled: Arc<CompiledProgram>,
}

impl CompiledProcessCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_COMPILED_PROCESS_CACHE_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        prewarm();
        Self {
            entries: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            capacity,
        }
    }

    pub fn get_or_compile(
        &mut self,
        artifact: &ModuleArtifact,
        process_ref: &ProcessRef,
        required_surface_ref: &RequiredSurfaceRef,
    ) -> Result<Arc<CompiledProgram>, crate::RuntimeError> {
        let key = CompiledProcessCacheKey::new(process_ref.clone(), required_surface_ref.clone());
        if let Some(entry) = self.entries.back()
            && entry.key == key
        {
            self.hits += 1;
            return Ok(entry.compiled.clone());
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.hits += 1;
            let entry = self
                .entries
                .remove(index)
                .expect("cache index came from existing entry");
            let compiled = entry.compiled.clone();
            self.entries.push_back(entry);
            return Ok(compiled);
        }

        self.misses += 1;
        let compiled = Arc::new(compile_module_artifact_process(artifact, process_ref)?);
        if self.capacity == 0 {
            return Ok(compiled);
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.evictions += 1;
        }
        self.entries.push_back(CachedCompiledProcess {
            key,
            compiled: compiled.clone(),
        });
        Ok(compiled)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    pub fn stats(&self) -> CompiledProgramCacheStats {
        CompiledProgramCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
            capacity: self.capacity,
        }
    }
}

impl Default for CompiledProcessCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CompiledProgramCache {
    entries: VecDeque<CachedCompiledProgram>,
    hits: u64,
    misses: u64,
    evictions: u64,
    capacity: usize,
}

struct CachedCompiledProgram {
    ast_hash: u64,
    program: Arc<Program>,
    compiled: Arc<CompiledProgram>,
}

impl CompiledProgramCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_COMPILED_PROGRAM_CACHE_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        prewarm();
        Self {
            entries: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            capacity,
        }
    }

    pub fn get_or_compile(
        &mut self,
        source: &str,
    ) -> Result<Arc<CompiledProgram>, crate::parser::ParseError> {
        let program = Arc::new(crate::parse(source)?);
        let ast_hash = program_hash(&program);
        if let Some(entry) = self.entries.back()
            && program_matches(entry, ast_hash, &program)
        {
            self.hits += 1;
            return Ok(entry.compiled.clone());
        }

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| program_matches(entry, ast_hash, &program))
        {
            self.hits += 1;
            let entry = self
                .entries
                .remove(index)
                .expect("cache index came from existing entry");
            let compiled = entry.compiled.clone();
            self.entries.push_back(entry);
            return Ok(compiled);
        }

        self.misses += 1;
        let compiled = Arc::new(compile_program_internal(&program));
        if self.capacity == 0 {
            return Ok(compiled);
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.evictions += 1;
        }
        self.entries.push_back(CachedCompiledProgram {
            ast_hash,
            program,
            compiled: compiled.clone(),
        });
        Ok(compiled)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    pub fn stats(&self) -> CompiledProgramCacheStats {
        CompiledProgramCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
            capacity: self.capacity,
        }
    }
}

impl Default for CompiledProgramCache {
    fn default() -> Self {
        Self::new()
    }
}

fn program_matches(entry: &CachedCompiledProgram, ast_hash: u64, program: &Program) -> bool {
    entry.ast_hash == ast_hash && entry.program.as_ref() == program
}

fn program_hash(program: &Program) -> u64 {
    let mut hasher = FxHasher::default();
    AST_CACHE_VERSION.hash(&mut hasher);
    serde_json::to_string(program)
        .expect("lashlang AST should serialize for cache hashing")
        .hash(&mut hasher);
    hasher.finish()
}
