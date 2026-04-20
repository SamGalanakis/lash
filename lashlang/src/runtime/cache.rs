use super::{CompiledProgram, compile_source, prewarm};
use rustc_hash::FxHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const DEFAULT_COMPILED_PROGRAM_CACHE_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompiledProgramCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub capacity: usize,
}

pub struct CompiledProgramCache {
    entries: VecDeque<CachedCompiledProgram>,
    hits: u64,
    misses: u64,
    evictions: u64,
    capacity: usize,
}

struct CachedCompiledProgram {
    source_hash: u64,
    source: Arc<str>,
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
        if let Some(entry) = self.entries.back()
            && source_matches(entry, source)
        {
            self.hits += 1;
            return Ok(entry.compiled.clone());
        }

        let source_hash = source_hash(source);
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.source_hash == source_hash && entry.source.as_ref() == source)
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
        let compiled = Arc::new(compile_source(source)?);
        if self.capacity == 0 {
            return Ok(compiled);
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.evictions += 1;
        }
        self.entries.push_back(CachedCompiledProgram {
            source_hash,
            source: Arc::<str>::from(source),
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

fn source_matches(entry: &CachedCompiledProgram, source: &str) -> bool {
    entry.source.len() == source.len()
        && (std::ptr::eq(entry.source.as_ptr(), source.as_ptr()) || entry.source.as_ref() == source)
}

fn source_hash(source: &str) -> u64 {
    let mut hasher = FxHasher::default();
    source.hash(&mut hasher);
    hasher.finish()
}
