use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PerfPhase {
    RenderBuild,
    DiffScan,
    AnsiQueue,
    FlushSyscall,
    Layout,
    Widget,
}

impl PerfPhase {
    const ALL: [Self; 6] = [
        Self::RenderBuild,
        Self::DiffScan,
        Self::AnsiQueue,
        Self::FlushSyscall,
        Self::Layout,
        Self::Widget,
    ];

    const fn index(self) -> usize {
        match self {
            Self::RenderBuild => 0,
            Self::DiffScan => 1,
            Self::AnsiQueue => 2,
            Self::FlushSyscall => 3,
            Self::Layout => 4,
            Self::Widget => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStat {
    pub count: u64,
    pub total_nanos: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub changed_cells: u64,
    pub changed_rows: u64,
    pub bytes_queued: u64,
    pub continuation_cells: u64,
    pub wide_glyph_updates: u64,
    pub sync_frames: u64,
    pub sync_fallback_frames: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PerfCounters {
    phases: [PhaseStat; PerfPhase::ALL.len()],
    pub frame: FrameStats,
}

impl PerfCounters {
    pub fn phase(&self, phase: PerfPhase) -> PhaseStat {
        self.phases[phase.index()]
    }

    pub fn record_phase_nanos(&mut self, phase: PerfPhase, nanos: u64) {
        let stat = &mut self.phases[phase.index()];
        stat.count = stat.count.saturating_add(1);
        stat.total_nanos = stat.total_nanos.saturating_add(nanos);
    }

    pub fn scope(&mut self, phase: PerfPhase) -> PerfScope<'_> {
        PerfScope {
            counters: self,
            phase,
            started: Instant::now(),
        }
    }
}

pub struct PerfScope<'a> {
    counters: &'a mut PerfCounters,
    phase: PerfPhase,
    started: Instant,
}

impl Drop for PerfScope<'_> {
    fn drop(&mut self) {
        self.counters
            .record_phase_nanos(self.phase, self.started.elapsed().as_nanos() as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::{PerfCounters, PerfPhase};

    #[test]
    fn perf_scope_accumulates_time() {
        let mut counters = PerfCounters::default();
        {
            let _scope = counters.scope(PerfPhase::Layout);
        }
        assert_eq!(counters.phase(PerfPhase::Layout).count, 1);
    }
}
