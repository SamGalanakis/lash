use serde::Serialize;

pub(crate) const BENCH_WIDTH: u16 = 220;
pub(crate) const BENCH_HEIGHT: u16 = 72;
const FULL_TURN_COUNT: usize = 480;
const FULL_SURFACE_ROW_COUNT: usize = 1_600;
pub(crate) const SCROLL_DELTA: usize = 3;
pub(crate) const SELECTION_SCROLL_DELTA: usize = 2;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UiPerfScenario {
    HistoryRender,
    WorkspaceSurface,
    WorkspaceOverlay,
    StreamingReactor,
    SlowSnapshot,
    FileIndexStorm,
    TimelineProjection,
    ActivityProjection,
    HtmlExport,
}

impl UiPerfScenario {
    pub(crate) const DEFAULTS: [Self; 9] = [
        Self::HistoryRender,
        Self::WorkspaceSurface,
        Self::WorkspaceOverlay,
        Self::StreamingReactor,
        Self::SlowSnapshot,
        Self::FileIndexStorm,
        Self::TimelineProjection,
        Self::ActivityProjection,
        Self::HtmlExport,
    ];

    pub(crate) const KNOWN: [Self; 9] = Self::DEFAULTS;

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "history_render" | "history" => Some(Self::HistoryRender),
            "workspace_surface" | "workspace" => Some(Self::WorkspaceSurface),
            "workspace_overlay" => Some(Self::WorkspaceOverlay),
            "streaming_reactor" => Some(Self::StreamingReactor),
            "slow_snapshot" => Some(Self::SlowSnapshot),
            "file_index_storm" | "file-index-storm" => Some(Self::FileIndexStorm),
            "timeline_projection" | "projection" => Some(Self::TimelineProjection),
            "activity_projection" | "activity" => Some(Self::ActivityProjection),
            "html_export" | "export" => Some(Self::HtmlExport),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::HistoryRender => "history_render",
            Self::WorkspaceSurface => "workspace_surface",
            Self::WorkspaceOverlay => "workspace_overlay",
            Self::StreamingReactor => "streaming_reactor",
            Self::SlowSnapshot => "slow_snapshot",
            Self::FileIndexStorm => "file_index_storm",
            Self::TimelineProjection => "timeline_projection",
            Self::ActivityProjection => "activity_projection",
            Self::HtmlExport => "html_export",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UiPerfProfile {
    Quick,
    Full,
    Stress,
}

impl UiPerfProfile {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "quick" => Some(Self::Quick),
            "full" => Some(Self::Full),
            "stress" => Some(Self::Stress),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Full => "full",
            Self::Stress => "stress",
        }
    }

    pub(crate) fn workload(self) -> UiPerfWorkload {
        match self {
            Self::Quick => UiPerfWorkload {
                turn_count: 120,
                surface_row_count: 420,
                scroll_passes: 1,
                selection_frames: 80,
                stream_deltas: 240,
                control_events: 48,
                snapshot_jobs: 4,
                snapshot_timeout_ms: 18,
                file_source_changes: 2,
                ignored_path_events: 120,
            },
            Self::Full => UiPerfWorkload {
                turn_count: FULL_TURN_COUNT,
                surface_row_count: FULL_SURFACE_ROW_COUNT,
                scroll_passes: 2,
                selection_frames: 320,
                stream_deltas: 1_200,
                control_events: 180,
                snapshot_jobs: 8,
                snapshot_timeout_ms: 24,
                file_source_changes: 6,
                ignored_path_events: 1_200,
            },
            Self::Stress => UiPerfWorkload {
                turn_count: 900,
                surface_row_count: 4_000,
                scroll_passes: 3,
                selection_frames: 700,
                stream_deltas: 5_000,
                control_events: 700,
                snapshot_jobs: 18,
                snapshot_timeout_ms: 30,
                file_source_changes: 16,
                ignored_path_events: 6_000,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct UiPerfWorkload {
    pub(crate) turn_count: usize,
    pub(crate) surface_row_count: usize,
    pub(crate) scroll_passes: usize,
    pub(crate) selection_frames: usize,
    pub(crate) stream_deltas: usize,
    pub(crate) control_events: usize,
    pub(crate) snapshot_jobs: usize,
    pub(crate) snapshot_timeout_ms: u64,
    pub(crate) file_source_changes: usize,
    pub(crate) ignored_path_events: usize,
}
