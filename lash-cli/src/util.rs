/// Format a duration in milliseconds as a human-readable string.
///
/// - `< 1s` → "42ms"
/// - `1s – 60s` → "14.6s"
/// - `1m – 60m` → "2m 13s"
/// - `≥ 1h` → "1h 5m"
pub fn format_duration_ms(ms: u64) -> String {
    if ms < 1_000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else if ms < 3_600_000 {
        let m = ms / 60_000;
        let s = (ms % 60_000) / 1_000;
        format!("{}m {}s", m, s)
    } else {
        let h = ms / 3_600_000;
        let m = (ms % 3_600_000) / 60_000;
        format!("{}h {}m", h, m)
    }
}
