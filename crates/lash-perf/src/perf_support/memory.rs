#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessMemorySample {
    pub rss_kb: Option<u64>,
    pub hwm_kb: Option<u64>,
}

pub fn process_memory_sample() -> ProcessMemorySample {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return ProcessMemorySample::default();
    };

    let mut sample = ProcessMemorySample::default();
    for line in status.lines() {
        if sample.rss_kb.is_none()
            && let Some(value) = parse_status_kb(line, "VmRSS:")
        {
            sample.rss_kb = Some(value);
        }
        if sample.hwm_kb.is_none()
            && let Some(value) = parse_status_kb(line, "VmHWM:")
        {
            sample.hwm_kb = Some(value);
        }
        if sample.rss_kb.is_some() && sample.hwm_kb.is_some() {
            break;
        }
    }
    sample
}

pub fn diff_opt_i64(before: Option<u64>, after: Option<u64>) -> Option<i64> {
    Some(after? as i64 - before? as i64)
}

fn parse_status_kb(line: &str, key: &str) -> Option<u64> {
    let value = line.strip_prefix(key)?.trim();
    value.split_whitespace().next()?.parse::<u64>().ok()
}
