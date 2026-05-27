use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn system_time_from_epoch_ms(epoch_ms: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(epoch_ms)
}

pub fn epoch_ms_from_system_time(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
