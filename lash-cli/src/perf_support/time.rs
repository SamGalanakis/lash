use std::time::Instant;

pub(crate) fn elapsed_ms(started: Instant) -> f64 {
    round3(started.elapsed().as_secs_f64() * 1000.0)
}

pub(crate) fn nanos_to_ms(nanos: u64) -> f64 {
    round3(nanos as f64 / 1_000_000.0)
}

pub(crate) fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}
