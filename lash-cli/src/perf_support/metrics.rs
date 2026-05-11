use serde::Serialize;

use super::time::round3;

#[derive(Debug, Clone, Serialize)]
#[cfg(feature = "runtime-perf")]
pub(crate) struct BasicMetricSummary {
    pub(crate) min: f64,
    pub(crate) median: f64,
    pub(crate) max: f64,
    pub(crate) mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PercentileMetricSummary {
    pub(crate) p50: f64,
    pub(crate) p95: f64,
    pub(crate) p99: f64,
    pub(crate) max: f64,
    pub(crate) mean: f64,
}

#[cfg(feature = "runtime-perf")]
pub(crate) fn basic_summary(mut values: Vec<f64>) -> BasicMetricSummary {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let min = *values.first().unwrap_or(&0.0);
    let max = *values.last().unwrap_or(&0.0);
    let median = if values.is_empty() {
        0.0
    } else if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    BasicMetricSummary {
        min: round3(min),
        median: round3(median),
        max: round3(max),
        mean: round3(mean),
    }
}

#[cfg(feature = "runtime-perf")]
pub(crate) fn optional_basic_summary(values: Vec<f64>) -> Option<BasicMetricSummary> {
    if values.is_empty() {
        None
    } else {
        Some(basic_summary(values))
    }
}

pub(crate) fn percentile_summary(mut values: Vec<f64>) -> PercentileMetricSummary {
    values.sort_by(f64::total_cmp);
    PercentileMetricSummary {
        p50: round3(percentile_sorted(&values, 0.50)),
        p95: round3(percentile_sorted(&values, 0.95)),
        p99: round3(percentile_sorted(&values, 0.99)),
        max: round3(*values.last().unwrap_or(&0.0)),
        mean: round3(values.iter().sum::<f64>() / values.len().max(1) as f64),
    }
}

pub(crate) fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let rank = percentile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = rank - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}
