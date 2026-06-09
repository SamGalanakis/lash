use std::path::Path;

use anyhow::Context;
use serde::Serialize;

pub fn resolve_named_scenarios<T>(
    filters: &[String],
    defaults: &[T],
    known: &[T],
    parse: impl Fn(&str) -> Option<T>,
    name: impl Fn(T) -> &'static str,
    label: &str,
) -> anyhow::Result<Vec<T>>
where
    T: Copy + PartialEq,
{
    if filters.is_empty() {
        return Ok(defaults.to_vec());
    }

    let mut scenarios = Vec::with_capacity(filters.len());
    for filter in filters {
        if filter == "all" {
            for scenario in known {
                push_unique(&mut scenarios, *scenario);
            }
            continue;
        }
        let scenario = parse(filter).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown {label} scenario `{filter}`; expected one of: {}, all",
                known
                    .iter()
                    .map(|scenario| name(*scenario))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        push_unique(&mut scenarios, scenario);
    }
    Ok(scenarios)
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

pub fn ensure_parent_dir(path: &Path, label: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} dir {}", parent.display()))?;
    }
    Ok(())
}

pub fn write_json_report(path: &Path, report: &impl Serialize) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write benchmark report {}", path.display()))
}
