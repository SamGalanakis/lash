//! Emit a `run.json` compatible with `scripts/bench_ui.py` so SWE-bench runs
//! land in the same dashboard as terminal-bench runs.
//!
//! `scripts/terminalbench_results.py::SCHEMA_VERSION` is the source of truth
//! for the shape. We fill only the fields the UI actually reads (see
//! `load_run_summaries` + `load_run`) and leave the rest at defaults.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::InstanceResult;
use crate::RunSettings;

pub fn write_dashboard_run_json(
    run_dir: &Path,
    settings: &RunSettings,
    results: &[InstanceResult],
    started_at: &str,
    finished_at: &str,
    duration_seconds: f64,
) -> Result<()> {
    let trials: Vec<Value> = results.iter().map(trial_record).collect();
    let global_stats = global_stats(&trials);
    let task_rollups = task_rollups(&trials);
    let requested_ids: Vec<String> = results.iter().map(|r| r.instance_id.clone()).collect();
    let task_scope = json!({
        "selection_mode": "exact",
        "requested_tasks": requested_ids,
        "requested_task_count": results.len(),
        "executed_tasks": results.iter().map(|r| &r.instance_id).collect::<Vec<_>>(),
        "executed_task_count": results.len(),
        "missing_requested_tasks": Vec::<String>::new(),
        "unexpected_executed_tasks": Vec::<String>::new(),
        "scope_mismatch": false,
    });

    let payload = json!({
        "schema_version": 9,
        "run_id": settings.run_id,
        "exported_at": finished_at,
        "job_name": settings.run_id,
        "source_job_dir": run_dir.display().to_string(),
        "params": {
            "agent": "lash",
            "dataset": settings.dataset_label,
            "execution_mode": settings.execution_mode_label,
            "preset": settings.preset,
            "preset_source": settings.preset.as_ref().map(|_| "explicit"),
            "requested_model": settings.model,
            "variant": settings.variant,
            "context_approach": settings.context_approach_label,
            "provider": {
                "active_provider": settings.provider_kind,
                "active_provider_type": settings.provider_kind,
                "available_providers": Vec::<Value>::new(),
            },
            "harbor_env": "local",
            "registry_url": "",
            "n_concurrent": settings.batch_size,
            "attempts": 1,
            "timeout_multiplier": 1.0,
            "delete_after_run": false,
            "debug": false,
            "binary_path": Option::<String>::None,
            "task_patterns": Vec::<String>::new(),
            "exact_tasks": results.iter().map(|r| &r.instance_id).collect::<Vec<_>>(),
            "task_scope": task_scope,
            "exclude_patterns": Vec::<String>::new(),
            "extra_args": Vec::<String>::new(),
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        },
        "global_stats": global_stats,
        "task_rollups": task_rollups,
        "trials": trials,
    });

    let path = run_dir.join("run.json");
    let text = serde_json::to_string_pretty(&payload)?;
    std::fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn trial_record(r: &InstanceResult) -> Value {
    let status = match r.grade.as_str() {
        "pass" => "pass",
        "fail" => "fail",
        "error" => "error",
        _ => "no-reward",
    };
    let reward = if status == "pass" {
        Some(1.0f64)
    } else {
        Some(0.0f64)
    };
    json!({
        "trial_name": r.instance_id,
        "task_name": r.instance_id,
        "task_source": "swebench",
        "status": status,
        "reward": reward,
        "timing": {
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "trial_seconds": r.elapsed_seconds,
            "environment_setup_seconds": r.checkout_seconds,
            "agent_setup_seconds": 0.0,
            "agent_execution_seconds": r.turn_seconds,
            "verifier_seconds": 0.0,
        },
        "duration_display": format_duration(r.elapsed_seconds),
        "tokens": {
            "input": r.tokens.input,
            "output": r.tokens.output,
            "reasoning": r.tokens.reasoning,
            "cache": r.tokens.cache,
            "cache_read": r.tokens.cache,
            "cache_write": 0,
            "non_cache_total": r.tokens.input + r.tokens.output + r.tokens.reasoning,
            "total": r.tokens.input + r.tokens.output + r.tokens.reasoning + r.tokens.cache,
        },
        "cost_usd": Option::<f64>::None,
        "resource_usage": {
            "scope": "all_commands",
            "sample_count": 0,
            "cpu_seconds_sum": 0.0,
            "wall_clock_seconds_sum": r.elapsed_seconds,
            "max_rss_kb_max": Option::<i64>::None,
            "command_phase_counts": {"main": 1, "overhead": 0, "total": 1},
            "all_commands": {"sample_count": 0, "cpu_seconds_sum": 0.0, "wall_clock_seconds_sum": r.elapsed_seconds, "max_rss_kb_max": Option::<i64>::None},
            "main_commands": {"sample_count": 0, "cpu_seconds_sum": 0.0, "wall_clock_seconds_sum": r.elapsed_seconds, "max_rss_kb_max": Option::<i64>::None},
            "overhead_commands": {"sample_count": 0, "cpu_seconds_sum": 0.0, "wall_clock_seconds_sum": 0.0, "max_rss_kb_max": Option::<i64>::None},
        },
        "metadata": {
            "agent": "lash",
            "execution_mode": r.execution_mode_label,
            "requested_model": r.model,
            "resolved_models": vec![&r.model],
            "llm_request_count": r.llm_calls,
            "llm_record_count": r.llm_calls,
            "llm_turn_count": r.iterations,
            "llm_call_count": r.llm_calls,
            "tool_call_count": r.tool_calls,
            "tool_batch_count": 0,
            "tool_call_breakdown": &r.tool_breakdown,
            "repo": r.repo,
            "base_commit": r.base_commit,
            "patch_bytes": r.model_patch.len(),
            "empty_patch": r.model_patch.trim().is_empty(),
            "turn_status": r.turn_status,
            "done_reason": r.done_reason,
        },
        "failure_reason": r.failure_reason,
        "logs": {
            "assistant_excerpt": shorten(&r.assistant_text, 2400, true),
            "verifier_excerpt": Option::<String>::None,
            "stderr_excerpt": Option::<String>::None,
        },
        "artifacts": {
            "files": {
                "predictions_jsonl": "predictions.jsonl",
                "instance_result": format!("instances/{}/result.json", r.instance_id),
                "instance_patch": format!("instances/{}/model.patch", r.instance_id),
                "instance_events": format!("instances/{}/events.jsonl", r.instance_id),
                "instance_trace": format!("instances/{}/session.trace.jsonl", r.instance_id),
                "instance_prompt": format!("instances/{}/prompt.txt", r.instance_id),
            },
            "commands": Vec::<Value>::new(),
            "setup": {},
            "sessions": Vec::<Value>::new(),
        },
    })
}

fn global_stats(trials: &[Value]) -> Value {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut errored = 0usize;
    let mut duration_sum = 0.0f64;
    let mut tokens_total = TokenAcc::default();
    let mut activity = ActivityAcc::default();
    for trial in trials {
        match trial.get("status").and_then(Value::as_str).unwrap_or("") {
            "pass" => passed += 1,
            "fail" => failed += 1,
            "error" => errored += 1,
            _ => {}
        }
        if let Some(secs) = trial
            .pointer("/timing/trial_seconds")
            .and_then(Value::as_f64)
        {
            duration_sum += secs;
        }
        if let Some(tok) = trial.get("tokens") {
            tokens_total.add(tok);
        }
        if let Some(meta) = trial.get("metadata") {
            activity.add(meta);
        }
    }
    let total = trials.len() as f64;
    let pass_rate = if total > 0.0 {
        passed as f64 / total
    } else {
        0.0
    };
    json!({
        "trials_total": trials.len(),
        "trials_passed": passed,
        "trials_failed": failed,
        "trials_errors": errored,
        "trials_without_reward": 0,
        "pass_rate": pass_rate,
        "reward_mean": pass_rate,
        "duration_seconds_sum": duration_sum,
        "duration_seconds_avg": if total > 0.0 { duration_sum / total } else { 0.0 },
        "tokens_total": tokens_total.snapshot(),
        "tokens_avg": tokens_total.avg(total),
        "activity_total": activity.snapshot(),
        "activity_avg": activity.avg(total),
        "resource_usage": empty_resource(),
        "resource_usage_all_commands": empty_resource(),
        "resource_usage_overhead_commands": empty_resource(),
        "status_counts": {
            "pass": passed,
            "fail": failed,
            "error": errored,
        },
    })
}

fn task_rollups(trials: &[Value]) -> Vec<Value> {
    trials
        .iter()
        .map(|trial| {
            let name = trial
                .get("task_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let status = trial
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let pass_rate = if status == "pass" { 1.0f64 } else { 0.0f64 };
            json!({
                "task_name": name,
                "attempts": 1,
                "pass_rate": pass_rate,
                "status_counts": {&status: 1},
                "reward_mean": pass_rate,
                "duration_seconds_avg": trial.pointer("/timing/trial_seconds").cloned().unwrap_or(json!(0.0)),
                "duration_seconds_sum": trial.pointer("/timing/trial_seconds").cloned().unwrap_or(json!(0.0)),
                "tokens_total": trial.get("tokens").cloned().unwrap_or(json!({})),
                "tokens_avg": trial.get("tokens").cloned().unwrap_or(json!({})),
                "activity_total": activity_from_meta(trial.get("metadata")),
                "activity_avg": activity_from_meta(trial.get("metadata")),
                "resource_usage": empty_resource(),
                "trial_names": [trial.get("trial_name").cloned().unwrap_or(json!(""))],
            })
        })
        .collect()
}

fn activity_from_meta(meta: Option<&Value>) -> Value {
    let get_u = |k: &str| {
        meta.and_then(|m| m.get(k))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    json!({
        "llm_records": get_u("llm_record_count"),
        "llm_turns": get_u("llm_turn_count"),
        "llm_calls": get_u("llm_call_count"),
        "tool_calls": get_u("tool_call_count"),
        "tool_batches": get_u("tool_batch_count"),
    })
}

fn empty_resource() -> Value {
    json!({
        "sample_count": 0,
        "cpu_seconds_sum": 0.0,
        "cpu_seconds_avg": Option::<f64>::None,
        "cpu_seconds_max": Option::<f64>::None,
        "wall_clock_seconds_sum": 0.0,
        "wall_clock_seconds_avg": Option::<f64>::None,
        "wall_clock_seconds_max": Option::<f64>::None,
        "cpu_percent_avg": Option::<f64>::None,
        "cpu_percent_max": Option::<f64>::None,
        "max_rss_kb_avg": Option::<f64>::None,
        "max_rss_kb_max": Option::<i64>::None,
    })
}

#[derive(Default)]
struct TokenAcc {
    input: u64,
    output: u64,
    reasoning: u64,
    cache: u64,
}

impl TokenAcc {
    fn add(&mut self, tok: &Value) {
        self.input += tok.get("input").and_then(Value::as_u64).unwrap_or(0);
        self.output += tok.get("output").and_then(Value::as_u64).unwrap_or(0);
        self.reasoning += tok.get("reasoning").and_then(Value::as_u64).unwrap_or(0);
        self.cache += tok.get("cache").and_then(Value::as_u64).unwrap_or(0);
    }

    fn snapshot(&self) -> Value {
        let non_cache = self.input + self.output + self.reasoning;
        json!({
            "input": self.input,
            "output": self.output,
            "reasoning": self.reasoning,
            "cache": self.cache,
            "cache_read": self.cache,
            "cache_write": 0,
            "non_cache_total": non_cache,
            "total": non_cache + self.cache,
        })
    }

    fn avg(&self, n: f64) -> Value {
        let safe = |v: u64| if n > 0.0 { v as f64 / n } else { 0.0 };
        json!({
            "input": safe(self.input),
            "output": safe(self.output),
            "reasoning": safe(self.reasoning),
            "cache": safe(self.cache),
            "cache_read": safe(self.cache),
            "cache_write": 0.0,
            "non_cache_total": safe(self.input + self.output + self.reasoning),
            "total": safe(self.input + self.output + self.reasoning + self.cache),
        })
    }
}

#[derive(Default)]
struct ActivityAcc {
    llm_records: u64,
    llm_turns: u64,
    llm_calls: u64,
    tool_calls: u64,
    tool_batches: u64,
}

impl ActivityAcc {
    fn add(&mut self, meta: &Value) {
        let get = |k: &str| meta.get(k).and_then(Value::as_u64).unwrap_or(0);
        self.llm_records += get("llm_record_count");
        self.llm_turns += get("llm_turn_count");
        self.llm_calls += get("llm_call_count");
        self.tool_calls += get("tool_call_count");
        self.tool_batches += get("tool_batch_count");
    }

    fn snapshot(&self) -> Value {
        json!({
            "llm_records": self.llm_records,
            "llm_turns": self.llm_turns,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_batches": self.tool_batches,
        })
    }

    fn avg(&self, n: f64) -> Value {
        let safe = |v: u64| if n > 0.0 { v as f64 / n } else { 0.0 };
        json!({
            "llm_records": safe(self.llm_records),
            "llm_turns": safe(self.llm_turns),
            "llm_calls": safe(self.llm_calls),
            "tool_calls": safe(self.tool_calls),
            "tool_batches": safe(self.tool_batches),
        })
    }
}

fn format_duration(secs: f64) -> String {
    let total = secs.round() as i64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else {
        format!("{minutes}m{seconds:02}s")
    }
}

fn shorten(text: &str, limit: usize, from_end: bool) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() <= limit {
        return Some(trimmed.to_string());
    }
    if from_end {
        let tail: String = trimmed
            .chars()
            .rev()
            .take(limit)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        Some(format!("...{tail}"))
    } else {
        let head: String = trimmed.chars().take(limit).collect();
        Some(format!("{head}..."))
    }
}
