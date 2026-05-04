#!/usr/bin/env python3
"""Summarize lash runtime/UI perf reports and dhat heap profiles.

Usage:
  perfreport.py REPORT.json                  # human summary
  perfreport.py REPORT.json --diff BASELINE  # before/after comparison
  perfreport.py PROFILE.dhat.json --top 25   # top heap consumers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def fmt_bytes(n: float | int) -> str:
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            if unit == "B":
                return f"{sign}{int(n)}{unit}"
            return f"{sign}{n:.2f}{unit}"
        n /= 1024
    return f"{sign}{n:.2f}GiB"


def fmt_kb(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    return fmt_bytes(float(n) * 1024)


def fmt_ms(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.2f}ms"


def is_dhat(payload: dict[str, Any]) -> bool:
    return "dhatFileVersion" in payload and "ftbl" in payload


def is_runtime_report(payload: dict[str, Any]) -> bool:
    return "summary" in payload and any(
        "phase_summary" in s for s in payload.get("summary", []) if isinstance(s, dict)
    )


def is_ui_report(payload: dict[str, Any]) -> bool:
    return (
        "parameters" in payload
        and isinstance(payload.get("scenarios"), list)
        and any("budgets" in s for s in payload.get("scenarios", []) if isinstance(s, dict))
    )


def summarize_runtime(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"# runtime-perf report  ({report.get('created_at', '?')[:19]}, {report.get('runs')} runs × {report.get('chat_turns')} turns)"
    )
    lines.append(f"version: {report.get('version', '?')}  scenarios: {', '.join(report.get('scenarios', []))}")
    if report.get("dhat_out"):
        lines.append(f"dhat profile: {report['dhat_out']}")
    lines.append("")

    for s in report.get("summary", []):
        scenario = s["scenario"]
        lines.append(f"## scenario: {scenario}  ({s['runs']} runs × {s['chat_turns']} turns)")
        lines.append("")
        lines.append("phase totals (median across runs):")
        lines.append(
            f"  build_runtime         {fmt_ms(s['build_runtime_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['build_runtime_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['build_runtime_live_bytes']['median']):>10s}"
        )
        lines.append(
            f"  seed_state            {fmt_ms(s['seed_state_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['seed_state_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['seed_state_live_bytes']['median']):>10s}"
        )
        lines.append(
            f"  run_turn (sum)        {fmt_ms(s['run_turn_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['run_turn_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['run_turn_live_bytes']['median']):>10s}"
        )
        lines.append(
            f"  await_background      {fmt_ms(s['await_background_work_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['await_background_work_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['await_background_work_live_bytes']['median']):>10s}"
        )
        lines.append(
            f"  export_state          {fmt_ms(s['export_state_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['export_state_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['export_state_live_bytes']['median']):>10s}"
        )
        lines.append(
            f"  TOTAL                 {fmt_ms(s['total_ms']['median']):>10s}  "
            f"alloc={fmt_bytes(s['total_alloc_bytes']['median']):>10s}  "
            f"live={fmt_bytes(s['total_live_bytes']['median']):>10s}"
        )
        lines.append("")

        rss = s.get("rss_after_export_kb")
        if rss is not None:
            growth = s.get("rss_growth_kb")
            hwm = s.get("hwm_growth_kb")
            lines.append(
                f"memory: rss_after={fmt_kb(rss['median'])}  "
                f"rss_growth={fmt_kb(growth['median']) if growth else 'n/a'}  "
                f"hwm_growth={fmt_kb(hwm['median']) if hwm else 'n/a'}"
            )
            lines.append("")

        sample = (
            f"session_nodes={s['sample_session_nodes']}  "
            f"active_path_messages={s['sample_active_path_messages']}"
        )
        lines.append(sample)
        lines.append("")

        ps = s.get("phase_summary", {})
        if ps:
            lines.append("hot phases by median duration:")
            ranked = sorted(ps.items(), key=lambda kv: -kv[1]["duration_ms"]["median"])
            for name, m in ranked:
                lines.append(
                    f"  {name:30s}  dur={fmt_ms(m['duration_ms']['median']):>9s}  "
                    f"alloc={fmt_bytes(m['alloc_bytes']['median']):>10s}  "
                    f"live={fmt_bytes(m['live_bytes']['median']):>10s}"
                )
            lines.append("")
            lines.append("hot phases by median allocation bytes:")
            ranked = sorted(ps.items(), key=lambda kv: -kv[1]["alloc_bytes"]["median"])
            for name, m in ranked:
                lines.append(
                    f"  {name:30s}  alloc={fmt_bytes(m['alloc_bytes']['median']):>10s}  "
                    f"dur={fmt_ms(m['duration_ms']['median']):>9s}"
                )
            lines.append("")

        first = s.get("first_turn") or {}
        last = s.get("last_turn") or {}
        steady = s.get("steady_state_turn") or {}
        if first and last:
            d_total = last["total_ms"]["median"] - first["total_ms"]["median"]
            d_alloc = last["total_alloc_bytes"]["median"] - first["total_alloc_bytes"]["median"]
            d_live = last["total_live_bytes"]["median"] - first["total_live_bytes"]["median"]
            lines.append("turn growth (last vs first, median across runs):")
            lines.append(
                f"  total_ms     first={fmt_ms(first['total_ms']['median']):>9s}  "
                f"steady={fmt_ms(steady.get('total_ms', {}).get('median')) if steady else 'n/a':>9s}  "
                f"last={fmt_ms(last['total_ms']['median']):>9s}  Δ={d_total:+.2f}ms"
            )
            lines.append(
                f"  alloc_bytes  first={fmt_bytes(first['total_alloc_bytes']['median']):>10s}  "
                f"steady={fmt_bytes(steady.get('total_alloc_bytes', {}).get('median', 0)):>10s}  "
                f"last={fmt_bytes(last['total_alloc_bytes']['median']):>10s}  Δ={fmt_bytes(d_alloc)}"
            )
            lines.append(
                f"  live_bytes   first={fmt_bytes(first['total_live_bytes']['median']):>10s}  "
                f"steady={fmt_bytes(steady.get('total_live_bytes', {}).get('median', 0)):>10s}  "
                f"last={fmt_bytes(last['total_live_bytes']['median']):>10s}  Δ={fmt_bytes(d_live)}"
            )
            lines.append("")

        # per-turn drift across the run (signal of O(n) or O(n²) regressions)
        results_for_scenario = [r for r in report.get("results", []) if r["scenario"] == scenario]
        if results_for_scenario:
            r0 = results_for_scenario[0]
            turns = r0.get("turns", [])
            if len(turns) >= 4:
                lines.append("per-turn drift (run #1, picks signs of O(n) growth):")
                lines.append(
                    f"  {'turn':>4}  {'run_ms':>8}  {'alloc':>11}  {'live_Δ':>11}  {'rss_kb':>8}"
                )
                for t in turns:
                    a = t["allocations"]["total"]
                    rss = t["memory"].get("rss_after_await_kb")
                    lines.append(
                        f"  {t['turn_index']:>4}  "
                        f"{t['run_turn_ms']:>6.2f}ms  "
                        f"{fmt_bytes(a['bytes_allocated']):>11s}  "
                        f"{fmt_bytes(a['net_live_bytes']):>11s}  "
                        f"{rss if rss is not None else 'n/a':>8}"
                    )
                lines.append("")

        lines.append("")
    return "\n".join(lines)


def summarize_ui(report: dict[str, Any]) -> str:
    params = report.get("parameters", {})
    profile = params.get("profile", "?")
    scenario_names = ", ".join(params.get("scenarios", []))
    lines: list[str] = []
    lines.append(
        f"# ui-perf report  ({report.get('created_at', '?')[:19]}, {params.get('runs')} runs, {profile} profile)"
    )
    git = report.get("git", {})
    dirty = "dirty" if git.get("dirty") else "clean"
    lines.append(
        f"version: {report.get('version', '?')}  build={report.get('build_mode', '?')}  "
        f"git={git.get('sha', '?')} ({dirty})  scenarios: {scenario_names}"
    )
    if params.get("dhat_out"):
        lines.append(f"dhat profile: {params['dhat_out']}")
    if params.get("compare_inputs"):
        lines.append("comparison inputs: " + ", ".join(str(p) for p in params["compare_inputs"]))
    lines.append("")

    interesting = [
        "initial_render_ms",
        "height_cache_rebuild_ms",
        "steady_scroll_selection_render_ms",
        "render_build_ms",
        "diff_scan_ms",
        "foreground_handler_ms",
        "input_control_latency_ms",
        "render_frame_ms",
        "snapshot_ms",
        "file_index_suggestion_query_ms",
        "file_index_refresh_visible_ms",
        "total_ms",
    ]

    for scenario in report.get("scenarios", []):
        lines.append(f"## scenario: {scenario.get('scenario', '?')}")
        summary = scenario.get("summary", {})
        for metric in interesting:
            if metric not in summary:
                continue
            m = summary[metric]
            lines.append(
                f"  {metric:36s} p50={fmt_ms(m.get('p50')):>9s}  "
                f"p95={fmt_ms(m.get('p95')):>9s}  p99={fmt_ms(m.get('p99')):>9s}  "
                f"max={fmt_ms(m.get('max')):>9s}"
            )
        budgets = scenario.get("budgets", [])
        if budgets:
            failed = [b for b in budgets if not b.get("passed")]
            if failed:
                lines.append("  budget failures:")
                for b in failed:
                    lines.append(
                        f"    {b['metric']} {b['statistic']} {fmt_ms(b['actual_ms'])} > {fmt_ms(b['budget_ms'])}"
                    )
            else:
                lines.append("  budgets: pass")
        counters = scenario.get("counters", {})
        if counters:
            short = ", ".join(f"{k}={v}" for k, v in counters.items())
            lines.append(f"  counters: {short}")
        lines.append("")
    return "\n".join(lines)


def metric_pairs(name: str, baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    rows: list[str] = []

    def cmp(metric: str, b: float | None, c: float | None, fmt) -> str:
        if b is None or c is None:
            return f"  {metric:30s} baseline={'n/a':>12s}  current={'n/a':>12s}"
        delta = c - b
        pct = (delta / b * 100.0) if b else 0.0
        delta_str = fmt(delta)
        if delta >= 0 and not delta_str.startswith("+"):
            delta_str = "+" + delta_str
        return (
            f"  {metric:30s} baseline={fmt(b):>12s}  current={fmt(c):>12s}  "
            f"Δ={delta_str:>12s} ({pct:+.1f}%)"
        )

    rows.append(f"### {name}")
    rows.append(
        cmp("run_turn_ms",
            baseline["run_turn_ms"]["median"], current["run_turn_ms"]["median"], lambda v: fmt_ms(v))
    )
    rows.append(
        cmp("total_ms",
            baseline["total_ms"]["median"], current["total_ms"]["median"], lambda v: fmt_ms(v))
    )
    rows.append(
        cmp("run_turn_alloc_bytes",
            baseline["run_turn_alloc_bytes"]["median"], current["run_turn_alloc_bytes"]["median"], fmt_bytes)
    )
    rows.append(
        cmp("total_alloc_bytes",
            baseline["total_alloc_bytes"]["median"], current["total_alloc_bytes"]["median"], fmt_bytes)
    )
    rows.append(
        cmp("total_live_bytes",
            baseline["total_live_bytes"]["median"], current["total_live_bytes"]["median"], fmt_bytes)
    )
    if baseline.get("rss_growth_kb") and current.get("rss_growth_kb"):
        rows.append(
            cmp("rss_growth",
                baseline["rss_growth_kb"]["median"], current["rss_growth_kb"]["median"], fmt_kb)
        )

    bp = baseline.get("phase_summary", {})
    cp = current.get("phase_summary", {})
    if bp and cp:
        rows.append("  phase deltas (median duration):")
        for ph in sorted(set(bp) | set(cp)):
            b = bp.get(ph, {}).get("duration_ms", {}).get("median")
            c = cp.get(ph, {}).get("duration_ms", {}).get("median")
            if b is None or c is None:
                continue
            delta = c - b
            pct = (delta / b * 100.0) if b else 0.0
            rows.append(
                f"    {ph:28s} baseline={fmt_ms(b):>9s}  current={fmt_ms(c):>9s}  "
                f"Δ={delta:+.2f}ms ({pct:+.1f}%)"
            )
    return rows


def diff_runtime(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    lines = ["# runtime-perf diff", ""]
    lines.append(f"baseline: {baseline.get('created_at', '?')[:19]}  scenarios: {', '.join(baseline.get('scenarios', []))}")
    lines.append(f"current:  {current.get('created_at', '?')[:19]}  scenarios: {', '.join(current.get('scenarios', []))}")
    lines.append("")
    bs = {s["scenario"]: s for s in baseline.get("summary", [])}
    cs = {s["scenario"]: s for s in current.get("summary", [])}
    for name in sorted(set(bs) & set(cs)):
        lines.extend(metric_pairs(name, bs[name], cs[name]))
        lines.append("")
    return "\n".join(lines)


def diff_ui(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    lines = ["# ui-perf diff", ""]
    bp = baseline.get("parameters", {})
    cp = current.get("parameters", {})
    lines.append(
        f"baseline: {baseline.get('created_at', '?')[:19]}  profile={bp.get('profile', '?')}"
    )
    lines.append(
        f"current:  {current.get('created_at', '?')[:19]}  profile={cp.get('profile', '?')}"
    )
    lines.append("")
    bs = {s["scenario"]: s for s in baseline.get("scenarios", [])}
    cs = {s["scenario"]: s for s in current.get("scenarios", [])}
    for name in sorted(set(bs) & set(cs)):
        lines.append(f"### {name}")
        bsum = bs[name].get("summary", {})
        csum = cs[name].get("summary", {})
        for metric in sorted(set(bsum) & set(csum)):
            for stat in ("p95", "p99", "max"):
                b = bsum[metric].get(stat)
                c = csum[metric].get(stat)
                if b is None or c is None:
                    continue
                delta = c - b
                pct = (delta / b * 100.0) if b else 0.0
                lines.append(
                    f"  {metric:36s} {stat:>3s} baseline={fmt_ms(b):>9s}  "
                    f"current={fmt_ms(c):>9s}  Δ={delta:+.2f}ms ({pct:+.1f}%)"
                )
        lines.append("")
    return "\n".join(lines)


def summarize_dhat(payload: dict[str, Any], top: int) -> str:
    ftbl: list[str] = payload["ftbl"]
    pps: list[dict[str, Any]] = payload["pps"]
    cmd = payload.get("cmd", "?")
    total_bytes = sum(p["tb"] for p in pps)
    total_blocks = sum(p["tbk"] for p in pps)
    total_max_bytes = sum(p["mb"] for p in pps)

    def frame_label(idx: int) -> str:
        s = ftbl[idx]
        # Strip leading hex address and any " (path:line:col)" tail.
        if s.startswith("0x"):
            sp = s.find(": ")
            if sp != -1:
                s = s[sp + 2 :]
        if " (" in s:
            s = s.split(" (")[0]
        return s

    def pretty_stack(fs: list[int], depth: int = 6) -> list[str]:
        # dhat stacks are root-first; reverse to user-first then keep top.
        labels = [frame_label(i) for i in fs]
        # Skip the dhat allocator hook frames at the top.
        skip = 0
        while skip < len(labels) and (
            "dhat::Alloc" in labels[skip]
            or "__rust_alloc" in labels[skip]
            or "RawVecInner" in labels[skip]
            or "raw_vec" in labels[skip]
            or "alloc::alloc" in labels[skip]
        ):
            skip += 1
        labels = labels[skip:]
        return labels[:depth]

    def fmt_block(p: dict[str, Any]) -> list[str]:
        blocks = pretty_stack(p["fs"], depth=8)
        out = [
            f"  total={fmt_bytes(p['tb']):>10s}  blocks={p['tbk']:>7d}  "
            f"max_live={fmt_bytes(p['mb']):>10s}  end_live={fmt_bytes(p['gb']):>10s}"
        ]
        for label in blocks:
            out.append(f"    {label}")
        return out

    lines = ["# dhat heap summary", ""]
    lines.append(f"command: {cmd}")
    lines.append(
        f"total alloc={fmt_bytes(total_bytes)}  blocks={total_blocks}  "
        f"max_live(sum-of-pps)={fmt_bytes(total_max_bytes)}  pps={len(pps)}"
    )
    lines.append("")

    lines.append(f"## top {top} call stacks by total bytes allocated")
    for p in sorted(pps, key=lambda p: -p["tb"])[:top]:
        lines.extend(fmt_block(p))
        lines.append("")

    lines.append(f"## top {top} call stacks by max live bytes")
    for p in sorted(pps, key=lambda p: -p["mb"])[:top]:
        lines.extend(fmt_block(p))
        lines.append("")

    lines.append(f"## top {top} call stacks by block count")
    for p in sorted(pps, key=lambda p: -p["tbk"])[:top]:
        lines.extend(fmt_block(p))
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("report", type=Path, help="runtime-perf JSON, ui-perf JSON, or *.dhat.json")
    parser.add_argument("--diff", type=Path, help="baseline runtime-perf JSON to diff against")
    parser.add_argument("--top", type=int, default=20, help="top-N call stacks for dhat output (default 20)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text())
    if is_dhat(payload):
        print(summarize_dhat(payload, args.top))
        return 0
    if args.diff:
        baseline = json.loads(args.diff.read_text())
        if is_runtime_report(payload) and is_runtime_report(baseline):
            print(diff_runtime(baseline, payload))
            return 0
        if is_ui_report(payload) and is_ui_report(baseline):
            print(diff_ui(baseline, payload))
            return 0
        print("error: --diff expects matching runtime-perf or ui-perf JSON pairs", file=sys.stderr)
        return 2
    if is_runtime_report(payload):
        print(summarize_runtime(payload))
        return 0
    if is_ui_report(payload):
        print(summarize_ui(payload))
        return 0
    print(f"error: unrecognized report format at {args.report}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
