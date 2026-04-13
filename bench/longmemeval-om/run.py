#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = REPO_ROOT / ".benchmarks" / "longmemeval-om"
DATA_DIR = STATE_ROOT / "data"
RUNS_DIR = STATE_ROOT / "runs"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
DEFAULT_DATASET_PATH = DATA_DIR / "longmemeval_s_cleaned.json"
DEFAULT_MODEL = "gpt-5.4-mini"

LONGMEMEVAL_GUIDANCE = textwrap.dedent(
    """\
    You are an assistant with excellent long-term memory capabilities being evaluated on the LongMemEval benchmark.

    Your primary objective is to accurately answer questions based on information from past conversations.

    Key behaviors:
    1. Always check your memory for relevant information before answering.
    2. Reference specific details from previous messages when available.
    3. If you do not have the information needed to answer a question, clearly state "I don't have that information in my memory" or equivalent.
    4. Track temporal relationships between events carefully.
    5. Handle updates to previously stated information by using the most recent version.
    6. Be precise and avoid making assumptions beyond what you remember.

    Question types you may encounter:
    - Information extraction
    - Multi-session reasoning
    - Temporal reasoning
    - Knowledge updates
    - Abstention

    Remember: accuracy is more important than being generally helpful. Only answer based on what you actually remember.
    """
).strip()

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
def empty_usage() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def normalized_usage(raw: dict | None) -> dict[str, int]:
    usage = empty_usage()
    if not isinstance(raw, dict):
        return usage
    for field in TOKEN_FIELDS:
        value = raw.get(field, 0)
        usage[field] = int(value) if isinstance(value, int | float) else 0
    return usage


def usage_totals(raw: dict[str, int]) -> dict[str, int]:
    usage = normalized_usage(raw)
    usage["total_tokens"] = (
        usage["input_tokens"] + usage["output_tokens"] + usage["reasoning_tokens"]
    )
    usage["context_total_tokens"] = usage["total_tokens"] + usage["cached_input_tokens"]
    return usage


def add_usage(target: dict[str, int], delta: dict[str, int]) -> None:
    for field in TOKEN_FIELDS:
        target[field] += int(delta.get(field, 0))


def summarize_entries(entries: list[dict]) -> dict:
    total = empty_usage()
    by_source: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    source_model_rows: list[dict] = []

    for entry in entries:
        source = str(entry["source"])
        model = str(entry["model"])
        usage = normalized_usage(entry.get("usage"))
        add_usage(total, usage)
        add_usage(by_source.setdefault(source, empty_usage()), usage)
        add_usage(by_model.setdefault(model, empty_usage()), usage)
        source_model_rows.append(
            {
                "source": source,
                "model": model,
                "usage": usage_totals(usage),
            }
        )

    source_model_rows.sort(key=lambda row: (row["source"], row["model"]))
    return {
        "entry_count": len(entries),
        "usage": usage_totals(total),
        "by_source": {
            source: usage_totals(usage)
            for source, usage in sorted(by_source.items(), key=lambda item: item[0])
        },
        "by_model": {
            model: usage_totals(usage)
            for model, usage in sorted(by_model.items(), key=lambda item: item[0])
        },
        "by_source_model": source_model_rows,
    }


def format_usage_line(label: str, usage: dict) -> str:
    return (
        f"{label}: input={usage['input_tokens']:,} cached={usage['cached_input_tokens']:,} "
        f"output={usage['output_tokens']:,} reasoning={usage['reasoning_tokens']:,} "
        f"total={usage['total_tokens']:,} context_total={usage['context_total_tokens']:,}"
    )


def render_summary_text(summary: dict) -> str:
    lines = [
        f"Run: {summary['run_id']}",
        f"Questions completed: {summary['completed_question_count']}/{summary['question_count']}",
        f"Started: {summary['started_at']}",
        f"Finished: {summary['finished_at']}",
        f"Duration seconds: {summary['duration_seconds']}",
        "",
        format_usage_line("Total", summary["token_usage"]["usage"]),
        "",
        "By phase:",
    ]
    for phase, phase_summary in summary["token_usage"]["by_phase"].items():
        lines.append(f"- {format_usage_line(phase, phase_summary['usage'])}")
    lines.extend(["", "By source:"])
    for source, usage in summary["token_usage"]["by_source"].items():
        lines.append(f"- {format_usage_line(source, usage)}")
    lines.extend(["", "By model:"])
    for model, usage in summary["token_usage"]["by_model"].items():
        lines.append(f"- {format_usage_line(model, usage)}")
    return "\n".join(lines) + "\n"


def write_run_summary(
    output_dir: Path,
    manifest: dict,
    started_at: str,
    finished_at: str,
    question_summaries: list[dict],
    phase_entries: dict[str, list[dict]],
    all_entries: list[dict],
    turn_counts: dict[str, int],
) -> None:
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    token_summary = summarize_entries(all_entries)
    summary = {
        "run_id": manifest["run_id"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "provider_id": manifest["provider_id"],
        "variant": manifest["variant"],
        "execution_mode": manifest["execution_mode"],
        "context_approach": manifest["context_approach"],
        "question_count": manifest["question_count"],
        "completed_question_count": len(question_summaries),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": int((finished_dt - started_dt).total_seconds()),
        "turn_counts": turn_counts,
        "token_usage": {
            **token_summary,
            "by_phase": {
                phase: summarize_entries(entries)
                for phase, entries in sorted(phase_entries.items(), key=lambda item: item[0])
            },
        },
        "questions": question_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "summary.txt").write_text(render_summary_text(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LongMemEval with Lash using gpt-5.4-mini, observational_memory, and rlm by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lash-binary",
        type=Path,
        default=REPO_ROOT / "target" / "release" / "lash",
        help="Path to the lash CLI binary to execute.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the cleaned LongMemEval JSON dataset.",
    )
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_DATASET_URL,
        help="Download URL used when the dataset file is missing.",
    )
    parser.add_argument("--run-id", help="Run identifier used under .benchmarks/longmemeval-om/runs/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory for run artifacts and hypotheses.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name passed to lash.")
    parser.add_argument(
        "--provider-id",
        default="codex",
        help="Provider id to activate inside the isolated LASH_HOME for the benchmark run.",
    )
    parser.add_argument("--variant", help="Provider-native model variant passed to lash.")
    parser.add_argument("--limit", type=int, help="Maximum number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many dataset entries first.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Run only the specified question id. Repeat for multiple ids.",
    )
    parser.add_argument(
        "--keep-existing-run",
        action="store_true",
        help="Allow writing into an existing output directory instead of failing.",
    )
    parser.add_argument(
        "--om-observation-message-tokens",
        type=int,
        help="Override the OM observation message-token threshold.",
    )
    parser.add_argument(
        "--om-observation-buffer-tokens",
        type=int,
        help="Override the OM observation buffer size in tokens.",
    )
    parser.add_argument(
        "--om-observation-block-after-tokens",
        type=int,
        help="Override the OM observation hard block-after threshold.",
    )
    parser.add_argument(
        "--om-observation-max-tokens-per-batch",
        type=int,
        help="Override the OM observer batch size limit.",
    )
    parser.add_argument(
        "--om-previous-observer-tokens",
        type=int,
        help="Override how much prior OM memory the observer sees.",
    )
    parser.add_argument(
        "--om-reflection-observation-tokens",
        type=int,
        help="Override the OM reflection activation threshold.",
    )
    parser.add_argument(
        "--om-reflection-buffer-activation-percent",
        type=int,
        help="Override when async reflection buffering begins, as a percent.",
    )
    parser.add_argument(
        "--om-reflection-block-after-tokens",
        type=int,
        help="Override the OM reflection hard block-after threshold.",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_dataset(dataset_path: Path, dataset_url: str) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        return
    print(f"Downloading dataset to {dataset_path}...", file=sys.stderr)
    urllib.request.urlretrieve(dataset_url, dataset_path)


def prepare_lash_home(lash_home: Path, provider_id: str) -> None:
    lash_home.mkdir(parents=True, exist_ok=True)
    (lash_home / "sessions").mkdir(parents=True, exist_ok=True)
    source_config = Path.home() / ".lash" / "config.json"
    target_config = lash_home / "config.json"
    if not source_config.exists() or target_config.exists():
        return

    config = json.loads(source_config.read_text())
    providers = config.get("providers")
    if not isinstance(providers, dict) or provider_id not in providers:
        available = ", ".join(sorted(providers)) if isinstance(providers, dict) else "<none>"
        raise SystemExit(
            f"provider `{provider_id}` is not configured in {source_config}; available providers: {available}"
        )
    config["active_provider"] = provider_id
    target_config.write_text(json.dumps(config, indent=2) + "\n")


def load_entries(dataset_path: Path) -> list[dict]:
    with dataset_path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {dataset_path}")
    return data


def select_entries(entries: list[dict], question_ids: list[str], offset: int, limit: int | None) -> list[dict]:
    if question_ids:
        wanted = set(question_ids)
        entries = [entry for entry in entries if entry.get("question_id") in wanted]
    if offset:
        entries = entries[offset:]
    if limit is not None:
        entries = entries[:limit]
    return entries


def render_transcript(session_turns: list[dict]) -> str:
    lines: list[str] = []
    for turn in session_turns:
        role = str(turn.get("role", "unknown")).strip().upper()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_ingest_prompt(entry: dict, session_index: int, session_total: int, session_date: str, session_turns: list[dict]) -> str:
    transcript = render_transcript(session_turns)
    return textwrap.dedent(
        f"""\
        Historical memory ingestion step for LongMemEval.

        The structured benchmark payload for this turn is also bound in lashlang globals as `benchmark` and `input`.
        Internalize the following dated conversation into long-term memory.
        Do not answer the conversation itself.
        Do not continue the conversation.
        Reply with exactly this single word and nothing else:
        stored

        Question ID: {entry["question_id"]}
        Session: {session_index}/{session_total}
        Session date: {session_date}

        Transcript:
        {transcript}
        """
    ).strip()


def build_question_prompt(entry: dict) -> str:
    return textwrap.dedent(
        f"""\
        LongMemEval evaluation question.

        The structured benchmark payload for this turn is also bound in lashlang globals as `benchmark` and `input`.

        Today's date: {entry["question_date"]}
        Question type: {entry["question_type"]}

        Answer the following question using only what you remember from the previously ingested sessions in this same benchmark run.
        If the answer is unavailable, say you do not have that information in your memory.
        Be concise and answer directly.

        Question: {entry["question"]}
        """
    ).strip()


def build_ingest_globals(
    entry: dict,
    session_index: int,
    session_total: int,
    session_date: str,
    session_turns: list[dict],
) -> dict:
    return {
        "benchmark": {
            "name": "LongMemEval",
            "question_id": entry["question_id"],
            "phase": "ingest",
        },
        "input": {
            "kind": "ingest",
            "question_id": entry["question_id"],
            "session_index": session_index,
            "session_total": session_total,
            "session_date": session_date,
            "session_turns": session_turns,
            "transcript": render_transcript(session_turns),
        },
    }


def build_question_globals(entry: dict) -> dict:
    return {
        "benchmark": {
            "name": "LongMemEval",
            "question_id": entry["question_id"],
            "phase": "answer",
        },
        "input": {
            "kind": "question",
            "question_id": entry["question_id"],
            "question_date": entry["question_date"],
            "question_type": entry["question_type"],
            "question": entry["question"],
        },
    }


def build_lash_command(
    args: argparse.Namespace,
    session_filename: str,
    prompt: str,
    vars_path: Path | None,
    usage_json_path: Path,
) -> list[str]:
    cmd = [
        str(args.lash_binary),
        "--resume",
        session_filename,
        "--model",
        args.model,
        "--execution-mode",
        "rlm",
        "--context-approach",
        "observational_memory",
        "--await-background-work",
        "--prompt-append",
        f"guidance={LONGMEMEVAL_GUIDANCE}",
        "--print",
        prompt,
        "--turn-usage-json",
        str(usage_json_path),
    ]
    if vars_path is not None:
        cmd.extend(["--rlm-vars-file", str(vars_path)])
    if args.variant:
        cmd.extend(["--variant", args.variant])

    om_flags = {
        "--om-observation-message-tokens": args.om_observation_message_tokens,
        "--om-observation-buffer-tokens": args.om_observation_buffer_tokens,
        "--om-observation-block-after-tokens": args.om_observation_block_after_tokens,
        "--om-observation-max-tokens-per-batch": args.om_observation_max_tokens_per_batch,
        "--om-previous-observer-tokens": args.om_previous_observer_tokens,
        "--om-reflection-observation-tokens": args.om_reflection_observation_tokens,
        "--om-reflection-buffer-activation-percent": args.om_reflection_buffer_activation_percent,
        "--om-reflection-block-after-tokens": args.om_reflection_block_after_tokens,
    }
    for flag, value in om_flags.items():
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)


def run_lash_turn(
    args: argparse.Namespace,
    lash_home: Path,
    session_filename: str,
    prompt: str,
    turn_dir: Path,
    phase: str,
    globals_payload: dict | None = None,
) -> dict:
    turn_dir.mkdir(parents=True, exist_ok=True)
    usage_json_path = turn_dir / "usage.json"
    vars_path = None
    if globals_payload is not None:
        vars_path = turn_dir / "rlm-vars.json"
        vars_path.write_text(json.dumps(globals_payload, indent=2) + "\n")
    cmd = build_lash_command(args, session_filename, prompt, vars_path, usage_json_path)
    (turn_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (turn_dir / "prompt.txt").write_text(prompt)
    env = os.environ.copy()
    env["LASH_HOME"] = str(lash_home)

    stdout_path = turn_dir / "stdout.txt"
    stderr_path = turn_dir / "stderr.txt"
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    (turn_dir / "returncode.txt").write_text(f"{result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"lash turn failed for {session_filename}: rc={result.returncode}\n{stderr_text.strip()}"
        )
    usage_artifact = json.loads(usage_json_path.read_text())
    delta_entries = usage_artifact.get("delta_entries", [])
    return {
        "output": stdout_text.strip(),
        "usage_entries": delta_entries,
        "usage_artifact": usage_artifact,
    }


def main() -> int:
    args = parse_args()
    if not args.lash_binary.exists():
        raise SystemExit(f"lash binary not found at {args.lash_binary}")

    ensure_dataset(args.dataset, args.dataset_url)
    entries = select_entries(load_entries(args.dataset), args.question_id, args.offset, args.limit)
    if not entries:
        raise SystemExit("no benchmark entries selected")

    run_id = args.run_id or utc_stamp()
    output_dir = args.output_dir or (RUNS_DIR / run_id)
    if output_dir.exists() and not args.keep_existing_run:
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lash_home = output_dir / "lash-home"
    prepare_lash_home(lash_home, args.provider_id)
    started_at = datetime.now(UTC).isoformat()

    manifest = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "dataset_url": args.dataset_url,
        "model": args.model,
        "provider_id": args.provider_id,
        "variant": args.variant,
        "execution_mode": "rlm",
        "context_approach": "observational_memory",
        "question_count": len(entries),
        "started_at": started_at,
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    question_summaries: list[dict] = []
    phase_entries: dict[str, list[dict]] = {"ingest": [], "answer": []}
    all_entries: list[dict] = []
    turn_counts = {"ingest": 0, "answer": 0}
    hypotheses_path = output_dir / "hypotheses.jsonl"
    with hypotheses_path.open("w") as hypotheses_file:
        for index, entry in enumerate(entries, start=1):
            question_id = str(entry["question_id"])
            question_dir = output_dir / "questions" / safe_slug(question_id)
            question_dir.mkdir(parents=True, exist_ok=True)
            session_filename = f"{safe_slug(question_id)}.db"
            question_entries: list[dict] = []
            question_phase_entries: dict[str, list[dict]] = {"ingest": [], "answer": []}

            print(f"[{index}/{len(entries)}] ingesting {question_id}", file=sys.stderr)
            sessions = list(zip(entry["haystack_dates"], entry["haystack_sessions"]))
            for session_index, (session_date, session_turns) in enumerate(sessions, start=1):
                prompt = build_ingest_prompt(
                    entry,
                    session_index,
                    len(sessions),
                    session_date,
                    session_turns,
                )
                result = run_lash_turn(
                    args,
                    lash_home,
                    session_filename,
                    prompt,
                    question_dir / "turns" / f"{session_index:03d}-ingest",
                    "ingest",
                    build_ingest_globals(
                        entry,
                        session_index,
                        len(sessions),
                        session_date,
                        session_turns,
                    ),
                )
                question_entries.extend(result["usage_entries"])
                question_phase_entries["ingest"].extend(result["usage_entries"])
                phase_entries["ingest"].extend(result["usage_entries"])
                all_entries.extend(result["usage_entries"])
                turn_counts["ingest"] += 1

            print(f"[{index}/{len(entries)}] answering {question_id}", file=sys.stderr)
            answer_result = run_lash_turn(
                args,
                lash_home,
                session_filename,
                build_question_prompt(entry),
                question_dir / "turns" / "999-answer",
                "answer",
                build_question_globals(entry),
            )
            question_entries.extend(answer_result["usage_entries"])
            question_phase_entries["answer"].extend(answer_result["usage_entries"])
            phase_entries["answer"].extend(answer_result["usage_entries"])
            all_entries.extend(answer_result["usage_entries"])
            turn_counts["answer"] += 1
            answer = answer_result["output"]
            (question_dir / "answer.txt").write_text(answer + "\n")
            question_summary = {
                "question_id": question_id,
                "question_type": entry.get("question_type"),
                "session_count": len(sessions),
                "turn_counts": {"ingest": len(sessions), "answer": 1},
                "token_usage": {
                    **summarize_entries(question_entries),
                    "by_phase": {
                        "ingest": summarize_entries(question_phase_entries["ingest"]),
                        "answer": summarize_entries(answer_result["usage_entries"]),
                    },
                },
            }
            (question_dir / "summary.json").write_text(json.dumps(question_summary, indent=2) + "\n")
            question_summaries.append(question_summary)
            hypotheses_file.write(
                json.dumps({"question_id": question_id, "hypothesis": answer}, ensure_ascii=True)
                + "\n"
            )
            hypotheses_file.flush()
            write_run_summary(
                output_dir,
                manifest,
                started_at,
                datetime.now(UTC).isoformat(),
                question_summaries,
                phase_entries,
                all_entries,
                turn_counts,
            )

    write_run_summary(
        output_dir,
        manifest,
        started_at,
        datetime.now(UTC).isoformat(),
        question_summaries,
        phase_entries,
        all_entries,
        turn_counts,
    )
    print(hypotheses_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
