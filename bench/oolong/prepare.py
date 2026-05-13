#!/usr/bin/env python3
"""Prepare compact OOLONG JSONL slices for the native Lash runner."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["synth", "synth-with-labels", "real"], default="synth")
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--dataset", dest="dataset_name", default="trec_coarse")
    p.add_argument("--context-len", type=int, default=131072)
    p.add_argument("--task-group")
    p.add_argument("--task")
    p.add_argument("--config", default="dnd", help="OOLONG-real config: dnd or toy_dnd")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int)
    return p.parse_args()


def parse_answer(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (str, int, float, bool, list, dict)) or parsed is None:
            return parsed
    except Exception:
        pass
    return text


def rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.suite in {"synth", "synth-with-labels"}:
        ds = load_dataset("oolongbench/oolong-synth", split=args.split, streaming=True)
        selected = []
        for row in tqdm(ds, desc="scan oolong-synth"):
            if args.dataset_name and row.get("dataset") != args.dataset_name:
                continue
            if args.context_len and int(row.get("context_len") or 0) != args.context_len:
                continue
            if args.task_group and row.get("task_group") != args.task_group:
                continue
            if args.task and row.get("task") != args.task:
                continue
            selected.append(row)
            if args.shuffle_seed is None and args.limit and len(selected) >= args.offset + args.limit:
                break
        if args.shuffle_seed is not None:
            import random

            random.Random(args.shuffle_seed).shuffle(selected)
        selected = selected[args.offset :]
        if args.limit:
            selected = selected[: args.limit]
        for row in selected:
            context = (
                row.get("context_window_text_with_labels")
                if args.suite == "synth-with-labels"
                else row.get("context_window_text")
            )
            question = row.get("question") or ""
            yield {
                "question_id": str(row.get("id")),
                "suite": "synth_with_labels" if args.suite == "synth-with-labels" else "synth",
                "split": args.split,
                "dataset": row.get("dataset"),
                "config": None,
                "context_len": row.get("context_len"),
                "context_window_id": row.get("context_window_id"),
                "task_group": row.get("task_group"),
                "task": row.get("task"),
                "answer_type": row.get("answer_type"),
                "input_subset": row.get("input_subset"),
                "context": context,
                "question": question,
                "prompt": f"{context}\n{question}",
                "answer": parse_answer(row.get("answer")),
                "source": row,
            }
        return

    ds = load_dataset("oolongbench/oolong-real", args.config, split=args.split, streaming=True)
    selected = []
    for row in tqdm(ds, desc=f"scan oolong-real/{args.config}"):
        selected.append(row)
        if args.shuffle_seed is None and args.limit and len(selected) >= args.offset + args.limit:
            break
    if args.shuffle_seed is not None:
        import random

        random.Random(args.shuffle_seed).shuffle(selected)
    selected = selected[args.offset :]
    if args.limit:
        selected = selected[: args.limit]
    for idx, row in enumerate(selected):
        context = row.get("context_window_text") or row.get("context") or ""
        question = row.get("question") or ""
        qid = row.get("id") or row.get("question_id") or f"{args.config}-{args.split}-{idx}"
        yield {
            "question_id": str(qid),
            "suite": "real",
            "split": args.split,
            "dataset": row.get("dataset"),
            "config": args.config,
            "context_len": row.get("context_len"),
            "context_window_id": row.get("context_window_id"),
            "task_group": row.get("task_group"),
            "task": row.get("task"),
            "answer_type": row.get("answer_type"),
            "input_subset": row.get("input_subset"),
            "context": context,
            "question": question,
            "prompt": f"{context}\n{question}",
            "answer": parse_answer(row.get("answer")),
            "source": row,
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or args.output_dir / (
        "oolong_synth_with_labels.jsonl"
        if args.suite == "synth-with-labels"
        else f"oolong_{args.suite}.jsonl"
    )
    selected = list(rows(args))
    if not selected:
        raise SystemExit("no rows matched the requested OOLONG slice")
    with output.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows to {output}")
    print("example:")
    print(json.dumps({k: selected[0].get(k) for k in [
        "question_id",
        "suite",
        "dataset",
        "context_len",
        "task_group",
        "task",
        "answer",
    ]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
