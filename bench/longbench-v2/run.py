#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_ROOT = Path('.benchmarks/longbench-v2')
DEFAULT_MODEL = 'google/gemini-3-flash-preview'
DEFAULT_PROVIDER = 'openai-compatible'
DEFAULT_EXECUTION_MODE = 'standard'
DEFAULT_CONTEXT_APPROACH = 'rolling_history'
LONG_BENCH_ARCHIVE_URL = 'https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip'
DATASET_PRESETS: dict[str, str] = {
    'narrativeqa': 'narrativeqa',
    'qasper': 'qasper',
    'multifieldqa_en': 'multifieldqa_en',
    'multifieldqa_zh': 'multifieldqa_zh',
    'hotpotqa': 'hotpotqa',
    '2wikimqa': '2wikimqa',
    'musique': 'musique',
    'dureader': 'dureader',
    'gov_report': 'gov_report',
    'qmsum': 'qmsum',
    'multi_news': 'multi_news',
    'vcsum': 'vcsum',
    'trec': 'trec',
    'triviaqa': 'triviaqa',
    'samsum': 'samsum',
    'lsht': 'lsht',
    'passage_count': 'passage_count',
    'passage_retrieval_en': 'passage_retrieval_en',
    'passage_retrieval_zh': 'passage_retrieval_zh',
    'lcc': 'lcc',
    'repobench-p': 'repobench-p',
}


@dataclass
class Example:
    row_id: str
    dataset: str
    input: str
    context: str
    answers: list[str]
    all_classes: Any
    length: Any
    language: Any
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run Lash on LongBench-style datasets.')
    p.add_argument('--dataset-path', type=Path)
    p.add_argument('--dataset-url')
    p.add_argument('--dataset-preset', choices=sorted(DATASET_PRESETS.keys()))
    p.add_argument('--run-id')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--dataset-name', action='append', default=[])
    p.add_argument('--limit', type=int)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--provider-id', default=DEFAULT_PROVIDER)
    p.add_argument('--base-url')
    p.add_argument('--api-key')
    p.add_argument('--execution-mode', default=DEFAULT_EXECUTION_MODE)
    p.add_argument('--context-approach', default=DEFAULT_CONTEXT_APPROACH)
    p.add_argument('--variant')
    p.add_argument('--max-concurrency', type=int, default=1)
    p.add_argument('--prompt-template', choices=['default', 'qa'], default='default')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def lazy_download(url: str, dest: Path) -> Path:
    ensure_parent(dest)
    if dest.exists():
        return dest
    print(f'Downloading dataset to {dest}...', file=sys.stderr)
    with urllib.request.urlopen(url) as response, dest.open('wb') as out:
        out.write(response.read())
    return dest


def ensure_longbench_dataset(repo_root: Path, preset: str) -> Path:
    state_root = repo_root / STATE_ROOT
    data_dir = state_root / 'data'
    archive_path = data_dir / 'longbench-data.zip'
    dataset_name = DATASET_PRESETS[preset]
    direct_path = data_dir / f'{dataset_name}.jsonl'
    nested_path = data_dir / 'data' / f'{dataset_name}.jsonl'

    if direct_path.exists():
        return direct_path
    if nested_path.exists():
        return nested_path

    lazy_download(LONG_BENCH_ARCHIVE_URL, archive_path)
    print(f'Extracting {archive_path}...', file=sys.stderr)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(data_dir)

    if direct_path.exists():
        return direct_path
    if nested_path.exists():
        return nested_path
    raise SystemExit(f'Preset dataset {preset!r} was not found after extracting {archive_path}')


def resolve_dataset_path(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.dataset_path:
        return args.dataset_path.resolve()

    if args.dataset_preset:
        return ensure_longbench_dataset(repo_root, args.dataset_preset)

    STATE_ROOT_FULL = repo_root / STATE_ROOT
    data_dir = STATE_ROOT_FULL / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_url:
        name = Path(args.dataset_url.split('?', 1)[0]).name or 'dataset.jsonl'
        return lazy_download(args.dataset_url, data_dir / name)

    raise SystemExit('One of --dataset-path, --dataset-url, or --dataset-preset is required')


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    stripped = text.lstrip()
    if path.suffix.lower() == '.jsonl':
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if stripped.startswith('['):
        return json.loads(text)
    if stripped.startswith('{'):
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ('data', 'examples', 'rows'):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError(f'Unsupported JSON object dataset shape in {path}')
    raise ValueError(f'Unsupported dataset format: {path}')


def normalize_rows(rows: list[dict[str, Any]]) -> list[Example]:
    out: list[Example] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        dataset = str(row.get('dataset') or 'longbench')
        answers = row.get('answers') or []
        if isinstance(answers, str):
            answers = [answers]
        out.append(
            Example(
                row_id=str(row.get('_id') or row.get('id') or i),
                dataset=dataset,
                input=str(row.get('input') or ''),
                context=str(row.get('context') or ''),
                answers=[str(x) for x in answers],
                all_classes=row.get('all_classes'),
                length=row.get('length'),
                language=row.get('language'),
                raw=row,
            )
        )
    return out


def build_prompt(example: Example, template: str) -> str:
    if template == 'qa':
        return (
            'Read the context and answer the question as accurately as possible. '
            'Return only the final answer.\n\n'
            f'Context:\n{example.context}\n\n'
            f'Question:\n{example.input}\n'
        )
    return (
        'You are answering a LongBench-style benchmark example. '
        'Use the provided context to answer the question. '
        'Keep the answer concise and directly responsive.\n\n'
        f'Dataset: {example.dataset}\n'
        f'Language: {example.language or "unknown"}\n\n'
        f'Context:\n{example.context}\n\n'
        f'Question:\n{example.input}\n'
    )


def run_one(repo_root: Path, args: argparse.Namespace, example: Example) -> dict[str, Any]:
    prompt = build_prompt(example, args.prompt_template)
    cmd = [
        'cargo', 'run', '--release', '-p', 'lash-cli', '--',
        '--print', prompt,
        '--model', args.model,
        '--execution-mode', args.execution_mode,
        '--context-approach', args.context_approach,
    ]
    if args.variant:
        cmd.extend(['--variant', args.variant])
    if args.base_url:
        cmd.extend(['--base-url', args.base_url])
    env = os.environ.copy()
    if args.api_key:
        env['OPENAI_COMPATIBLE_API_KEY'] = args.api_key
    started = time.time()
    if args.dry_run:
        pred = ''
        exit_code = 0
        stderr = ''
    else:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
        )
        pred = proc.stdout.strip()
        exit_code = proc.returncode
        stderr = proc.stderr.strip()
    elapsed = time.time() - started
    return {
        'row_id': example.row_id,
        'dataset': example.dataset,
        'pred': pred,
        'answers': example.answers,
        'all_classes': example.all_classes,
        'length': example.length,
        'language': example.language,
        'elapsed_seconds': elapsed,
        'status': 'ok' if exit_code == 0 else 'failed',
        'exit_code': exit_code,
        'stderr': stderr,
        'input': example.input,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = resolve_dataset_path(args, repo_root)
    rows = normalize_rows(load_rows(dataset_path))
    if args.dataset_name:
        wanted = set(args.dataset_name)
        rows = [row for row in rows if row.dataset in wanted]
    if args.offset:
        rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit('No dataset rows selected')

    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run_dir = (args.output_dir or (repo_root / STATE_ROOT / 'runs' / run_id)).resolve()
    pred_model_name = f'lash-{run_id}'
    pred_dir = run_dir / 'pred' / pred_model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'run_id': run_id,
        'created_at': now_iso(),
        'dataset_path': str(dataset_path),
        'dataset_url': args.dataset_url,
        'dataset_preset': args.dataset_preset,
        'selected_count': len(rows),
        'dataset_names': sorted({row.dataset for row in rows}),
        'model': args.model,
        'provider_id': args.provider_id,
        'execution_mode': args.execution_mode,
        'context_approach': args.context_approach,
        'variant': args.variant,
        'max_concurrency': args.max_concurrency,
        'prompt_template': args.prompt_template,
        'pred_model_name': pred_model_name,
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    results: list[dict[str, Any]] = []
    if args.max_concurrency <= 1:
        for row in rows:
            results.append(run_one(repo_root, args, row))
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            futures = [pool.submit(run_one, repo_root, args, row) for row in rows]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: (r['dataset'], str(r['row_id'])))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result['dataset']].append(result)

    for dataset_name, items in grouped.items():
        path = pred_dir / f'{dataset_name}.jsonl'
        with path.open('w') as fh:
            for item in items:
                fh.write(json.dumps({
                    'pred': item['pred'],
                    'answers': item['answers'],
                    'all_classes': item['all_classes'],
                    'length': item['length'],
                }, ensure_ascii=False) + '\n')

    summary = {
        'run_id': run_id,
        'created_at': manifest['created_at'],
        'finished_at': now_iso(),
        'result_count': len(results),
        'ok_count': sum(1 for r in results if r['status'] == 'ok'),
        'failed_count': sum(1 for r in results if r['status'] != 'ok'),
        'dataset_counts': {k: len(v) for k, v in grouped.items()},
        'results': results,
    }
    (run_dir / 'results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(str(run_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
