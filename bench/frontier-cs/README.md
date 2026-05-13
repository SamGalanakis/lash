# Frontier-CS

Native Lash harness for [Frontier-CS](https://github.com/FrontierCS/Frontier-CS).

The runner generates one solution per selected Frontier-CS problem through
Lash's lashlang-backed RLM mode, writes full Lash trace artifacts, then scores
the solution with Frontier-CS's own evaluator.

By default this runs the algorithmic track because it is local and repeatable:
C++17 single-file solutions evaluated by Frontier-CS's Docker/go-judge stack.
The research track is also selectable, but its official default backend is
SkyPilot and many tasks need problem-specific GPU or system resources.

## Setup

```bash
bench/frontier-cs/setup.sh
```

This clones Frontier-CS into `.benchmarks/frontier-cs/source` and runs
`uv sync`. Docker is required for local algorithmic evaluation.

## Run

```bash
bench/frontier-cs/run.sh --track algorithmic --max-problems 1
```

Useful knobs:

```bash
bench/frontier-cs/run.sh \
  --track algorithmic \
  --problem-id 0 \
  --model openai/gpt-5.2 \
  --variant high \
  --batch-size 1 \
  --max-turns 50 \
  --max-context-tokens 1000000 \
  --max-output-tokens 125000
```

The root and child sessions use the locked-down benchmark RLM surface:

- `llm_query`
- `spawn_agent` with `capability: "default"`
- `continue_as`
- `list_async_handles`

No shell, filesystem, web, MCP, monitor, or task-control tools are registered
inside Lash. Evaluation happens after generation through Frontier-CS.

## Outputs

```
.benchmarks/frontier-cs/runs/<run-id>/
  manifest.json
  results.json
  predictions.jsonl
  index.html
  problems/<track>/<problem-id>/
    problem.txt
    solution.cpp or solution.py
    answer.txt
    events.jsonl
    session.db
    session.trace.jsonl
    system_prompt.txt
    trace.html
    evaluation.json
    result.json
```

## Evaluate

```bash
bench/frontier-cs/evaluate.sh .benchmarks/frontier-cs/runs/<run-id>
```
