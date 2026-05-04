# AppWorld

This harness runs Lash against [AppWorld](https://github.com/StonyBrookNLP/appworld) through AppWorld's MCP terminal-agent path, using the AppWorld snapshot vendored by [context-labs/halo](https://github.com/context-labs/halo). That keeps the setup comparable with Halo's published AppWorld loop.

For each task, the runner:

1. Starts local AppWorld environment, API, and MCP servers.
2. Activates the task world through the environment server.
3. Writes a benchmark-local Lash config that adds the AppWorld MCP server via `npx mcp-remote`.
4. Runs `lash --print --tool-surface appworld` in an empty per-task workspace.
5. Saves and evaluates the final AppWorld state.

Generated state lives under ignored `.benchmarks/appworld/`.

The AppWorld tool surface is explicit: it registers RLM, AppWorld MCP tools,
RLM tool discovery, and subagent/handoff tools. AppWorld MCP tools are
callable and discoverable, but not injected into the initial `Available Tools`
prompt. It does not register local shell, local file/search/edit/web tools,
autoresearch, host docs, the shell-backed `monitor` tool, or `tasks_list` /
`tasks_stop` (RLM exposes the equivalent through lashlang async handles).

## Quickstart

```bash
bench/appworld/setup.sh
bench/appworld/run.sh --dataset dev --limit 1 --provider-id openai-compatible --model gpt-5.4 --variant high --execution-mode rlm
```

The setup step clones Halo into `.benchmarks/appworld/vendor/halo`, installs Halo's `demo/appworld` package into `.benchmarks/appworld/venv`, unpacks AppWorld's encrypted bundles, and downloads the benchmark data into `.benchmarks/appworld/root`.

## Useful Flags

```bash
bench/appworld/run.sh --dataset dev --limit 5 --provider-id openai-compatible --model gpt-5.4 --variant high
bench/appworld/run.sh --dataset dev --max-concurrency 4 --provider-id openai-compatible --model gpt-5.4 --variant high
bench/appworld/run.sh --task-id 59fae45_2
bench/appworld/run.sh --dataset train --offset 10 --limit 10
bench/appworld/run.sh --dataset dev --provider-id openai-compatible --model anthropic/claude-sonnet-4.6 --variant high --execution-mode rlm
bench/appworld/run.sh --dry-run --dataset dev --limit 1
```

Dataset names are AppWorld's native splits: `train`, `dev`, `test_normal`, and `test_challenge`. For Halo-comparable numbers, tune on `dev` and report `test_normal`; Halo's README reports both TGC and SGC from AppWorld's evaluator.
Non-dry benchmark runs require explicit `--provider-id`, `--model`, and `--variant` so saved manifests and per-task commands are comparable.

## Requirements

- `uv`
- `node`/`npx` for `mcp-remote`
- `~/.lash/config.json` with an active provider

The runner keeps AppWorld data out of the agent's working directory, but this is a local convenience harness, not a hardened sandbox. AppWorld's own guide recommends stronger isolation for official terminal-agent evaluations.

## Outputs

Each run writes:

```text
.benchmarks/appworld/runs/<run-id>/
├── manifest.json
├── results.json
└── tasks/<task-id>/
    ├── command.txt
    ├── prompt.txt
    ├── stdout.txt
    ├── stderr.txt
    ├── initialize.json
    ├── lash-home/
    └── evaluation.json
```

`results.json` includes per-task pass/fail status. The AppWorld evaluator output in `evaluation.json` is the source for task success; full-split comparable TGC/SGC can be computed with AppWorld's dataset evaluator once all tasks in a split have been run under the same experiment name.
