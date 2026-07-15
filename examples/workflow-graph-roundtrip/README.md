# Workflow graph round-trip backend

This example is the Rust half of a visual Lashlang workflow editor. It exposes
the source → graph → edited graph → canonical source seam over HTTP, then runs
the saved version and streams node-correlated display events over SSE.

The backend owns in-memory versions and execution. Lashlang owns graph
projection, graph validation/rendering, and execution-site correlation. Canvas
layout is deliberately frontend-owned and never appears in source or API graph
documents.

Run it from the repository root:

```sh
CARGO_TARGET_DIR=/tmp/lash-workflow-graph \
  cargo run -p workflow-graph-roundtrip
```

The code default is `http://127.0.0.1:3031`. The conventional demo uses
`WORKFLOW_GRAPH_ADDR=127.0.0.1:3057`; set that variable to any available
`IP:PORT`. See [CONTRACT.md](CONTRACT.md) for the complete API used by the
frontend.

The server serves files from `frontend/dist/` (or directly from `frontend/`)
when present. A frontend dev server
can also run separately because the API permits cross-origin GET, POST, and
OPTIONS requests.

See [RUNBOOK.md](RUNBOOK.md) for the production frontend build, server startup,
operator checks, and integration verification command.
