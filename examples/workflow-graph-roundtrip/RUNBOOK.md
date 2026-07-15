# Workflow graph round-trip operator runbook

From the repository root, install the pinned frontend dependencies and build
the production assets served by the Rust backend:

```sh
npm --prefix examples/workflow-graph-roundtrip/frontend ci
npm --prefix examples/workflow-graph-roundtrip/frontend run build
```

Start the server. It listens on `127.0.0.1:3031` unless
`WORKFLOW_GRAPH_ADDR` is set:

```sh
CARGO_TARGET_DIR=/tmp/lash-workflow-graph \
  cargo run -p workflow-graph-roundtrip
```

Open <http://127.0.0.1:3031>. The operator loop is:

1. Select any built-in workflow.
2. Edit a field or delete/reorder a node and save the graph.
3. Confirm the backend returns the canonical, reprojected document.
4. Play the saved version and watch correlated node states plus display updates
   arrive over SSE until the terminal node succeeds.

This proves the source → graph → edited graph → canonical source lens and the
saved-version → runtime execution-site → graph-node correlation used by a
visual host. The deterministic integration pass builds the frontend and runs
the backend catalog, edit, typed-error, and SSE tests:

```sh
just workflow-graph-integration-verify
```

Browser UI automation is intentionally out of scope; the integration pass
verifies the edit → save → run contract at the backend seam.
