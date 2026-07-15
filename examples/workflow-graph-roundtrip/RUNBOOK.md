# Workflow graph round-trip operator runbook

From the repository root, install the pinned frontend dependencies and build
the production assets served by the Rust backend:

```sh
npm --prefix examples/workflow-graph-roundtrip/frontend ci
npm --prefix examples/workflow-graph-roundtrip/frontend run build
```

Start the server. The code default is `127.0.0.1:3031`:

```sh
CARGO_TARGET_DIR=/tmp/lash-workflow-graph \
  cargo run -p workflow-graph-roundtrip
```

The conventional demo port is `3057`; select it explicitly when needed:

```sh
WORKFLOW_GRAPH_ADDR=127.0.0.1:3057 \
  CARGO_TARGET_DIR=/tmp/lash-workflow-graph \
  cargo run -p workflow-graph-roundtrip
```

The backend serves `frontend/dist` from disk on each request, so a frontend
rebuild is visible on a browser refresh alone. The Rust backend is **not**
hot-reloaded, however: after any change to backend source (`src/*.rs`, or the
lash crates it depends on) you must rebuild and restart the server process, or
it keeps serving the old projection. A half-refreshed server — new frontend,
stale binary — is the most likely cause of a field that renders empty or a
graph that ignores an edit; rebuild and relaunch before debugging further.

Open the address used above. Exercise the newly editable fields with this
operator loop:

1. Select **Branching Approval**, change the inner literal `if` condition from
   `true` to `false`, and save. Confirm the returned canonical source contains
   `if false`, then Play and observe the `Approval needs review` branch.
2. Select **Counter Loop**. On `for pct`, enter ` pct ` for the binding and
   `[15]` for the iterable, then save. Confirm the reprojected source
   canonicalizes the binding to `pct` and contains `for pct in [15]`; Play and
   observe a successful run ending at 15% progress.
3. No built-in catalog workflow contains a list comprehension, so there is no
   honest catalog run for editing `clauses`. Comprehension edit/reprojection is
   covered by the Lashlang property and focused unit tests instead.
4. Select **Onboarding**. On the `Wait for approval` effect, enter
   ` approval ` for `binding`, save, and confirm the canonical source restores
   `approval = wait_signal(...)`; Play and observe a successful correlated run.

Delete/reorder checks may still be performed on any workflow. After every
save, replace the displayed document with the returned canonical projection
before running it.

This proves the source → graph → edited graph → canonical source lens and the
saved-version → runtime execution-site → graph-node correlation used by a
visual host. The deterministic integration pass builds the frontend and runs
the backend catalog, edit, typed-error, and SSE tests:

```sh
just workflow-graph-integration-verify
```

Browser UI automation is intentionally out of scope; the integration pass
verifies the edit → save → reproject → run contract at the backend seam.
