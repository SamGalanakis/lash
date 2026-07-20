# Workflow graph round-trip runbooks

The canonical browser journey now lives in the repository runbook suite:
[workflow-editor-authoring](../../runbooks/workflow-editor-authoring/runbook.md).
It includes production frontend boot, blank-workflow authoring in Steps and
Canvas, typed-error recovery, save reconciliation, and terminal execution.

The deterministic integration gate remains:

```sh
just workflow-graph-integration-verify
```

This compatibility file intentionally remains at its original path so existing
links and automation references continue to resolve without maintaining a
second operator procedure.
