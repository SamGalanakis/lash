# Restate Postgres Workers E2E

The full distributed harness runs with:

```sh
just restate-postgres-workers-e2e
```

That starts Postgres, MinIO, Restate, a mock OpenAI-compatible provider, two
workers, the h2c proxy, and the runner. Alongside the process, durable-wait,
frame-switch, and storage gates, the runner verifies first-party turn control:

- cancellation from the runner process with no Lash session handle;
- cancellation before the worker starts the addressed turn;
- first-writer-wins cancellation versus completion sealing;
- cancellation replayed by a peer after the original worker exits;
- terminal attachment returning the exact cancellation evidence; and
- Restate Admin invocation hard-kill remaining break-glass rather than
  manufacturing a Lash `Cancelled` terminal.

The final `turn-control gates passed:` line is the deterministic evidence for
those assertions. Session and turn IDs used by this test are routing identity,
not authorization. Production hosts must authorize callers before exposing the
same driver, and cancellation remains cooperative: detached effects are not
guaranteed to stop.

The package-level build/unit check is lighter and does not start the distributed
services:

```sh
cargo test -p lash-restate-postgres-workers-e2e --all-targets
```

## Local Postgres Conformance

The Postgres store conformance tests require `LASH_POSTGRES_DATABASE_URL`.
Without it, the Postgres conformance binary reports a skip. To run the process
registry conformance locally without the full E2E stack:

```sh
docker run --rm --name lash-postgres-conformance \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=lash_conformance \
  -p 55432:5432 \
  -d postgres:16

LASH_POSTGRES_DATABASE_URL=postgres://postgres:postgres@localhost:55432/lash_conformance \
  cargo test -p lash-postgres-store --locked \
  postgres_process_registry_satisfies_conformance_when_configured

docker rm -f lash-postgres-conformance
```

Use a fresh database for each run when debugging registry persistence semantics.
