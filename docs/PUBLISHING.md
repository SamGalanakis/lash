# Publishing to crates.io

The workspace publishes as one lockstep release: every publishable crate shares
the `[workspace.package]` version and pins its internal dependencies to that
exact version (centralized in the root `[workspace.dependencies]`). A release
publishes them all together, in dependency order.

## What gets published

- **Published:** every workspace member without `publish = false`. The public
  entry point is `lash-runtime` (imported as `lash`); embedders also pull in
  provider crates (`lash-provider-*`), stores (`lash-turso-store`,
  `lash-restate`), the remote-embedding DTOs (`lash-remote-protocol`), and
  à-la-carte capability crates (`lash-plugin-mcp`, `lash-subagents`,
  `lash-plugin-plan-mode`, `lash-llm-tools`).
- **Not published:** anything marked `publish = false` — the CLI (`lash-cli`),
  TUI crates, examples, and dev/internal tooling (`lash-harness-opt`,
  `lash-trace-viewer`, `lash-export`, `lash-file-index`, `lash-autoresearch`).

Because of the exact `=` version pins, a published crate's internal deps must
already be on crates.io at the same version — so it is **all-or-nothing**:
`cargo publish --workspace` publishes the whole set in topological order and
waits for index propagation between crates.

## How a release runs (CI)

1. Bump the version (`scripts/release_version.py set <version>` — edits the
   root `[workspace.package]` version and the root `[workspace.dependencies]`
   pins, the only places versions live now).
2. Tag it `vX.Y.Z` and push (or use the `Release` workflow's
   `workflow_dispatch`).
3. `.github/workflows/release.yml` → `publish-crates` job runs the core tests,
   then `cargo publish --workspace --locked`. Already-published versions are
   skipped, so a failed run can be re-run to resume.

## Auth

The `publish-crates` job uses a `CARGO_REGISTRY_TOKEN` repository secret (a
crates.io API token).

A token (not Trusted Publishing) is required for the **first** publish of a
brand-new crate, because crates.io Trusted Publishing can only be configured on
a crate that already exists.

### One-time bootstrap (creates the 30 not-yet-published crates)

All public crates except `lash-core`, `lashlang`, `lash-sansio`, `lash-trace`,
and `lash-provider-auth` have never been published (the published five are also
stale at `alpha.1`). To create them at the current version:

```bash
# from a clean checkout at the release version, with a crates.io token:
cargo login          # or export CARGO_REGISTRY_TOKEN=...
cargo publish --workspace --locked
```

or set the `CARGO_REGISTRY_TOKEN` secret and run the `Release` workflow once.

### Upgrade to Trusted Publishing (recommended, after bootstrap)

Once every crate exists on crates.io, configure a trusted publisher for each
(crate settings → Trusted Publishing → repo `SamGalanakis/lash`, workflow
`release.yml`), then in `publish-crates`:

- add `id-token: write` to `permissions`,
- add a `rust-lang/crates-io-auth-action@v1` step (`id: auth`) before publish,
- set `CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}`,
- delete the long-lived `CARGO_REGISTRY_TOKEN` secret.

## Adding a new crate

New publishable crates inherit the version automatically (`version.workspace =
true`) and reference internal deps via `{ workspace = true }`. Add the crate's
own version pin to the root `[workspace.dependencies]` so dependents can use it.
Keep internal-only crates `publish = false`.
