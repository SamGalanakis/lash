# Publishing to crates.io

The workspace publishes as one lockstep release: every publishable crate shares
the `[workspace.package]` version and pins its internal dependencies to that
exact version (centralized in the root `[workspace.dependencies]`). A release
publishes them all together, in dependency order.

## What gets published

- **Published:** every workspace member without `publish = false`. The public
  entry point is `lash-runtime` (imported as `lash`); embedders also pull in
  provider crates (`lash-provider-*`), stores (`lash-sqlite-store`,
  `lash-postgres-store`, `lash-s3-store`,
  `lash-restate`), the remote protocol DTOs (`lash-remote-protocol`), and
  a-la-carte capability crates (`lash-tools`, `lash-plugin-mcp`,
  `lash-subagents`, `lash-plugin-plan-mode`,
  `lash-plugin-tool-output-budget`, `lash-llm-tools`).
- **Not published:** anything marked `publish = false` — the CLI (`lash-cli`),
  TUI crates, examples, E2E harnesses, and dev/internal tooling
  (`lash-harness-opt`, `lash-perf`, `lash-trace-viewer`, `lash-export`,
  `lash-file-index`, `lash-autoresearch`).

Because of the exact `=` version pins, a published crate's internal deps must
already be on crates.io at the same version — so it is **all-or-nothing**:
`scripts/publish_workspace.py` publishes the whole set one crate at a time in
dependency order and waits for crates.io visibility between crates.

## How a release runs (CI)

1. Bump the version (`scripts/release_version.py set <version>` — edits the
   root `[workspace.package]` version, root `[workspace.dependencies]` pins,
   checked-in docs snippets, and lockfile entries for workspace members that
   use `version.workspace = true`). Private fixed-version members, such as E2E
   harness crates at `0.0.0`, are intentionally left alone.
2. Tag it `vX.Y.Z` and push (or use the `Release` workflow's
   `workflow_dispatch`).
3. `.github/workflows/release.yml` validates `cargo metadata --locked`, then
   the `publish-crates` job runs `python3 .release-tools/scripts/publish_workspace.py`.
   Already-published versions are skipped, so a failed run can be re-run to
   resume.
4. The same release workflow builds the CLI release assets and publishes the
   GitHub release. On `main`, CI prepares automated releases in two passes:
   the first passing run may push the version-bump commit, and the next CI run
   validates that exact versioned commit before creating the tag and dispatching
   `release.yml`. The normal CI workflow does not auto-release when its commit
   message contains `[skip release]`.

The main CI workflow also runs:

```bash
python3 scripts/test_release_version.py
python3 scripts/test_publish_workspace.py
python3 scripts/test_release_notes.py
```

Those tests pin the lockstep/private-crate version behavior, the publisher's
transient retry classification, and the release-notes extraction rules.

## Release notes (required)

Every release ships curated notes. Any commit that should contribute
user-facing notes carries a `Release-Notes:` section in its body — everything
after the marker line, written as Markdown:

```text
Add durable suspension to processes

Implementation details for reviewers...

Release-Notes:
- Processes now suspend durably while waiting on signals or timers.
- Signals are named and typed; the unnamed `wait_signal()` is removed.
```

`prepare-release` runs `scripts/release_notes.py collect --require` before
mutating anything: if no commit in `previous-tag..HEAD` carries a section, the
release fails loudly (push with `[skip release]` if the range deliberately
cuts no release). The publish job collects the same range's sections (oldest
first) into the GitHub release body; the auto-generated commit list is
appended below. The previous tag is resolved by graph ancestry
(`git describe`), not version sorting, so tags from unrelated history lines
are ignored. Release-automation commits ("Release vX", staging version syncs)
never contribute notes.

## Docs code snippets

Every Rust code block on a published docs page is compiled. The sources live in
`examples/docs-snippets/` (one module per page) inside
`// docs:start:<id>` / `// docs:end:<id>` regions, and each page block carries
`<pre data-snippet="<module>#<id>">`. CI runs
`cargo check -p docs-snippets --locked` (snippets must build against the
current API) and `python3 scripts/lint_docs.py` (the HTML must match the
regions byte-for-byte). To change a snippet, edit the `.rs` source and run
`python3 scripts/lint_docs.py --fix-snippets` to re-inject the HTML (and the
README hero block). Display-only blocks (shell transcripts, Lashlang, API-shape
excerpts) are marked `data-lang="..."` instead.

## Docs search index

The static site checks in its Pagefind bundle and generated index under
`docs/pagefind/`. When adding, removing, or renaming docs pages, regenerate the
index with the pinned Pagefind version used by the checked-in bundle:

```bash
rm -f docs/pagefind/fragment/*.pf_fragment docs/pagefind/index/*.pf_index docs/pagefind/pagefind.*.pf_meta
npx -y pagefind@1.5.2 --site docs --output-path docs/pagefind
```

Run `python3 scripts/lint_docs.py` after regeneration; the linter verifies the
hand-authored registry, links, snippets, and static pagers that Pagefind indexes.

## Auth

The `publish-crates` job uses a `CARGO_REGISTRY_TOKEN` repository secret (a
crates.io API token).

A token (not Trusted Publishing) is required for the **first** publish of a
brand-new crate, because crates.io Trusted Publishing can only be configured on
a crate that already exists.

### First publish of a new crate

New publishable crates inherit the lockstep version automatically
(`version.workspace = true`) and reference internal deps through
`{ workspace = true }`. Add the crate's own pin to root
`[workspace.dependencies]` so dependents can use it, and keep internal-only
crates `publish = false`.

To publish the workspace at the current checked-out release version:

```bash
# from a clean checkout at the release tag, with a crates.io token:
cargo login          # or export CARGO_REGISTRY_TOKEN=...
python3 scripts/publish_workspace.py
```

The helper asks crates.io whether each `(crate, version)` is already visible,
skips published versions, publishes ready crates with `cargo publish -p <crate>
--no-verify --locked`, retries transient registry/network failures, waits for
API visibility, then continues to dependents.

### Upgrade to Trusted Publishing (recommended, after bootstrap)

Once every crate exists on crates.io, configure a trusted publisher for each
(crate settings → Trusted Publishing → repo `SamGalanakis/lash`, workflow
`release.yml`), then in `publish-crates`:

- add `id-token: write` to `permissions`,
- add a `rust-lang/crates-io-auth-action@v1` step (`id: auth`) before publish,
- set `CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}`,
- delete the long-lived `CARGO_REGISTRY_TOKEN` secret.
