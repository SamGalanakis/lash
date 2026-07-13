# Contributing to Lash

Feature requests and bug reports are welcome — open an
[issue](https://github.com/Ascending-AI/lash/issues).

At this alpha stage, detailed write-ups help more than drive-by PRs. The
internals are still moving fast, so open an issue before starting a substantial
implementation and agree on the shape first.

To understand how the runtime fits together, start at <https://lash.run/>. The
architecture chapters cover the crate layout, turn/effect boundary, and plugin
model.

## Development workflow

Lash uses trunk-based development. `main` is the only long-lived branch and is
kept releasable.

1. Update `main` and create a short-lived branch.
2. Make one focused change and run the relevant local checks.
3. Open a pull request into `main`.
4. Keep the branch current and merge only after required CI is green.
5. Delete the branch after merge.

There is no `staging` branch. Preview work belongs in pull requests, while the
merged product state lives on `main`.

## Releases

Merging to `main` does not release. A maintainer manually runs the GitHub
`Release` workflow after selecting a green commit on `main`; leaving
`release_sha` blank selects the current head. The workflow verifies main-branch
CI, requires curated release notes, computes the next version, tags the exact
commit, builds assets, and publishes.

Never create release tags or publish crates and artifacts by hand. See
`docs/PUBLISHING.md` for the complete release contract.
