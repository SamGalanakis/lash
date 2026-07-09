# Repository workflow

Lash uses trunk-based development with `main` as the only long-lived branch.

- Start every change from an up-to-date `main` on a short-lived branch.
- Ship through a pull request into `main`; do not push product changes directly
  to `main` and do not create or use a `staging` branch.
- Keep each pull request focused, keep it current with `main`, and merge only
  after the required CI checks pass.
- Ordinary merges never publish a release. Releases are manual: dispatch the
  `Release` workflow for a green commit on `main` (blank `release_sha` selects
  the current head). Do not create release tags or publish artifacts by hand.
- Add a `Release-Notes:` section to a commit in every releasable range. The
  manual release workflow validates the notes before it creates a tag.

See `CONTRIBUTING.md` and `docs/PUBLISHING.md` for the human workflow and
release details.
