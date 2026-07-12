# The CLI is an independent Host Application

The `lash` terminal application and its private support crates live in
[`SamGalanakis/lash-cli`](https://github.com/SamGalanakis/lash-cli), outside the
Lash runtime repository. The CLI repository owns its TUI, UI extensions, file
index, transcript exporter, autoresearch integration, benchmark support,
operator harness, installer, self-update policy, and binary releases.

Lash owns reusable runtime, protocol, provider, persistence, plugin, tooling,
and performance contracts. The CLI consumes those contracts at one reviewed,
exact Lash revision and advances that revision through an explicit compatibility
change. Lash releases publish the SDK crates and no longer publish the `lash`
binary or installer assets.

This boundary makes the CLI an honest external embedder: it can choose plugin
composition and Execution Modes without forcing runtime releases, while changes
to Lash must remain usable without private workspace paths. A private support
crate stays in the Host Application repository while the CLI is its only real
consumer; it moves into Lash only when it becomes a stable, frontend-independent
contract with credible use by another host.
