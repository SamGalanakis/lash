# Lash development rules

Follow `AGENTS.md` for all repository work. In particular, Lash is trunk based:
`main` is the sole long-lived branch, changes ship by short-lived pull requests,
and there is no staging branch or staging release channel.

Do not push product changes directly to `main`. Do not tag or publish releases
manually. A maintainer releases a green `main` commit by manually dispatching
the GitHub `Release` workflow.
