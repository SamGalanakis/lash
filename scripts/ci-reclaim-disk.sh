#!/usr/bin/env bash
# Reclaim disk on CI runners before heavy cargo jobs.
#
# Written for the GitHub-hosted ubuntu-latest image, which ships ~25GB of
# preinstalled toolchains this repo never uses (dotnet, Android SDK, GHC,
# CodeQL). The workspace debug cache plus sccache plus per-shard test-binary
# codegen exceeds the ~14GB that remains, which surfaces as `No space left on
# device` and linker Bus errors mid-shard.
#
# The jobs now run on Blacksmith runners, whose image does not necessarily
# carry those toolchains; every removal below is failure-tolerant, so this
# degrades to a no-op rather than breaking. Once Blacksmith disk headroom is
# measured, this can likely be dropped from the jobs that call it.
set -euo pipefail

echo "before:"; df -h / | tail -1
sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc \
  /opt/hostedtoolcache/CodeQL /usr/local/share/boost 2>/dev/null || true
sudo docker image prune --all --force >/dev/null 2>&1 || true
echo "after:"; df -h / | tail -1
