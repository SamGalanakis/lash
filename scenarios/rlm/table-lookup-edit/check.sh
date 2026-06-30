#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "${1:?workspace path required}" && pwd)"
scenario_dir="$(cd "${2:?scenario path required}" && pwd)"

cd "$workspace"
sh test.sh
cmp -s test.sh "$scenario_dir/workspace/test.sh"
grep -Eq 'green[)] echo "#00ff00"' lookup.sh

