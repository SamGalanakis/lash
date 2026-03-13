#!/usr/bin/env bash
set -euo pipefail

iterations="${1:-10000}"

exec cargo run -p lashlang --release --example profile -- "${iterations}"
