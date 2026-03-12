#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <target-triple> <asset-name>" >&2
  exit 1
fi

target="$1"
asset_name="$2"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary_path="${repo_root}/target/${target}/release/lash"
dist_dir="${repo_root}/dist"
stage_dir="$(mktemp -d)"
trap 'rm -rf "${stage_dir}"' EXIT

if [[ ! -x "${binary_path}" ]]; then
  echo "expected compiled binary at ${binary_path}" >&2
  exit 1
fi

mkdir -p "${dist_dir}"
cp "${binary_path}" "${stage_dir}/lash"
chmod 0755 "${stage_dir}/lash"
tar -C "${stage_dir}" -czf "${dist_dir}/${asset_name}" lash

echo "${dist_dir}/${asset_name}"
