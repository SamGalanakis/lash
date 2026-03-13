#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Render a release-pinned install_lash.sh asset.

Usage:
  scripts/render-installer.sh <release-tag> <output-path>
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 1
fi

release_tag="$1"
output_path="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
template_path="${repo_root}/install_lash.sh"

escaped_tag="$(printf '%s' "${release_tag}" | sed 's/[\/&]/\\&/g')"
mkdir -p "$(dirname "${output_path}")"
sed "0,/__LASH_RELEASE_VERSION__/s//${escaped_tag}/" "${template_path}" > "${output_path}"
chmod +x "${output_path}"
