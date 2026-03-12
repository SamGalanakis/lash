#!/usr/bin/env bash
set -euo pipefail

repo="${LASH_REPO:-SamGalanakis/lash}"
version="${LASH_VERSION:-}"
install_dir="${LASH_INSTALL_DIR:-$HOME/.local/bin}"

usage() {
  cat <<'EOF'
Install lash from GitHub release assets.

Usage:
  install_lash.sh
  LASH_VERSION=v0.2.0 install_lash.sh
  LASH_INSTALL_DIR=/usr/local/bin install_lash.sh

Environment:
  LASH_REPO         GitHub repo in owner/name form (default: SamGalanakis/lash)
  LASH_VERSION      Release tag to install; defaults to the latest GitHub release
  LASH_INSTALL_DIR  Destination directory for the lash binary (default: ~/.local/bin)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "unexpected arguments" >&2
  usage >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd tar
require_cmd mktemp
require_cmd install

detect_os() {
  case "$(uname -s)" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    *)
      echo "unsupported operating system: $(uname -s)" >&2
      exit 1
      ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "x86_64" ;;
    arm64|aarch64) echo "aarch64" ;;
    *)
      echo "unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
}

verify_checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c -
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c -
    return
  fi

  echo "missing required command: sha256sum or shasum" >&2
  exit 1
}

os="$(detect_os)"
arch="$(detect_arch)"
asset_name="lash-${os}-${arch}.tar.gz"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

if [[ -n "${version}" ]]; then
  base_url="https://github.com/${repo}/releases/download/${version}"
else
  base_url="https://github.com/${repo}/releases/latest/download"
fi

archive_path="${tmp_dir}/${asset_name}"
checksums_path="${tmp_dir}/SHA256SUMS"

echo "Downloading ${asset_name} from ${repo}..."
curl -fsSL "${base_url}/${asset_name}" -o "${archive_path}"
curl -fsSL "${base_url}/SHA256SUMS" -o "${checksums_path}"

expected_line="$(awk -v asset="${asset_name}" '$2 == asset { print }' "${checksums_path}")"
if [[ -z "${expected_line}" ]]; then
  echo "failed to find checksum entry for ${asset_name}" >&2
  exit 1
fi

(
  cd "${tmp_dir}"
  printf '%s\n' "${expected_line}" | verify_checksum
)

mkdir -p "${install_dir}"
tar -xzf "${archive_path}" -C "${tmp_dir}" lash
install -m 0755 "${tmp_dir}/lash" "${install_dir}/lash"

echo "Installed lash to ${install_dir}/lash"

case ":${PATH}:" in
  *":${install_dir}:"*) ;;
  *)
    echo "Add ${install_dir} to PATH if it is not already available in your shell."
    ;;
esac
