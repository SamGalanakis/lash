#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/appworld"
VENV_DIR="${STATE_DIR}/venv"
APPWORLD_ROOT="${STATE_DIR}/root"
VENDOR_DIR="${STATE_DIR}/vendor/halo"
APPWORLD_VENDOR_DIR="${VENDOR_DIR}/demo/appworld"
HALO_REPO_URL="https://github.com/context-labs/halo.git"
HALO_LFS_BASE_URL="https://media.githubusercontent.com/media/context-labs/halo/main/demo/appworld"

SKIP_DATA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --help)
      cat <<'EOF'
Usage: bench/appworld/setup.sh [--skip-data]

Installs AppWorld into .benchmarks/appworld/venv and downloads data into
.benchmarks/appworld/root unless --skip-data is passed.
EOF
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to set up AppWorld (https://docs.astral.sh/uv/)." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required because Lash connects to AppWorld's HTTP MCP server through mcp-remote." >&2
  exit 1
fi

mkdir -p "${STATE_DIR}/vendor" "${APPWORLD_ROOT}"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone --depth 1 "${HALO_REPO_URL}" "${VENDOR_DIR}"
fi

download_lfs_file_if_pointer() {
  local rel_path="$1"
  local path="${APPWORLD_VENDOR_DIR}/${rel_path}"
  if [[ ! -f "${path}" ]]; then
    echo "error: expected vendored AppWorld file missing: ${path}" >&2
    exit 1
  fi
  if head -n 1 "${path}" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
    echo "Downloading Git LFS media for ${rel_path}"
    curl -L --fail --silent --show-error \
      "${HALO_LFS_BASE_URL}/${rel_path}" \
      -o "${path}"
  fi
}

download_lfs_file_if_pointer "src/appworld/.source/apps.bundle"
download_lfs_file_if_pointer "src/appworld/.source/tests.bundle"
download_lfs_file_if_pointer "generate/.source/data.bundle"
download_lfs_file_if_pointer "generate/.source/tasks.bundle"

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
uv pip install --python "${VENV_DIR}/bin/python" -q --upgrade pip
uv pip install --python "${VENV_DIR}/bin/python" -q -e "${APPWORLD_VENDOR_DIR}[mcp]"

if ! "${VENV_DIR}/bin/appworld" serve mcp --help >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Installed AppWorld does not expose `appworld serve mcp`.

The AppWorld README documents MCP support on the current GitHub main branch.
Install a release that includes MCP support, or install from source with Git LFS
available so the encrypted .bundle files are present.
EOF
  exit 1
fi

(
  cd "${APPWORLD_VENDOR_DIR}"
  "${VENV_DIR}/bin/appworld" install --repo
)

if [[ "${SKIP_DATA}" -eq 0 ]]; then
  "${VENV_DIR}/bin/appworld" download data --root "${APPWORLD_ROOT}"
fi

echo "AppWorld benchmark state ready under ${STATE_DIR}"
echo "  venv: ${VENV_DIR}"
echo "  root: ${APPWORLD_ROOT}"
