#!/usr/bin/env bash
# Download python-build-standalone, install dill, generate pyo3-config.txt.
# Usage: ./scripts/fetch-python.sh [TARGET_TRIPLE]
# Example: ./scripts/fetch-python.sh x86_64-unknown-linux-gnu
set -euo pipefail

PYTHON_VERSION="3.14.3"
PYTHON_MAJOR_MINOR="3.14"
PBS_RELEASE="20260211"

TARGET="${1:-$(rustc -vV | sed -n 's/host: //p')}"
DEST="${2:-target/python-standalone}"

# Map target to PBS variant
case "$TARGET" in
    x86_64-apple-darwin)
        PBS_TRIPLE="x86_64-apple-darwin"
        PBS_FLAVOR="debug-full"
        ;;
    aarch64-apple-darwin)
        PBS_TRIPLE="aarch64-apple-darwin"
        PBS_FLAVOR="debug-full"
        ;;
    x86_64-unknown-linux-gnu)
        PBS_TRIPLE="x86_64-unknown-linux-gnu"
        PBS_FLAVOR="pgo+lto-full"
        ;;
    aarch64-unknown-linux-gnu)
        PBS_TRIPLE="aarch64-unknown-linux-gnu"
        PBS_FLAVOR="pgo+lto-full"
        ;;
    *) echo "Unsupported target: $TARGET"; exit 1 ;;
esac

FILENAME="cpython-${PYTHON_VERSION}+${PBS_RELEASE}-${PBS_TRIPLE}-${PBS_FLAVOR}.tar.zst"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/${FILENAME}"
MARKER="${DEST}/.version"
EXPECTED="${PYTHON_VERSION}+${PBS_RELEASE}+${PBS_FLAVOR}"

# Check cache
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$EXPECTED" ]; then
    echo "Python standalone already cached at $DEST"
    exit 0
fi

echo "Downloading python-build-standalone ${EXPECTED} for ${PBS_TRIPLE}..."
rm -rf "$DEST"
mkdir -p "$DEST"

# Download
curl -fSL --retry 3 -o "${DEST}/${FILENAME}" "$URL"

# Decompress zstd then extract tar
zstd -d -f "${DEST}/${FILENAME}" -o "${DEST}/python.tar"
tar xf "${DEST}/python.tar" -C "$DEST"

# Clean up archives
rm -f "${DEST}/${FILENAME}" "${DEST}/python.tar"

INSTALL_DIR="${DEST}/python/install"
BUILD_LIB_DIR="${DEST}/python/build/lib"
SITE_PACKAGES="${INSTALL_DIR}/lib/python${PYTHON_MAJOR_MINOR}/site-packages"

# Copy bundled static libs (mpdec, zstd, readline, etc.) into install/lib
# so the linker finds them via the -L path pyo3 already sets
if [ -d "$BUILD_LIB_DIR" ]; then
    echo "Copying bundled static libs into install/lib..."
    cp -n "$BUILD_LIB_DIR"/*.a "$INSTALL_DIR/lib/" 2>/dev/null || true
fi

# Install dill (skip for cross-compilation where we can't execute the binary)
HOST_TRIPLE="$(rustc -vV | sed -n 's/host: //p')"
if [ "$TARGET" = "$HOST_TRIPLE" ]; then
    if [ ! -d "${SITE_PACKAGES}/dill" ]; then
        echo "Installing dill..."
        "${INSTALL_DIR}/bin/python3" -m pip install --target "$SITE_PACKAGES" --no-deps --quiet dill
    fi
else
    echo "Cross-compiling for $TARGET (host: $HOST_TRIPLE), skipping dill install"
fi

# Generate pyo3-config.txt
LIB_DIR="${INSTALL_DIR}/lib"
case "$TARGET" in
    *x86_64*|*aarch64*) POINTER_WIDTH=64 ;;
    *) POINTER_WIDTH=32 ;;
esac

cat > "${DEST}/pyo3-config.txt" <<EOF
implementation=CPython
version=${PYTHON_MAJOR_MINOR}
shared=false
lib_name=python${PYTHON_MAJOR_MINOR}
lib_dir=${PWD}/${LIB_DIR}
pointer_width=${POINTER_WIDTH}
build_flags=
suppress_build_script_link_lines=false
EOF

echo "$EXPECTED" > "$MARKER"
echo "Python standalone ready at $DEST"
echo "PYO3_CONFIG_FILE=${PWD}/${DEST}/pyo3-config.txt"
