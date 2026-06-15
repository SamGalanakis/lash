#!/usr/bin/env bash
set -euo pipefail

production_limit="${LASH_PRODUCTION_RUST_LINE_LIMIT:-1600}"
test_limit="${LASH_TEST_RUST_LINE_LIMIT:-2500}"

if (($#)); then
  roots=("$@")
else
  roots=(".")
fi

is_test_rust_file() {
  local file="$1"
  case "$file" in
    */tests/*|*/test/*|*/testing/*|*/src/tests.rs|*/src/test.rs|*/src/*/tests.rs|*/src/*/test.rs|*/language/support.rs)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

failures=()
while IFS= read -r -d '' file; do
  rel="${file#./}"
  if is_test_rust_file "$rel"; then
    limit="$test_limit"
    kind="test"
  else
    limit="$production_limit"
    kind="production"
  fi

  lines=$(wc -l < "$file")
  if ((lines > limit)); then
    failures+=("$kind:$lines:$rel")
  fi
done < <(
  find "${roots[@]}" \
    \( \
      -path '*/.git' -o \
      -path '*/.git/*' -o \
      -path '*/target' -o \
      -path '*/target/*' -o \
      -path '*/vendor' -o \
      -path '*/vendor/*' -o \
      -path '*/vendored' -o \
      -path '*/vendored/*' -o \
      -path '*/generated' -o \
      -path '*/generated/*' \
    \) -prune -o \
    -type f -name '*.rs' -print0
)

if ((${#failures[@]})); then
  echo "Rust files over line budget:" >&2
  echo "  production limit: ${production_limit} lines" >&2
  echo "  test/support limit: ${test_limit} lines" >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
