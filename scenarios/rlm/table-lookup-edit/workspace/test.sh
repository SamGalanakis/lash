#!/bin/sh
set -eu

check() {
  name="$1"
  expected="$2"
  actual="$(sh lookup.sh "$name")"
  if [ "$actual" != "$expected" ]; then
    echo "for $name: expected $expected, got $actual" >&2
    exit 1
  fi
}

check red "#ff0000"
check green "#00ff00"
check blue "#0000ff"
check missing "unknown"

echo ok

