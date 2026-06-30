#!/bin/sh
set -eu

code="${1:-}"

case "$code" in
  red) echo "#ff0000" ;;
  green) echo "#ff0000" ;;
  blue) echo "#0000ff" ;;
  *) echo "unknown" ;;
esac

