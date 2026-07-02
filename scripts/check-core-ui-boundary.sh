#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

forbidden='PluginSurfaceEvent|PromptRequest|PromptResponse|PromptSelectionMode|PromptPanel|PromptSlot::Cli|CliAutonomous|CliRlm|PanelUpsert|PanelAppend|PanelClear|ModeIndicator|desktop_notification'

if command -v rg >/dev/null 2>&1; then
  search=(rg -n "$forbidden" crates/lash-sansio/src crates/lash-core/src crates/lash/src)
else
  # Portable fallback when ripgrep is unavailable: grep -E over the same source
  # trees. -r recurses like rg's directory walk, -n prints line numbers, so the
  # match output and the pass/fail behavior are identical.
  search=(grep -rEn "$forbidden" crates/lash-sansio/src crates/lash-core/src crates/lash/src)
fi

if "${search[@]}"; then
  echo "core UI boundary check failed: UI-only vocabulary found in sansio/core/facade source" >&2
  exit 1
fi
