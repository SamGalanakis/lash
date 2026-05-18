#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

forbidden='PluginSurfaceEvent|PromptRequest|PromptResponse|PromptSelectionMode|PromptPanel|PromptSlot::Cli|CliAutonomous|CliRlm|PanelUpsert|PanelAppend|PanelClear|ModeIndicator|desktop_notification'

if rg -n "$forbidden" crates/lash-sansio/src crates/lash-core/src crates/lash/src; then
  echo "core UI boundary check failed: UI-only vocabulary found in sansio/core/facade source" >&2
  exit 1
fi
