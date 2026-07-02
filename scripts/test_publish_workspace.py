#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import pathlib
import subprocess
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_publish_workspace_module():
    module_path = ROOT / "scripts" / "publish_workspace.py"
    spec = importlib.util.spec_from_file_location("publish_workspace", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublishWorkspaceTest(unittest.TestCase):
    def test_publish_retries_cargo_registry_http2_failures(self) -> None:
        publish_workspace = load_publish_workspace_module()
        args = argparse.Namespace(
            publish_timeout_seconds=600,
            publish_attempts=2,
            retry_delay_seconds=0,
            visibility_timeout_seconds=1,
            visibility_delay_seconds=0,
        )
        package = {"name": "lash-provider-anthropic", "version": "0.1.0-alpha.40"}
        first = subprocess.CompletedProcess(
            ["cargo", "publish"],
            101,
            "",
            "\n".join(
                [
                    "error: download of config.json failed",
                    "Caused by:",
                    "  curl failed",
                    "Caused by:",
                    "  [16] Error in the HTTP2 framing layer",
                ]
            ),
        )
        second = subprocess.CompletedProcess(["cargo", "publish"], 0, "", "")

        with (
            mock.patch.object(publish_workspace, "crate_version_visible", return_value=False),
            mock.patch.object(publish_workspace, "wait_for_crate_version") as wait,
            mock.patch.object(publish_workspace.time, "sleep"),
            mock.patch.object(publish_workspace.subprocess, "run", side_effect=[first, second]) as run,
        ):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                publish_workspace.publish_package(package, args)

        self.assertEqual(run.call_count, 2)
        wait.assert_called_once_with(package["name"], package["version"], args)

    def test_dev_dependencies_do_not_gate_publish_ordering(self) -> None:
        # A crate that dev-depends on itself (to enable one of its own
        # features for integration tests) and a provider that dev-depends
        # back onto the transport's test support must not deadlock the
        # publish planner: dev-deps are stripped from published packages.
        publish_workspace = load_publish_workspace_module()
        transport_dir = str(ROOT / "crates" / "lash-llm-transport")
        provider_dir = str(ROOT / "crates" / "lash-provider-anthropic")
        metadata = {
            "workspace_members": ["transport-id", "provider-id"],
            "packages": [
                {
                    "id": "transport-id",
                    "name": "lash-llm-transport",
                    "version": "0.0.1",
                    "publish": None,
                    "manifest_path": f"{transport_dir}/Cargo.toml",
                    "dependencies": [
                        {"name": "lash-llm-transport", "path": transport_dir, "kind": "dev"},
                    ],
                },
                {
                    "id": "provider-id",
                    "name": "lash-provider-anthropic",
                    "version": "0.0.1",
                    "publish": None,
                    "manifest_path": f"{provider_dir}/Cargo.toml",
                    "dependencies": [
                        {"name": "lash-llm-transport", "path": transport_dir, "kind": None},
                        {"name": "lash-llm-transport", "path": transport_dir, "kind": "dev"},
                    ],
                },
            ],
        }
        with mock.patch.object(publish_workspace, "run_json", return_value=metadata):
            packages = publish_workspace.load_publishable_workspace_packages()

        self.assertEqual(packages["transport-id"]["workspace_dependencies"], set())
        self.assertEqual(
            packages["provider-id"]["workspace_dependencies"], {"transport-id"}
        )


if __name__ == "__main__":
    unittest.main()
