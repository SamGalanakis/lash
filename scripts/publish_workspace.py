#!/usr/bin/env python3
"""Publish Cargo workspace crates to crates.io one package at a time.

`cargo publish --workspace` can fail after partial progress when crates.io index
propagation lags behind newly uploaded workspace dependencies. This helper is
idempotent: it skips versions already visible on crates.io, publishes only crates
whose workspace dependencies are visible, and waits after each upload before
continuing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


USER_AGENT = "lash-release-publisher/1.0"
TRANSIENT_PUBLISH_ERRORS = (
    "no matching package named",
    "failed to select a version",
    "perhaps a crate was updated and forgotten to be re-vendored",
)
ALREADY_PUBLISHED_ERRORS = (
    "already uploaded",
    "already exists",
    "previously uploaded",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-timeout-seconds", type=int, default=600)
    parser.add_argument("--publish-attempts", type=int, default=6)
    parser.add_argument("--retry-delay-seconds", type=int, default=15)
    parser.add_argument("--visibility-timeout-seconds", type=int, default=600)
    parser.add_argument("--visibility-delay-seconds", type=int, default=5)
    args = parser.parse_args()

    packages = load_publishable_workspace_packages()
    completed: set[str] = set()
    pending = set(packages)

    for package_id, package in packages.items():
        if crate_version_visible(package["name"], package["version"]):
            print(f"already published: {package['name']} {package['version']}")
            completed.add(package_id)
            pending.remove(package_id)

    while pending:
        ready = sorted(
            (
                package_id
                for package_id in pending
                if packages[package_id]["workspace_dependencies"] <= completed
            ),
            key=lambda package_id: packages[package_id]["name"],
        )
        if not ready:
            remaining = ", ".join(packages[package_id]["name"] for package_id in sorted(pending))
            raise RuntimeError(f"publish dependency cycle or unresolved publish set: {remaining}")

        for package_id in ready:
            package = packages[package_id]
            publish_package(package, args)
            pending.remove(package_id)
            completed.add(package_id)

    print(f"published or confirmed {len(packages)} workspace crates")
    return 0


def load_publishable_workspace_packages() -> dict[str, dict]:
    metadata = run_json(["cargo", "metadata", "--format-version", "1", "--locked", "--no-deps"])
    workspace_members = set(metadata["workspace_members"])
    package_by_id = {
        package["id"]: package
        for package in metadata["packages"]
        if package["id"] in workspace_members and package.get("publish") != []
    }
    package_id_by_dir = {
        Path(package["manifest_path"]).parent.resolve(): package_id
        for package_id, package in package_by_id.items()
    }

    result = {}
    for package_id, package in package_by_id.items():
        workspace_dependencies = set()
        for dependency in package.get("dependencies", []):
            dependency_path = dependency.get("path")
            if not dependency_path:
                continue
            dependency_id = package_id_by_dir.get(Path(dependency_path).resolve())
            if dependency_id in package_by_id:
                workspace_dependencies.add(dependency_id)
        result[package_id] = {
            "id": package_id,
            "name": package["name"],
            "version": package["version"],
            "workspace_dependencies": workspace_dependencies,
        }
    return result


def publish_package(package: dict, args: argparse.Namespace) -> None:
    name = package["name"]
    version = package["version"]
    if crate_version_visible(name, version):
        print(f"already published: {name} {version}")
        return

    command = ["cargo", "publish", "-p", name, "--no-verify", "--locked"]
    deadline = time.monotonic() + args.publish_timeout_seconds
    last_output = ""

    for attempt in range(1, args.publish_attempts + 1):
        print(f"publishing {name} {version} (attempt {attempt}/{args.publish_attempts})")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        output = result.stdout + result.stderr
        last_output = output
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)

        if result.returncode == 0:
            wait_for_crate_version(name, version, args)
            return

        normalized = output.lower()
        if any(marker in normalized for marker in ALREADY_PUBLISHED_ERRORS):
            wait_for_crate_version(name, version, args)
            return

        if not any(marker in normalized for marker in TRANSIENT_PUBLISH_ERRORS):
            raise RuntimeError(f"cargo publish failed for {name} {version}")

        if attempt == args.publish_attempts or time.monotonic() >= deadline:
            break
        print(f"waiting for crates.io dependency propagation before retrying {name}")
        time.sleep(args.retry_delay_seconds)

    raise RuntimeError(f"cargo publish did not complete for {name} {version}\n{last_output}")


def wait_for_crate_version(name: str, version: str, args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.visibility_timeout_seconds
    while time.monotonic() < deadline:
        if crate_version_visible(name, version):
            print(f"visible on crates.io: {name} {version}")
            return
        print(f"waiting for crates.io visibility: {name} {version}")
        time.sleep(args.visibility_delay_seconds)
    raise RuntimeError(f"timed out waiting for crates.io visibility: {name} {version}")


def crate_version_visible(name: str, version: str) -> bool:
    encoded_name = urllib.parse.quote(name, safe="")
    encoded_version = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{encoded_name}/{encoded_version}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        print(f"crates.io lookup failed for {name} {version}: HTTP {error.code}", file=sys.stderr)
        return False
    except OSError as error:
        print(f"crates.io lookup failed for {name} {version}: {error}", file=sys.stderr)
        return False


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return json.loads(result.stdout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
