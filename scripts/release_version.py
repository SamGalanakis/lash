#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
CARGO_LOCK = ROOT / "Cargo.lock"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# Pre-release versions (e.g. "0.1.0-alpha.1") are honoured but never
# auto-bumped — during an alpha series the workspace version is set
# manually and CI's auto-sync no-ops.
PRERELEASE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)-[A-Za-z0-9.-]+$")


def main() -> int:
    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1]
    if command == "print-next":
        print(compute_next_version())
        return 0
    if command == "set":
        if len(sys.argv) != 3:
            print("set requires a version argument", file=sys.stderr)
            return 1
        value = sys.argv[2]
        if not (VERSION_RE.fullmatch(value) or PRERELEASE_RE.fullmatch(value)):
            print(f"unsupported version format: {value!r}", file=sys.stderr)
            return 1
        apply_version_text(value)
        return 0

    print(f"unknown command: {command}", file=sys.stderr)
    print_usage()
    return 1


def print_usage() -> None:
    print(
        "Usage:\n"
        "  scripts/release_version.py print-next\n"
        "  scripts/release_version.py set <version>",
        file=sys.stderr,
    )


def compute_next_version() -> str:
    workspace_version = read_workspace_version()
    # Pre-release versions ("0.1.0-alpha.1") are kept as-is — they are
    # managed by hand. Auto-bumping resumes once the workspace promotes
    # to a clean semver release.
    if PRERELEASE_RE.fullmatch(workspace_version):
        return workspace_version
    latest_tag = read_latest_release_tag()
    if latest_tag is None:
        return workspace_version

    latest_version = latest_tag.removeprefix("v")
    workspace_tuple = parse_version(workspace_version)
    latest_tuple = parse_version(latest_version)
    if workspace_tuple > latest_tuple:
        return workspace_version
    return format_version((latest_tuple[0], latest_tuple[1], latest_tuple[2] + 1))


def read_workspace_version() -> str:
    cargo_toml = CARGO_TOML.read_text()
    match = re.search(
        r"(?ms)^\[workspace\.package\]\s+version = \"([^\"]+)\"$",
        cargo_toml,
    )
    if not match:
        raise SystemExit("failed to find [workspace.package] version in Cargo.toml")
    value = match.group(1)
    # accept either clean semver or a pre-release tag; neither shape
    # raises here so the helper can still report the current version
    if not (VERSION_RE.fullmatch(value) or PRERELEASE_RE.fullmatch(value)):
        raise SystemExit(f"unsupported workspace version: {value!r}")
    return value


def read_latest_release_tag() -> str | None:
    result = subprocess.run(
        ["git", "tag", "--list", "v*", "--sort=-version:refname"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        tag = line.strip()
        if tag:
            parse_version(tag.removeprefix("v"))
            return tag
    return None


def parse_version(value: str) -> tuple[int, int, int]:
    match = VERSION_RE.fullmatch(value)
    if not match:
        raise SystemExit(f"unsupported version format: {value!r}")
    return tuple(int(part) for part in match.groups())


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def apply_version(version: tuple[int, int, int]) -> None:
    apply_version_text(format_version(version))


def apply_version_text(version_text: str) -> None:
    update_workspace_version(version_text)
    update_lockfile_versions(version_text)


def update_workspace_version(version: str) -> None:
    cargo_toml = CARGO_TOML.read_text()
    updated, count = re.subn(
        r"(?m)^version = \"[^\"]+\"$",
        f'version = "{version}"',
        cargo_toml,
        count=1,
    )
    if count != 1:
        raise SystemExit("failed to update workspace version in Cargo.toml")
    CARGO_TOML.write_text(updated)


def update_lockfile_versions(version: str) -> None:
    workspace_packages = read_workspace_package_names()
    lines = CARGO_LOCK.read_text().splitlines()
    current_name: str | None = None
    updated_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[[package]]":
            current_name = None
        elif stripped.startswith('name = "'):
            current_name = stripped[len('name = "') : -1]
        elif (
            current_name in workspace_packages
            and stripped.startswith('version = "')
        ):
            line = f'version = "{version}"'
        updated_lines.append(line)
    CARGO_LOCK.write_text("\n".join(updated_lines) + "\n")


def read_workspace_package_names() -> set[str]:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
            str(CARGO_TOML),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    workspace_members = set(metadata["workspace_members"])
    return {
        package["name"]
        for package in metadata["packages"]
        if package["id"] in workspace_members
    }


if __name__ == "__main__":
    raise SystemExit(main())
