#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CARGO_TOML = ROOT / "Cargo.toml"
CARGO_LOCK = ROOT / "Cargo.lock"
WORKSPACE_PACKAGES = ("lash-cli", "lash-core", "lashlang", "xtask")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


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
        version = parse_version(sys.argv[2])
        apply_version(version)
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
    parse_version(match.group(1))
    return match.group(1)


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
    version_text = format_version(version)
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
            current_name in WORKSPACE_PACKAGES
            and stripped.startswith('version = "')
        ):
            line = f'version = "{version}"'
        updated_lines.append(line)
    CARGO_LOCK.write_text("\n".join(updated_lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
