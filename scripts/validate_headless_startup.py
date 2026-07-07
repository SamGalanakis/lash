#!/usr/bin/env python3
"""Headless startup validation for lash CLI credential/config changes.

Simulates `docker compose exec` (stdin is a pipe, not a TTY) and checks that
autonomous modes fail fast with actionable errors instead of raw-mode crashes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_lash_bin() -> Path:
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    if target_dir:
        candidate = Path(target_dir) / "debug" / "lash"
        if candidate.is_file():
            return candidate
    candidate = REPO_ROOT / "target" / "debug" / "lash"
    if candidate.is_file():
        return candidate
    return candidate


LASH_BIN = resolve_lash_bin()
VALID_CONFIG = {
    "active_provider": "openai-compatible",
    "providers": {
        "openai-compatible": {
            "type": "openai-compatible",
            "api_key": "fake-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
    },
}
INVALID_CONFIG = {
    **VALID_CONFIG,
    "unexpected_top_level_field": True,
}


@dataclass
class Case:
    name: str
    lash_home_key: str  # "valid" | "invalid" | "missing"
    args: list[str]
    env: dict[str, str]
    timeout_s: float
    expect_stderr: list[str]
    expect_stdout: list[str]
    forbid_stderr: list[str]
    forbid_stdout: list[str]
    allow_timeout: bool = False


def write_configs(base: Path) -> None:
    (base / "valid").mkdir(parents=True)
    (base / "invalid").mkdir(parents=True)
    (base / "missing").mkdir(parents=True)
    (base / "valid" / "config.json").write_text(json.dumps(VALID_CONFIG, indent=2))
    (base / "invalid" / "config.json").write_text(json.dumps(INVALID_CONFIG, indent=2))


def run_case(base: Path, case: Case) -> tuple[int, str, str]:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "USER": os.environ.get("USER", ""),
    }
    if case.lash_home_key == "missing":
        env["LASH_HOME"] = str(base / "missing")
    else:
        env["LASH_HOME"] = str(base / case.lash_home_key)
    for key, value in case.env.items():
        if value:
            env[key] = value

    proc = subprocess.run(
        [str(LASH_BIN), *case.args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=case.timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def matches(pattern: str, haystack: str) -> bool:
    return re.search(pattern, haystack, re.IGNORECASE | re.MULTILINE) is not None


def main() -> int:
    if not LASH_BIN.is_file():
        print(
            "error: lash binary not found; run `cargo build -p lash-cli --bin lash` "
            f"(checked {LASH_BIN})"
        )
        return 1
    print(f"using lash binary: {LASH_BIN}")

    cases = [
        Case(
            name="valid-info",
            lash_home_key="valid",
            args=["--info"],
            env={},
            timeout_s=8,
            expect_stdout=[r"provider:"],
            expect_stderr=[],
            forbid_stderr=[r"enable raw mode"],
            forbid_stdout=[r"provider: \(not configured\)"],
        ),
        Case(
            name="invalid-print",
            lash_home_key="invalid",
            args=["--print", "hi"],
            env={},
            timeout_s=8,
            expect_stderr=[r"no usable lash config", r"config present but invalid"],
            expect_stdout=[],
            forbid_stderr=[r"enable raw mode"],
            forbid_stdout=[],
        ),
        Case(
            name="missing-print",
            lash_home_key="missing",
            args=["--print", "hi"],
            env={},
            timeout_s=8,
            expect_stderr=[r"no usable lash config", r"config file not found"],
            expect_stdout=[],
            forbid_stderr=[r"enable raw mode"],
            forbid_stdout=[],
        ),
        Case(
            name="invalid-info",
            lash_home_key="invalid",
            args=["--info"],
            env={},
            timeout_s=8,
            expect_stdout=[r"config: present but invalid", r"provider: \(not configured\)"],
            expect_stderr=[],
            forbid_stderr=[r"enable raw mode"],
            forbid_stdout=[],
        ),
        Case(
            name="openrouter-bypass",
            lash_home_key="invalid",
            args=[
                "--base-url",
                "https://openrouter.ai/api/v1",
                "--print",
                "hi",
            ],
            env={
                "OPENROUTER_API_KEY": "or-fake-key",
                "OPENAI_API_KEY": "",
                "OPENAI_COMPATIBLE_API_KEY": "",
            },
            timeout_s=20,
            expect_stderr=[r"(401|unauthorized|invalid|error|api)"],
            expect_stdout=[],
            forbid_stderr=[r"enable raw mode", r"no usable lash config"],
            forbid_stdout=[],
            allow_timeout=True,
        ),
        Case(
            name="provider-flag-headless",
            lash_home_key="valid",
            args=["--provider", "--print", "hi"],
            env={},
            timeout_s=8,
            expect_stderr=[r"`--provider` cannot be used"],
            expect_stdout=[],
            forbid_stderr=[r"enable raw mode"],
            forbid_stdout=[],
        ),
    ]

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lash-headless-validate-") as tmp:
        base = Path(tmp)
        write_configs(base)

        for case in cases:
            print(f"=== {case.name} ===")
            try:
                code, stdout, stderr = run_case(base, case)
            except subprocess.TimeoutExpired:
                if case.allow_timeout:
                    print("timed out (allowed for LLM reachability check)")
                    continue
                failures.append(f"{case.name}: timed out after {case.timeout_s}s")
                continue

            print(f"exit={code}")
            if stderr.strip():
                print("stderr:", stderr.strip().splitlines()[0])
            if stdout.strip():
                print("stdout:", stdout.strip().splitlines()[0])

            combined_out = stdout
            combined_err = stderr
            for pattern in case.expect_stderr:
                if not matches(pattern, combined_err):
                    failures.append(f"{case.name}: expected stderr /{pattern}/")
            for pattern in case.expect_stdout:
                if not matches(pattern, combined_out):
                    failures.append(f"{case.name}: expected stdout /{pattern}/")
            for pattern in case.forbid_stderr:
                if matches(pattern, combined_err):
                    failures.append(f"{case.name}: forbidden stderr /{pattern}/")
            for pattern in case.forbid_stdout:
                if matches(pattern, combined_out):
                    failures.append(f"{case.name}: forbidden stdout /{pattern}/")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(f"PASSED all {len(cases)} headless scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
