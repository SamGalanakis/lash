#!/usr/bin/env python3
"""Collect curated release notes from commit messages.

Convention: a commit that should contribute user-facing release notes carries a
`Release-Notes:` line in its body; everything after that line (to the end of
the message) is the note, written as Markdown. Notes are collected across the
release range (previous release tag → the release ref), oldest first, so the
release body reads chronologically. The publish-time version-injection flow
authors no synthetic commits (no version bump, no staging sync), so every
commit in range is a real change eligible to contribute notes.

The release pipeline uses two entry points:
  - the fail-early `release-notes-gate` job runs `collect --require` at t=0 on
    a main push, so a range with no curated notes fails in under a minute
    instead of after the full matrix; the release-cut job relies on the gate.
  - the publish job runs `collect --end <tag> --out <file>` and feeds the file
    to the GitHub release body (the auto-generated commit list is appended
    below it). Empty output is allowed there so manually pushed tags still
    publish.

Uses only the Python standard library, like the sibling release scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MARKER = "Release-Notes:"
RECORD_SEPARATOR = "\x1e"
FIELD_SEPARATOR = "\x1f"


def previous_tag(end: str) -> str | None:
    """The nearest release tag reachable from (and not equal to) `end`.

    Graph ancestry, not version sorting: the repo's tag namespace contains
    tags from unrelated history lines, so "previous release" means the
    nearest `v*` ancestor on this line.
    """
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*", f"{end}^"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def end_is_released(end: str) -> bool:
    """True when `end` resolves to a commit that already carries a release tag.

    Keeps `--require` reruns idempotent: a re-run on an already-tagged commit
    has an empty range by definition and must not fail the gate.
    """
    result = subprocess.run(
        ["git", "tag", "--list", "v*", "--points-at", end],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def extract_note(body: str) -> str | None:
    """The Markdown after a standalone `Release-Notes:` line, or None."""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == MARKER:
            note = "\n".join(lines[index + 1 :]).strip()
            return note or None
        # Also accept inline form: "Release-Notes: text on the same line".
        if line.strip().startswith(MARKER):
            first = line.strip()[len(MARKER) :].strip()
            rest = "\n".join(lines[index + 1 :]).strip()
            note = "\n".join(part for part in (first, rest) if part).strip()
            return note or None
    return None


def collect_notes(end: str) -> list[str]:
    prev = previous_tag(end)
    range_spec = f"{prev}..{end}" if prev else end
    result = subprocess.run(
        [
            "git",
            "log",
            "--reverse",
            f"--format=%H{FIELD_SEPARATOR}%s{FIELD_SEPARATOR}%B{RECORD_SEPARATOR}",
            range_spec,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    notes: list[str] = []
    for record in result.stdout.split(RECORD_SEPARATOR):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(FIELD_SEPARATOR)
        if len(parts) != 3:
            continue
        _sha, _subject, body = parts
        note = extract_note(body)
        if note:
            notes.append(note)
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="collect notes for a release range")
    collect.add_argument(
        "--end",
        default="HEAD",
        help="end of the release range: a release tag or HEAD (default)",
    )
    collect.add_argument(
        "--out",
        type=Path,
        help="write collected notes to this file (always written; may be empty)",
    )
    collect.add_argument(
        "--require",
        action="store_true",
        help="fail when the range contains no Release-Notes sections",
    )
    args = parser.parse_args()

    notes = collect_notes(args.end)
    rendered = "\n\n".join(notes).strip()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n" if rendered else "", encoding="utf-8")
    if args.require and not rendered and args.end == "HEAD" and end_is_released("HEAD"):
        sys.stderr.write("HEAD already carries a release tag; nothing new to gate.\n")
        return 0
    if args.require and not rendered:
        prev = previous_tag(args.end)
        sys.stderr.write(
            "no release notes found in "
            f"{prev or 'history start'}..{args.end}.\n"
            "Add a `Release-Notes:` section to at least one commit body\n"
            "(Markdown, everything after the marker line), or push with\n"
            "`[skip release]` if this range should not cut a release.\n"
        )
        return 2
    if rendered and not args.out:
        sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
