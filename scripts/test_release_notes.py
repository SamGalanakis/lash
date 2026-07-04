#!/usr/bin/env python3
"""Tests for release_notes.py (pure functions; no git fixtures needed)."""

from __future__ import annotations

import release_notes


def test_extract_note_standalone_marker() -> None:
    body = "Subject\n\nDetails here.\n\nRelease-Notes:\n- Added X\n- Fixed Y\n"
    assert release_notes.extract_note(body) == "- Added X\n- Fixed Y"


def test_extract_note_inline_marker() -> None:
    body = "Subject\n\nRelease-Notes: Single line note.\n"
    assert release_notes.extract_note(body) == "Single line note."


def test_extract_note_inline_marker_with_continuation() -> None:
    body = "Subject\n\nRelease-Notes: First line.\nSecond line.\n"
    assert release_notes.extract_note(body) == "First line.\nSecond line."


def test_extract_note_absent_or_empty() -> None:
    assert release_notes.extract_note("Subject\n\nNo marker here.\n") is None
    assert release_notes.extract_note("Subject\n\nRelease-Notes:\n\n") is None


def test_marker_must_start_the_line() -> None:
    body = "Subject\n\nSee the Release-Notes: convention for details.\n"
    assert release_notes.extract_note(body) is None


def main() -> int:
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok   {name}")
            except AssertionError as err:
                failures += 1
                print(f"FAIL {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
