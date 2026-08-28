#!/usr/bin/env python3
"""Verify PACKAGE_SHA256SUMS.txt using Python's plaintext file reads."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKSUM_PATH = PROJECT_ROOT / "PACKAGE_SHA256SUMS.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not CHECKSUM_PATH.is_file():
        print(f"Missing {CHECKSUM_PATH}", file=sys.stderr)
        return 2

    checked = 0
    failed = 0
    for line_number, line in enumerate(
        CHECKSUM_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            print(f"Malformed checksum line {line_number}", file=sys.stderr)
            failed += 1
            continue
        path = PROJECT_ROOT / relative.removeprefix("./")
        if not path.is_file():
            print(f"MISSING: {relative}", file=sys.stderr)
            failed += 1
            continue
        actual = _sha256(path)
        if actual != expected:
            print(f"FAILED: {relative}", file=sys.stderr)
            failed += 1
        checked += 1

    if failed:
        print(f"FAIL: {failed} of {checked} files did not verify.", file=sys.stderr)
        return 1
    print(f"PASS: verified {checked} packaged files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
