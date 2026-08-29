"""tests/test_no_em_dashes.py: Guard against U+2014 (em dash) in cfr/ text files.

Standalone by design (cambia-1019): stdlib only, no import of anything under
cfr/src/. Runnable either via pytest or as a plain script:

    python3 -m pytest tests/test_no_em_dashes.py -q
    python3 tests/test_no_em_dashes.py

Walks all tracked-like text files under cfr/ and fails if any contains an
em dash (U+2014). Project writing style (see CLAUDE.md) forbids em dashes in
favor of colons, semicolons, commas, parentheses, simple dashes, or deletion.
"""

from __future__ import annotations

import sys
from pathlib import Path

EM_DASH = chr(0x2014)  # em dash, spelled via codepoint so this file has none literally

CFR_ROOT = Path(__file__).resolve().parent.parent

# Directory names skipped outright, wherever they appear in the tree.
SKIP_DIR_NAMES = {
    "runs",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}

# File extensions never treated as text worth scanning.
SKIP_SUFFIXES = {
    ".so",
    ".pyc",
    ".pyd",
    ".pyo",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".bin",
}


def _should_skip_dir(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIR_NAMES or name.endswith(".egg-info")


def _iter_candidate_files(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if not _should_skip_dir(entry.name):
                    stack.append(entry)
                continue
            if entry.is_file():
                yield entry


def _read_text_or_none(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None  # binary heuristic
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def find_em_dash_occurrences(root: Path) -> list[tuple[str, int, str]]:
    """Return (relative_path, line_number, line_text) for every em dash found."""
    hits: list[tuple[str, int, str]] = []
    for path in _iter_candidate_files(root):
        text = _read_text_or_none(path)
        if text is None:
            continue
        if EM_DASH not in text:
            continue
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                hits.append((rel, lineno, line))
    hits.sort()
    return hits


def test_no_em_dashes_in_cfr() -> None:
    hits = find_em_dash_occurrences(CFR_ROOT)
    if hits:
        detail = "\n".join(f"  {rel}:{lineno}: {line}" for rel, lineno, line in hits)
        raise AssertionError(
            f"Found {len(hits)} em dash (U+2014) occurrence(s) under cfr/, "
            f"which the project style forbids:\n{detail}"
        )


if __name__ == "__main__":
    found = find_em_dash_occurrences(CFR_ROOT)
    if found:
        for rel, lineno, line in found:
            print(f"{rel}:{lineno}: {line}")
        print(f"FAIL: {len(found)} em dash occurrence(s) found under cfr/")
        sys.exit(1)
    print("OK: no em dashes found under cfr/")
    sys.exit(0)
