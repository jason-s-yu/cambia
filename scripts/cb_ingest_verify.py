#!/usr/bin/env python3
"""Verify the Codebridge ingest of `.docs/` against the live hub, and report drift.

The ingest (`/cb:ingest`) makes the hub the searchable home for this repo's
gitignored `.docs/` prose. That index drifts over time: docs get edited, added,
or deleted on disk; hub rows get hand-edited or lifecycle-flipped; an ingest is
half-applied. This script is the re-runnable reconciliation that finds all of it.

It is DETERMINISTIC and READ-ONLY: it never mutates `.docs/`, the hub, or the
manifest. It classifies every doc and prints the exact remediation commands; you
(or the `cambia-ingest-drift` workflow) decide what to apply.

Three sources, reconciled:

  1. DISK     - the live `.docs/` tree, hashed by the canonical parser
                (`parse_docs.py`, so the content_hash algorithm matches the hub's).
  2. MANIFEST - `.codebridge/ingest-manifest.json`, the committed ingest state
                (per-doc hub number + content_hash + path_hint).
  3. HUB      - the live Document rows (content_hash, lifecycle_status, deleted_at),
                fetched over the bearer-authed REST shim.

Drift classes (per doc):

  in_sync          disk == manifest == hub, lifecycle matches the current classifier
  changed_on_disk  disk body differs from the manifest (parser reingest_action=supersede)
  moved_on_disk    same body, new path (reingest_action=rename)
  added_on_disk    on disk, absent from the manifest (reingest_action=new)
  deleted_on_disk  in the manifest, gone from disk -> its hub row is now source-orphaned
  missing_on_hub   in the manifest, the hub has no live row (deleted/lost on the hub)
  hub_hash_drift   hub content_hash != manifest content_hash (hub edited out of band)
  lifecycle_drift  hub lifecycle_status != what the current config classifier assigns

Plus an aggregate orphan check: hub project doc count vs manifest doc count (a
per-doc orphan identity needs a hub doc-list endpoint, which cbmcp does not yet
expose -- see KNOWN LIMITATIONS at the bottom).

Exit code: 0 = fully in sync, 1 = drift detected, 2 = operational error.

Usage:
    python3 scripts/cb_ingest_verify.py                  # verify cwd's ingest
    python3 scripts/cb_ingest_verify.py --root /path     # explicit repo root
    python3 scripts/cb_ingest_verify.py --json out.json  # also write a JSON report
    python3 scripts/cb_ingest_verify.py --quiet          # summary + exit code only

Auth/URL resolution mirrors the ingest scripts: hub_url from `.codebridge/config.yaml`
(or --url / $CODEBRIDGE_URL); secret from $CODEBRIDGE_AUTH_SECRET or
`~/.config/codebridge/credentials` ([default] auth_secret). stdlib only.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HUB_URL = "http://127.0.0.1:8000"
CRED_PATH = Path.home() / ".config" / "codebridge" / "credentials"
HUB_GET_CHUNK = 20  # docs per /api/manage_docs get batch


# --------------------------------------------------------------------------- #
# Resolution: root, config, manifest, hub url, secret, parse_docs.py
# --------------------------------------------------------------------------- #
def find_root(start: Path) -> Path:
    """Walk up from *start* for the dir holding `.codebridge/config.yaml`."""
    d = start.resolve()
    while True:
        if (d / ".codebridge" / "config.yaml").is_file():
            return d
        if d.parent == d:
            return start.resolve()
        d = d.parent


def read_flat_yaml_key(path: Path, key: str) -> str | None:
    """First `key: value` line (quotes/whitespace stripped), no YAML dep."""
    if not path.is_file():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            v = line[len(prefix):].strip()
            if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
                v = v[1:-1].strip()
            return v or None
    return None


def resolve_hub_url(cli_url: str | None, config_path: Path) -> str:
    if cli_url and cli_url.strip():
        return cli_url.strip().rstrip("/")
    env = os.environ.get("CODEBRIDGE_URL", "").strip()
    if env:
        return env.rstrip("/")
    cfg = read_flat_yaml_key(config_path, "hub_url")
    if cfg:
        return cfg.rstrip("/")
    return DEFAULT_HUB_URL


def resolve_secret(cli_secret: str | None) -> str | None:
    if cli_secret:
        return cli_secret
    env = os.environ.get("CODEBRIDGE_AUTH_SECRET")
    if env:
        return env
    if CRED_PATH.is_file():
        cp = configparser.ConfigParser()
        try:
            cp.read(CRED_PATH)
            if cp.has_option("default", "auth_secret"):
                return cp.get("default", "auth_secret")
        except configparser.Error:
            return None
    return None


def find_parse_docs(cli_path: str | None) -> Path | None:
    """Locate the canonical parse_docs.py (CLI flag > env > glob of cb skills)."""
    if cli_path:
        p = Path(cli_path).expanduser()
        return p if p.is_file() else None
    env = os.environ.get("CB_INGEST_PARSE_DOCS")
    if env and Path(env).is_file():
        return Path(env)
    patterns = [
        str(Path.home() / ".claude*/skills/cb/skills/ingest/parse_docs.py"),
        str(Path.home() / ".claude*/plugins/**/cb/skills/ingest/parse_docs.py"),
        str(Path.home() / ".config/claude*/**/cb/skills/ingest/parse_docs.py"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return None


# --------------------------------------------------------------------------- #
# Disk pass: re-parse `.docs/` with the prior manifest -> reingest_action + hash
# --------------------------------------------------------------------------- #
def run_disk_pass(parse_docs: Path, docs_root: Path, manifest: Path) -> dict:
    """Run parse_docs --prior-manifest into a temp preview; return its dict.

    Using the canonical parser guarantees the content_hash + lifecycle
    classification (incl. config.yaml overrides) match what an actual re-ingest
    would produce. reingest_action encodes the disk-vs-manifest comparison.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "verify-preview.json"
        cmd = [
            sys.executable, str(parse_docs), str(docs_root),
            "--prior-manifest", str(manifest), "--out", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"parse_docs failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )
        return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Hub pass: fetch each manifest doc's live state (content_hash, lifecycle, deleted)
# --------------------------------------------------------------------------- #
def hub_post(base_url: str, tool: str, body: dict, secret: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        f"{base_url}/api/{tool}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hub_get_docs(base_url: str, numbers: list[int], secret: str) -> dict[int, dict]:
    """Batch-get docs by global number. On a batch error (e.g. a missing id makes
    the all-or-nothing batch fail), fall back to per-doc gets to isolate it.
    Returns {number: {content_hash, lifecycle_status, deleted_at, path_hint}};
    a number absent from the result is missing on the hub."""
    found: dict[int, dict] = {}

    def ingest_result(res: dict) -> None:
        for d in res.get("result", []):
            num = d.get("number") or d.get("id")
            if num is not None:
                found[int(num)] = {
                    "content_hash": d.get("content_hash"),
                    "lifecycle_status": d.get("lifecycle_status"),
                    "deleted_at": d.get("deleted_at"),
                    "path_hint": d.get("path_hint"),
                    "display_handle": d.get("display_handle") or d.get("display_number"),
                }

    for i in range(0, len(numbers), HUB_GET_CHUNK):
        chunk = numbers[i:i + HUB_GET_CHUNK]
        ops = [{"op": "get", "doc_id": n} for n in chunk]
        try:
            ingest_result(hub_post(base_url, "manage_docs", {"ops": ops}, secret))
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
            for n in chunk:  # isolate the offender(s)
                try:
                    ingest_result(hub_post(base_url, "manage_docs", {"ops": [{"op": "get", "doc_id": n}]}, secret))
                except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
                    pass  # left out of `found` -> reported missing_on_hub
    return found


def hub_doc_count(base_url: str, project: str, secret: str) -> int | None:
    """Project doc count via get_session_context.doc_refs_total (the session/context
    tool takes an unwrapped {project} body over the REST shim). None if unavailable."""
    try:
        res = hub_post(base_url, "get_session_context", {"project": project}, secret)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None
    return res.get("doc_refs_total")


# --------------------------------------------------------------------------- #
# Reconcile
# --------------------------------------------------------------------------- #
def reconcile(manifest: dict, preview: dict, hub: dict[int, dict]) -> dict:
    """Three-way reconcile. Returns {class: [records]} + counts."""
    man_docs = manifest.get("documents", [])
    prev_docs = preview.get("documents", [])
    prev_by_path = {d["source_path"]: d for d in prev_docs}
    prev_paths = set(prev_by_path)

    # manifest keyed by source_path (its stable on-disk locator). Supersession
    # chains leave multiple rows per path; the live end of the chain is the row
    # whose content_hash matches what the parser sees on disk now. Older chain
    # rows are history, not drift.
    man_by_path: dict[str, dict] = {}
    for d in man_docs:
        sp = d["source_path"]
        cur = man_by_path.get(sp)
        if cur is None:
            man_by_path[sp] = d
            continue
        disk_hash = (prev_by_path.get(sp) or {}).get("content_hash")
        if disk_hash and d.get("content_hash") == disk_hash:
            man_by_path[sp] = d
        elif disk_hash and cur.get("content_hash") == disk_hash:
            pass  # already holding the live row
        else:
            man_by_path[sp] = d  # no disk match on either row: last wins
    man_paths = set(man_by_path)

    buckets: dict[str, list] = {
        "in_sync": [], "changed_on_disk": [], "moved_on_disk": [], "added_on_disk": [],
        "deleted_on_disk": [], "missing_on_hub": [], "hub_hash_drift": [], "lifecycle_drift": [],
    }

    # Disk-side: classify every doc the parser currently sees.
    for d in prev_docs:
        sp = d["source_path"]
        action = d.get("reingest_action", "new")
        rec = {
            "source_path": sp,
            "reingest_action": action,
            "disk_hash": (d.get("content_hash") or "")[:16],
            "classified_lifecycle": d.get("lifecycle_status"),
        }
        if action == "new":
            buckets["added_on_disk"].append(rec)
            continue
        if action == "rename":
            buckets["moved_on_disk"].append(rec)
        elif action == "supersede":
            buckets["changed_on_disk"].append(rec)

        # For docs the manifest knows (skip/rename/supersede), check the hub row.
        man = man_by_path.get(sp)
        if not man:
            continue
        num = int(man["number"])
        rec["number"] = num
        hub_row = hub.get(num)
        if hub_row is None or hub_row.get("deleted_at") is not None:
            buckets["missing_on_hub"].append({**rec, "hub": "absent"})
            continue
        # hash drift: hub vs manifest (out-of-band hub edit)
        if hub_row.get("content_hash") and man.get("content_hash") and \
                hub_row["content_hash"] != man["content_hash"]:
            buckets["hub_hash_drift"].append({
                **rec, "manifest_hash": man["content_hash"][:16],
                "hub_hash": hub_row["content_hash"][:16],
            })
        # lifecycle drift: hub vs current classifier
        hub_lc = hub_row.get("lifecycle_status")
        cls_lc = d.get("lifecycle_status")
        if hub_lc and cls_lc and hub_lc != cls_lc:
            buckets["lifecycle_drift"].append({
                **rec, "hub_lifecycle": hub_lc, "classified_lifecycle": cls_lc,
            })
        if action == "skip" and num in hub and hub_row.get("deleted_at") is None \
                and hub_lc == cls_lc \
                and not (hub_row.get("content_hash") and man.get("content_hash")
                         and hub_row["content_hash"] != man["content_hash"]):
            buckets["in_sync"].append(rec)

    # Deletions: manifest paths the parser no longer sees on disk.
    for sp in sorted(man_paths - prev_paths):
        man = man_by_path[sp]
        num = int(man["number"])
        hub_row = hub.get(num)
        buckets["deleted_on_disk"].append({
            "source_path": sp, "number": num,
            "hub": "absent" if (hub_row is None or hub_row.get("deleted_at")) else "present-orphaned",
        })

    counts = {k: len(v) for k, v in buckets.items()}
    return {"buckets": buckets, "counts": counts}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render(report: dict, *, quiet: bool) -> None:
    c = report["counts"]
    drift_classes = ["changed_on_disk", "moved_on_disk", "added_on_disk",
                     "deleted_on_disk", "missing_on_hub", "hub_hash_drift", "lifecycle_drift"]
    total_drift = sum(c[k] for k in drift_classes)
    orphan = report.get("orphan_count")

    print(f"\ncb-ingest-verify: project '{report['project']}' @ {report['hub_url']}")
    print(f"  manifest docs: {report['manifest_doc_count']}   hub docs: "
          f"{report['hub_doc_count'] if report['hub_doc_count'] is not None else '?'}   "
          f"in_sync: {c['in_sync']}")
    print(f"  {'IN SYNC' if total_drift == 0 and not orphan else 'DRIFT DETECTED'}: "
          f"{total_drift} doc(s) across {sum(1 for k in drift_classes if c[k])} class(es)"
          + (f"; orphan-count gap {orphan:+d}" if orphan else ""))

    order = [
        ("changed_on_disk", "edited on disk since ingest"),
        ("added_on_disk", "new on disk, not ingested"),
        ("moved_on_disk", "moved on disk (same body)"),
        ("deleted_on_disk", "removed from disk; hub row now source-orphaned"),
        ("missing_on_hub", "in manifest, no live hub row"),
        ("hub_hash_drift", "hub body edited out of band"),
        ("lifecycle_drift", "hub lifecycle != current classifier"),
    ]
    if not quiet:
        for key, desc in order:
            recs = report["buckets"][key]
            if not recs:
                continue
            print(f"\n  [{key}] {len(recs)} - {desc}")
            for r in recs[:50]:
                extra = ""
                if key == "lifecycle_drift":
                    extra = f"  ({r['hub_lifecycle']} -> {r['classified_lifecycle']})"
                elif key == "hub_hash_drift":
                    extra = f"  (manifest {r['manifest_hash']} vs hub {r['hub_hash']})"
                elif key == "deleted_on_disk":
                    extra = f"  (hub: {r['hub']})"
                print(f"    - {r['source_path']}{extra}")
            if len(recs) > 50:
                print(f"    ... {len(recs) - 50} more (see --json)")

    # Remediation
    print("\n  remediation:")
    if total_drift == 0 and not orphan:
        print("    none - hub matches .docs/ and the manifest.")
    else:
        if c["changed_on_disk"] or c["added_on_disk"] or c["moved_on_disk"] or c["missing_on_hub"]:
            print("    re-ingest (content_hash supersession: supersede/rename/new, recreate missing):")
            print("      python3 <cb>/parse_docs.py .docs --prior-manifest "
                  ".codebridge/ingest-manifest.json --out .codebridge/ingest-preview.json")
            print("      # review the preview, then:")
            print("      python3 <cb>/commit_ingest.py --force --preview .codebridge/ingest-preview.json")
        if c["lifecycle_drift"]:
            print("    lifecycle drift: pin the intended status in .codebridge/config.yaml "
                  "`ingest.overrides`, then re-ingest;")
            print("      or apply directly via manage_docs (archive/supersede/invalidate/unarchive).")
        if c["deleted_on_disk"]:
            print("    deleted-on-disk: the hub row is now source-orphaned - archive or delete it "
                  "(manage_docs), or restore the file. JUDGMENT CALL - not auto-applied.")
        if orphan:
            print(f"    orphan-count gap {orphan:+d}: hub has more docs than the manifest - "
                  "likely hand-created rows. Per-doc identity needs a hub doc-list endpoint "
                  "(see KNOWN LIMITATIONS). JUDGMENT CALL.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the Codebridge ingest of .docs/ and report drift.")
    ap.add_argument("--root", help="repo root (default: walk up from cwd for .codebridge/config.yaml)")
    ap.add_argument("--url", help="hub base URL (default: config.yaml hub_url / $CODEBRIDGE_URL / localhost)")
    ap.add_argument("--secret", help="bearer secret (default: $CODEBRIDGE_AUTH_SECRET / credentials file)")
    ap.add_argument("--parse-docs", help="path to the cb ingest parse_docs.py (default: auto-locate)")
    ap.add_argument("--json", help="also write the full drift report to this path")
    ap.add_argument("--quiet", action="store_true", help="summary + exit code only")
    args = ap.parse_args(argv)

    root = find_root(Path(args.root) if args.root else Path.cwd())
    config_path = root / ".codebridge" / "config.yaml"
    manifest_path = root / ".codebridge" / "ingest-manifest.json"
    docs_root = root / ".docs"

    if not manifest_path.is_file():
        print(f"error: no manifest at {manifest_path} - project not ingested yet.", file=sys.stderr)
        return 2
    if not docs_root.is_dir():
        print(f"error: no docs root at {docs_root}.", file=sys.stderr)
        return 2

    project = read_flat_yaml_key(config_path, "project") or root.name
    hub_url = resolve_hub_url(args.url, config_path)
    secret = resolve_secret(args.secret)
    if not secret:
        print("error: no hub secret ($CODEBRIDGE_AUTH_SECRET or ~/.config/codebridge/credentials).", file=sys.stderr)
        return 2
    parse_docs = find_parse_docs(args.parse_docs)
    if not parse_docs:
        print("error: could not locate parse_docs.py - pass --parse-docs or set $CB_INGEST_PARSE_DOCS.", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    man_docs = manifest.get("documents", [])

    try:
        preview = run_disk_pass(parse_docs, docs_root, manifest_path)
        numbers = [int(d["number"]) for d in man_docs]
        hub = hub_get_docs(hub_url, numbers, secret)
        hub_count = hub_doc_count(hub_url, project, secret)
    except Exception as exc:  # operational failure - report and bail clean
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = reconcile(manifest, preview, hub)
    report["project"] = project
    report["hub_url"] = hub_url
    report["manifest_doc_count"] = len(man_docs)
    report["hub_doc_count"] = hub_count
    report["orphan_count"] = (hub_count - len(man_docs)) if isinstance(hub_count, int) else None

    render(report, quiet=args.quiet)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")

    drift = sum(report["counts"][k] for k in report["counts"] if k != "in_sync")
    return 1 if (drift or report.get("orphan_count")) else 0


if __name__ == "__main__":
    sys.exit(main())

# KNOWN LIMITATIONS
# - Per-doc ORPHAN identity (hub rows with no manifest entry) is not enumerable:
#   cbmcp exposes no project-scoped doc-list endpoint. This script flags the
#   orphan CONDITION via the count gap (hub doc_refs_total vs manifest count) but
#   cannot name the orphan rows. A `list_documents(project=)` tool would close it.
# - hub_get_docs pulls full bodies (manage_docs `get` has no metadata-only mode);
#   bodies are parsed for content_hash/lifecycle and discarded. Fine for a verify
#   run; a lean metadata fetch would cut the payload.
