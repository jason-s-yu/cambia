# Codebridge ingest: verification and drift runbook

cambia's `.docs/` tree is gitignored and lives, searchable, on the Codebridge hub
(`https://codebridge.example.internal`, project `cambia`). The hub is the
authoritative store for this prose. This runbook covers keeping the hub in sync
with `.docs/` after the initial ingest.

## What's where

| Artifact | Path | Tracked | Purpose |
|-|-|-|-|
| Hub binding | `.codebridge/config.yaml` | yes | project + hub_url + ingest path/override rules |
| Ingest manifest | `.codebridge/ingest-manifest.json` | no (gitignored) | committed ingest state: per-doc hub number + content_hash + path_hint |
| Preview | `.codebridge/ingest-preview.json` | no | editable Phase-1 dry-run output |
| Verifier | `scripts/cb_ingest_verify.py` | yes | deterministic three-way reconciliation (this runbook) |
| Drift workflow | `.claude/workflows/cambia-ingest-drift.js` | no (`.claude/` ignored) | orchestrated verify -> diagnose -> optional reconcile |
| Ingest scripts | `~/.claude*/skills/cb/skills/ingest/` | n/a (cb plugin) | `parse_docs.py`, `commit_ingest.py`, `uningest.py` |

The bearer secret is never in the repo: it lives in `~/.config/codebridge/credentials`
(`[default] auth_secret`). The verifier and ingest scripts read it from there.

## Lifecycle decisions baked into the binding

cambia versioning is non-monotonic: `v0.4` (PRT-CFR) is the re-keyed latest,
superseding the older `v2` and `v3` trees. The default integer `version_dir_pattern`
reads `v3` as the max live version (`v0.4` fails `^v\d+$`), so it would wrongly keep
`v3/` active. `config.yaml` `ingest.overrides` pins reality:

- `v3/*` (DESCA, gate-failed) -> `superseded` (the cfr-ceiling research + v0.4 supersede it).
- `index.md` -> `active` (the default classifier false-archives it on a body CLOSED marker).
- `roadmap.md` -> `superseded` (historical SoG/GT-CFR direction).
- `v2/*` archives by the version rule; `archive/*` by prefix; three `archive/*` docs are `invalidated` by body markers (guardrails that surface at full weight).

When a new doc lands under `v3/`, add an override before re-ingesting, or it will
classify `superseded` only if a body marker says so (it will otherwise be `active`).

## Verify (read-only, run anytime)

```fish
python3 scripts/cb_ingest_verify.py            # human summary + exit code
python3 scripts/cb_ingest_verify.py --json .codebridge/drift-report.json
```

It reconciles three sources: DISK (`.docs/`, hashed by the canonical `parse_docs.py`),
MANIFEST, and the live HUB. Exit 0 = in sync, 1 = drift, 2 = error (CI-usable). It
mutates nothing. Drift classes:

| Class | Meaning | Remediation |
|-|-|-|
| `changed_on_disk` | doc body edited since ingest | re-ingest (supersession) |
| `added_on_disk` | new doc on disk, not ingested | re-ingest (new) |
| `moved_on_disk` | same body, new path | re-ingest (rename) |
| `deleted_on_disk` | in manifest, gone from disk; hub row source-orphaned | judgment: archive/delete the hub row, or restore the file |
| `missing_on_hub` | in manifest, no live hub row | re-ingest (recreate) |
| `hub_hash_drift` | hub body edited out of band | re-ingest, or accept the hub edit |
| `lifecycle_drift` | hub status != current classifier | pin in `ingest.overrides` + re-ingest, or `manage_docs` lifecycle op |

## Address drift

### Safe, deterministic (re-ingest)

`changed`, `added`, `moved`, and `missing_on_hub` are all reconciled by one re-ingest
over the live manifest, which annotates each doc skip / rename / supersede / new by
content_hash:

```fish
set CB (ls -d ~/.claude*/skills/cb/skills/ingest 2>/dev/null | head -1)
python3 $CB/parse_docs.py .docs --prior-manifest .codebridge/ingest-manifest.json --out .codebridge/ingest-preview.json
# review the preview (reingest_action per doc), then commit:
python3 $CB/commit_ingest.py --force --preview .codebridge/ingest-preview.json
```

`--force` is required because the project is already ingested (a live manifest exists).
Index-only is the default; `.docs/` is gitignored so `--materialize` is not applicable.

### Lifecycle drift

Pin the intended status in `.codebridge/config.yaml` under `ingest.overrides`
(`<docs-relative-path>: { lifecycle_status: <status> }`) and re-ingest, or apply
directly with `manage_docs` (`archive` / `unarchive` / `supersede` / `invalidate`).

### Judgment calls (never auto-applied)

- `deleted_on_disk`: the source file is gone but the hub row remains. Decide: archive
  the hub row, delete it, or restore the file. The verifier surfaces it; it does not act.
- Orphan-count gap (hub doc count > manifest count): hub has rows the manifest doesn't
  know, usually hand-created. Per-doc orphan identity is not enumerable (see below).

## Full reset / un-ingest

- Un-ingest (reverse the index, leave `.docs/` untouched):
  `python3 $CB/uningest.py --manifest .codebridge/ingest-manifest.json`
- Hard reset of all hub state for the project (items + docs + memories), bearer-authed
  admin route, gated by a type-the-name interlock:
  `curl -X POST -H "Authorization: Bearer $SECRET" -d '{"project":"cambia","confirm":"cambia"}' $HUB/api/admin/purge_project`
  Then re-bind + re-run the two ingest phases.

## Orchestrated workflow

`.claude/workflows/cambia-ingest-drift.js` runs verify -> diagnose -> (optional) reconcile.
Default is read-only (verify + a prioritized reconciliation plan). Pass `{apply:true}`
to also apply the safe deterministic steps and re-verify; judgment calls are always
surfaced, never auto-applied. Invoke it via the Workflow tool (it is not a worker-runnable
skill).

## Known limitations

- Per-doc ORPHAN identity is not enumerable: cbmcp exposes no project-scoped doc-list
  endpoint. The verifier flags the orphan CONDITION via the count gap but cannot name
  the rows. A `list_documents(project=)` tool would close this.
- `manage_docs get` returns full bodies (no metadata-only mode); the verifier parses
  them for content_hash + lifecycle and discards the bodies. A lean fetch would cut the
  per-run payload.
