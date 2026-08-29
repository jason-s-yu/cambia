"""cfr/scripts/gate_vs_served_gap.py

X2 gate metric vs served SD-CFR policy: exact NashConv of both objects on the
corrected {A,6} tiny tree (cambia-737).

Why this exists
---------------
The X2 gate scores one object and the eval/serving stack plays another, and the
two have never been compared:

  gate (src/cfr/prtcfr_eval.materialize_policy_incremental, the number the
  stability controller records and the X2R pre-registration rules read):

      b_gate(I)[a] = sum_{s<=t} w_s * sigma_s(I)[a] / sum_{s<=t} w_s,  w_s = s

    a per-decision, reach-UNWEIGHTED linear-iteration mean of the per-snapshot
    regret-matched strategies.

  served (src/evaluate_agents.PRTCFRAgentWrapper -> src/cfr/prtcfr_mixture):
    one snapshot is sampled per EPISODE proportional to w_s = s and plays the
    whole game. That is a mixture over behavioral strategies, not a
    per-decision average.

Under perfect recall a mixture over behavioral strategies is realization
equivalent (Kuhn) to the single behavioral strategy that weights each snapshot
by the acting player's OWN reach to the infoset:

      b_served(I)[a] = sum_{s<=t} w_s * pi_i^{sigma_s}(I) * sigma_s(I)[a]
                       / sum_{s<=t} w_s * pi_i^{sigma_s}(I)

Realization equivalence preserves the distribution over terminal histories
against any opponent, so it preserves both best-response values and the
on-policy value: exploitability(b_served) IS the exploitability of the sampled
mixture the wrapper plays. No sampling is used anywhere in this script; both
numbers are exact on the enumerated tree.

The tiny tree is perfect-recall keyed by construction (tools/tiny_solver
build_tree(perfect_recall=True): pkey = ("PR", priv_init, priv_draw, pub_path)),
which is what makes the own-reach of an infoset well defined. The prepare stage
proves it per run rather than assuming it: every node sharing a pkey must agree
on (parent own infoset, parent action slot), and no pkey may be acted on by both
players. A violation aborts.

Scoring under the run's own pinned commit
-----------------------------------------
Each X2R run pins a commit in its ``jobspec.json``. The tokenizer changed after
those launches (cambia-528/529 F1/F2 -> TOKENIZER_VERSION 3, vocab 325 -> 327),
so scoring a pre-F1 checkpoint with current code produces a different token
stream and a different, wrong NashConv (measured: iteration-1 C-rep reads
1.582023 on current master vs the recorded 1.417861). This script therefore
materializes each run's pinned commit with ``git archive`` into a cache
directory and imports ``src`` / ``tools`` from THERE, per the X2R5 procedural
pin (cambia-615). The run directory itself is only ever read.

The gate reproduction check is the guard on that: the recomputed gate NashConv
at each checkpoint is compared against the number recorded in the run's own
``resume_state.json`` controller history.

Throughput
----------
CPU-only. The dominant cost is one full-tree forward per snapshot (69636
infosets) for every snapshot in 1..t, and the SD-CFR average needs all of them.
Two things make that tractable:

  - Trie-shared GRU encoding. The 69636 perfect-recall token streams share
    prefixes: 3,964,608 total tokens collapse to 498,547 distinct prefixes
    (7.95x). The encoder walks the prefix trie depth by depth, one GRU call per
    depth with the parent hidden states gathered, so each distinct prefix is
    encoded once. Identical recurrence, hence identical numbers up to float32
    reassociation; ``--verify-forward`` checks it against the gate's own
    ``strategy_from_tokens`` path.
  - Sharding by snapshot range across single-threaded worker processes. The
    CPU GRU scales badly with intra-op threads (measured on this box: 150
    core-seconds per full-tree forward at 1 thread, 515 at 8), so N single-
    threaded workers beat one N-threaded worker by ~3x.

Shard boundaries are aligned to the requested checkpoints so a checkpoint's
accumulators are an exact sum of whole shards.

Usage (from cfr/, with PYTHONPATH=$PWD):
    python scripts/gate_vs_served_gap.py run \\
        --run-dir /home/jasonyu/dev/cambia/runs/v0.4-x2r-crep-xpu \\
        --checkpoints 350,700,1000 --workers 16 --work-dir /tmp/x2gap
    python scripts/gate_vs_served_gap.py report --work-dir /tmp/x2gap

Stages (``run`` drives them; each is also callable on its own):
    prepare  build the tree under the pinned commit, extract the infoset token
             trie + legal-action layout + own-reach parent structure
    shard    fold one contiguous snapshot range into the two accumulators
    score    combine shards per checkpoint, materialize both policies, run the
             exact NashConv on each
    report   render the collected results table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default checkpoints: the A3 read-out window (700) plus the mid-run and
# full-horizon reads the X2R record quotes.
DEFAULT_CHECKPOINTS = (350, 700, 1000)

# Snapshot filename pattern, mirrored from prtcfr_eval._SNAPSHOT_RE (this script
# never imports that module at driver level, only inside a pinned-source stage).
SNAPSHOT_GLOB_RE = r"prtcfr_snapshot_iter_(\d+)\.pt$"


# ===========================================================================
# Pure reconstruction math (no pinned-source imports; unit tested directly)
# ===========================================================================


def own_reaches(
    strat_flat: np.ndarray,
    legal_off: np.ndarray,
    parent_iset: np.ndarray,
    parent_slot: np.ndarray,
    depth_order: np.ndarray,
    depth_start: np.ndarray,
) -> np.ndarray:
    """Own-reach probability of every infoset under ONE snapshot's strategy.

    ``pi_i(I) = prod over the acting player's own ancestor decisions of
    sigma(ancestor)[action taken to get here]``: chance and the opponent are
    excluded, which is what realization equivalence weights by.

    Args:
        strat_flat: (total_legal,) per-infoset strategy vectors concatenated in
            infoset order, each slice already a distribution over that
            infoset's legal actions.
        legal_off: (N+1,) offsets into ``strat_flat``.
        parent_iset: (N,) index of the acting player's previous own decision
            infoset, -1 at an own-root.
        parent_slot: (N,) which legal-action slot of that parent leads here.
        depth_order: (N,) infoset indices sorted by own-decision depth.
        depth_start: (D+1,) boundaries of each depth block in ``depth_order``.

    Returns (N,) float64 reaches.
    """
    n = legal_off.shape[0] - 1
    pi = np.empty(n, dtype=np.float64)
    for d in range(depth_start.shape[0] - 1):
        idx = depth_order[depth_start[d] : depth_start[d + 1]]
        if idx.size == 0:
            continue
        if d == 0:
            pi[idx] = 1.0
            continue
        par = parent_iset[idx]
        pi[idx] = pi[par] * strat_flat[legal_off[par] + parent_slot[idx]]
    return pi


def finalize_gate(
    acc_gate: np.ndarray, weight_sum: float, legal_off: np.ndarray
) -> np.ndarray:
    """Gate policy: the reach-unweighted linear-iteration mean, renormalized.

    Mirrors ``prtcfr_eval.IncrementalPolicyAccumulator.policy``: divide the
    weighted sum by the total weight, then renormalize each infoset's slice
    (uniform where the slice sums to ~0).
    """
    out = np.array(acc_gate, dtype=np.float64, copy=True)
    if weight_sum > 0.0:
        out /= weight_sum
    _renormalize_slices(out, legal_off, weight_sum > 0.0)
    return out


def finalize_served(
    acc_served: np.ndarray, reach_weight: np.ndarray, legal_off: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Served policy: the own-reach-weighted mixture average, renormalized.

    ``reach_weight[i] = sum_s w_s * pi_s(I_i)`` is the realization weight of
    infoset ``i``. Where it is zero the infoset is unreachable under every
    snapshot, so the mixture's behavior there is unconstrained and cannot
    affect any best-response value, the on-policy value, or the terminal
    distribution; those slices fall back to uniform. The returned boolean array
    flags them so the caller can report how many there were.
    """
    out = np.array(acc_served, dtype=np.float64, copy=True)
    n = legal_off.shape[0] - 1
    unreached = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = int(legal_off[i]), int(legal_off[i + 1])
        w = float(reach_weight[i])
        if w > 0.0:
            out[lo:hi] /= w
            s = out[lo:hi].sum()
            if s > 1e-12:
                out[lo:hi] /= s
            else:  # pragma: no cover - defensive, w>0 implies s>0 analytically
                out[lo:hi] = 1.0 / (hi - lo)
        else:
            unreached[i] = True
            out[lo:hi] = 1.0 / (hi - lo)
    return out, unreached


def _renormalize_slices(flat: np.ndarray, legal_off: np.ndarray, have_weight: bool):
    n = legal_off.shape[0] - 1
    for i in range(n):
        lo, hi = int(legal_off[i]), int(legal_off[i + 1])
        if not have_weight:
            flat[lo:hi] = 1.0 / (hi - lo)
            continue
        s = flat[lo:hi].sum()
        if s > 1e-12:
            flat[lo:hi] /= s
        else:
            flat[lo:hi] = 1.0 / (hi - lo)


def accumulate_snapshot(
    acc_gate: np.ndarray,
    acc_served: np.ndarray,
    reach_weight: np.ndarray,
    strat_flat: np.ndarray,
    weight: float,
    legal_off: np.ndarray,
    parent_iset: np.ndarray,
    parent_slot: np.ndarray,
    depth_order: np.ndarray,
    depth_start: np.ndarray,
    counts: np.ndarray,
) -> float:
    """Fold one snapshot into both accumulators; returns its weight.

    ``counts`` is (N,) the per-infoset legal-action count, used to expand the
    per-infoset reach to the flat slice layout.
    """
    acc_gate += weight * strat_flat
    pi = own_reaches(
        strat_flat, legal_off, parent_iset, parent_slot, depth_order, depth_start
    )
    acc_served += (weight * np.repeat(pi, counts)) * strat_flat
    reach_weight += weight * pi
    return weight


# ===========================================================================
# Pinned-source materialization
# ===========================================================================


def _atomic_savez(out: Path, compressed: bool = False, **arrays) -> None:
    """Write an .npz that is either fully present or absent after a crash.

    The driver's resume path treats an existing shard as computed, so a
    half-written file from a killed worker (or a host restart: this box has
    already lost one) would silently poison a checkpoint's accumulator. Writing
    to a sibling temp file, fsyncing, then renaming makes the final path appear
    only once the bytes are on disk.
    """
    tmp = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        if compressed:
            np.savez_compressed(fh, **arrays)
        else:
            np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


def _atomic_write_text(out: Path, text: str) -> None:
    tmp = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


def _is_complete(payload: Path, sidecar_suffix: str) -> bool:
    """A stage's output counts as done only with its sidecar alongside it.

    The sidecar (``.meta.json`` for the tree, ``.info.json`` for a shard) is
    written strictly after the payload, so its presence is the completion
    marker; the payload alone can be a rename that landed before the process
    died.
    """
    return payload.is_file() and Path(str(payload) + sidecar_suffix).is_file()


def _sweep_partials(directory: Path) -> int:
    """Delete leftover ``*.partial`` temp files from an interrupted stage."""
    removed = 0
    if directory.is_dir():
        for stale in directory.glob("*.partial"):
            stale.unlink()
            removed += 1
    return removed


def repo_root_of(path: Path) -> Path:
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def materialize_pinned_source(repo: Path, commit: str, cache_root: Path) -> Path:
    """Extract ``cfr/`` at ``commit`` into ``cache_root/<commit>/cfr``.

    The run's own code is the only correct scorer for its checkpoints: the
    tokenizer's vocabulary and observation frames changed after these runs were
    launched. Idempotent; a ``.complete`` marker guards partial extractions.
    """
    dest = cache_root / commit
    marker = dest / ".complete"
    if marker.is_file():
        return dest / "cfr"
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "archive", commit, "cfr"], stdout=subprocess.PIPE
    )
    tar = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=proc.stdout)
    proc.stdout.close()
    tar.communicate()
    if tar.returncode != 0 or proc.wait() != 0:
        raise RuntimeError(f"git archive {commit} failed")
    marker.write_text(commit + "\n", encoding="utf-8")
    return dest / "cfr"


def read_run_meta(run_dir: Path) -> Dict[str, Any]:
    """Commit, config path, recorded gate series and snapshot inventory."""
    jobspec = json.loads((run_dir / "jobspec.json").read_text(encoding="utf-8"))
    resume = json.loads((run_dir / "resume_state.json").read_text(encoding="utf-8"))
    hist_raw = resume.get("controller", {}).get("history", []) or []
    recorded: Dict[int, float] = {}
    for entry in hist_raw:
        if isinstance(entry, dict):
            recorded[int(entry["iteration"])] = float(entry["metric"])
        else:
            recorded[int(entry[0])] = float(entry[1])
    import re

    pat = re.compile(SNAPSHOT_GLOB_RE)
    iters = sorted(
        int(m.group(1))
        for m in (pat.search(name) for name in os.listdir(run_dir / "snapshots"))
        if m
    )
    return {
        "run_name": run_dir.name,
        "commit": jobspec["commit"],
        "source_config": jobspec.get("config"),
        "recorded_gate": recorded,
        "snapshot_iters": iters,
    }


# ===========================================================================
# Stage: prepare
# ===========================================================================


def _install_pinned(src_root: str) -> None:
    """Put a pinned ``cfr/`` checkout at the head of sys.path and assert it wins.

    The editable install in the shared venv resolves ``src`` through a
    MetaPathFinder, so an unchecked import can silently pull the CURRENT source
    tree instead of the pinned one and score a checkpoint against a tokenizer it
    was never trained on.
    """
    sys.path.insert(0, src_root)
    import src  # noqa: F401  (import for the resolution check only)
    import tools  # noqa: F401

    got = os.path.realpath(src.__file__)
    want = os.path.realpath(src_root)
    if not got.startswith(want):
        raise RuntimeError(
            f"pinned-source import check failed: src resolved to {got}, expected "
            f"a path under {want}. Refusing to score against unpinned code."
        )


def _tree_and_infosets(cfg_path: str, seq_cap: int):
    from src.cfr.prtcfr_eval import build_tiny_tree, enumerate_infosets

    root, _isets, n_nodes, aborted = build_tiny_tree(cfg_path, seq_cap=seq_cap)
    nodes = enumerate_infosets(root)
    return root, nodes, n_nodes, aborted


def _extract_parent_structure(root, index_of_pkey: Dict[Any, int], n: int):
    """(parent infoset, parent action slot, own depth) per infoset, proved consistent.

    Walks every history. At each decision node the acting player's previous own
    decision along that path is the candidate parent. Perfect recall says every
    history in an infoset agrees on it; a disagreement means the tree is not
    perfect-recall keyed and the own-reach of an infoset is not well defined, so
    the realization-equivalence argument would not hold. That aborts.
    """
    parent_iset = np.full(n, -1, dtype=np.int64)
    parent_slot = np.full(n, -1, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    # stack entries: (node, last_own) with last_own a 2-tuple of (iset, slot) or None
    stack: List[Tuple[Any, Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]]]
    stack = [(root, (None, None))]
    while stack:
        node, last_own = stack.pop()
        kind = node.kind
        if kind == "T":
            continue
        if kind == "C":
            for child in node.children:
                stack.append((child, last_own))
            continue
        p = node.player
        i = index_of_pkey[node.pkey]
        cand = last_own[p]
        cand_iset = -1 if cand is None else cand[0]
        cand_slot = -1 if cand is None else cand[1]
        if seen[i]:
            if owner[i] != p:
                raise RuntimeError(
                    f"infoset {i} is acted on by both players; a single merged "
                    f"policy dict would conflate them"
                )
            if parent_iset[i] != cand_iset or parent_slot[i] != cand_slot:
                raise RuntimeError(
                    f"infoset {i} has two distinct own-action predecessors "
                    f"({parent_iset[i]},{parent_slot[i]}) vs ({cand_iset},{cand_slot}); "
                    f"the tree is not perfect-recall keyed and own-reach is undefined"
                )
        else:
            seen[i] = True
            owner[i] = p
            parent_iset[i] = cand_iset
            parent_slot[i] = cand_slot
        for k, child in enumerate(node.children):
            new_last = list(last_own)
            new_last[p] = (i, k)
            stack.append((child, (new_last[0], new_last[1])))
    if not seen.all():
        raise RuntimeError("some enumerated infoset was never visited in the walk")
    depth = np.full(n, -1, dtype=np.int64)
    frontier = np.flatnonzero(parent_iset < 0)
    depth[frontier] = 0
    d = 0
    remaining = n - frontier.size
    while remaining > 0:
        nxt = np.flatnonzero((depth < 0) & (depth[parent_iset] == d))
        if nxt.size == 0:
            raise RuntimeError("own-decision parent graph has a cycle or a gap")
        depth[nxt] = d + 1
        remaining -= nxt.size
        d += 1
    return parent_iset, parent_slot, owner, depth


def _build_token_trie(tok_flat: np.ndarray, tok_off: np.ndarray, n: int):
    """Prefix trie over the infoset token streams, laid out depth by depth.

    Returns (node_token, node_parent, depth_start, row_end):
      node_token[k]   the token consumed entering trie node k
      node_parent[k]  the trie node one token shorter (-1 at depth 0)
      depth_start     (D+1,) node-id boundaries; nodes are numbered in depth order
      row_end[i]      the trie node holding infoset i's LAST token
    """
    node_token: List[int] = []
    node_parent: List[int] = []
    depth_start = [0]
    row_end = np.full(n, -1, dtype=np.int64)
    lengths = np.diff(tok_off)
    active = np.flatnonzero(lengths > 0)
    cur_node = np.full(n, -1, dtype=np.int64)
    d = 0
    while active.size:
        level: Dict[Tuple[int, int], int] = {}
        for r in active:
            tok = int(tok_flat[tok_off[r] + d])
            key = (int(cur_node[r]), tok)
            nid = level.get(key)
            if nid is None:
                nid = len(node_token)
                level[key] = nid
                node_token.append(tok)
                node_parent.append(key[0])
            cur_node[r] = nid
        depth_start.append(len(node_token))
        done = active[lengths[active] == d + 1]
        row_end[done] = cur_node[done]
        active = active[lengths[active] > d + 1]
        d += 1
    if (row_end < 0).any():
        raise RuntimeError("an infoset has an empty token stream")
    return (
        np.asarray(node_token, dtype=np.int64),
        np.asarray(node_parent, dtype=np.int64),
        np.asarray(depth_start, dtype=np.int64),
        row_end,
    )


def stage_prepare(args) -> None:
    t0 = time.time()
    _install_pinned(args.src_root)
    from src.encoding import NUM_ACTIONS, action_to_index, encode_action_mask
    from src.sequence_encoding import PAD_ID

    root, nodes, n_nodes, aborted = _tree_and_infosets(args.config, args.seq_cap)
    n = len(nodes)
    index_of_pkey = {nd.pkey: i for i, nd in enumerate(nodes)}
    if len(index_of_pkey) != n:
        raise RuntimeError("enumerate_infosets returned duplicate pkeys")

    counts = np.asarray([len(nd.actions) for nd in nodes], dtype=np.int64)
    legal_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=legal_off[1:])
    legal_flat = np.empty(int(legal_off[-1]), dtype=np.int64)
    mask_rows = np.zeros((n, NUM_ACTIONS), dtype=bool)
    tok_lists = []
    for i, nd in enumerate(nodes):
        lo = int(legal_off[i])
        for k, a in enumerate(nd.actions):
            legal_flat[lo + k] = action_to_index(a)
        mask_rows[i] = encode_action_mask(nd.actions)
        toks = list(nd.seq_tokens)
        if len(toks) > args.seq_cap:  # keep-most-recent, matching _pad_tokens
            toks = toks[-args.seq_cap :]
        tok_lists.append(toks)

    tok_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(t) for t in tok_lists], out=tok_off[1:])
    tok_flat = np.fromiter(
        (t for row in tok_lists for t in row), dtype=np.int64, count=int(tok_off[-1])
    )
    if tok_flat.size and int(tok_flat.min()) <= PAD_ID:
        raise RuntimeError(
            f"a real token equals PAD_ID ({PAD_ID}); the packed-length shortcut "
            f"this script and the gate scorer both rely on is invalid"
        )

    parent_iset, parent_slot, owner, depth = _extract_parent_structure(
        root, index_of_pkey, n
    )
    depth_order = np.argsort(depth, kind="stable")
    depth_start = np.zeros(int(depth.max()) + 2, dtype=np.int64)
    np.cumsum(np.bincount(depth, minlength=int(depth.max()) + 1), out=depth_start[1:])

    node_token, node_parent, trie_depth_start, row_end = _build_token_trie(
        tok_flat, tok_off, n
    )

    pkey_digest = hashlib.sha256()
    for nd in nodes:
        pkey_digest.update(repr(nd.pkey).encode("utf-8"))
        pkey_digest.update(b"\x00")

    out = Path(args.out)
    _atomic_savez(
        out,
        compressed=True,
        counts=counts,
        legal_off=legal_off,
        legal_flat=legal_flat,
        mask_rows=np.packbits(mask_rows, axis=1),
        tok_flat=tok_flat,
        tok_off=tok_off,
        parent_iset=parent_iset,
        parent_slot=parent_slot,
        owner=owner,
        depth=depth,
        depth_order=depth_order,
        depth_start=depth_start,
        node_token=node_token,
        node_parent=node_parent,
        trie_depth_start=trie_depth_start,
        row_end=row_end,
    )
    meta = {
        "num_infosets": n,
        "num_nodes": n_nodes,
        "aborted_deals": aborted,
        "num_actions": int(NUM_ACTIONS),
        "total_legal": int(legal_off[-1]),
        "total_tokens": int(tok_off[-1]),
        "trie_nodes": int(node_token.size),
        "trie_reduction": float(int(tok_off[-1]) / max(1, int(node_token.size))),
        "max_own_depth": int(depth.max()),
        "pkey_sha256": pkey_digest.hexdigest(),
        "seq_cap": args.seq_cap,
        "config": args.config,
        "src_root": args.src_root,
        "player_split": np.bincount(owner, minlength=2).tolist(),
        "prepare_seconds": round(time.time() - t0, 2),
    }
    # Sidecar last: its presence is what the resume path reads as "tree done".
    _atomic_write_text(Path(str(out) + ".meta.json"), json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


# ===========================================================================
# Stage: shard
# ===========================================================================


class TrieEncoder:
    """Encode every infoset's token stream by walking the shared prefix trie.

    One GRU call per trie depth, with each level's initial hidden state gathered
    from its parents at the previous level. This is the identical recurrence the
    gate's ``strategy_from_tokens`` runs (``PRTCFRNet._embed_pack_gru`` packs the
    padded rows and takes the top-layer hidden AT THE LAST REAL TOKEN), just with
    each distinct prefix encoded once instead of once per infoset that shares it.
    Sequences are fed one token at a time with all-real tokens, so no packing is
    needed and none is done.
    """

    def __init__(self, arrays: Dict[str, np.ndarray], torch_mod):
        self.torch = torch_mod
        t = torch_mod
        self.node_token = t.as_tensor(arrays["node_token"], dtype=t.long)
        self.node_parent = t.as_tensor(arrays["node_parent"], dtype=t.long)
        self.depth_start = arrays["trie_depth_start"]
        row_end = arrays["row_end"]
        # Per depth: which infoset rows terminate there, and where in that
        # depth's node block. Precomputed once so the per-snapshot loop is pure
        # arithmetic.
        self.terminals: List[Tuple[Any, Any]] = []
        for d in range(self.depth_start.shape[0] - 1):
            lo, hi = int(self.depth_start[d]), int(self.depth_start[d + 1])
            rows = np.flatnonzero((row_end >= lo) & (row_end < hi))
            self.terminals.append(
                (
                    t.as_tensor(rows, dtype=t.long),
                    t.as_tensor(row_end[rows] - lo, dtype=t.long),
                )
            )

    def encode(self, net) -> Any:
        """Top-layer raw (pre-LayerNorm) hidden state per infoset: (N, hidden)."""
        t = self.torch
        embed = net.encoder["embed"]
        gru = net.encoder["gru"]
        n_rows = sum(int(rows.numel()) for rows, _ in self.terminals)
        h_top = t.empty((n_rows, net.hidden_dim), dtype=t.float32)
        h_prev = None
        prev_lo = 0
        with t.no_grad():
            for d in range(self.depth_start.shape[0] - 1):
                lo, hi = int(self.depth_start[d]), int(self.depth_start[d + 1])
                toks = self.node_token[lo:hi].unsqueeze(1)  # (B, 1)
                emb = embed(toks)
                if d == 0:
                    h0 = None
                else:
                    par_local = self.node_parent[lo:hi] - prev_lo
                    h0 = h_prev.index_select(1, par_local).contiguous()
                _out, h_new = gru(emb, h0)
                rows, local = self.terminals[d]
                if rows.numel():
                    h_top[rows] = h_new[-1].index_select(0, local)
                h_prev = h_new
                prev_lo = lo
        return h_top


def _strategies_from_hidden(net, h_top, mask_bits, n, num_actions, legal_flat,
                            legal_off, chunk, torch_mod) -> np.ndarray:
    """Regret-matched strategy per infoset, projected to its legal-action slice.

    ``strategy_from_hidden`` is the net's own read-time path (LayerNorm -> head
    -> regret matching), the same one ``strategy_from_tokens`` ends in.
    """
    t = torch_mod

    out = np.empty(int(legal_off[-1]), dtype=np.float64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        mask_np = np.unpackbits(mask_bits[lo:hi], axis=1, count=num_actions).astype(bool)
        mask_t = t.as_tensor(mask_np)
        with t.no_grad():
            strat = net.strategy_from_hidden(h_top[lo:hi], mask_t)
        strat_np = strat.detach().to("cpu", dtype=t.float64).numpy()
        flat_lo, flat_hi = int(legal_off[lo]), int(legal_off[hi])
        rows = np.repeat(
            np.arange(hi - lo, dtype=np.int64), np.diff(legal_off[lo : hi + 1])
        )
        out[flat_lo:flat_hi] = strat_np[rows, legal_flat[flat_lo:flat_hi]]
    return out


def _reference_strategies(net, arrays, n, num_actions, chunk, torch_mod, rows_idx):
    """The gate's own forward path on a subset of rows, for cross-validation."""
    t = torch_mod
    tok_flat, tok_off = arrays["tok_flat"], arrays["tok_off"]
    width = int(np.diff(tok_off).max())
    toks = np.zeros((rows_idx.size, width), dtype=np.int64)
    for j, i in enumerate(rows_idx):
        lo, hi = int(tok_off[i]), int(tok_off[i + 1])
        toks[j, : hi - lo] = tok_flat[lo:hi]
    mask_np = np.unpackbits(
        arrays["mask_rows"][rows_idx], axis=1, count=num_actions
    ).astype(bool)
    with t.no_grad():
        strat = net.strategy_from_tokens(t.as_tensor(toks), t.as_tensor(mask_np))
    return strat.detach().to("cpu", dtype=t.float64).numpy()


def stage_shard(args) -> None:
    t0 = time.time()
    _install_pinned(args.src_root)
    import torch

    torch.set_num_threads(args.threads)
    from src.cfr.prtcfr_eval import _load_net
    from src.encoding import NUM_ACTIONS

    z = np.load(args.tree, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    counts = arrays["counts"]
    legal_off = arrays["legal_off"]
    legal_flat = arrays["legal_flat"]
    n = counts.shape[0]
    total_legal = int(legal_off[-1])

    encoder = TrieEncoder(arrays, torch)
    acc_gate = np.zeros(total_legal, dtype=np.float64)
    acc_served = np.zeros(total_legal, dtype=np.float64)
    reach_weight = np.zeros(n, dtype=np.float64)
    weight_sum = 0.0

    snapdir = Path(args.snapshot_dir)
    verify: Optional[Dict[str, float]] = None
    for it in range(args.lo, args.hi + 1):
        fp = snapdir / f"prtcfr_snapshot_iter_{it}.pt"
        net = _load_net(str(fp), device="cpu")
        h_top = encoder.encode(net)
        strat_flat = _strategies_from_hidden(
            net, h_top, arrays["mask_rows"], n, NUM_ACTIONS, legal_flat, legal_off,
            args.chunk, torch,
        )
        if args.verify_forward and verify is None:
            rng = np.random.default_rng(0)
            rows = rng.choice(n, size=min(args.verify_forward, n), replace=False)
            rows.sort()
            ref = _reference_strategies(net, arrays, n, NUM_ACTIONS, args.chunk, torch, rows)
            diffs = []
            for j, i in enumerate(rows):
                lo, hi = int(legal_off[i]), int(legal_off[i + 1])
                diffs.append(
                    np.max(np.abs(strat_flat[lo:hi] - ref[j, legal_flat[lo:hi]]))
                )
            verify = {
                "rows": int(rows.size),
                "max_abs_diff": float(np.max(diffs)),
                "iteration": it,
            }
        weight_sum += accumulate_snapshot(
            acc_gate, acc_served, reach_weight, strat_flat, float(it), legal_off,
            arrays["parent_iset"], arrays["parent_slot"], arrays["depth_order"],
            arrays["depth_start"], counts,
        )
        del net

    out = Path(args.out)
    _atomic_savez(
        out,
        acc_gate=acc_gate,
        acc_served=acc_served,
        reach_weight=reach_weight,
        weight_sum=np.float64(weight_sum),
        lo=np.int64(args.lo),
        hi=np.int64(args.hi),
    )
    info = {
        "lo": args.lo,
        "hi": args.hi,
        "snapshots": args.hi - args.lo + 1,
        "seconds": round(time.time() - t0, 2),
        "threads": args.threads,
    }
    if verify is not None:
        info["verify_forward"] = verify
    # Sidecar last: its presence is what the resume path reads as "shard done".
    _atomic_write_text(Path(str(out) + ".info.json"), json.dumps(info))
    print(json.dumps(info), flush=True)


# ===========================================================================
# Stage: score
# ===========================================================================


def stage_score(args) -> None:
    t0 = time.time()
    _install_pinned(args.src_root)
    from tools.tiny_solver import exploitability

    z = np.load(args.tree, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    meta = json.loads(Path(str(args.tree) + ".meta.json").read_text(encoding="utf-8"))
    legal_off = arrays["legal_off"]
    counts = arrays["counts"]
    n = counts.shape[0]

    root, nodes, _n_nodes, _ab = _tree_and_infosets(meta["config"], meta["seq_cap"])
    if len(nodes) != n:
        raise RuntimeError(f"tree rebuilt with {len(nodes)} infosets, expected {n}")
    digest = hashlib.sha256()
    for nd in nodes:
        digest.update(repr(nd.pkey).encode("utf-8"))
        digest.update(b"\x00")
    if digest.hexdigest() != meta["pkey_sha256"]:
        raise RuntimeError(
            "rebuilt tree's infoset order differs from the prepared one; "
            "the accumulators cannot be aligned to it"
        )
    build_seconds = time.time() - t0

    shards = []
    for path in sorted(Path(args.shard_dir).glob("shard_*.npz")):
        if not _is_complete(path, ".info.json"):
            raise RuntimeError(
                f"{path.name} has no .info.json sidecar: it is a half-written "
                f"shard. Delete it and relaunch the driver to recompute it."
            )
        s = np.load(path, allow_pickle=False)
        shards.append(
            {
                "lo": int(s["lo"]),
                "hi": int(s["hi"]),
                "acc_gate": s["acc_gate"],
                "acc_served": s["acc_served"],
                "reach_weight": s["reach_weight"],
                "weight_sum": float(s["weight_sum"]),
                "seconds": json.loads(
                    Path(str(path) + ".info.json").read_text(encoding="utf-8")
                )["seconds"],
            }
        )
    shards.sort(key=lambda d: d["lo"])
    covered = set()
    for sh in shards:
        covered.update(range(sh["lo"], sh["hi"] + 1))

    results = []
    for target in args.checkpoints:
        if not all(i in covered for i in range(1, target + 1)):
            print(f"[skip] checkpoint {target}: shards do not cover 1..{target}", flush=True)
            continue
        parts = [sh for sh in shards if sh["hi"] <= target]
        if sum(sh["hi"] - sh["lo"] + 1 for sh in parts) != target:
            raise RuntimeError(
                f"checkpoint {target} is not a shard boundary; shards must align "
                f"to the requested checkpoints"
            )
        acc_gate = np.zeros(int(legal_off[-1]), dtype=np.float64)
        acc_served = np.zeros(int(legal_off[-1]), dtype=np.float64)
        reach_weight = np.zeros(n, dtype=np.float64)
        weight_sum = 0.0
        fold_seconds = 0.0
        for sh in parts:
            acc_gate += sh["acc_gate"]
            acc_served += sh["acc_served"]
            reach_weight += sh["reach_weight"]
            weight_sum += sh["weight_sum"]
            fold_seconds += sh["seconds"]

        gate_flat = finalize_gate(acc_gate, weight_sum, legal_off)
        served_flat, unreached = finalize_served(acc_served, reach_weight, legal_off)

        gate_pol = {}
        served_pol = {}
        for i, nd in enumerate(nodes):
            lo, hi = int(legal_off[i]), int(legal_off[i + 1])
            gate_pol[nd.pkey] = gate_flat[lo:hi]
            served_pol[nd.pkey] = served_flat[lo:hi]

        t1 = time.time()
        gate_nc, gate_comp = exploitability(root, gate_pol)
        t2 = time.time()
        served_nc, served_comp = exploitability(root, served_pol)
        t3 = time.time()

        # Total variation between the two policies, reach-weighted by the served
        # mixture's own realization weight: how different the objects actually
        # are where play actually goes.
        tv = np.empty(n, dtype=np.float64)
        for i in range(n):
            lo, hi = int(legal_off[i]), int(legal_off[i + 1])
            tv[i] = 0.5 * np.abs(gate_flat[lo:hi] - served_flat[lo:hi]).sum()
        rw = reach_weight / max(weight_sum, 1e-300)

        results.append(
            {
                "iteration": target,
                "num_snapshots": target,
                "gate_nashconv": float(gate_nc),
                "served_nashconv": float(served_nc),
                "gap_abs": float(served_nc - gate_nc),
                "gap_rel": float((served_nc - gate_nc) / gate_nc) if gate_nc else None,
                "gate_components": [float(x) for x in gate_comp],
                "served_components": [float(x) for x in served_comp],
                "unreached_infosets": int(unreached.sum()),
                "mean_tv": float(tv.mean()),
                "reach_weighted_mean_tv": float(
                    (tv * rw).sum() / rw.sum() if rw.sum() > 0 else 0.0
                ),
                "max_tv": float(tv.max()),
                "forward_seconds": round(fold_seconds, 1),
                "gate_score_seconds": round(t2 - t1, 2),
                "served_score_seconds": round(t3 - t2, 2),
            }
        )
        print(
            f"[score] t={target} gate={gate_nc:.8f} served={served_nc:.8f} "
            f"gap={served_nc - gate_nc:+.8f} ({100 * (served_nc - gate_nc) / gate_nc:+.2f}%)",
            flush=True,
        )

    out = {
        "tree_meta": meta,
        "tree_build_seconds": round(build_seconds, 1),
        "results": results,
    }
    _atomic_write_text(Path(args.out), json.dumps(out, indent=2))
    print(f"[score] wrote {args.out}", flush=True)


# ===========================================================================
# Driver
# ===========================================================================


def _shard_plan(checkpoints: Sequence[int], max_iter: int, size: int) -> List[Tuple[int, int]]:
    """Contiguous snapshot ranges whose boundaries land on every checkpoint.

    Stops at the last requested checkpoint: snapshots past it are never folded
    into any accumulator, so encoding them would be pure waste.
    """
    usable = [c for c in checkpoints if c <= max_iter]
    top = min(max_iter, max(usable)) if usable else max_iter
    bounds = sorted({0, *usable, top})
    plan: List[Tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        lo = a + 1
        while lo <= b:
            hi = min(lo + size - 1, b)
            plan.append((lo, hi))
            lo = hi + 1
    return plan


def _run_stage(argv: List[str], env: Dict[str, str], cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *argv],
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stage_run(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    work = Path(args.work_dir).resolve()
    here = Path(__file__).resolve().parent.parent  # cfr/
    repo = repo_root_of(here)
    meta = read_run_meta(run_dir)
    src_root = str(materialize_pinned_source(repo, meta["commit"], work / "src-cache"))
    rundir_work = work / meta["run_name"]
    (rundir_work / "shards").mkdir(parents=True, exist_ok=True)

    max_iter = max(meta["snapshot_iters"])
    checkpoints = [c for c in args.checkpoints if c <= max_iter]
    dropped = [c for c in args.checkpoints if c > max_iter]
    print(
        f"[run] {meta['run_name']} commit={meta['commit'][:8]} snapshots=1..{max_iter} "
        f"checkpoints={checkpoints}" + (f" (dropped {dropped}: no snapshots)" if dropped else ""),
        flush=True,
    )
    (rundir_work / "run_meta.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in meta.items() if k != "snapshot_iters"},
                "recorded_gate": {str(k): v for k, v in meta["recorded_gate"].items()},
                "max_snapshot_iter": max_iter,
                "checkpoints": checkpoints,
                "src_root": src_root,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = src_root
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        base_env[var] = str(args.threads)

    # Resume: drop any temp files an interrupted stage left behind, then skip
    # every stage whose output is complete (payload + sidecar). A relaunch with
    # the same --work-dir picks up exactly where the last one stopped; --force
    # recomputes everything.
    swept = _sweep_partials(rundir_work) + _sweep_partials(rundir_work / "shards")
    if swept:
        print(f"[run] swept {swept} partial file(s) from an interrupted stage", flush=True)

    tree = rundir_work / "tree.npz"
    if not _is_complete(tree, ".meta.json") or args.force:
        cfg = args.config or str(run_dir / "config.yaml")
        proc = _run_stage(
            ["prepare", "--src-root", src_root, "--config", cfg,
             "--seq-cap", str(args.seq_cap), "--out", str(tree)],
            base_env, str(here),
        )
        _drain(proc, "prepare")
    else:
        print("[run] tree.npz present; skipping prepare", flush=True)

    plan = _shard_plan(checkpoints or [max_iter], max_iter, args.shard_size)
    pending = [
        (lo, hi)
        for lo, hi in plan
        if args.force
        or not _is_complete(
            rundir_work / "shards" / f"shard_{lo:05d}_{hi:05d}.npz", ".info.json"
        )
    ]
    print(
        f"[run] {len(plan)} shards, {len(plan) - len(pending)} already on disk, "
        f"{len(pending)} to compute, {args.workers} workers",
        flush=True,
    )
    t0 = time.time()
    running: List[Tuple[subprocess.Popen, str]] = []
    done = 0
    while pending or running:
        while pending and len(running) < args.workers:
            lo, hi = pending.pop(0)
            out = rundir_work / "shards" / f"shard_{lo:05d}_{hi:05d}.npz"
            proc = _run_stage(
                ["shard", "--src-root", src_root, "--tree", str(tree),
                 "--snapshot-dir", str(run_dir / "snapshots"),
                 "--lo", str(lo), "--hi", str(hi), "--out", str(out),
                 "--threads", str(args.threads), "--chunk", str(args.chunk)]
                + (["--verify-forward", str(args.verify_forward)] if args.verify_forward else []),
                base_env, str(here),
            )
            running.append((proc, f"shard {lo}-{hi}"))
        time.sleep(1.0)
        still = []
        for proc, label in running:
            if proc.poll() is None:
                still.append((proc, label))
                continue
            _drain(proc, label)
            done += 1
            print(
                f"[run] {done}/{len(plan)} shards done, {time.time() - t0:.0f}s elapsed",
                flush=True,
            )
        running = still

    proc = _run_stage(
        ["score", "--src-root", src_root, "--tree", str(tree),
         "--shard-dir", str(rundir_work / "shards"),
         "--checkpoints", ",".join(str(c) for c in checkpoints),
         "--out", str(rundir_work / "results.json")],
        base_env, str(here),
    )
    _drain(proc, "score")
    print(f"[run] {meta['run_name']} complete in {time.time() - t0:.0f}s", flush=True)


def _drain(proc: subprocess.Popen, label: str) -> None:
    out = proc.stdout.read() if proc.stdout else ""
    rc = proc.wait()
    for line in out.splitlines():
        print(f"  [{label}] {line}", flush=True)
    if rc != 0:
        raise RuntimeError(f"stage {label} failed with exit code {rc}")


# ===========================================================================
# Stage: report
# ===========================================================================


def stage_report(args) -> None:
    work = Path(args.work_dir).resolve()
    rows = []
    for meta_path in sorted(work.glob("*/run_meta.json")):
        rundir_work = meta_path.parent
        res_path = rundir_work / "results.json"
        if not res_path.is_file():
            continue
        rmeta = json.loads(meta_path.read_text(encoding="utf-8"))
        res = json.loads(res_path.read_text(encoding="utf-8"))
        recorded = {int(k): v for k, v in rmeta["recorded_gate"].items()}
        for r in res["results"]:
            rec = recorded.get(r["iteration"])
            rows.append(
                {
                    "run": rundir_work.name,
                    "commit": rmeta["commit"][:8],
                    **r,
                    "recorded_gate": rec,
                    "repro_abs": (r["gate_nashconv"] - rec) if rec is not None else None,
                    "repro_rel": (
                        (r["gate_nashconv"] - rec) / rec if rec else None
                    ),
                }
            )
    rows.sort(key=lambda d: (d["run"], d["iteration"]))
    bar = args.bar
    print()
    print(f"X2 gate metric vs served SD-CFR mixture, exact NashConv (bar_respec = {bar})")
    print()
    hdr = (
        f"{'run':<20} {'iter':>5} {'gate':>10} {'recorded':>10} {'repro d':>10} "
        f"{'served':>10} {'gap_abs':>10} {'gap_rel':>9} {'g<bar':>6} {'s<bar':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rec = r["recorded_gate"]
        rec_s = "n/a" if rec is None else "%.6f" % rec
        rep_s = "n/a" if r["repro_abs"] is None else "%+.2e" % r["repro_abs"]
        g_bar = "yes" if r["gate_nashconv"] < bar else "no"
        s_bar = "yes" if r["served_nashconv"] < bar else "no"
        print(
            f"{r['run']:<20} {r['iteration']:>5} {r['gate_nashconv']:>10.6f} "
            f"{rec_s:>10} {rep_s:>10} "
            f"{r['served_nashconv']:>10.6f} {r['gap_abs']:>+10.6f} "
            f"{100 * r['gap_rel']:>8.2f}% {g_bar:>6} {s_bar:>6}"
        )
    print()
    print(
        f"{'run':<20} {'iter':>5} {'mean TV':>9} {'reach-w TV':>11} {'max TV':>8} "
        f"{'unreached':>10} {'fwd s':>9} {'score s':>8}"
    )
    for r in rows:
        print(
            f"{r['run']:<20} {r['iteration']:>5} {r['mean_tv']:>9.5f} "
            f"{r['reach_weighted_mean_tv']:>11.5f} {r['max_tv']:>8.5f} "
            f"{r['unreached_infosets']:>10} {r['forward_seconds']:>9.0f} "
            f"{r['gate_score_seconds'] + r['served_score_seconds']:>8.1f}"
        )
    print()
    if args.json_out:
        _atomic_write_text(Path(args.json_out), json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================


def _csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("run", help="prepare + shard + score one run directory")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--work-dir", required=True)
    r.add_argument("--checkpoints", type=_csv_ints, default=list(DEFAULT_CHECKPOINTS))
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--shard-size", type=int, default=25)
    r.add_argument("--chunk", type=int, default=16384)
    r.add_argument("--seq-cap", type=int, default=256)
    r.add_argument("--config", default=None, help="override the run's own config.yaml")
    r.add_argument("--verify-forward", type=int, default=256,
                   help="rows to cross-check trie vs gate forward path (0 disables)")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=stage_run)

    q = sub.add_parser("prepare", help="build the tree and extract its structure")
    q.add_argument("--src-root", required=True)
    q.add_argument("--config", required=True)
    q.add_argument("--seq-cap", type=int, default=256)
    q.add_argument("--out", required=True)
    q.set_defaults(func=stage_prepare)

    s = sub.add_parser("shard", help="fold one snapshot range into the accumulators")
    s.add_argument("--src-root", required=True)
    s.add_argument("--tree", required=True)
    s.add_argument("--snapshot-dir", required=True)
    s.add_argument("--lo", type=int, required=True)
    s.add_argument("--hi", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--chunk", type=int, default=16384)
    s.add_argument("--verify-forward", type=int, default=0)
    s.set_defaults(func=stage_shard)

    c = sub.add_parser("score", help="combine shards and score both policies")
    c.add_argument("--src-root", required=True)
    c.add_argument("--tree", required=True)
    c.add_argument("--shard-dir", required=True)
    c.add_argument("--checkpoints", type=_csv_ints, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=stage_score)

    p2 = sub.add_parser("report", help="render the results table")
    p2.add_argument("--work-dir", required=True)
    p2.add_argument("--bar", type=float, default=0.057)
    p2.add_argument("--json-out", default=None)
    p2.set_defaults(func=stage_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
