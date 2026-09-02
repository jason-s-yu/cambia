"""src/cfr/prtcfr_eval.py

X2 gate scorer: exact NashConv of a trained PRT-CFR neural policy on the tiny
{A,6} 2-card plateau game.

What this proves
----------------
The X1 keystone (tools/tiny_solver.py --perfect-recall) showed that a TABULAR
policy keyed on the genuine perfect-recall information state drives the {A,6}
2-card NashConv to ~0 (5.22e-06), versus the 0.0832 plateau under the production
imperfect-recall belief abstraction. X2 asks the next question: does a NEURAL
PRT-CFR policy, trained on the same perfect-recall token stream, also reach the
~0 region? The contract bar is NashConv < 0.05 (contract.md, X2).

How it scores
-------------
The tiny game is built by ``tools.tiny_solver.build_tree`` in PERFECT-RECALL +
TOKENIZE mode. With ``tokenize=True`` the builder stores each acting player's
perfect-recall observation-action token stream on ``Decision.seq_tokens`` (via
the one tokenizer, ``src.sequence_encoding.encode_observation_sequence``). For
each distinct infoset we:

  1. Read the node's tokens via core's single-sourced helper
     ``tiny_node_to_tokens`` (src/cfr/prtcfr_net.py), which returns
     ``node.seq_tokens``. This module never tokenizes on its own: a token path
     that diverges from training is the train/eval mismatch (RC-B) that caused
     the original wall, so the seam is single-sourced through the builder + that
     helper.
  2. Realize the SD-CFR AVERAGE policy deterministically. SD-CFR serves a
     MIXTURE: ``prtcfr_mixture.sample_episode`` draws one snapshot per episode
     with probability proportional to w_t = t and plays the whole game with it.
     Under perfect recall that mixture is realization equivalent (Kuhn) to the
     single behavior strategy weighting each snapshot by the acting player's OWN
     reach of the infoset:

         strategy(I)[a] = sum_t w_t pi_i^{sigma_t}(I) sigma_t(I)[a]
                          / sum_t w_t pi_i^{sigma_t}(I),
         sigma_t(I) = regret_match(net_t(tokens, mask)),
         w_t = t   (linear weighting; matches deep_trainer sd_cfr_snapshot_weighting)

     That is the ``SERVED`` object this module scores by default (cambia-708).
     The reach-UNWEIGHTED per-decision mean ``sum_t w_t sigma_t(I) / sum_t w_t``
     is a different strategy nothing plays; it was the gate's scored object
     until cambia-708 and stays reachable as ``objective=PER_DECISION``. See
     the "two SD-CFR policy objects" section below.

     Each ``net_t`` is one snapshot; the per-net regret-matched strategy over the
     146-action space comes from ``PRTCFRNet.strategy_from_tokens(tokens, mask)``.
     A single checkpoint is the degenerate one-snapshot case (weight 1.0), where
     the two objects coincide.
  3. Materialize a plain ``{pkey: strategy_vector(nA)}`` dict aligned to the
     node's legal-action order (the 146-vector entries are read back per legal
     action via ``src.encoding.action_to_index``), then call
     ``tools.tiny_solver.exploitability``. The dict is pre-materialized so the
     solver's existing dict-lookup path (``_lookup`` -> bare ``node.pkey``)
     scores it untouched; tiny_solver's ``_lookup`` is not modified.

@chief runs the real X2 verdict at integration by pointing
``score_policy_on_tiny_game`` (or the parametrized gate test) at a trained
snapshot directory / checkpoint produced by the PRT-CFR trainer.

Core interface dependency (src/cfr/prtcfr_net.py)
-------------------------------------------------
This module imports two primitives and owns everything else (snapshot loading,
SD-CFR averaging, dict materialization, scoring):

  - ``tiny_node_to_tokens(node) -> list[int]``
        The acting player's perfect-recall token stream, stored on the node by
        ``build_tree(..., tokenize=True)``. The SAME tokens training uses.
  - ``PRTCFRNet`` (torch.nn.Module) with
        ``strategy_from_tokens(tokens: Tensor[B, L], mask: Tensor[B, 146]) -> Tensor[B, 146]``
        a regret-matched distribution over the full 146-action space (legal
        entries selected by ``mask``). Checkpoint format is
        ``{encoder_state_dict, head_state_dict, iteration}`` loaded via
        ``PRTCFRNet.load_encoder_head``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Core-owned primitives. Tolerant import: the gate test can inject a stub before
# this resolves if core has not landed yet (parallel development).
try:  # pragma: no cover - real path exercised at integration
    from src.cfr.prtcfr_net import PRTCFRNet, tiny_node_to_tokens
except Exception:  # noqa: BLE001 - core may be unwritten during parallel dev
    PRTCFRNet = None  # type: ignore[assignment]
    tiny_node_to_tokens = None  # type: ignore[assignment]

# Snapshot-shape architecture inference (cambia-341): the pinned checkpoint
# format carries no net-dims config, so a checkpoint trained with non-default
# GRU dims (e.g. a tiny-config gate run's shrunk net) must not be loaded
# against PRTCFRNet's production-default constructor args -- the shapes would
# mismatch and load_state_dict would raise. prtcfr_mixture already solved this
# for the full-game eval mixture (X4); reused here rather than re-derived.
try:  # pragma: no cover - same tolerant-import posture as PRTCFRNet above
    from src.cfr.prtcfr_mixture import _build_net_from_state
except Exception:  # noqa: BLE001
    _build_net_from_state = None  # type: ignore[assignment]

from src.config import load_config
from src.encoding import NUM_ACTIONS, action_to_index, encode_action_mask
from src.sequence_encoding import PAD_ID, SEQ_CAP, TOKENIZER_VERSION
from tools.tiny_solver import build_tree, exploitability
from tools import tiny_exact

# Tiny {A,6} 2-card plateau game: the EXACT X1 cause-isolation game.
TINY_2CARD_CONFIG = "config/tiny_2card_plateau.yaml"

# Build parameters matching the X1 keystone run (config header + X1 verdict):
# enumerated draw-chance, K=5 deals, seed0=0, perfect-recall keying. tokenize=True
# populates Decision.seq_tokens with the single-sourced token stream.
TINY_N_DEALS = 5
TINY_SEED0 = 0
TINY_MAX_NODES_PER_DEAL = 2_000_000

# X2 gate bar: neural-policy NashConv strictly below this. Re-derived for the
# corrected {A,6} tree per the frozen pre-registration (.docs/v0.4/
# phase2-throughput-pilot/x2-respec-preregistration.md, cambia-517):
# bar = 0.0340 * U where U = 1.6727709 (exact uniform-policy NashConv,
# tools/measure_uniform_nashconv.py). Supersedes the old-tree 0.05; the
# informal 0.045 was voided for provenance (cambia-516).
X2_NASHCONV_BAR = 0.057

# SD-CFR snapshot filename pattern: prtcfr_snapshot_iter_{t}.pt
_SNAPSHOT_RE = re.compile(r"prtcfr_snapshot_iter_(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Token-stream provenance gate (cambia-612)
# ---------------------------------------------------------------------------


def _recorded_tokenizer_version(checkpoint_or_snapshot_dir: str) -> Optional[int]:
    """Read ``tokenizer_version`` from the run_meta.json governing a checkpoint.

    ``checkpoint_or_snapshot_dir`` is a single .pt checkpoint, a snapshot
    directory, or a run directory. Walks up from it to the nearest ancestor
    holding a run_meta.json (the run-dir provenance record written by
    run_db.write_run_meta_json) and returns its integer ``tokenizer_version``.

    Returns None when no run_meta.json is found, or it carries no
    ``tokenizer_version`` -- the case for every run created before cambia-612
    stamped it. None means "unknown provenance", handled loudly (not silently)
    by ``_resolve_scoring_obs_path``.
    """
    p = Path(checkpoint_or_snapshot_dir)
    start = p if p.is_dir() else p.parent
    for d in (start, *start.parents):
        meta_path = d / "run_meta.json"
        if meta_path.is_file():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                return None
            v = meta.get("tokenizer_version")
            return int(v) if v is not None else None
    return None


def _resolve_scoring_obs_path(recorded_version: Optional[int]) -> bool:
    """Validate a checkpoint's recorded tokenizer version against the live
    tokenizer and return the observation path to score it under.

    Returns ``production_obs``: True to build the scorer tree through the
    production worker observation path (post-draw drawn frame + peek-result
    frames, cambia-528/529), which is what training tokens carry from v2 on;
    False for the legacy analysis_tools BR path (matches the pre-F1 v1 stream,
    which surfaced the drawn card at draw time and dropped peeked cards).

    Version handling:
      - recorded == live TOKENIZER_VERSION: score on the matching path
        (production_obs = live >= 2).
      - recorded != live: hard error. A net trained on one token stream scored
        on another produces silently-wrong NashConv -- the RC-B train/eval
        mismatch this gate exists to prevent.
      - recorded is None (every run predating cambia-612's stamp carries none):
        unknown provenance. Emit a loud warning and proceed on the legacy path;
        the pre-F1 pin is handled procedurally (X2R5, note cambia-615).
    """
    live = TOKENIZER_VERSION
    if recorded_version is None:
        warnings.warn(
            "TOKENIZER-VERSION PROVENANCE UNKNOWN: this checkpoint's run carries no "
            f"recorded tokenizer_version (live tokenizer is v{live}). Scoring "
            "proceeds on the LEGACY observation path; the resulting NashConv is "
            "trustworthy ONLY if the net was trained under the pre-F1 (v1) "
            "tokenizer. Confirm the training-era tokenizer before acting on this "
            "number (cambia-612; pre-F1 pin handled procedurally per X2R5/cambia-615).",
            stacklevel=2,
        )
        return False
    if recorded_version != live:
        raise ValueError(
            f"TOKENIZER-VERSION MISMATCH: this checkpoint was trained under tokenizer "
            f"v{recorded_version} but the live tokenizer is v{live}. A policy trained on one "
            f"token stream scored on another yields silently-wrong NashConv. Score "
            f"this checkpoint with the training-era (v{recorded_version}) code, or retrain "
            f"under v{live}. Refusing to score (cambia-612)."
        )
    return live >= 2


# ---------------------------------------------------------------------------
# Tiny-game construction + infoset enumeration
# ---------------------------------------------------------------------------


def build_tiny_tree(
    config_path: str = TINY_2CARD_CONFIG,
    seq_cap: int = SEQ_CAP,
    exact_weights: bool = False,
    production_obs: Optional[bool] = None,
    backend: str = "go",
):
    """Build the perfect-recall + tokenized {A,6} tiny tree.

    Returns (root, isets, n, aborted). Reuses tools.tiny_solver.build_tree with
    ``perfect_recall=True, tokenize=True`` so the tree, its infoset partition, and
    the per-node token streams are identical to what the PRT-CFR worker trains on.

    exact_weights (default off): also attach exact-rational chance mass
    (Chance.wfrac) for the NashConv certifier (tools/tiny_exact.py, cambia-530).
    The float ``weights`` used by the fast-path scorer are unchanged.

    production_obs (cambia-612, default None = unspecified): build each node's
    token stream
    through the PRODUCTION worker observation path (peek-result + post-draw drawn
    frames) instead of the analysis_tools BR path. The scoring entry points set
    this from the checkpoint's recorded tokenizer version (>= 2 -> True) so the
    scorer tokens match what the net was trained on; the default preserves every
    legacy caller's behavior byte-for-byte.
    """
    cfg = load_config(config_path)
    root, isets, nnodes, aborted = build_tree(
        cfg,
        n_deals=TINY_N_DEALS,
        seed0=TINY_SEED0,
        max_nodes_per_deal=TINY_MAX_NODES_PER_DEAL,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=True,
        seq_cap=seq_cap,
        exact_weights=exact_weights,
        production_obs=production_obs,
        backend=backend,
    )
    if aborted:
        raise RuntimeError(
            f"tiny tree truncated ({aborted} deals hit the node cap); "
            f"raise TINY_MAX_NODES_PER_DEAL"
        )
    return root, isets, nnodes, aborted


def enumerate_infosets(root) -> List[Any]:
    """Collect one representative Decision node per distinct perfect-recall pkey.

    Perfect-recall keying makes ``node.pkey`` determine both the legal-action
    count and the legal-action SET (proved in tests/test_tiny_solver_perfect_recall.py),
    so every node sharing a pkey has identical ``.actions`` repr-order. We keep
    the first representative per pkey and assert consistency on the rest, then
    realize one strategy vector per infoset.
    """
    reps: Dict[Any, Any] = {}
    stack = [root]
    while stack:
        nd = stack.pop()
        kind = nd.kind
        if kind == "T":
            continue
        if kind == "C":
            stack.extend(nd.children)
            continue
        # decision node
        prev = reps.get(nd.pkey)
        if prev is None:
            reps[nd.pkey] = nd
        else:
            if [repr(a) for a in prev.actions] != [repr(a) for a in nd.actions]:
                raise AssertionError(
                    f"perfect-recall pkey {nd.pkey!r} maps to two legal sets; "
                    f"tiny_solver PR keying broken"
                )
        stack.extend(nd.children)
    return list(reps.values())


# ---------------------------------------------------------------------------
# Snapshot loading + SD-CFR averaging
# ---------------------------------------------------------------------------


def discover_snapshots(path: str) -> List[Tuple[int, str]]:
    """Resolve a checkpoint/snapshot location into [(iter, filepath), ...] sorted.

    Accepts:
      - a directory containing prtcfr_snapshot_iter_{t}.pt files (SD-CFR mode):
        returns every snapshot, sorted by iteration.
      - a single .pt file (checkpoint mode): returns [(iter_or_1, file)]. The
        iteration is parsed from the filename if it matches the snapshot pattern,
        else 1 (a single-snapshot average is weight-invariant).
    """
    if os.path.isdir(path):
        out: List[Tuple[int, str]] = []
        for name in os.listdir(path):
            m = _SNAPSHOT_RE.search(name)
            if m:
                out.append((int(m.group(1)), os.path.join(path, name)))
        if not out:
            raise FileNotFoundError(
                f"no prtcfr_snapshot_iter_*.pt files in directory {path!r}"
            )
        out.sort(key=lambda t: t[0])
        return out
    if os.path.isfile(path):
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        return [(int(m.group(1)) if m else 1, path)]
    raise FileNotFoundError(f"checkpoint/snapshot path not found: {path!r}")


def _load_net(filepath: str, device: str = "cpu") -> Any:
    """Load one PRTCFRNet snapshot from the pinned checkpoint format.

    The only accepted format is ``{encoder_state_dict, head_state_dict,
    iteration}`` (prtcfr_net docstring), loaded via
    ``PRTCFRNet.load_encoder_head``. This is the only place that touches
    checkpoint internals.

    Loading is hardened against poisoned pickles rsync-written under ``runs/``
    (cambia-552): ``weights_only=True`` refuses any pickle that would execute
    code on load (e.g. a whole ``nn.Module`` via ``__reduce__``), so only plain
    tensor/state-dict payloads deserialize. Every writer in the repo emits
    exactly the pinned shape (prtcfr_trainer ``_save_snapshot`` /
    ``_save_checkpoint``; prtcfr_mixture already loads it under
    ``weights_only=True``); a file that does not match is rejected, not coerced.

    Net dims (embed/hidden/layers/head-hidden) are read back from the loaded
    tensor shapes (cambia-341), not the module's production defaults: a
    checkpoint trained with non-default GRU dims (e.g. a tiny-gate config's
    shrunk net) would otherwise mismatch PRTCFRNet's default-arg constructor
    and ``load_state_dict``/``load_encoder_head`` would raise. This makes
    ``_load_net`` correct for any net width the checkpoint was actually
    trained at, independent of any config file.
    """
    import torch

    if PRTCFRNet is None:
        raise RuntimeError(
            "src.cfr.prtcfr_net.PRTCFRNet unavailable. Core (prtcfr-core) has not "
            "landed; the gate test injects a stub for plumbing runs."
        )
    obj = torch.load(filepath, map_location=device, weights_only=True)

    # Pinned format only: split encoder/head state dicts. The whole-module and
    # combined-state_dict fallbacks were removed with cambia-552 -- the former
    # was the RCE landing point (a pickled ``nn.Module``) and is unreachable
    # under ``weights_only=True`` anyway; neither format is emitted by any
    # writer in the repo.
    if not (
        isinstance(obj, dict) and "encoder_state_dict" in obj and "head_state_dict" in obj
    ):
        raise ValueError(
            f"snapshot {filepath!r} is not in the pinned PRT-CFR format "
            "{encoder_state_dict, head_state_dict, iteration}; refusing to load "
            "(security hardening, cambia-552)."
        )

    if _build_net_from_state is not None:
        return _build_net_from_state(
            obj["encoder_state_dict"], obj["head_state_dict"], device
        )
    # Mixture helper unavailable (defensive; e.g. a stubbed core during
    # parallel dev that lacks _regret_match) -- fall back to the module's
    # production-default dims, the pre-cambia-341 behavior.
    net = PRTCFRNet(device=device)
    net.load_encoder_head(obj["encoder_state_dict"], obj["head_state_dict"])
    net.eval()
    return net


def _pad_tokens(tokens: List[int], seq_cap: int = SEQ_CAP) -> np.ndarray:
    """Right-pad (keep-most-recent on overflow) a token list to width seq_cap."""
    arr = np.full(seq_cap, PAD_ID, dtype=np.int64)
    if not tokens:
        return arr
    if len(tokens) > seq_cap:
        tokens = tokens[-seq_cap:]
    arr[: len(tokens)] = np.asarray(tokens, dtype=np.int64)
    return arr


def _net_strategy_over_legal(
    net: Any,
    tokens: List[int],
    legal_actions: List[Any],
    seq_cap: int = SEQ_CAP,
) -> np.ndarray:
    """Regret-matched strategy from one net, projected to the node's legal order.

    Calls the core contract ``strategy_from_tokens(tokens[B,L], mask[B,146]) ->
    [B,146]`` with B=1, then reads back the entry for each legal action via
    ``action_to_index`` so the returned vector is length-nA in the node's
    legal-action (repr-sorted) order. ``action_to_index`` is a bijection over the
    146 space, so there are no collisions. Renormalized defensively.

    This is the single call site for the core net contract; adapt only here if
    the signature changes at integration.
    """
    import torch

    nA = len(legal_actions)
    tok_arr = _pad_tokens(tokens, seq_cap=seq_cap)
    mask146 = encode_action_mask(legal_actions)  # (146,) bool
    dev = getattr(net, "device", None) or torch.device("cpu")
    tok_t = torch.as_tensor(tok_arr, dtype=torch.long, device=dev).unsqueeze(0)  # (1, L)
    mask_t = torch.as_tensor(mask146, dtype=torch.bool, device=dev).unsqueeze(
        0
    )  # (1, 146)
    with torch.no_grad():
        strat146 = net.strategy_from_tokens(tok_t, mask_t)
    strat146 = np.asarray(strat146.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    if strat146.shape[0] != NUM_ACTIONS:
        raise ValueError(
            f"strategy_from_tokens returned width {strat146.shape[0]}, "
            f"expected {NUM_ACTIONS}"
        )
    out = np.empty(nA, dtype=np.float64)
    for i, a in enumerate(legal_actions):
        out[i] = strat146[action_to_index(a)]
    s = out.sum()
    if s > 1e-12:
        out = out / s
    else:
        out = np.ones(nA, dtype=np.float64) / nA
    return out


def sd_cfr_average_strategy(
    nets_by_iter: List[Tuple[int, Any]],
    tokens: List[int],
    legal_actions: List[Any],
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
) -> np.ndarray:
    """PER-DECISION mean of the per-net strategies at ONE infoset. NOT the served
    policy.

    ``sum_t w_t sigma^t(I) / sum_t w_t`` with w_t = t for linear weighting
    (default; matches the trainer's snapshot weighting), w_t = 1 for uniform.
    Renormalized defensively.

    This is the ``PER_DECISION`` object, kept as the per-infoset reference for
    ``materialize_policy(objective="per_decision")``. It is NOT what
    ``prtcfr_mixture.sample_episode`` serves: the served policy weights each
    snapshot by the acting player's own reach of the infoset under that
    snapshot, a quantity that depends on the whole tree and so cannot be formed
    one infoset at a time. Use ``combine_snapshot_policies`` (or the
    ``objective="served"`` default of the materializers) for the served object.
    """
    nA = len(legal_actions)
    acc = np.zeros(nA, dtype=np.float64)
    wsum = 0.0
    for it, net in nets_by_iter:
        w = float(it) if weighting == "linear" else 1.0
        if w <= 0.0:
            continue
        acc += w * _net_strategy_over_legal(net, tokens, legal_actions, seq_cap=seq_cap)
        wsum += w
    if wsum <= 0.0:
        return np.ones(nA, dtype=np.float64) / nA
    avg = acc / wsum
    s = avg.sum()
    if s > 1e-12:
        avg = avg / s
    else:
        avg = np.ones(nA, dtype=np.float64) / nA
    return avg


# ---------------------------------------------------------------------------
# The two SD-CFR policy objects
# ---------------------------------------------------------------------------
#
# SERVED (the default, and what the eval/serving stack plays). PRT-CFR keeps no
# strategy net: the average strategy is realized by SD-CFR snapshot sampling
# (prtcfr_mixture.sample_episode) -- one snapshot drawn per EPISODE with
# probability proportional to w_s, playing the whole game. That is a mixture
# over behavioral strategies. Under perfect recall Kuhn's theorem makes it
# realization equivalent to the single behavioral strategy that weights each
# snapshot by the acting player's OWN reach of the infoset:
#
#     b_served(I)[a] = sum_s w_s pi_i^{sigma_s}(I) sigma_s(I)[a]
#                      / sum_s w_s pi_i^{sigma_s}(I)
#
# Realization equivalence preserves the distribution over terminal histories
# against any opponent, hence both best-response values and the on-policy
# value, so exploitability(b_served) IS the exploitability of the mixture the
# wrapper plays.
#
# PER_DECISION (option, not the served policy). The reach-unweighted mean
#
#     b_per_decision(I)[a] = sum_s w_s sigma_s(I)[a] / sum_s w_s
#
# is a different strategy that nothing serves. It was the X2 gate's scored
# object until cambia-708; cambia-737 measured the resulting gap on the X2R
# runs (.docs/v0.4/phase2-throughput-pilot/x2-gate-vs-served-gap-2026-08-29/).
# It stays reachable so the two can be compared and historical numbers
# reproduced, never as a default.
#
# The own reach pi_i^{sigma_s}(I) excludes chance and the opponent, so it is
# the product of sigma_s over the acting player's OWN ancestor decisions.
# Perfect-recall keying is what makes it well defined per infoset;
# ``own_decision_structure`` proves that per tree rather than assuming it.

SERVED = "served"
PER_DECISION = "per_decision"
POLICY_OBJECTIVES = (SERVED, PER_DECISION)
DEFAULT_OBJECTIVE = SERVED


def _check_objective(objective: str) -> str:
    if objective not in POLICY_OBJECTIVES:
        raise ValueError(
            f"objective={objective!r} is not one of {POLICY_OBJECTIVES}; "
            f"{SERVED!r} is the policy prtcfr_mixture actually serves"
        )
    return objective


def _legal_layout(nodes: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-infoset legal-action counts and the offsets of the flat slice layout.

    Strategies are carried as one concatenated ``(total_legal,)`` float64 array
    in infoset order; ``flat[legal_off[i]:legal_off[i + 1]]`` is infoset ``i``'s
    distribution over its legal actions.
    """
    counts = np.array([len(nd.actions) for nd in nodes], dtype=np.int64)
    legal_off = np.zeros(len(nodes) + 1, dtype=np.int64)
    if counts.size:
        np.cumsum(counts, out=legal_off[1:])
    return counts, legal_off


def own_decision_structure(
    root: Any, nodes: List[Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Own-decision parent chain per infoset, proved consistent with perfect recall.

    Walks every history. At each decision node the acting player's previous own
    decision along that path is the candidate parent. Perfect recall says every
    history in an infoset agrees on it; a disagreement means own reach is not
    well defined per infoset and the realization-equivalence argument does not
    hold, so that raises rather than silently producing a number.

    Returns ``(parent_iset, parent_slot, owner, depth_order, depth_start)``:
      parent_iset  (N,) index of the acting player's previous own decision
                   infoset, -1 at an own-root.
      parent_slot  (N,) which legal-action slot of that parent leads here, -1
                   at an own-root.
      owner        (N,) the acting player.
      depth_order  (N,) infoset indices sorted by own-decision depth.
      depth_start  (D+1,) block boundaries of each depth in ``depth_order``.
    """
    n = len(nodes)
    index_of_pkey = {nd.pkey: i for i, nd in enumerate(nodes)}
    parent_iset = np.full(n, -1, dtype=np.int64)
    parent_slot = np.full(n, -1, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    # stack entries: (node, (last_own_p0, last_own_p1)), each last_own either
    # None or a (infoset index, action slot) pair.
    stack: List[Tuple[Any, Tuple[Optional[Tuple[int, int]], ...]]] = [
        (root, (None, None))
    ]
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
                    f"infoset {i} ({node.pkey!r}) is acted on by both players; "
                    f"one merged policy dict would conflate them"
                )
            if parent_iset[i] != cand_iset or parent_slot[i] != cand_slot:
                raise RuntimeError(
                    f"infoset {i} ({node.pkey!r}) has two distinct own-action "
                    f"predecessors ({parent_iset[i]},{parent_slot[i]}) vs "
                    f"({cand_iset},{cand_slot}); the tree is not perfect-recall "
                    f"keyed and own reach is undefined"
                )
        else:
            seen[i] = True
            owner[i] = p
            parent_iset[i] = cand_iset
            parent_slot[i] = cand_slot
        for k, child in enumerate(node.children):
            new_last = list(last_own)
            new_last[p] = (i, k)
            stack.append((child, tuple(new_last)))
    if n and not seen.all():
        raise RuntimeError(
            "some enumerated infoset was never reached by the history walk"
        )
    depth = np.full(n, -1, dtype=np.int64)
    roots = np.flatnonzero(parent_iset < 0)
    depth[roots] = 0
    remaining = n - roots.size
    d = 0
    while remaining > 0:
        nxt = np.flatnonzero((depth < 0) & (depth[parent_iset] == d))
        if nxt.size == 0:
            raise RuntimeError("own-decision parent graph has a cycle or a gap")
        depth[nxt] = d + 1
        remaining -= nxt.size
        d += 1
    depth_order = np.argsort(depth, kind="stable")
    per_depth = np.bincount(depth, minlength=1) if n else np.zeros(1, dtype=np.int64)
    depth_start = np.zeros(per_depth.size + 1, dtype=np.int64)
    np.cumsum(per_depth, out=depth_start[1:])
    return parent_iset, parent_slot, owner, depth_order, depth_start


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
    sigma(ancestor)[slot taken to get here]``: chance and the opponent are
    excluded, which is exactly what realization equivalence weights by. Swept
    depth by depth so each level is one vectorized gather.
    """
    n = int(legal_off.shape[0]) - 1
    pi = np.empty(n, dtype=np.float64)
    for d in range(int(depth_start.shape[0]) - 1):
        idx = depth_order[depth_start[d] : depth_start[d + 1]]
        if idx.size == 0:
            continue
        if d == 0:
            pi[idx] = 1.0
            continue
        par = parent_iset[idx]
        pi[idx] = pi[par] * strat_flat[legal_off[par] + parent_slot[idx]]
    return pi


def _renormalize_slices(flat: np.ndarray, legal_off: np.ndarray) -> None:
    """In place: make every infoset slice a distribution (uniform if it sums to 0)."""
    for i in range(int(legal_off.shape[0]) - 1):
        lo, hi = int(legal_off[i]), int(legal_off[i + 1])
        s = flat[lo:hi].sum()
        if s > 1e-12:
            flat[lo:hi] /= s
        else:
            flat[lo:hi] = 1.0 / (hi - lo)


def _finalize_per_decision(
    acc_flat: np.ndarray, weight_sum: float, legal_off: np.ndarray
) -> np.ndarray:
    """Per-decision mean: divide by the total weight, renormalize each slice."""
    out = np.array(acc_flat, dtype=np.float64, copy=True)
    if weight_sum <= 0.0:
        for i in range(int(legal_off.shape[0]) - 1):
            lo, hi = int(legal_off[i]), int(legal_off[i + 1])
            out[lo:hi] = 1.0 / (hi - lo)
        return out
    out /= weight_sum
    _renormalize_slices(out, legal_off)
    return out


def _finalize_served(
    acc_flat: np.ndarray, reach_weight: np.ndarray, legal_off: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Served policy: divide each slice by its realization weight, renormalize.

    ``reach_weight[i] = sum_s w_s pi_s(I_i)``. Where it is zero the infoset is
    unreachable under every snapshot, so the mixture's behavior there is
    unconstrained: it cannot move the terminal distribution, either
    best-response value, or the on-policy value. Those slices fall back to
    uniform, and the returned boolean array flags them.
    """
    out = np.array(acc_flat, dtype=np.float64, copy=True)
    n = int(legal_off.shape[0]) - 1
    unreached = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = int(legal_off[i]), int(legal_off[i + 1])
        w = float(reach_weight[i])
        if w <= 0.0:
            unreached[i] = True
            out[lo:hi] = 1.0 / (hi - lo)
            continue
        out[lo:hi] /= w
        s = out[lo:hi].sum()
        if s > 1e-12:
            out[lo:hi] /= s
        else:  # pragma: no cover - defensive; w > 0 implies s > 0 analytically
            out[lo:hi] = 1.0 / (hi - lo)
    return out, unreached


def _policy_dict(
    nodes: List[Any], flat: np.ndarray, legal_off: np.ndarray
) -> Dict[Any, np.ndarray]:
    """Flat slice layout -> the ``{pkey: strategy_vector(nA)}`` dict tiny_solver scores."""
    return {
        nd.pkey: np.array(flat[int(legal_off[i]) : int(legal_off[i + 1])], copy=True)
        for i, nd in enumerate(nodes)
    }


def combine_snapshot_policies(
    root: Any,
    weighted_policies: List[Tuple[float, Dict[Any, np.ndarray]]],
    objective: str = DEFAULT_OBJECTIVE,
    nodes: Optional[List[Any]] = None,
) -> Dict[Any, np.ndarray]:
    """Combine per-snapshot policy dicts into one policy under ``objective``.

    The net-free core of both materializers: ``weighted_policies`` is
    ``[(w_s, {pkey: vector}), ...]``, one entry per SD-CFR snapshot, each dict
    covering every infoset of ``root``. Returns the same dict shape.

    ``objective=SERVED`` (default) produces the own-reach-weighted mixture
    average -- the realized behavior strategy of what ``prtcfr_mixture`` serves.
    ``objective=PER_DECISION`` produces the reach-unweighted mean, which nothing
    serves.
    """
    _check_objective(objective)
    if nodes is None:
        nodes = enumerate_infosets(root)
    n = len(nodes)
    if n == 0:
        return {}
    counts, legal_off = _legal_layout(nodes)
    total = int(legal_off[-1])
    acc_flat = np.zeros(total, dtype=np.float64)
    wsum = 0.0
    if objective == SERVED:
        parent_iset, parent_slot, _owner, depth_order, depth_start = (
            own_decision_structure(root, nodes)
        )
        reach_weight = np.zeros(n, dtype=np.float64)
    for w, pol in weighted_policies:
        w = float(w)
        if w <= 0.0:
            continue
        strat_flat = np.empty(total, dtype=np.float64)
        for i, nd in enumerate(nodes):
            strat_flat[int(legal_off[i]) : int(legal_off[i + 1])] = np.asarray(
                pol[nd.pkey], dtype=np.float64
            )
        if objective == SERVED:
            pi = own_reaches(
                strat_flat, legal_off, parent_iset, parent_slot, depth_order, depth_start
            )
            acc_flat += (w * np.repeat(pi, counts)) * strat_flat
            reach_weight += w * pi
        else:
            acc_flat += w * strat_flat
        wsum += w
    if objective == SERVED:
        flat, _unreached = _finalize_served(acc_flat, reach_weight, legal_off)
    else:
        flat = _finalize_per_decision(acc_flat, wsum, legal_off)
    return _policy_dict(nodes, flat, legal_off)


# ---------------------------------------------------------------------------
# Policy materialization + scoring
# ---------------------------------------------------------------------------


def materialize_policy(
    root,
    nets_by_iter: List[Tuple[int, Any]],
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
    objective: str = DEFAULT_OBJECTIVE,
) -> Dict[Any, np.ndarray]:
    """Build the ``{pkey: strategy_vector(nA)}`` dict for tiny_solver.exploitability.

    One vector per distinct perfect-recall infoset, keyed by the BARE ``node.pkey``
    so the solver's existing ``_lookup`` dict path scores it without modification.

    ``objective`` selects which SD-CFR object is materialized: ``SERVED``
    (default, the own-reach-weighted mixture average that ``prtcfr_mixture``
    realizes by sampling) or ``PER_DECISION`` (the reach-unweighted mean, which
    nothing serves). See the module's "two SD-CFR policy objects" section.

    Batched: every infoset's padded token row and 146-mask are stacked into one
    ``(N, seq_cap)`` / ``(N, 146)`` pair, and each net's
    ``strategy_from_tokens`` is called ONCE over the whole batch (not once per
    infoset per net). The {A,6} tree has ~13k infosets; the per-infoset-per-net
    loop is tens of thousands of B=1 GRU forwards (minutes on CPU), the batched
    path is one forward per net (seconds). The two paths produce the same numbers
    up to float order; ``sd_cfr_average_strategy`` is the per-infoset reference
    for the ``PER_DECISION`` object.
    """
    import torch

    _check_objective(objective)
    if tiny_node_to_tokens is None:
        raise RuntimeError(
            "src.cfr.prtcfr_net.tiny_node_to_tokens unavailable. Core has not "
            "landed; the gate test injects a stub for plumbing runs."
        )

    nodes = enumerate_infosets(root)
    n = len(nodes)
    if n == 0:
        return {}

    # Stack token rows + 146-masks once; record each node's legal head indices.
    counts, legal_off = _legal_layout(nodes)
    total = int(legal_off[-1])
    tok_rows = np.empty((n, seq_cap), dtype=np.int64)
    mask_rows = np.zeros((n, NUM_ACTIONS), dtype=bool)
    legal_flat = np.empty(total, dtype=np.int64)
    for i, node in enumerate(nodes):
        tok_rows[i] = _pad_tokens(tiny_node_to_tokens(node), seq_cap=seq_cap)
        mask_rows[i] = encode_action_mask(node.actions)
        legal_flat[int(legal_off[i]) : int(legal_off[i + 1])] = [
            action_to_index(a) for a in node.actions
        ]

    if objective == SERVED:
        parent_iset, parent_slot, _owner, depth_order, depth_start = (
            own_decision_structure(root, nodes)
        )
        reach_weight = np.zeros(n, dtype=np.float64)

    # SD-CFR weighted accumulation in the flat legal layout, one batched forward
    # per net.
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    acc_flat = np.zeros(total, dtype=np.float64)
    wsum = 0.0
    for it, net in nets_by_iter:
        w = float(it) if weighting == "linear" else 1.0
        if w <= 0.0:
            continue
        dev = getattr(net, "device", None) or torch.device("cpu")
        tok_t = torch.as_tensor(tok_rows, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(mask_rows, dtype=torch.bool, device=dev)
        with torch.no_grad():
            strat = net.strategy_from_tokens(tok_t, mask_t)
        strat = np.asarray(strat.detach().cpu().numpy(), dtype=np.float64)
        if strat.shape != (n, NUM_ACTIONS):
            raise ValueError(
                f"strategy_from_tokens returned shape {strat.shape}, "
                f"expected {(n, NUM_ACTIONS)}"
            )
        strat_flat = strat[rows, legal_flat]
        if objective == SERVED:
            pi = own_reaches(
                strat_flat, legal_off, parent_iset, parent_slot, depth_order, depth_start
            )
            acc_flat += (w * np.repeat(pi, counts)) * strat_flat
            reach_weight += w * pi
        else:
            acc_flat += w * strat_flat
        wsum += w

    if objective == SERVED:
        flat, _unreached = _finalize_served(acc_flat, reach_weight, legal_off)
    else:
        flat = _finalize_per_decision(acc_flat, wsum, legal_off)
    return _policy_dict(nodes, flat, legal_off)


class IncrementalPolicyAccumulator:
    """Incremental, memory-bounded SD-CFR policy materializer.

    ``materialize_policy`` stacks EVERY infoset's token row into one
    ``(N, seq_cap)`` batch and calls ``strategy_from_tokens`` ONCE over the
    whole thing per net -- fine at small N, but at production net dims
    (``PRTCFRNet`` defaults: embed_dim=64, hidden_dim=256) and N in the tens
    of thousands (the {A,6} tree post-S1W11 snap-legality fix: 69636
    perfect-recall infosets, up from 12884 pre-fix), a single N-row forward
    allocates tens of GB: the pre-pack embedding tensor alone is
    N * seq_cap * embed_dim * 4 bytes (~4.6GB at N=69636, seq_cap=256,
    embed_dim=64), and ``pack_padded_sequence``'s sort-reorder
    ``index_select`` (``enforce_sorted=False``) copies it again before the
    GRU forward even starts. Confirmed by a bounded (``ulimit -v``) repro
    that failed inside exactly that ``index_select`` at exactly this size
    (test_x2_plumbing_random_net's reported 28.7GB RSS OOM, S1W11 gate-fix
    sprint).

    This accumulator computes the numerically equivalent SD-CFR weighted
    average (same accumulation order as ``materialize_policy``: per-net
    ``w * strategy``, summed in ``nets_by_iter`` order, normalized once at
    the end) via ``chunk_size``-row forwards, so peak memory is bounded by
    the chunk regardless of tree size. It also skips any ``(iter, net)``
    pair already folded in, so repeated calls across a growing snapshot
    horizon (the real gate's per-eval-checkpoint usage during training) cost
    each snapshot ONE forward ever, not once per subsequent eval (linear in
    the horizon instead of quadratic) -- mirrors the incremental-accumulation
    technique prototyped in the S1W11 X2 revalidation launcher
    (``cfr/scratch/prtcfr_x2_s1w11_gpu.py``, uncommitted).

    Equivalence to ``materialize_policy`` (same ``root``, ``nets_by_iter``,
    ``weighting``, ``seq_cap``, ``objective``) holds up to float32
    matmul-reordering noise
    (chunking only changes how rows are grouped into the net's float32 batched
    matmuls; each row's own arithmetic is independent of which other rows
    share its batch since neither the GRU nor LayerNorm mixes across the
    batch dimension -- the outer SD-CFR accumulation is float64 regardless of
    chunking) -- verified on a small tree by
    tests/test_prtcfr_x2_gate.py::test_incremental_policy_matches_materialize_policy_small_tree
    (empirically ~1.8e-6 max abs diff, well inside float32 precision).

    ``objective`` (cambia-708) selects which SD-CFR object accumulates:
    ``SERVED`` (default) folds each snapshot weighted by ``w_s`` times its own
    reach of the infoset, the realized behavior strategy of the sampled
    mixture; ``PER_DECISION`` folds by ``w_s`` alone. The own reach is a
    per-snapshot, whole-tree quantity, so under ``SERVED`` each snapshot's
    strategy is completed over every infoset before it is folded -- which the
    chunked accumulation already does, one net at a time. ``SERVED`` costs one
    extra ``(total_legal,)`` float64 buffer per net plus one depth sweep; the
    per-net forwards, the dominant cost, are unchanged.
    """

    def __init__(
        self,
        root: Any,
        weighting: str = "linear",
        seq_cap: int = SEQ_CAP,
        chunk_size: int = 2048,
        objective: str = DEFAULT_OBJECTIVE,
    ):
        if tiny_node_to_tokens is None:
            raise RuntimeError(
                "src.cfr.prtcfr_net.tiny_node_to_tokens unavailable. Core has not "
                "landed; the gate test injects a stub for plumbing runs."
            )
        _check_objective(objective)
        self.weighting = weighting
        self.seq_cap = seq_cap
        self.chunk_size = max(1, int(chunk_size))
        self.objective = objective
        self.nodes = enumerate_infosets(root)
        n = len(self.nodes)
        self._counts, self._legal_off = _legal_layout(self.nodes)
        total = int(self._legal_off[-1])
        self._tok_rows = np.empty((n, seq_cap), dtype=np.int64)
        self._mask_rows = np.zeros((n, NUM_ACTIONS), dtype=bool)
        self._legal_flat = np.empty(total, dtype=np.int64)
        for i, node in enumerate(self.nodes):
            self._tok_rows[i] = _pad_tokens(tiny_node_to_tokens(node), seq_cap=seq_cap)
            self._mask_rows[i] = encode_action_mask(node.actions)
            self._legal_flat[int(self._legal_off[i]) : int(self._legal_off[i + 1])] = [
                action_to_index(a) for a in node.actions
            ]
        self._acc_flat = np.zeros(total, dtype=np.float64)
        self._wsum = 0.0
        self._accumulated: set = set()
        self._reach_weight = np.zeros(n, dtype=np.float64)
        # Number of infosets the served finalize fell back to uniform on
        # (unreachable under every folded snapshot); set by ``policy()``.
        self.unreached_infosets = 0
        if objective == SERVED:
            (
                self._parent_iset,
                self._parent_slot,
                _owner,
                self._depth_order,
                self._depth_start,
            ) = own_decision_structure(root, self.nodes)

    def accumulate(self, nets_by_iter: List[Tuple[int, Any]]) -> None:
        """Fold every ``(iter, net)`` not already accumulated into the
        running weighted sum, in the given order, one chunked forward per
        net (bounds peak memory to ``chunk_size`` rows regardless of N)."""
        import torch

        n = len(self.nodes)
        rows = np.repeat(np.arange(n, dtype=np.int64), self._counts)
        for it, net in nets_by_iter:
            if it in self._accumulated:
                continue
            self._accumulated.add(it)
            w = float(it) if self.weighting == "linear" else 1.0
            if w <= 0.0:
                continue
            dev = getattr(net, "device", None) or torch.device("cpu")
            strat_flat = np.empty(int(self._legal_off[-1]), dtype=np.float64)
            for lo in range(0, n, self.chunk_size):
                hi = min(lo + self.chunk_size, n)
                tok_t = torch.as_tensor(
                    self._tok_rows[lo:hi], dtype=torch.long, device=dev
                )
                mask_t = torch.as_tensor(
                    self._mask_rows[lo:hi], dtype=torch.bool, device=dev
                )
                with torch.no_grad():
                    strat = net.strategy_from_tokens(tok_t, mask_t)
                strat_np = strat.detach().to("cpu", dtype=torch.float64).numpy()
                if strat_np.shape != (hi - lo, NUM_ACTIONS):
                    raise ValueError(
                        f"strategy_from_tokens returned shape {strat_np.shape}, "
                        f"expected {(hi - lo, NUM_ACTIONS)}"
                    )
                flat_lo, flat_hi = int(self._legal_off[lo]), int(self._legal_off[hi])
                strat_flat[flat_lo:flat_hi] = strat_np[
                    rows[flat_lo:flat_hi] - lo, self._legal_flat[flat_lo:flat_hi]
                ]
            if self.objective == SERVED:
                pi = own_reaches(
                    strat_flat,
                    self._legal_off,
                    self._parent_iset,
                    self._parent_slot,
                    self._depth_order,
                    self._depth_start,
                )
                self._acc_flat += (w * np.repeat(pi, self._counts)) * strat_flat
                self._reach_weight += w * pi
            else:
                self._acc_flat += w * strat_flat
            self._wsum += w

    def policy(self) -> Dict[Any, np.ndarray]:
        """Materialize the current ``{pkey: strategy_vector(nA)}`` dict from
        the running accumulation state (same normalization as
        ``materialize_policy`` under the same ``objective``)."""
        if self.objective == SERVED:
            flat, unreached = _finalize_served(
                self._acc_flat, self._reach_weight, self._legal_off
            )
            self.unreached_infosets = int(unreached.sum())
        else:
            flat = _finalize_per_decision(self._acc_flat, self._wsum, self._legal_off)
            self.unreached_infosets = 0
        return _policy_dict(self.nodes, flat, self._legal_off)


def materialize_policy_incremental(
    root: Any,
    nets_by_iter: List[Tuple[int, Any]],
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
    chunk_size: int = 2048,
    objective: str = DEFAULT_OBJECTIVE,
) -> Dict[Any, np.ndarray]:
    """Drop-in, memory-bounded replacement for ``materialize_policy`` (same
    signature; numerically equivalent up to float summation order -- see
    ``IncrementalPolicyAccumulator``): computes the SD-CFR weighted average
    via chunked per-net forwards so peak memory is bounded by ``chunk_size``
    rows instead of one N-row forward. Prefer this over ``materialize_policy``
    whenever N (infosets) times production net dims makes a single-shot batch
    forward too large -- which is always, for the real {A,6} tree at
    production net dims (see the class docstring for the concrete OOM this
    avoids); ``materialize_policy`` is the single-batch equivalence reference.

    ``objective`` defaults to ``SERVED``, the policy ``prtcfr_mixture`` plays."""
    acc = IncrementalPolicyAccumulator(
        root,
        weighting=weighting,
        seq_cap=seq_cap,
        chunk_size=chunk_size,
        objective=objective,
    )
    acc.accumulate(nets_by_iter)
    return acc.policy()


def score_policy_on_tiny_game(
    checkpoint_or_snapshot_dir: str,
    config_path: str = TINY_2CARD_CONFIG,
    weighting: str = "linear",
    device: str = "cpu",
    seq_cap: int = SEQ_CAP,
    objective: str = DEFAULT_OBJECTIVE,
) -> Dict[str, Any]:
    """End-to-end X2 scorer: load -> enumerate -> tokenize -> average -> score.

    Args:
        checkpoint_or_snapshot_dir: a single .pt checkpoint, or a directory of
            prtcfr_snapshot_iter_{t}.pt SD-CFR snapshots.
        config_path: tiny-game config (default the {A,6} plateau game).
        weighting: "linear" (w_t=t, default) or "uniform".
        device: torch map_location for loading nets.
        objective: ``SERVED`` (default, the policy prtcfr_mixture plays) or
            ``PER_DECISION`` (the reach-unweighted mean, which nothing serves).

    Returns a dict:
        {"nashconv": float, "components": (br0, br1, onp0, onp1),
         "num_infosets": int, "num_snapshots": int,
         "snapshot_iters": [int, ...], "objective": str, "passed": bool}
    where ``passed`` == (nashconv < X2_NASHCONV_BAR).

    @chief invokes this for the real verdict with a trained snapshot dir.
    """
    _check_objective(objective)
    snaps = discover_snapshots(checkpoint_or_snapshot_dir)
    # Provenance gate (cambia-612): refuse a version-mismatched checkpoint before
    # loading it, and pick the observation path that matches its training-era
    # tokenizer (>= v2 -> production peek/post-draw frames; unknown -> legacy).
    production_obs = _resolve_scoring_obs_path(
        _recorded_tokenizer_version(checkpoint_or_snapshot_dir)
    )
    nets_by_iter = [(it, _load_net(fp, device=device)) for it, fp in snaps]
    root, _isets, _n, _ab = build_tiny_tree(
        config_path, seq_cap=seq_cap, production_obs=production_obs
    )
    # materialize_policy_incremental, not materialize_policy: at production net
    # dims (embed=64, hidden=256) and the real {A,6} tree's 69636 infosets, a
    # single N-row batched forward OOMs (see IncrementalPolicyAccumulator's
    # docstring); the chunked accumulator is numerically equivalent.
    policy = materialize_policy_incremental(
        root, nets_by_iter, weighting=weighting, seq_cap=seq_cap, objective=objective
    )
    nashconv, components = exploitability(root, policy)
    return {
        "nashconv": float(nashconv),
        "components": tuple(float(x) for x in components),
        "num_infosets": len(policy),
        "num_snapshots": len(snaps),
        "snapshot_iters": [it for it, _ in snaps],
        "objective": objective,
        "passed": bool(nashconv < X2_NASHCONV_BAR),
    }


def score_with_loaded_nets(
    nets_by_iter: List[Tuple[int, Any]],
    config_path: str = TINY_2CARD_CONFIG,
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
    tokenizer_version: Optional[int] = None,
    objective: str = DEFAULT_OBJECTIVE,
) -> Dict[str, Any]:
    """Same as score_policy_on_tiny_game but with nets already in memory.

    Used by the plumbing test (random-init net) and any caller that has built
    PRTCFRNet instances directly. ``nets_by_iter`` is [(iter, net), ...].

    tokenizer_version (cambia-612): the version the nets were trained under, so
    the same provenance gate applies as in the path-based entry points -- hard
    error on mismatch with the live tokenizer, loud warning + legacy path when
    None (unknown, the default for in-memory nets with no run_meta to consult).
    """
    _check_objective(objective)
    production_obs = _resolve_scoring_obs_path(tokenizer_version)
    root, _isets, _n, _ab = build_tiny_tree(
        config_path, seq_cap=seq_cap, production_obs=production_obs
    )
    # See score_policy_on_tiny_game: incremental/chunked, not the single-batch
    # materialize_policy, to stay well under a few GB RSS at production dims.
    policy = materialize_policy_incremental(
        root, nets_by_iter, weighting=weighting, seq_cap=seq_cap, objective=objective
    )
    nashconv, components = exploitability(root, policy)
    return {
        "nashconv": float(nashconv),
        "components": tuple(float(x) for x in components),
        "num_infosets": len(policy),
        "num_snapshots": len(nets_by_iter),
        "snapshot_iters": [it for it, _ in nets_by_iter],
        "objective": objective,
        "passed": bool(nashconv < X2_NASHCONV_BAR),
    }


def certify_policy_on_tiny_game(
    checkpoint_or_snapshot_dir: str,
    config_path: str = TINY_2CARD_CONFIG,
    weighting: str = "linear",
    device: str = "cpu",
    seq_cap: int = SEQ_CAP,
    objective: str = DEFAULT_OBJECTIVE,
) -> Dict[str, Any]:
    """Exact-rational X2 verdict: the authoritative scorer for the X2R5 ruling.

    Scores the ``SERVED`` object by default (cambia-708): the exact NashConv of
    the policy ``prtcfr_mixture`` actually plays, which puts the certifier on
    the same footing as the reach-weighted X1 tabular baseline. Pass
    ``objective=PER_DECISION`` to reproduce a pre-cambia-708 number.

    Same load -> enumerate -> tokenize -> SD-CFR average path as
    ``score_policy_on_tiny_game``, but the tree carries exact-rational chance
    mass (``build_tiny_tree(exact_weights=True)``) and NashConv is recomputed
    end-to-end in ``fractions.Fraction`` (tools/tiny_exact.py, cambia-530): no
    rounding, no ``limit_denominator``, no epsilon anywhere on the exact path.
    float64 remains the fast path and is reported for the differential, but the
    verdict (``passed``) is decided on the EXACT NashConv against
    ``bar_respec = 0.057 = Fraction(57, 1000)``.

    Returns ``score_policy_on_tiny_game``'s keys plus:
        {"nashconv": float (exact projected),
         "nashconv_float64": float (fast-path scorer),
         "nashconv_exact_str": "num/den" (exact rational, lossless),
         "components_exact_str": (…) exact components as "num/den",
         "margin_vs_bar": float (exact nashconv - 0.057; <0 => pass),
         "margin_vs_bar_str": "num/den" exact signed margin,
         "float_vs_exact_abs": float |float64 - exact|,
         "bar": float, "passed": bool (exact)}
    ``passed`` is the exact verdict; ``nashconv`` reports the exact value so the
    gate consumes exact by default.
    """
    _check_objective(objective)
    snaps = discover_snapshots(checkpoint_or_snapshot_dir)
    # Provenance gate (cambia-612): same version check + obs-path selection as
    # score_policy_on_tiny_game, applied to the exact-rational verdict path.
    production_obs = _resolve_scoring_obs_path(
        _recorded_tokenizer_version(checkpoint_or_snapshot_dir)
    )
    nets_by_iter = [(it, _load_net(fp, device=device)) for it, fp in snaps]
    root, _isets, _n, _ab = build_tiny_tree(
        config_path, seq_cap=seq_cap, exact_weights=True, production_obs=production_obs
    )
    policy = materialize_policy_incremental(
        root, nets_by_iter, weighting=weighting, seq_cap=seq_cap, objective=objective
    )
    nc_f, comp_f = exploitability(root, policy)
    cert = tiny_exact.certify(root, policy, bar=tiny_exact.BAR_RESPEC)
    nc_e = cert["nashconv"]
    return {
        "nashconv": float(nc_e),
        "nashconv_float64": float(nc_f),
        "nashconv_exact_str": f"{nc_e.numerator}/{nc_e.denominator}",
        "components": cert["components_float"],
        "components_exact_str": tuple(
            f"{c.numerator}/{c.denominator}" for c in cert["components"]
        ),
        "components_float64": tuple(float(x) for x in comp_f),
        "margin_vs_bar": cert["margin_float"],
        "margin_vs_bar_str": f"{cert['margin'].numerator}/{cert['margin'].denominator}",
        "float_vs_exact_abs": abs(float(nc_f) - float(nc_e)),
        "num_infosets": len(policy),
        "num_snapshots": len(snaps),
        "snapshot_iters": [it for it, _ in snaps],
        "objective": objective,
        "bar": float(tiny_exact.BAR_RESPEC),
        "passed": cert["passed"],
    }
