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
  2. Realize the SD-CFR AVERAGE policy deterministically. For SD-CFR the served
     strategy is the linear-iteration-weighted mean of per-snapshot regret-matched
     strategies:

         strategy(I) = sum_t w_t * regret_match(net_t(tokens, mask)) / sum_t w_t,
         w_t = t   (linear weighting; matches deep_trainer sd_cfr_snapshot_weighting)

     Each ``net_t`` is one snapshot; the per-net regret-matched strategy over the
     146-action space comes from ``PRTCFRNet.strategy_from_tokens(tokens, mask)``.
     A single checkpoint is the degenerate one-snapshot case (weight 1.0).
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

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Core-owned primitives. Tolerant import: the gate test can inject a stub before
# this resolves if core has not landed yet (parallel development).
try:  # pragma: no cover - real path exercised at integration
    from src.cfr.prtcfr_net import PRTCFRNet, tiny_node_to_tokens
except Exception:  # noqa: BLE001 - core may be unwritten during parallel dev
    PRTCFRNet = None  # type: ignore[assignment]
    tiny_node_to_tokens = None  # type: ignore[assignment]

from src.config import load_config
from src.encoding import NUM_ACTIONS, action_to_index, encode_action_mask
from src.sequence_encoding import PAD_ID, SEQ_CAP
from tools.tiny_solver import build_tree, exploitability

# Tiny {A,6} 2-card plateau game: the EXACT X1 cause-isolation game.
TINY_2CARD_CONFIG = "config/tiny_2card_plateau.yaml"

# Build parameters matching the X1 keystone run (config header + X1 verdict):
# enumerated draw-chance, K=5 deals, seed0=0, perfect-recall keying. tokenize=True
# populates Decision.seq_tokens with the single-sourced token stream.
TINY_N_DEALS = 5
TINY_SEED0 = 0
TINY_MAX_NODES_PER_DEAL = 2_000_000

# X2 gate bar (contract.md): neural-policy NashConv strictly below this.
X2_NASHCONV_BAR = 0.05

# SD-CFR snapshot filename pattern: prtcfr_snapshot_iter_{t}.pt
_SNAPSHOT_RE = re.compile(r"prtcfr_snapshot_iter_(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Tiny-game construction + infoset enumeration
# ---------------------------------------------------------------------------


def build_tiny_tree(config_path: str = TINY_2CARD_CONFIG, seq_cap: int = SEQ_CAP):
    """Build the perfect-recall + tokenized {A,6} tiny tree.

    Returns (root, isets, n, aborted). Reuses tools.tiny_solver.build_tree with
    ``perfect_recall=True, tokenize=True`` so the tree, its infoset partition, and
    the per-node token streams are identical to what the PRT-CFR worker trains on.
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
    """Load one PRTCFRNet snapshot.

    The pinned checkpoint format is ``{encoder_state_dict, head_state_dict,
    iteration}`` (prtcfr_net docstring); loaded via ``PRTCFRNet.load_encoder_head``.
    A whole-module pickle or a plain combined state_dict are also accepted as
    fallbacks. This is the only place that touches checkpoint internals.
    """
    import torch

    if PRTCFRNet is None:
        raise RuntimeError(
            "src.cfr.prtcfr_net.PRTCFRNet unavailable. Core (prtcfr-core) has not "
            "landed; the gate test injects a stub for plumbing runs."
        )
    obj = torch.load(filepath, map_location=device, weights_only=False)

    # Pinned format: split encoder/head state dicts.
    if isinstance(obj, dict) and "encoder_state_dict" in obj and "head_state_dict" in obj:
        net = PRTCFRNet(device=device)
        net.load_encoder_head(obj["encoder_state_dict"], obj["head_state_dict"])
        net.eval()
        return net

    # Whole-module pickle.
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # Combined / wrapped state_dict.
    state_dict = obj
    if isinstance(obj, dict) and not _looks_like_state_dict(obj):
        for key in ("model_state_dict", "state_dict", "net", "policy_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break
    net = PRTCFRNet(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _looks_like_state_dict(d: dict) -> bool:
    """Heuristic: a state_dict maps str -> tensor-like at the top level."""
    import torch

    if not d:
        return False
    return all(
        isinstance(k, str) and isinstance(v, (torch.Tensor, np.ndarray))
        for k, v in d.items()
    )


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
    """SD-CFR served strategy at one infoset: weighted mean of per-net strategies.

    w_t = t for linear weighting (default; matches the trainer), w_t = 1 for
    uniform. Renormalized defensively.
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
# Policy materialization + scoring
# ---------------------------------------------------------------------------


def materialize_policy(
    root,
    nets_by_iter: List[Tuple[int, Any]],
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
) -> Dict[Any, np.ndarray]:
    """Build the ``{pkey: strategy_vector(nA)}`` dict for tiny_solver.exploitability.

    One vector per distinct perfect-recall infoset, keyed by the BARE ``node.pkey``
    so the solver's existing ``_lookup`` dict path scores it without modification.

    Batched: every infoset's padded token row and 146-mask are stacked into one
    ``(N, seq_cap)`` / ``(N, 146)`` pair, and each net's
    ``strategy_from_tokens`` is called ONCE over the whole batch (not once per
    infoset per net). The {A,6} tree has ~13k infosets; the per-infoset-per-net
    loop is tens of thousands of B=1 GRU forwards (minutes on CPU), the batched
    path is one forward per net (seconds). The two paths produce the same numbers
    up to float order; ``sd_cfr_average_strategy`` is the per-infoset reference.
    """
    import torch

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
    tok_rows = np.empty((n, seq_cap), dtype=np.int64)
    mask_rows = np.zeros((n, NUM_ACTIONS), dtype=bool)
    legal_idx: List[List[int]] = []
    for i, node in enumerate(nodes):
        tok_rows[i] = _pad_tokens(tiny_node_to_tokens(node), seq_cap=seq_cap)
        mask_rows[i] = encode_action_mask(node.actions)
        legal_idx.append([action_to_index(a) for a in node.actions])

    # SD-CFR weighted accumulation in 146-space, one batched forward per net.
    acc146 = np.zeros((n, NUM_ACTIONS), dtype=np.float64)
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
        acc146 += w * strat
        wsum += w

    policy: Dict[Any, np.ndarray] = {}
    for i, node in enumerate(nodes):
        idx = legal_idx[i]
        nA = len(idx)
        if wsum <= 0.0:
            policy[node.pkey] = np.ones(nA, dtype=np.float64) / nA
            continue
        vec = acc146[i, idx] / wsum
        s = vec.sum()
        policy[node.pkey] = vec / s if s > 1e-12 else np.ones(nA, dtype=np.float64) / nA
    return policy


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
    ``weighting``, ``seq_cap``) holds up to float32 matmul-reordering noise
    (chunking only changes how rows are grouped into the net's float32 batched
    matmuls; each row's own arithmetic is independent of which other rows
    share its batch since neither the GRU nor LayerNorm mixes across the
    batch dimension -- the outer SD-CFR accumulation is float64 regardless of
    chunking) -- verified on a small tree by
    tests/test_prtcfr_x2_gate.py::test_incremental_policy_matches_materialize_policy_small_tree
    (empirically ~1.8e-6 max abs diff, well inside float32 precision).
    """

    def __init__(
        self,
        root: Any,
        weighting: str = "linear",
        seq_cap: int = SEQ_CAP,
        chunk_size: int = 2048,
    ):
        if tiny_node_to_tokens is None:
            raise RuntimeError(
                "src.cfr.prtcfr_net.tiny_node_to_tokens unavailable. Core has not "
                "landed; the gate test injects a stub for plumbing runs."
            )
        self.weighting = weighting
        self.seq_cap = seq_cap
        self.chunk_size = max(1, int(chunk_size))
        self.nodes = enumerate_infosets(root)
        n = len(self.nodes)
        self._tok_rows = np.empty((n, seq_cap), dtype=np.int64)
        self._mask_rows = np.zeros((n, NUM_ACTIONS), dtype=bool)
        self._legal_idx: List[List[int]] = []
        for i, node in enumerate(self.nodes):
            self._tok_rows[i] = _pad_tokens(tiny_node_to_tokens(node), seq_cap=seq_cap)
            self._mask_rows[i] = encode_action_mask(node.actions)
            self._legal_idx.append([action_to_index(a) for a in node.actions])
        self._acc146 = np.zeros((n, NUM_ACTIONS), dtype=np.float64)
        self._wsum = 0.0
        self._accumulated: set = set()

    def accumulate(self, nets_by_iter: List[Tuple[int, Any]]) -> None:
        """Fold every ``(iter, net)`` not already accumulated into the
        running weighted sum, in the given order, one chunked forward per
        net (bounds peak memory to ``chunk_size`` rows regardless of N)."""
        import torch

        n = len(self.nodes)
        for it, net in nets_by_iter:
            if it in self._accumulated:
                continue
            self._accumulated.add(it)
            w = float(it) if self.weighting == "linear" else 1.0
            if w <= 0.0:
                continue
            dev = getattr(net, "device", None) or torch.device("cpu")
            for lo in range(0, n, self.chunk_size):
                hi = min(lo + self.chunk_size, n)
                tok_t = torch.as_tensor(self._tok_rows[lo:hi], dtype=torch.long, device=dev)
                mask_t = torch.as_tensor(self._mask_rows[lo:hi], dtype=torch.bool, device=dev)
                with torch.no_grad():
                    strat = net.strategy_from_tokens(tok_t, mask_t)
                strat_np = strat.detach().to("cpu", dtype=torch.float64).numpy()
                if strat_np.shape != (hi - lo, NUM_ACTIONS):
                    raise ValueError(
                        f"strategy_from_tokens returned shape {strat_np.shape}, "
                        f"expected {(hi - lo, NUM_ACTIONS)}"
                    )
                self._acc146[lo:hi] += w * strat_np
            self._wsum += w

    def policy(self) -> Dict[Any, np.ndarray]:
        """Materialize the current ``{pkey: strategy_vector(nA)}`` dict from
        the running accumulation state (same normalization as
        ``materialize_policy``)."""
        out: Dict[Any, np.ndarray] = {}
        for i, node in enumerate(self.nodes):
            idx = self._legal_idx[i]
            nA = len(idx)
            if self._wsum <= 0.0:
                out[node.pkey] = np.ones(nA, dtype=np.float64) / nA
                continue
            vec = self._acc146[i, idx] / self._wsum
            s = vec.sum()
            out[node.pkey] = vec / s if s > 1e-12 else np.ones(nA, dtype=np.float64) / nA
        return out


def materialize_policy_incremental(
    root: Any,
    nets_by_iter: List[Tuple[int, Any]],
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
    chunk_size: int = 2048,
) -> Dict[Any, np.ndarray]:
    """Drop-in, memory-bounded replacement for ``materialize_policy`` (same
    signature; numerically equivalent up to float summation order -- see
    ``IncrementalPolicyAccumulator``): computes the SD-CFR weighted average
    via chunked per-net forwards so peak memory is bounded by ``chunk_size``
    rows instead of one N-row forward. Prefer this over ``materialize_policy``
    whenever N (infosets) times production net dims makes a single-shot batch
    forward too large -- which is always, for the real {A,6} tree at
    production net dims (see the class docstring for the concrete OOM this
    avoids); ``materialize_policy`` itself is kept unchanged as the
    equivalence-gate reference."""
    acc = IncrementalPolicyAccumulator(
        root, weighting=weighting, seq_cap=seq_cap, chunk_size=chunk_size
    )
    acc.accumulate(nets_by_iter)
    return acc.policy()


def score_policy_on_tiny_game(
    checkpoint_or_snapshot_dir: str,
    config_path: str = TINY_2CARD_CONFIG,
    weighting: str = "linear",
    device: str = "cpu",
    seq_cap: int = SEQ_CAP,
) -> Dict[str, Any]:
    """End-to-end X2 scorer: load -> enumerate -> tokenize -> average -> score.

    Args:
        checkpoint_or_snapshot_dir: a single .pt checkpoint, or a directory of
            prtcfr_snapshot_iter_{t}.pt SD-CFR snapshots.
        config_path: tiny-game config (default the {A,6} plateau game).
        weighting: "linear" (w_t=t, default) or "uniform".
        device: torch map_location for loading nets.

    Returns a dict:
        {"nashconv": float, "components": (br0, br1, onp0, onp1),
         "num_infosets": int, "num_snapshots": int,
         "snapshot_iters": [int, ...], "passed": bool}
    where ``passed`` == (nashconv < X2_NASHCONV_BAR).

    @chief invokes this for the real verdict with a trained snapshot dir.
    """
    snaps = discover_snapshots(checkpoint_or_snapshot_dir)
    nets_by_iter = [(it, _load_net(fp, device=device)) for it, fp in snaps]
    root, _isets, _n, _ab = build_tiny_tree(config_path, seq_cap=seq_cap)
    # materialize_policy_incremental, not materialize_policy: at production net
    # dims (embed=64, hidden=256) and the real {A,6} tree's 69636 infosets, a
    # single N-row batched forward OOMs (see IncrementalPolicyAccumulator's
    # docstring); the chunked accumulator is numerically equivalent.
    policy = materialize_policy_incremental(
        root, nets_by_iter, weighting=weighting, seq_cap=seq_cap
    )
    nashconv, components = exploitability(root, policy)
    return {
        "nashconv": float(nashconv),
        "components": tuple(float(x) for x in components),
        "num_infosets": len(policy),
        "num_snapshots": len(snaps),
        "snapshot_iters": [it for it, _ in snaps],
        "passed": bool(nashconv < X2_NASHCONV_BAR),
    }


def score_with_loaded_nets(
    nets_by_iter: List[Tuple[int, Any]],
    config_path: str = TINY_2CARD_CONFIG,
    weighting: str = "linear",
    seq_cap: int = SEQ_CAP,
) -> Dict[str, Any]:
    """Same as score_policy_on_tiny_game but with nets already in memory.

    Used by the plumbing test (random-init net) and any caller that has built
    PRTCFRNet instances directly. ``nets_by_iter`` is [(iter, net), ...].
    """
    root, _isets, _n, _ab = build_tiny_tree(config_path, seq_cap=seq_cap)
    # See score_policy_on_tiny_game: incremental/chunked, not the single-batch
    # materialize_policy, to stay well under a few GB RSS at production dims.
    policy = materialize_policy_incremental(root, nets_by_iter, weighting=weighting, seq_cap=seq_cap)
    nashconv, components = exploitability(root, policy)
    return {
        "nashconv": float(nashconv),
        "components": tuple(float(x) for x in components),
        "num_infosets": len(policy),
        "num_snapshots": len(nets_by_iter),
        "snapshot_iters": [it for it, _ in nets_by_iter],
        "passed": bool(nashconv < X2_NASHCONV_BAR),
    }
