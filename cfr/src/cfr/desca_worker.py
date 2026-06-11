"""
src/cfr/desca_worker.py

DESCA (Dense ESCHER with Semantic Action Abstraction) worker.

External-sampling MCCFR traversal with an ESCHER-style history-value baseline
over Cambia's 32 abstract-action space. Drives Phase 1 of the v3.1 CFR track.

Public entry point
------------------
``run_desca_iteration(updating_player, regret_net, avg_strategy_net,
history_value_net, *, iteration, config, traversals, device, rng=None)``
runs ``traversals`` independent external-sampling traversals for the given
``updating_player`` and returns three lists of ``ReservoirSample`` objects:
regret samples, strategy samples, and history-value samples. The trainer
buffers them and fits the networks; this module performs no optimization.

Algorithm
---------
Per traversal (cf. spec Section 8.1):

  1. Descend from a freshly dealt game.
  2. At an updating-player infoset I:
     - Compute the 257-dim v2 features of I via
       ``encode_infoset_eppbs_interleaved(..., encoding_version=2)``.
     - Compute a legal-abstract-action mask via
       ``action_abstraction.abstract_actions`` (32-dim bool).
     - Compute the current iteration policy ``sigma^t(I)`` by regret matching
       plus over ``regret_net(features)`` masked to legal abstract actions.
     - For each legal abstract action ``a``:
       * Save engine + agent snapshots.
       * ``unabstract`` to a concrete action and apply it.
       * Compute ``V_omni(h.apply(a))`` by encoding the resulting state's
         fair features (257-dim) and omniscient hidden cards (120-dim) and
         calling ``history_value_net`` in fair+omniscient mode.
       * Restore engine + agent snapshots.
       * Store v_hat[a].
     - Compute ESCHER regret targets:
       ``r_hat[a] = v_hat[a] - sum_b sigma[b] * v_hat[b]``.
       Buffer (features, r_hat, abstract_mask, iteration) into regret samples.
       Buffer (features, sigma, abstract_mask, iteration) into strategy samples.
     - Sample a single abstract action from ``sigma`` via np.random.choice,
       unabstract it, and recurse.
  3. At an opponent infoset:
     - Compute 257-dim features.
     - Compute policy from ``avg_strategy_net(features)`` masked to legal
       abstract actions. (During warmup or when avg_strategy_net is None,
       fall back to uniform over legal abstract actions.)
     - Sample one abstract action, unabstract, recurse. No regret or strategy
       samples are stored at opponent nodes.
  4. At a non-acting state (Cambia has no explicit chance nodes distinct from
     acting-player nodes; stock draws are resolved inside the engine's RNG
     when an action is applied), we treat the acting player as authoritative.
  5. At any visited state (regardless of acting player), also buffer a
     history-value sample: ``(257 + 120 concat features, terminal_payoff_p)``
     where ``terminal_payoff_p`` is the updating player's utility of the
     terminal reached by the current traversal. Buffered after the recursion
     returns so the target is ground truth.

Notes
-----
- Chance-node sampling is handled implicitly by the Go engine's RNG at draw
  actions; one stochastic sample per engine rollout satisfies spec 8.1 (3).
- Regret and strategy samples live only at updating-player infosets.
- V_omni is an asymmetric perfect-info critic: it sees all players' cards.
  Its gradients never reach the acting nets; the trainer optimizes V_omni
  with a separate optimizer on the history-value reservoir.
- Warmup: when ``iteration <= warmup_iters``, sample actions uniformly at
  both traverser and opponent nodes. Regret and strategy samples are still
  buffered to seed the reservoirs with coverage.
- Outputs are ``ReservoirSample`` objects; regret/strategy buffers use
  ``target_dim = NUM_ABSTRACT_ACTIONS_2P = 32``; the V_omni buffer uses
  ``input_dim = 257 + 120 = 377, target_dim = 1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard requirement at runtime
    torch = None  # type: ignore[assignment]

from ..action_abstraction import NUM_ABSTRACT_ACTIONS_2P, abstract_actions, unabstract
from ..agent_state import AgentState
from ..constants import GameAction
from ..encoding import encode_infoset_eppbs_interleaved_v2
from ..reservoir import ReservoirSample
from .omniscient import compute_omniscient_features, omniscient_dim

logger = logging.getLogger(__name__)

# Module-level once-only flag for the omniscient zero-fallback warning.
# Trips the first time _encode_omniscient is forced to return all zeros so the
# silent fallthrough does not mask a misconfigured env_factory in production
# (the bug landed during the Apr 2026 ablation launch, see Phase 1 audit).
_OMNISCIENT_FALLBACK_WARNED: bool = False

# Shared dims (keep in sync with encoding.py / omniscient.py).
FEATURE_DIM: int = 257
OMNISCIENT_DIM_2P: int = 2 * 6 * 10  # 120
VALUE_INPUT_DIM: int = FEATURE_DIM + OMNISCIENT_DIM_2P  # 377

# Numerical floor for regret matching normalization.
_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DESCAWorkerResult:
    """Per-traversal sample lists produced by ``run_desca_iteration``.

    Callers (the trainer) extend their reservoirs with each list.
    """

    regret_samples: List[ReservoirSample] = field(default_factory=list)
    strategy_samples: List[ReservoirSample] = field(default_factory=list)
    value_samples: List[ReservoirSample] = field(default_factory=list)
    # Diagnostic counters.
    traversals_started: int = 0
    terminals_reached: int = 0
    nodes_visited: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _regret_matching_plus(regrets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute sigma^t from regret-matching-plus over a legal-action mask.

    Args:
        regrets: (NUM_ABSTRACT_ACTIONS_2P,) float regret vector.
        mask: (NUM_ABSTRACT_ACTIONS_2P,) bool mask (True = legal).

    Returns:
        Probability distribution over abstract actions. Zero everywhere mask
        is False; uniform over legal actions when all masked regrets are
        non-positive.
    """
    clipped = np.where(mask, np.maximum(regrets, 0.0), 0.0).astype(np.float64)
    total = clipped.sum()
    if total < _EPS:
        legal_count = int(mask.sum())
        if legal_count == 0:
            return np.zeros_like(clipped, dtype=np.float64)
        uniform = np.zeros_like(clipped, dtype=np.float64)
        uniform[mask] = 1.0 / legal_count
        return uniform
    return clipped / total


def _masked_softmax(
    logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Stable softmax over legal actions. Zero for illegal actions."""
    x = logits.astype(np.float64)
    if temperature != 1.0 and temperature > 0.0:
        x = x / float(temperature)
    x = np.where(mask, x, -np.inf)
    if not np.any(np.isfinite(x)):
        legal_count = int(mask.sum())
        out = np.zeros_like(x, dtype=np.float64)
        if legal_count:
            out[mask] = 1.0 / legal_count
        return out
    x = x - np.max(x[mask])
    ex = np.where(mask, np.exp(x), 0.0)
    total = ex.sum()
    if total < _EPS:
        legal_count = int(mask.sum())
        out = np.zeros_like(x, dtype=np.float64)
        if legal_count:
            out[mask] = 1.0 / legal_count
        return out
    return ex / total


def _uniform(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=np.float64)
    legal_count = int(mask.sum())
    if legal_count:
        out[mask] = 1.0 / legal_count
    return out


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _forward_np(net: Any, x: np.ndarray, device: Any, use_bf16: bool = False) -> np.ndarray:
    """Run a torch ``nn.Module`` on a single-batch numpy input and return numpy.

    Returns a 1-D array over the module's output dimension.

    When ``use_bf16`` is True, wraps the forward in ``torch.autocast`` with
    dtype bfloat16 for ~16% speedup on Arc and compatible hardware. Falls back
    to fp32 silently if the device type doesn't appear in the autocast registry
    (e.g. old CPUs that lack bf16 ISA support).
    """
    if torch is None:
        raise RuntimeError("torch is required for DESCA worker network inference")
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    device_type = device.type if hasattr(device, "type") else "cpu"
    with torch.no_grad():
        if use_bf16:
            try:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                    out = net(t)
            except RuntimeError:
                # Device doesn't support bf16 autocast (e.g. older CPUs); fall back to fp32.
                out = net(t)
        else:
            out = net(t)
    return out.squeeze(0).detach().cpu().float().numpy()


def _forward_masked_softmax_net(
    net: Any, x: np.ndarray, mask: np.ndarray, device: Any, use_bf16: bool = False
) -> np.ndarray:
    """Forward an ``AvgStrategyNetwork`` that accepts (x, mask) and returns probs.

    Falls back to softmax(logits) masking if the network's forward signature
    does not accept a mask kwarg.

    When ``use_bf16`` is True, wraps the forward in ``torch.autocast`` with
    dtype bfloat16. Falls back to fp32 on devices without bf16 support.
    """
    if torch is None:
        raise RuntimeError("torch is required for DESCA worker network inference")
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    m = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_)).unsqueeze(0)
    if device is not None:
        t = t.to(device)
        m = m.to(device)
    device_type = device.type if hasattr(device, "type") else "cpu"
    with torch.no_grad():
        if use_bf16:
            try:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                    try:
                        out = net(t, m)
                    except TypeError:
                        out = net(t)
            except RuntimeError:
                try:
                    out = net(t, m)
                except TypeError:
                    out = net(t)
        else:
            try:
                out = net(t, m)
            except TypeError:
                out = net(t)
    probs = out.squeeze(0).detach().cpu().float().numpy().astype(np.float64)
    # Re-mask for safety: any downstream consumers expect zeros on illegal slots.
    probs = np.where(mask, probs, 0.0)
    total = probs.sum()
    if total < _EPS:
        return _uniform(mask)
    return probs / total


def _forward_value_net(
    net: Any,
    fair_features: np.ndarray,
    omni_features: Optional[np.ndarray],
    device: Any,
    use_bf16: bool = False,
) -> float:
    """Call ``HistoryValueNetwork`` and return a scalar value.

    When ``use_bf16`` is True, wraps the forward in ``torch.autocast`` with
    dtype bfloat16. Falls back to fp32 on devices without bf16 support.
    """
    if torch is None:
        raise RuntimeError("torch is required for DESCA worker network inference")
    t = torch.from_numpy(np.ascontiguousarray(fair_features, dtype=np.float32)).unsqueeze(0)
    if omni_features is not None:
        h = torch.from_numpy(np.ascontiguousarray(omni_features, dtype=np.float32)).unsqueeze(0)
    else:
        h = None
    if device is not None:
        t = t.to(device)
        if h is not None:
            h = h.to(device)
    device_type = device.type if hasattr(device, "type") else "cpu"
    with torch.no_grad():
        if use_bf16:
            try:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                    try:
                        out = net(t, h)
                    except TypeError:
                        out = net(t)
            except RuntimeError:
                try:
                    out = net(t, h)
                except TypeError:
                    out = net(t)
        else:
            try:
                out = net(t, h)
            except TypeError:
                out = net(t)
    v = out.squeeze().detach().cpu().float()
    try:
        return float(v.item())
    except Exception:
        return float(v.numpy().reshape(-1)[0])


# ---------------------------------------------------------------------------
# Engine adapter
# ---------------------------------------------------------------------------


def _legal_concrete_actions(engine: Any, agent: Any) -> List[GameAction]:
    """Return the list of concrete ``GameAction`` objects legal at the current state.

    This module supports two backends:
      * Python ``CambiaGameState`` via ``engine.get_legal_actions()``.
      * Go ``GoEngine`` via ``engine.legal_actions_mask()`` and a decoder
        provided by the trainer through ``engine._decode_action``.

    Callers pass a backend adapter with a ``legal_actions`` method; the
    worker never assumes which backend is in use.
    """
    fn = getattr(engine, "legal_actions", None)
    if callable(fn):
        return list(fn())
    raise RuntimeError(
        "Engine adapter must expose legal_actions() returning "
        "a list of concrete GameAction objects"
    )


# ---------------------------------------------------------------------------
# Core traversal
# ---------------------------------------------------------------------------


@dataclass
class _TraversalCtx:
    """Bundle of all per-traversal state passed between recursive calls."""

    updating_player: int
    regret_net: Any
    avg_strategy_net: Any
    history_value_net: Any
    iteration: int
    device: Any
    rng: np.random.Generator
    warmup: bool
    result: DESCAWorkerResult
    use_bf16: bool = True
    depth: int = 0
    recursion_limit: int = 512


def _encode_state(engine: Any, agent: Any) -> np.ndarray:
    """Encode a 257-dim v2 feature vector for ``agent`` at the current engine state."""
    decision_context = engine.get_decision_context()
    drawn_bucket = engine.get_drawn_card_bucket()
    return encode_infoset_eppbs_interleaved_v2(
        agent, decision_context, drawn_bucket
    ).astype(np.float32)


def _encode_omniscient(engine: Any) -> np.ndarray:
    """Return the 120-dim (for 2P) omniscient feature vector.

    The ``compute_omniscient_features`` helper requires a Go engine because it
    reads ground-truth hand card identities via the FFI. For a pure-Python
    engine (tests), the trainer injects ``engine._omniscient_features``
    returning the appropriate vector, so this function prefers that attribute
    when available.
    """
    cached = getattr(engine, "_omniscient_features", None)
    if callable(cached):
        return cached().astype(np.float32, copy=False)
    try:
        return compute_omniscient_features(engine).astype(np.float32, copy=False)
    except Exception as exc:
        global _OMNISCIENT_FALLBACK_WARNED
        if not _OMNISCIENT_FALLBACK_WARNED:
            logger.warning(
                "DESCA: omniscient features unavailable on engine type %s "
                "(no _omniscient_features and compute_omniscient_features raised: %s). "
                "Falling back to zero vector; V_omni critic will be NON-ASYMMETRIC. "
                "This is an env_factory misconfiguration, not a runtime fault. "
                "Suppressing further warnings this process.",
                type(engine).__name__,
                exc,
            )
            _OMNISCIENT_FALLBACK_WARNED = True
        num_players = getattr(engine, "num_players", 2) or 2
        dim = omniscient_dim(int(num_players))
        return np.zeros(dim, dtype=np.float32)


def _policy_at_infoset(
    ctx: _TraversalCtx,
    features: np.ndarray,
    mask: np.ndarray,
    is_traverser: bool,
) -> np.ndarray:
    """Compute the sampling distribution at the current infoset.

    Traverser: regret matching plus on ``regret_net`` output.
    Opponent: masked softmax (or native probs) on ``avg_strategy_net``.
    Warmup or missing nets: uniform over legal abstract actions.
    """
    if ctx.warmup:
        return _uniform(mask)

    if is_traverser:
        if ctx.regret_net is None:
            return _uniform(mask)
        try:
            regrets = _forward_np(ctx.regret_net, features, ctx.device, use_bf16=ctx.use_bf16)
        except Exception as e:
            logger.debug("regret_net forward failed: %s", e)
            return _uniform(mask)
        return _regret_matching_plus(regrets, mask)

    if ctx.avg_strategy_net is None:
        return _uniform(mask)
    try:
        return _forward_masked_softmax_net(
            ctx.avg_strategy_net, features, mask, ctx.device, use_bf16=ctx.use_bf16
        )
    except Exception as e:
        logger.debug("avg_strategy_net forward failed: %s", e)
        return _uniform(mask)


def _evaluate_v_omni(
    ctx: _TraversalCtx, engine: Any, agent_for_features: Any
) -> float:
    """Compute V_omni on the current engine state.

    Encodes 257-dim fair features for ``agent_for_features`` and 120-dim
    omniscient features, calls the history-value net, returns the scalar.
    Signed so it represents the updating player's expected terminal utility.
    """
    if ctx.history_value_net is None:
        return 0.0
    try:
        fair = _encode_state(engine, agent_for_features)
        omni = _encode_omniscient(engine)
        return _forward_value_net(ctx.history_value_net, fair, omni, ctx.device, use_bf16=ctx.use_bf16)
    except Exception as e:
        logger.debug("history_value_net forward failed: %s", e)
        return 0.0


def _sample_index(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Sample a categorical index from a (possibly zero-sum) probability vector."""
    total = float(probs.sum())
    if total < _EPS:
        # Fall back to uniform over nonzero support
        nonzero = np.where(probs >= 0)[0]
        if len(nonzero) == 0:
            return 0
        return int(rng.choice(nonzero))
    p = probs / total
    return int(rng.choice(len(p), p=p))


def _traverse(ctx: _TraversalCtx, engine: Any, agents: Sequence[Any]) -> float:
    """Recursive external-sampling traversal; returns terminal payoff for updating player."""
    ctx.result.nodes_visited += 1

    if ctx.depth >= ctx.recursion_limit:
        logger.warning("DESCA traversal hit recursion limit %d", ctx.recursion_limit)
        ctx.result.errors += 1
        return 0.0

    if engine.is_terminal():
        ctx.result.terminals_reached += 1
        return float(engine.get_utility()[ctx.updating_player])

    try:
        acting_player = engine.get_acting_player()
    except Exception as e:
        logger.warning("DESCA: failed to get acting player: %s", e)
        ctx.result.errors += 1
        return 0.0

    agent = agents[acting_player]

    try:
        legal = _legal_concrete_actions(engine, agent)
    except Exception as e:
        logger.warning("DESCA: failed to fetch legal actions: %s", e)
        ctx.result.errors += 1
        return 0.0

    if not legal:
        # No legal actions on a non-terminal state: treat as terminal.
        return float(engine.get_utility()[ctx.updating_player])

    mask = abstract_actions(legal, agent)
    if not mask.any():
        # Action abstraction did not map any legal action. Sample uniformly
        # over concrete actions and recurse; do not buffer samples.
        idx = int(ctx.rng.integers(0, len(legal)))
        engine.apply_action(legal[idx])
        for a in agents:
            a.update(engine)
        value = _traverse_from_child(ctx, engine, agents)
        return value

    features = _encode_state(engine, agent)
    sigma = _policy_at_infoset(
        ctx, features, mask, is_traverser=(acting_player == ctx.updating_player)
    )

    if acting_player == ctx.updating_player:
        # Evaluate V_omni for each legal abstract action by applying the
        # unabstracted concrete action, evaluating V_omni on the resulting
        # state, and rolling back via snapshot.
        #
        # Batched V_omni: collect (fair_features, omni_features) for all
        # non-terminal legal branches, then do ONE batched forward over all
        # of them instead of ~32 sequential single-sample forwards.
        # RNG calls to unabstract happen in the same order as the sequential
        # path (per-a_idx in ascending order) to preserve rng determinism.
        v_hat = np.zeros(NUM_ABSTRACT_ACTIONS_2P, dtype=np.float64)
        # _batch_indices: list of a_idx values whose V_omni needs the network.
        # _batch_fair / _batch_omni: accumulated feature rows for those entries.
        _batch_indices: List[int] = []
        _batch_fair: List[np.ndarray] = []
        _batch_omni: List[np.ndarray] = []

        for a_idx in range(NUM_ABSTRACT_ACTIONS_2P):
            if not mask[a_idx]:
                continue
            try:
                concrete = unabstract(
                    a_idx,
                    legal,
                    agent,
                    seed=int(ctx.rng.integers(0, 2**31 - 1)),
                )
            except ValueError:
                # Abstract class had no concrete action in this state; mark as
                # illegal and skip.
                mask[a_idx] = False
                continue

            snap = _snapshot(engine, agents)
            try:
                engine.apply_action(concrete)
                for a in agents:
                    a.update(engine)
                if engine.is_terminal():
                    # Terminal: write value directly; skip batched forward.
                    v_hat[a_idx] = float(
                        engine.get_utility()[ctx.updating_player]
                    )
                else:
                    # Non-terminal: encode features and defer forward to batch.
                    if ctx.history_value_net is not None:
                        try:
                            fair = _encode_state(engine, agents[ctx.updating_player])
                            omni = _encode_omniscient(engine)
                            _batch_indices.append(a_idx)
                            _batch_fair.append(fair)
                            _batch_omni.append(omni)
                        except Exception as enc_e:
                            logger.debug("DESCA: v_hat feature encode failed: %s", enc_e)
                            ctx.result.errors += 1
                    # If no history_value_net, v_hat[a_idx] stays 0.0 (fallback).
            except Exception as e:
                logger.debug("DESCA: v_hat branch failed: %s", e)
                ctx.result.errors += 1
            finally:
                _restore(engine, agents, snap)

        # Batched V_omni forward: one forward over all deferred non-terminal rows.
        if _batch_indices and ctx.history_value_net is not None:
            try:
                fair_stack = np.stack(_batch_fair, axis=0)   # [B, 257]
                omni_stack = np.stack(_batch_omni, axis=0)   # [B, 120]
                combined = np.concatenate([fair_stack, omni_stack], axis=1)  # [B, 377]
                t_in = torch.from_numpy(np.ascontiguousarray(combined, dtype=np.float32))
                if ctx.device is not None:
                    t_in = t_in.to(ctx.device)
                device_type = ctx.device.type if hasattr(ctx.device, "type") else "cpu"
                with torch.no_grad():
                    if ctx.use_bf16:
                        try:
                            with torch.autocast(
                                device_type=device_type,
                                dtype=torch.bfloat16,
                                enabled=True,
                            ):
                                # HistoryValueNetwork.forward(x, hidden_cards):
                                # split inside the net; pass combined as x with None hidden.
                                # But the network expects (fair, omni) split -- pass both.
                                t_fair = t_in[:, :FEATURE_DIM]
                                t_omni = t_in[:, FEATURE_DIM:]
                                try:
                                    out = ctx.history_value_net(t_fair, t_omni)
                                except TypeError:
                                    out = ctx.history_value_net(t_in)
                        except RuntimeError:
                            t_fair = t_in[:, :FEATURE_DIM]
                            t_omni = t_in[:, FEATURE_DIM:]
                            try:
                                out = ctx.history_value_net(t_fair, t_omni)
                            except TypeError:
                                out = ctx.history_value_net(t_in)
                    else:
                        t_fair = t_in[:, :FEATURE_DIM]
                        t_omni = t_in[:, FEATURE_DIM:]
                        try:
                            out = ctx.history_value_net(t_fair, t_omni)
                        except TypeError:
                            out = ctx.history_value_net(t_in)
                # out: [B, 1] or [B] -> scatter back into v_hat
                vals = out.squeeze(-1).detach().cpu().float().numpy()  # [B]
                for batch_pos, a_idx in enumerate(_batch_indices):
                    v_hat[a_idx] = float(vals[batch_pos])
            except Exception as e:
                logger.debug("DESCA: batched V_omni forward failed; falling back to zeros: %s", e)
                ctx.result.errors += 1

        # After possibly masking off a_idx in the loop, renormalize sigma.
        if not mask.any():
            # All branches failed; fall back to uniform recursion.
            idx = int(ctx.rng.integers(0, len(legal)))
            engine.apply_action(legal[idx])
            for a in agents:
                a.update(engine)
            return _traverse_from_child(ctx, engine, agents)
        sigma = np.where(mask, sigma, 0.0)
        total = sigma.sum()
        sigma = sigma / total if total > _EPS else _uniform(mask)

        baseline = float(np.sum(sigma * v_hat))
        r_hat = np.where(mask, v_hat - baseline, 0.0).astype(np.float32)

        ctx.result.regret_samples.append(
            ReservoirSample(
                features=features.astype(np.float32, copy=False),
                target=r_hat,
                action_mask=mask.astype(np.bool_, copy=True),
                iteration=ctx.iteration,
            )
        )
        ctx.result.strategy_samples.append(
            ReservoirSample(
                features=features.astype(np.float32, copy=False),
                target=sigma.astype(np.float32, copy=False),
                action_mask=mask.astype(np.bool_, copy=True),
                iteration=ctx.iteration,
            )
        )

        # Sample one action to recurse.
        a_sampled_abstract = _sample_index(ctx.rng, sigma)
        try:
            concrete = unabstract(
                a_sampled_abstract,
                legal,
                agent,
                seed=int(ctx.rng.integers(0, 2**31 - 1)),
            )
        except ValueError:
            # Extremely rare: renormalization picked a masked-off class. Bail
            # to uniform concrete.
            concrete = legal[int(ctx.rng.integers(0, len(legal)))]

        # Capture omniscient features BEFORE applying so the V buffer target
        # is the terminal payoff that follows from THIS state.
        omni_features_here = _encode_omniscient(engine)
        engine.apply_action(concrete)
        for a in agents:
            a.update(engine)
        value = _traverse_from_child(ctx, engine, agents)

        # Buffer history-value sample with the 377-dim (features + omni) input
        # and the realized terminal-payoff target.
        v_input = np.concatenate([features, omni_features_here]).astype(np.float32)
        ctx.result.value_samples.append(
            ReservoirSample(
                features=v_input,
                target=np.array([value], dtype=np.float32),
                action_mask=np.empty(0, dtype=np.bool_),
                iteration=ctx.iteration,
            )
        )
        return value

    # Opponent node: sample one abstract action from sigma, recurse.
    a_sampled_abstract = _sample_index(ctx.rng, sigma)
    try:
        concrete = unabstract(
            a_sampled_abstract,
            legal,
            agent,
            seed=int(ctx.rng.integers(0, 2**31 - 1)),
        )
    except ValueError:
        concrete = legal[int(ctx.rng.integers(0, len(legal)))]

    engine.apply_action(concrete)
    for a in agents:
        a.update(engine)
    return _traverse_from_child(ctx, engine, agents)


def _traverse_from_child(ctx: _TraversalCtx, engine: Any, agents: Sequence[Any]) -> float:
    """Helper that bumps depth before recursion and decrements on return."""
    ctx.depth += 1
    try:
        return _traverse(ctx, engine, agents)
    finally:
        ctx.depth -= 1


def _snapshot(engine: Any, agents: Sequence[Any]) -> Tuple[Any, List[Any]]:
    """Save engine + agent belief snapshots in a backend-agnostic way."""
    eng_snap = engine.save() if hasattr(engine, "save") else None
    agent_snaps = [a.clone() if hasattr(a, "clone") else None for a in agents]
    return eng_snap, agent_snaps


def _restore(engine: Any, agents: Sequence[Any], snap: Tuple[Any, List[Any]]) -> None:
    """Restore engine + agent belief snapshots previously captured by _snapshot."""
    eng_snap, agent_snaps = snap
    try:
        if eng_snap is not None and hasattr(engine, "restore"):
            engine.restore(eng_snap)
        if eng_snap is not None and hasattr(engine, "free_snapshot"):
            try:
                engine.free_snapshot(eng_snap)
            except Exception:
                pass
    except Exception as e:
        logger.debug("DESCA: engine restore failed: %s", e)
    for i, orig in enumerate(agents):
        backup = agent_snaps[i]
        if backup is None:
            continue
        try:
            # For GoAgentState, restoration requires swapping the live instance
            # with the backup; for Python AgentState, a deepcopy-style replace.
            if hasattr(orig, "close") and hasattr(orig, "_agent_h"):
                # Free the current handle and swap the backup's handle into orig.
                live_h = getattr(orig, "_agent_h", -1)
                if live_h >= 0:
                    try:
                        orig._lib.cambia_agent_free(live_h)
                    except Exception:
                        pass
                orig._agent_h = backup._agent_h
                backup._agent_h = -1  # prevent backup __del__ from freeing.
            else:
                agents[i].__dict__.update(backup.__dict__)  # type: ignore[index]
        except Exception as e:
            logger.debug("DESCA: agent restore failed: %s", e)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_desca_iteration(
    env_factory,
    updating_player: int,
    regret_net: Any,
    avg_strategy_net: Any,
    history_value_net: Any,
    *,
    iteration: int,
    traversals: int,
    device: Any = None,
    rng: Optional[np.random.Generator] = None,
    warmup: bool = False,
    recursion_limit: int = 512,
    use_bf16: bool = True,
) -> DESCAWorkerResult:
    """Run ``traversals`` DESCA traversals for ``updating_player``.

    Args:
        env_factory: Callable returning ``(engine, agents)`` for a fresh game.
            The engine exposes ``is_terminal()``, ``get_utility()``,
            ``get_acting_player()``, ``apply_action(concrete_action)``,
            ``legal_actions()``, ``save()``, ``restore(snap)``,
            ``free_snapshot(snap)``, ``get_decision_context()``,
            ``get_drawn_card_bucket()``, and optionally
            ``_omniscient_features()`` for Python-only backends.
            Agents expose ``update(engine)``, ``clone()``, plus the same
            attribute surface as ``src.agent_state.AgentState``.
        updating_player: Player index (0 or 1) whose regret we accumulate.
        regret_net: ``RegretNetwork`` instance (may be None during warmup).
        avg_strategy_net: ``AvgStrategyNetwork`` (may be None during warmup).
        history_value_net: ``HistoryValueNetwork`` (may be None during warmup).
        iteration: CFR iteration count; used as weight=iter in DCFR+ / LinearCFR.
        traversals: Number of independent traversals to perform.
        device: Torch device for network inference. None -> CPU.
        rng: Optional ``np.random.Generator`` for determinism.
        warmup: If True, use uniform policy everywhere; still buffer samples.
        recursion_limit: Safety cap on recursion depth.
        use_bf16: If True, wrap NN forward calls in ``torch.autocast`` with
            dtype=bfloat16 for ~16% speedup on Arc and compatible hardware.
            Falls back silently to fp32 on devices without bf16 support.
            Reads from DESCAConfig.use_bf16_inference when called by trainer.

    Returns:
        ``DESCAWorkerResult`` with three reservoir-sample lists and diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng()

    if updating_player not in (0, 1):
        raise ValueError(f"DESCA worker currently supports 2P only; got updating_player={updating_player}")

    result = DESCAWorkerResult()
    ctx = _TraversalCtx(
        updating_player=updating_player,
        regret_net=regret_net,
        avg_strategy_net=avg_strategy_net,
        history_value_net=history_value_net,
        iteration=iteration,
        device=device,
        rng=rng,
        warmup=warmup,
        result=result,
        use_bf16=use_bf16,
        recursion_limit=recursion_limit,
    )

    for _ in range(traversals):
        result.traversals_started += 1
        try:
            engine, agents = env_factory(rng)
        except TypeError:
            engine, agents = env_factory()
        ctx.depth = 0
        try:
            _traverse(ctx, engine, agents)
        except Exception as e:
            logger.warning("DESCA traversal aborted: %s", e, exc_info=True)
            result.errors += 1
        finally:
            # Caller's env_factory is responsible for providing cleanable
            # engine/agent instances; attempt best-effort close.
            _safe_close(engine, agents)

    return result


def _safe_close(engine: Any, agents: Sequence[Any]) -> None:
    """Best-effort close on engine and agents from a finished traversal."""
    for a in agents or ():
        close = getattr(a, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


__all__ = [
    "DESCAWorkerResult",
    "FEATURE_DIM",
    "OMNISCIENT_DIM_2P",
    "VALUE_INPUT_DIM",
    "run_desca_iteration",
]
