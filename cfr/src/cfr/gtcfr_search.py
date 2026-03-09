"""
src/cfr/gtcfr_search.py

GT-CFR growing-tree search engine for Phase 2.

Implements the GT-CFR algorithm:
  - Maintains a PUBLIC game tree (nodes = public states, not infostates)
  - Alternates between CFR updates on the current tree and PUCT-guided expansion
  - Uses a CVPN (Counterfactual Value-and-Policy Network) to evaluate leaf nodes
  - Returns average strategy and root CFVs after a fixed expansion budget
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..encoding import NUM_ACTIONS
from ..networks import CVPN
from ..pbs import (
    PBS,
    PBS_INPUT_DIM,
    NUM_HAND_TYPES,
    encode_pbs,
    make_public_features,
    PHASE_DRAW,
    PHASE_DISCARD,
    PHASE_ABILITY,
    PHASE_SNAP,
    PHASE_TERMINAL,
)

logger = logging.getLogger(__name__)

# Value target dimension: 2 * NUM_HAND_TYPES = 936
VALUE_DIM: int = 2 * NUM_HAND_TYPES

# Decision context integer → PBS phase index (copied from rebel_worker.py — coexistence policy)
_CTX_TO_PHASE: Dict[int, int] = {
    0: PHASE_DRAW,
    1: PHASE_DISCARD,
    2: PHASE_ABILITY,
    3: PHASE_SNAP,
    4: PHASE_SNAP,
    5: PHASE_TERMINAL,
}

_MAX_TURNS: int = 46
_STOCK_TOTAL: int = 46


def _build_pbs(game: Any, range_p0: np.ndarray, range_p1: np.ndarray) -> PBS:
    """Construct a PBS from a GoEngine state and range distributions.

    Copied from rebel_worker.py (coexistence policy: do not import from rebel).
    """
    ctx = game.decision_ctx()
    phase = _CTX_TO_PHASE.get(ctx, PHASE_DRAW)
    pub = make_public_features(
        turn=game.turn_number(),
        max_turns=_MAX_TURNS,
        phase=phase,
        discard_top_bucket=game.discard_top(),
        stockpile_remaining=game.stock_len(),
        stockpile_total=_STOCK_TOTAL,
    )
    return PBS(
        range_p0=range_p0.copy(),
        range_p1=range_p1.copy(),
        public_features=pub,
    )


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class GTCFRNode:
    """Node in the GT-CFR growing public tree."""

    # Identity
    depth: int
    acting_player: int          # 0 or 1; -1 for terminal nodes
    is_terminal: bool
    terminal_values: Optional[np.ndarray]  # (2,) float32 if terminal

    # Legal actions at this node
    legal_mask: np.ndarray      # (NUM_ACTIONS,) bool
    n_legal: int

    # Children (populated on expansion)
    children: Dict[int, "GTCFRNode"]
    is_expanded: bool

    # CFR state
    cumulative_regret: np.ndarray    # (NUM_ACTIONS,) float32
    cumulative_strategy: np.ndarray  # (NUM_ACTIONS,) float32
    cfr_visits: int

    # PUCT state
    visit_counts: np.ndarray        # (NUM_ACTIONS,) int32
    total_action_value: np.ndarray  # (NUM_ACTIONS,) float32 — sum of values
    policy_prior: np.ndarray        # (NUM_ACTIONS,) float32 — from CVPN

    # CVPN leaf evaluation (stored when first evaluated)
    leaf_values: Optional[np.ndarray]  # (2, NUM_HAND_TYPES) float32

    # Game engine handle (GoEngine instance)
    engine_handle: Any

    def current_strategy(self) -> np.ndarray:
        """Regret-matching strategy from cumulative regrets."""
        pos = np.maximum(self.cumulative_regret, 0.0)
        total = float(pos[self.legal_mask].sum())
        strat = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if total > 1e-10:
            strat[self.legal_mask] = pos[self.legal_mask] / total
        else:
            # Uniform fallback
            if self.n_legal > 0:
                strat[self.legal_mask] = 1.0 / self.n_legal
        return strat

    def average_strategy(self) -> np.ndarray:
        """Time-averaged strategy from cumulative strategy sums."""
        total = float(self.cumulative_strategy[self.legal_mask].sum())
        strat = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if total > 1e-10:
            strat[self.legal_mask] = self.cumulative_strategy[self.legal_mask] / total
        else:
            if self.n_legal > 0:
                strat[self.legal_mask] = 1.0 / self.n_legal
        return strat


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Result of a GT-CFR search."""

    policy: np.ndarray       # (NUM_ACTIONS,) float32 — root average strategy
    root_values: np.ndarray  # (VALUE_DIM,) = (936,) float32 — flattened (2, 468) CFVs
    tree_size: int           # total nodes in tree
    depth_stats: dict        # {"min": int, "max": int, "mean": float}


# ---------------------------------------------------------------------------
# GT-CFR search
# ---------------------------------------------------------------------------


class GTCFRSearch:
    """GT-CFR growing-tree search for imperfect-information games.

    At each search() call:
      1. Clone root game engine and evaluate with CVPN
      2. For each expansion step:
         a. Run cfr_iters_per_expansion CFR iterations on current tree
         b. Run 1 PUCT-guided expansion
      3. Return root average strategy and root CFVs
      4. Free all GoEngine handles

    The tree is a PUBLIC tree — nodes represent public game states.
    CFR produces MIXED strategies (not deterministic like MCTS).
    Ranges (range_p0, range_p1) are passed through but not Bayesian-updated
    during tree traversal (Phase 2 approximation; Phase 3 adds full range updates).
    """

    def __init__(
        self,
        cvpn: CVPN,
        expansion_budget: int = 100,
        c_puct: float = 2.0,
        cfr_iters_per_expansion: int = 10,
        expansion_k: int = 3,
        device: str = "cpu",
    ):
        self._cvpn = cvpn
        self._expansion_budget = expansion_budget
        self._c_puct = c_puct
        self._cfr_iters = cfr_iters_per_expansion
        self._expansion_k = expansion_k
        self._device = device
        self._cvpn.eval()

    def search(
        self,
        game: Any,               # GoEngine at current decision point (not modified)
        range_p0: np.ndarray,    # (NUM_HAND_TYPES,) current range for player 0
        range_p1: np.ndarray,    # (NUM_HAND_TYPES,) current range for player 1
    ) -> SearchResult:
        """Run GT-CFR search from the current game state.

        Clones the engine internally; the provided game is not modified.
        All engine handles are freed in the finally block.
        """
        root: Optional[GTCFRNode] = None
        root_game_clone = None

        try:
            root_game_clone = self._clone_engine(game)

            # Handle terminal root
            if root_game_clone.is_terminal():
                util = root_game_clone.get_utility().astype(np.float32)
                root = self._make_terminal_node(0, root_game_clone, util)
                cfvs = np.tile(util[:, None], (1, NUM_HAND_TYPES)).astype(np.float32)
                return SearchResult(
                    policy=np.zeros(NUM_ACTIONS, dtype=np.float32),
                    root_values=cfvs.ravel(),
                    tree_size=1,
                    depth_stats={"min": 0, "max": 0, "mean": 0.0},
                )

            # Evaluate root with CVPN
            root_leaf_values, root_prior = self._evaluate_node(
                root_game_clone, range_p0, range_p1
            )
            legal_mask = root_game_clone.legal_actions_mask().astype(bool)
            n_legal = int(legal_mask.sum())
            acting = root_game_clone.acting_player()

            root = GTCFRNode(
                depth=0,
                acting_player=acting,
                is_terminal=False,
                terminal_values=None,
                legal_mask=legal_mask,
                n_legal=n_legal,
                children={},
                is_expanded=False,
                cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
                cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
                cfr_visits=0,
                visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
                total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
                policy_prior=root_prior,
                leaf_values=root_leaf_values,
                engine_handle=root_game_clone,
            )

            # Main search loop: interleave CFR and expansion
            reach = np.ones(2, dtype=np.float32)
            last_cfvs = root_leaf_values.copy()

            for _ in range(self._expansion_budget):
                # CFR iterations on current tree
                for _ in range(self._cfr_iters):
                    last_cfvs = self._cfr_traverse(root, reach, range_p0, range_p1)

                # PUCT-guided expansion
                self._expand_once(root, root_game_clone, range_p0, range_p1)

            # Collect depth stats and build result
            depths: List[int] = []
            self._collect_depths(root, depths)

            return SearchResult(
                policy=root.average_strategy(),
                root_values=last_cfvs.ravel().astype(np.float32),
                tree_size=self._count_nodes(root),
                depth_stats={
                    "min": int(min(depths)) if depths else 0,
                    "max": int(max(depths)) if depths else 0,
                    "mean": float(np.mean(depths)) if depths else 0.0,
                },
            )

        finally:
            if root is not None:
                self._cleanup_tree(root)
            elif root_game_clone is not None:
                root_game_clone.close()

    # ---------------------------------------------------------------------------
    # CFR traversal
    # ---------------------------------------------------------------------------

    def _cfr_traverse(
        self,
        node: GTCFRNode,
        reach_probs: np.ndarray,  # (2,) counterfactual reach probabilities
        range_p0: np.ndarray,
        range_p1: np.ndarray,
    ) -> np.ndarray:
        """One CFR iteration on the growing tree.

        Returns:
            np.ndarray of shape (2, NUM_HAND_TYPES) — CFVs at this node.
        """
        # Terminal: broadcast scalar utilities across hand types
        if node.is_terminal:
            util = node.terminal_values  # (2,)
            return np.tile(util[:, None], (1, NUM_HAND_TYPES)).astype(np.float32)

        # Unexpanded leaf: return stored CVPN values
        if not node.is_expanded:
            if node.leaf_values is not None:
                return node.leaf_values.copy()
            return np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)

        # Expanded but no children (e.g. no legal actions)
        if not node.children:
            if node.leaf_values is not None:
                return node.leaf_values.copy()
            return np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)

        acting = node.acting_player
        strategy = node.current_strategy()  # (NUM_ACTIONS,)

        # Traverse all children, collecting per-child CFVs
        child_cfvs: Dict[int, np.ndarray] = {}
        for a, child in node.children.items():
            new_reach = reach_probs.copy()
            new_reach[acting] *= strategy[a]
            child_cfvs[a] = self._cfr_traverse(child, new_reach, range_p0, range_p1)

        # Node CFV = strategy-weighted sum of children CFVs
        node_cfvs = np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)
        for a, cv in child_cfvs.items():
            node_cfvs += strategy[a] * cv

        # Regret update: scalar per action, range-weighted
        # regret[a] = cf_reach * dot(range[acting], child_cfv[acting] - node_cfv[acting])
        cf_reach = float(reach_probs[1 - acting])
        r = range_p0 if acting == 0 else range_p1

        for a, cv in child_cfvs.items():
            delta = cv[acting] - node_cfvs[acting]  # (NUM_HAND_TYPES,)
            node.cumulative_regret[a] += cf_reach * float(np.dot(r, delta))

        # Strategy sum update (weighted by acting player's reach)
        node.cumulative_strategy += reach_probs[acting] * strategy
        node.cfr_visits += 1

        return node_cfvs

    # ---------------------------------------------------------------------------
    # PUCT-guided expansion
    # ---------------------------------------------------------------------------

    def _expand_once(
        self,
        root: GTCFRNode,
        root_game: Any,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
    ) -> int:
        """Run one PUCT simulation to expand a leaf node.

        Walks down the tree using π_select = 0.5·PUCT + 0.5·CFR.
        When reaching an unexpanded node, expands ALL legal children (k=∞).
        Backpropagates Q-values up the selection path.

        Returns:
            Number of new nodes added to the tree.
        """
        # Walk down to an unexpanded node
        path: List[Tuple[GTCFRNode, int]] = []  # (parent_node, action_idx)
        node = root

        while node.is_expanded and not node.is_terminal and node.children:
            action_idx = self._select_action(node)
            if action_idx not in node.children:
                break
            path.append((node, action_idx))
            node = node.children[action_idx]

        # Can't expand terminal or already-fully-expanded leaf with no children
        if node.is_terminal:
            return 0
        if node.is_expanded:
            return 0

        node_game = node.engine_handle
        if node_game is None:
            return 0

        legal_actions = [a for a in range(NUM_ACTIONS) if node.legal_mask[a]]
        if not legal_actions:
            node.is_expanded = True
            return 0

        if self._expansion_k > 0 and len(legal_actions) > self._expansion_k:
            # Select top-k actions by PUCT score
            puct_scores = self._puct_scores(node)
            legal_puct = [(a, puct_scores[a]) for a in legal_actions]
            legal_puct.sort(key=lambda x: x[1], reverse=True)
            legal_actions = [a for a, _ in legal_puct[:self._expansion_k]]

        # Create child engines for selected legal actions
        child_engines: List[Any] = []
        child_terminal_flags: List[bool] = []
        child_utilities: List[Optional[np.ndarray]] = []

        try:
            for a in legal_actions:
                child_eng = self._make_child_engine(node_game, a)
                is_term = child_eng.is_terminal()
                child_engines.append(child_eng)
                child_terminal_flags.append(is_term)
                if is_term:
                    child_utilities.append(child_eng.get_utility().astype(np.float32))
                else:
                    child_utilities.append(None)
        except Exception:
            # Free any engines created before the failure
            for eng in child_engines:
                try:
                    eng.close()
                except Exception:
                    pass
            raise

        # Batch CVPN evaluation for non-terminal children
        non_term_indices = [i for i, t in enumerate(child_terminal_flags) if not t]
        batch_leaf_values: Dict[int, np.ndarray] = {}
        batch_priors: Dict[int, np.ndarray] = {}

        if non_term_indices:
            pbs_encs: List[np.ndarray] = []
            legal_masks: List[np.ndarray] = []

            for i in non_term_indices:
                eng = child_engines[i]
                pbs = _build_pbs(eng, range_p0, range_p1)
                pbs_encs.append(encode_pbs(pbs))
                legal_masks.append(eng.legal_actions_mask().astype(bool))

            pbs_batch = torch.from_numpy(np.stack(pbs_encs)).float().to(self._device)
            mask_batch = torch.from_numpy(np.stack(legal_masks)).to(self._device)

            with torch.inference_mode():
                values_t, logits_t = self._cvpn(pbs_batch, mask_batch)

            values_np = values_t.cpu().numpy()                       # (N, 936)
            probs_np = F.softmax(logits_t, dim=-1).cpu().numpy()     # (N, 146)

            for j, i in enumerate(non_term_indices):
                batch_leaf_values[i] = values_np[j].reshape(2, NUM_HAND_TYPES)
                batch_priors[i] = probs_np[j].astype(np.float32)

        # Build child nodes
        n_added = 0
        node_value = 0.0  # value of expanded node for PUCT backprop

        for idx, a in enumerate(legal_actions):
            child_eng = child_engines[idx]
            is_term = child_terminal_flags[idx]

            if is_term:
                util = child_utilities[idx]
                child_node = self._make_terminal_node(node.depth + 1, child_eng, util)
            else:
                child_lm = child_eng.legal_actions_mask().astype(bool)
                child_nl = int(child_lm.sum())
                child_acting = child_eng.acting_player()
                leaf_vals = batch_leaf_values.get(idx)
                prior = batch_priors.get(idx, np.zeros(NUM_ACTIONS, dtype=np.float32))

                child_node = GTCFRNode(
                    depth=node.depth + 1,
                    acting_player=child_acting,
                    is_terminal=False,
                    terminal_values=None,
                    legal_mask=child_lm,
                    n_legal=child_nl,
                    children={},
                    is_expanded=False,
                    cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
                    cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
                    cfr_visits=0,
                    visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
                    total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
                    policy_prior=prior,
                    leaf_values=leaf_vals,
                    engine_handle=child_eng,
                )

            node.children[a] = child_node
            n_added += 1

        node.is_expanded = True

        # Value for PUCT backprop: collapse acting player's range-weighted CFV
        if node.leaf_values is not None and node.acting_player >= 0:
            r = range_p0 if node.acting_player == 0 else range_p1
            node_value = float(np.dot(r, node.leaf_values[node.acting_player]))

        # Backprop visit counts and Q-values up the selection path
        for parent_node, action in reversed(path):
            parent_node.visit_counts[action] += 1
            parent_node.total_action_value[action] += node_value

        return n_added

    def _select_action(self, node: GTCFRNode) -> int:
        """π_select = 0.5·PUCT + 0.5·CFR, normalized, then sample."""
        puct_scores = self._puct_scores(node)
        cfr_strategy = node.current_strategy()

        # Normalize PUCT scores to probabilities over legal actions
        puct_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal = node.legal_mask
        if legal.any():
            ls = puct_scores[legal]
            exp = np.exp(ls - ls.max())
            puct_probs[legal] = exp / exp.sum()

        blended = 0.5 * puct_probs + 0.5 * cfr_strategy
        blended[~legal] = 0.0
        total = blended.sum()
        if total < 1e-10:
            blended[legal] = 1.0 / max(node.n_legal, 1)
            total = blended.sum()
        blended /= total

        return int(np.random.choice(NUM_ACTIONS, p=blended))

    def _puct_scores(self, node: GTCFRNode) -> np.ndarray:
        """PUCT(s,a) = Q(s,a) + c_puct·prior(a)·√N_parent / (1 + N(s,a))."""
        N_parent = max(1, int(node.visit_counts.sum()))
        Q = np.where(
            node.visit_counts > 0,
            node.total_action_value / np.maximum(node.visit_counts, 1),
            0.0,
        ).astype(np.float32)
        exploration = (
            self._c_puct
            * node.policy_prior
            * np.sqrt(N_parent)
            / (1.0 + node.visit_counts)
        )
        scores = Q + exploration
        scores[~node.legal_mask] = -1e9
        return scores

    # ---------------------------------------------------------------------------
    # CVPN evaluation
    # ---------------------------------------------------------------------------

    def _evaluate_node(
        self,
        game: Any,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single game state with CVPN.

        Returns:
            (values, policy_prior):
              values:       (2, NUM_HAND_TYPES) float32
              policy_prior: (NUM_ACTIONS,)       float32 probability distribution
        """
        pbs = _build_pbs(game, range_p0, range_p1)
        pbs_enc = encode_pbs(pbs)  # (956,)

        pbs_t = torch.from_numpy(pbs_enc).float().unsqueeze(0).to(self._device)
        mask = (
            torch.from_numpy(game.legal_actions_mask().astype(bool))
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.inference_mode():
            values_t, logits_t = self._cvpn(pbs_t, mask)

        values = values_t[0].cpu().numpy().reshape(2, NUM_HAND_TYPES).astype(np.float32)
        probs = F.softmax(logits_t[0], dim=-1).cpu().numpy().astype(np.float32)
        return values, probs

    # ---------------------------------------------------------------------------
    # Engine management
    # ---------------------------------------------------------------------------

    def _clone_engine(self, engine: Any) -> Any:
        """Clone a GoEngine via save/restore.

        Creates a fresh GoEngine, saves the parent's full GameState (including
        HouseRules) into a snapshot, and restores it into the new handle.
        """
        from ..ffi.bridge import GoEngine

        snap_h = engine.save()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                child = GoEngine(seed=0)
            child.restore(snap_h)
        finally:
            engine.free_snapshot(snap_h)
        return child

    def _make_child_engine(self, parent_engine: Any, action_idx: int) -> Any:
        """Clone parent engine and apply action. Returns new engine."""
        child = self._clone_engine(parent_engine)
        child.apply_action(action_idx)
        return child

    def _cleanup_tree(self, node: GTCFRNode) -> None:
        """Recursively free all GoEngine handles in the tree."""
        if node.engine_handle is not None:
            node.engine_handle.close()
            node.engine_handle = None
        for child in node.children.values():
            self._cleanup_tree(child)

    # ---------------------------------------------------------------------------
    # Tree utilities
    # ---------------------------------------------------------------------------

    def _make_terminal_node(
        self, depth: int, engine: Any, util: np.ndarray
    ) -> GTCFRNode:
        """Create a terminal GTCFRNode."""
        return GTCFRNode(
            depth=depth,
            acting_player=-1,
            is_terminal=True,
            terminal_values=util.astype(np.float32),
            legal_mask=np.zeros(NUM_ACTIONS, dtype=bool),
            n_legal=0,
            children={},
            is_expanded=True,
            cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
            cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
            cfr_visits=0,
            visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
            total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
            policy_prior=np.zeros(NUM_ACTIONS, dtype=np.float32),
            leaf_values=None,
            engine_handle=engine,
        )

    def _count_nodes(self, node: GTCFRNode) -> int:
        """Count total nodes in tree (including root)."""
        return 1 + sum(self._count_nodes(c) for c in node.children.values())

    def _collect_depths(self, node: GTCFRNode, depths: List[int]) -> None:
        """Collect depths of all leaf nodes (no children or terminal)."""
        if not node.children:
            depths.append(node.depth)
        else:
            for child in node.children.values():
                self._collect_depths(child, depths)
