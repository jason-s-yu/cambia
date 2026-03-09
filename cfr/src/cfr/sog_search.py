"""
src/cfr/sog_search.py

SoG (Student of Games) search engine, Phase 3.

Extends GT-CFR search with:
  - Continual re-solving: tree persistence across sequential decisions
  - Budget decoupling: separate train/eval expansion budgets
  - Safety check: commitment value comparison from prior search

SoGSearch COMPOSES GTCFRSearch (does not subclass or duplicate it).
All tree operations delegate to self._inner (GTCFRSearch) methods.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from ..encoding import NUM_ACTIONS
from ..networks import CVPN
from ..pbs import NUM_HAND_TYPES
from .gtcfr_search import GTCFRSearch, GTCFRNode, SearchResult

logger = logging.getLogger(__name__)

VALUE_DIM: int = 2 * NUM_HAND_TYPES  # 936


class SoGSearch:
    """SoG search engine with continual re-solving and budget decoupling.

    Composes GTCFRSearch internally. At each search() call:
      1. Re-solve path (if prior_tree provided and action_taken in children):
         a. Extract subtree at action_taken
         b. Free root engine handle + all sibling subtrees
         c. Depth-prune to max_persist_depth relative to subtree root
         d. Handle-count check: fall back to fresh if exceeded
         e. Run expansion loop on subtree
         f. Safety check: retain prior strategy if values regressed vs commitment
      2. Fresh path (no prior tree, action not in children, or handle fallback):
         Builds tree from scratch (GTCFRSearch logic) without finally-cleanup.

    Tree is stored in self._last_tree for persistence across calls.
    """

    def __init__(
        self,
        cvpn: CVPN,
        train_budget: int = 50,
        eval_budget: int = 200,
        c_puct: float = 2.0,
        cfr_iters_per_expansion: int = 10,
        expansion_k: int = 3,
        device: str = "cpu",
        max_persist_depth: int = 8,
        max_persist_handles: int = 512,
        safety_margin: float = 0.01,
        safety_check_enabled: bool = True,
    ):
        self._cvpn = cvpn
        self._train_budget = train_budget
        self._eval_budget = eval_budget
        self._c_puct = c_puct
        self._cfr_iters = cfr_iters_per_expansion
        self._expansion_k = expansion_k
        self._device = device
        self._max_persist_depth = max_persist_depth
        self._max_persist_handles = max_persist_handles
        self._safety_margin = safety_margin
        self._safety_check_enabled = safety_check_enabled
        self._current_budget = train_budget
        self._last_tree: Optional[GTCFRNode] = None
        self._inner: GTCFRSearch = self._build_inner()

    # ---------------------------------------------------------------------------
    # Budget toggling
    # ---------------------------------------------------------------------------

    def use_train_budget(self) -> None:
        """Switch to training budget (smaller, faster)."""
        if self._current_budget != self._train_budget:
            self._current_budget = self._train_budget
            self._inner = self._build_inner()

    def use_eval_budget(self) -> None:
        """Switch to eval budget (larger, higher quality)."""
        if self._current_budget != self._eval_budget:
            self._current_budget = self._eval_budget
            self._inner = self._build_inner()

    def get_tree(self) -> Optional[GTCFRNode]:
        """Return last search tree for persistence (pass as prior_tree next call)."""
        return self._last_tree

    def cleanup(self) -> None:
        """Free all GoEngine handles in the persisted tree."""
        if self._last_tree is not None:
            self._inner._cleanup_tree(self._last_tree)
            self._last_tree = None

    # ---------------------------------------------------------------------------
    # Public search interface
    # ---------------------------------------------------------------------------

    def search(
        self,
        game: Any,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        prior_tree: Optional[GTCFRNode] = None,
        action_taken: Optional[int] = None,
    ) -> SearchResult:
        """Run SoG search with optional tree persistence.

        Args:
            game:         GoEngine at current decision point (not modified).
            range_p0:     (NUM_HAND_TYPES,) range for player 0.
            range_p1:     (NUM_HAND_TYPES,) range for player 1.
            prior_tree:   GTCFRNode from previous search (or None for fresh).
            action_taken: Action index taken from prior_tree root (or None).

        Returns:
            SearchResult with policy, root_values, tree_size, depth_stats.
        """
        # Attempt re-solve path
        if (
            prior_tree is not None
            and action_taken is not None
            and action_taken in prior_tree.children
        ):
            result = self._resolve_search(
                game, range_p0, range_p1, prior_tree, action_taken
            )
            if result is not None:
                return result
            # Fell back to fresh; prior_tree already cleaned up inside _resolve_search
            return self._fresh_search(game, range_p0, range_p1, prior_tree=None)

        # Fresh path (clean up prior_tree if provided but not usable)
        return self._fresh_search(game, range_p0, range_p1, prior_tree=prior_tree)

    # ---------------------------------------------------------------------------
    # Re-solve path
    # ---------------------------------------------------------------------------

    def _resolve_search(
        self,
        game: Any,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        prior_tree: GTCFRNode,
        action_taken: int,
    ) -> Optional[SearchResult]:
        """Continue search from action_taken subtree.

        Returns None if fallback to fresh search is needed.
        """
        # Extract subtree
        subtree = prior_tree.children[action_taken]

        # Free root engine handle
        if prior_tree.engine_handle is not None:
            prior_tree.engine_handle.close()
            prior_tree.engine_handle = None

        # Free all sibling subtrees
        for a, sibling in list(prior_tree.children.items()):
            if a != action_taken:
                self._inner._cleanup_tree(sibling)
        prior_tree.children.clear()

        # Depth-prune subtree relative to its new root position
        self._prune_depth(subtree, relative_depth=0)

        # Handle count check: fall back to fresh if too many nodes
        node_count = self._inner._count_nodes(subtree)
        if node_count > self._max_persist_handles:
            logger.debug(
                "SoGSearch: subtree has %d nodes (> %d max), falling back to fresh.",
                node_count, self._max_persist_handles,
            )
            self._inner._cleanup_tree(subtree)
            return None

        # Terminal subtree root: return immediately
        if subtree.is_terminal:
            util = (
                subtree.terminal_values
                if subtree.terminal_values is not None
                else np.zeros(2, dtype=np.float32)
            )
            cfvs = np.tile(util[:, None], (1, NUM_HAND_TYPES)).astype(np.float32)
            self._last_tree = subtree
            return SearchResult(
                policy=subtree.average_strategy(),
                root_values=cfvs.ravel(),
                tree_size=1,
                depth_stats={"min": 0, "max": 0, "mean": 0.0},
            )

        # Need engine handle to re-solve
        if subtree.engine_handle is None:
            return None

        # Save prior strategy (before new CFR iterations) for safety check
        prior_strategy = subtree.average_strategy().copy()

        # Store commitment value from CVPN evaluation at expansion time
        commitment_v0: Optional[float] = None
        commitment_v1: Optional[float] = None
        if subtree.leaf_values is not None:
            commitment_v0 = float(np.dot(range_p0, subtree.leaf_values[0]))
            commitment_v1 = float(np.dot(range_p1, subtree.leaf_values[1]))

        # Run expansion loop on subtree
        reach = np.ones(2, dtype=np.float32)
        last_cfvs: np.ndarray = (
            subtree.leaf_values.copy()
            if subtree.leaf_values is not None
            else np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)
        )

        subtree_game = subtree.engine_handle
        for _ in range(self._current_budget):
            for _ in range(self._cfr_iters):
                last_cfvs = self._inner._cfr_traverse(
                    subtree, reach, range_p0, range_p1
                )
            self._inner._expand_once(subtree, subtree_game, range_p0, range_p1)

        # Collect depth stats
        depths: List[int] = []
        self._inner._collect_depths(subtree, depths)

        new_policy = subtree.average_strategy()

        # Safety check: retain prior strategy if acting player's value regressed.
        # Disabled during training (budget too low to beat trained CVPN estimates,
        # causing 100% rejection and blocking all policy learning signal).
        # Active at eval/play for deployment soundness.
        if (self._safety_check_enabled
                and commitment_v0 is not None and commitment_v1 is not None):
            new_v0 = float(np.dot(range_p0, last_cfvs[0]))
            new_v1 = float(np.dot(range_p1, last_cfvs[1]))
            ap = subtree.acting_player
            if ap == 0:
                regressed = new_v0 < commitment_v0 - self._safety_margin
            elif ap == 1:
                regressed = new_v1 < commitment_v1 - self._safety_margin
            else:
                regressed = False  # terminal or unknown: skip check
            if regressed:
                logger.warning(
                    "SoGSearch safety check (player %d): new=(%.4f,%.4f) "
                    "< commit=(%.4f,%.4f) margin=%.4f. Retaining prior strategy.",
                    ap, new_v0, new_v1, commitment_v0, commitment_v1,
                    self._safety_margin,
                )
                new_policy = prior_strategy

        self._last_tree = subtree
        return SearchResult(
            policy=new_policy,
            root_values=last_cfvs.ravel().astype(np.float32),
            tree_size=self._inner._count_nodes(subtree),
            depth_stats={
                "min": int(min(depths)) if depths else 0,
                "max": int(max(depths)) if depths else 0,
                "mean": float(np.mean(depths)) if depths else 0.0,
            },
        )

    # ---------------------------------------------------------------------------
    # Fresh search path
    # ---------------------------------------------------------------------------

    def _fresh_search(
        self,
        game: Any,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        prior_tree: Optional[GTCFRNode] = None,
    ) -> SearchResult:
        """Build a new search tree from scratch without freeing it on exit.

        Mirrors GTCFRSearch.search() but retains the tree in self._last_tree
        instead of cleaning up in a finally block.
        """
        # Clean up any prior tree passed in that was not usable
        if prior_tree is not None:
            self._inner._cleanup_tree(prior_tree)

        # Clean up any previous last tree
        if self._last_tree is not None:
            self._inner._cleanup_tree(self._last_tree)
            self._last_tree = None

        root: Optional[GTCFRNode] = None
        root_game_clone = None
        stored = False

        try:
            root_game_clone = self._inner._clone_engine(game)

            # Terminal root
            if root_game_clone.is_terminal():
                util = root_game_clone.get_utility().astype(np.float32)
                root = self._inner._make_terminal_node(0, root_game_clone, util)
                root_game_clone = None  # now owned by root.engine_handle
                cfvs = np.tile(util[:, None], (1, NUM_HAND_TYPES)).astype(np.float32)
                self._last_tree = root
                stored = True
                return SearchResult(
                    policy=np.zeros(NUM_ACTIONS, dtype=np.float32),
                    root_values=cfvs.ravel(),
                    tree_size=1,
                    depth_stats={"min": 0, "max": 0, "mean": 0.0},
                )

            # Evaluate root with CVPN
            root_leaf_values, root_prior = self._inner._evaluate_node(
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
            root_game_clone = None  # now owned by root.engine_handle

            # Main search loop
            reach = np.ones(2, dtype=np.float32)
            last_cfvs = root_leaf_values.copy()

            for _ in range(self._current_budget):
                for _ in range(self._cfr_iters):
                    last_cfvs = self._inner._cfr_traverse(
                        root, reach, range_p0, range_p1
                    )
                self._inner._expand_once(root, root.engine_handle, range_p0, range_p1)

            depths: List[int] = []
            self._inner._collect_depths(root, depths)

            self._last_tree = root
            stored = True
            return SearchResult(
                policy=root.average_strategy(),
                root_values=last_cfvs.ravel().astype(np.float32),
                tree_size=self._inner._count_nodes(root),
                depth_stats={
                    "min": int(min(depths)) if depths else 0,
                    "max": int(max(depths)) if depths else 0,
                    "mean": float(np.mean(depths)) if depths else 0.0,
                },
            )

        except Exception:
            if not stored:
                if root is not None:
                    self._inner._cleanup_tree(root)
                elif root_game_clone is not None:
                    root_game_clone.close()
            self._last_tree = None
            raise

    # ---------------------------------------------------------------------------
    # Depth pruning
    # ---------------------------------------------------------------------------

    def _prune_depth(self, node: GTCFRNode, relative_depth: int) -> None:
        """Prune children at or beyond max_persist_depth (relative to subtree root).

        At the cap depth, clears all children and marks node unexpanded so
        the re-solve expansion loop can grow from there.
        """
        if not node.children:
            return
        if relative_depth >= self._max_persist_depth:
            for child in node.children.values():
                self._inner._cleanup_tree(child)
            node.children.clear()
            node.is_expanded = False
            return
        for child in node.children.values():
            self._prune_depth(child, relative_depth + 1)

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _build_inner(self) -> GTCFRSearch:
        """Construct GTCFRSearch with current budget settings."""
        return GTCFRSearch(
            cvpn=self._cvpn,
            expansion_budget=self._current_budget,
            c_puct=self._c_puct,
            cfr_iters_per_expansion=self._cfr_iters,
            expansion_k=self._expansion_k,
            device=self._device,
        )
