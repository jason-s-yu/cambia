"""Go-engine backend for the tiny-Cambia exact tree (cambia-1429).

tools/tiny_solver.py used to expand the tree by recursing the Python reference
engine. The Go backend (``build_tree(..., backend="go")``, now the default) does
it on GoEngine.from_deck with the token-inclusive state_save/state_restore pair
for backtracking, so the exact-tree consumers -- the X2 gate scorer and the
NashConv certifier -- no longer depend on src.game.engine.

Four things are pinned here:

  1. The deal generator. go_deal_decks reproduces CambiaGameState._setup_game's
     RNG consumption without importing the engine; the test asserts the decks and
     starting seats it emits ARE the engine's own deal, card for card.

  2. Cross-engine tree parity, on config/tiny_norecall.yaml -- the one tiny config
     whose enumerated tree never exhausts the stockpile. Both backends must agree
     on node count, infoset count, and the whole structural profile.

  3. The suit-collapse trap. src.card.Card declares ``suit`` as
     ``field(compare=False)``, so Card objects hash by rank alone; a key component
     holding raw Card objects merges all four suits of a rank. The test asserts
     the trap is real and that _go_card_key sidesteps it, because a regression
     here is silent (the tree stays structurally perfect and only the infoset
     partition collapses).

  4. The {A,6} divergence (the recorded 230,206 / 69,636 counts are NOT
     reproducible on the Go engine). See test_a6_tree_divergence_is_diagnosed for
     the numbers and the mechanism.
"""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest

from src.config import load_config
from src.encoding import NUM_ACTIONS, action_to_index
from src.constants import ActionDrawStockpile
from tools.tiny_solver import (
    TabularCFR,
    build_tree,
    exploitability,
    go_deal_decks,
    _go_action_table,
    _go_actions_from_mask,
    _go_card_key,
)

pytest.importorskip("cffi")

# The reshuffle-free control config: deck {A,6}, ONE card per player, 3 engine
# turns. Its enumerated tree never empties the stockpile, so the deck-order
# channel the Go builder enumerates through stays live for the whole tree and the
# two backends must produce the same tree exactly.
CONTROL_CONFIG = "config/tiny_norecall.yaml"
CONTROL_NODES = 19592
CONTROL_ISETS = 8043

# The X1/X2 config. Its enumerated tree DOES exhaust the stockpile (see the
# divergence test).
A6_CONFIG = "config/tiny_2card_plateau.yaml"

# Recorded (Python-backend) counts for A6_CONFIG at 5 deals / seed0=0, the
# provenance of the X1 number and the X2 bar.
A6_RECORDED_NODES = 230206
A6_RECORDED_ISETS = 69636


def _lib_available() -> bool:
    try:
        from src.ffi import bridge

        bridge._get_lib()
        return True
    except Exception:  # noqa: BLE001 - any load failure means skip
        return False


requires_lib = pytest.mark.skipif(
    not _lib_available(), reason="libcambia.so not loadable"
)


def _build(config, backend, deals=5, **kw):
    cfg = load_config(config)
    return build_tree(
        cfg,
        n_deals=deals,
        seed0=0,
        max_nodes_per_deal=2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        backend=backend,
        **kw,
    )


def _profile(root):
    """Aggregate fingerprint of a tree, independent of key representation.

    Multisets, not shape: node kinds, per-seat decision counts, action-count and
    chance-fan histograms, and the sorted terminal utilities. It deliberately does
    NOT compare the two trees node-for-node in traversal order, because the child
    ORDER of a nested chance node cannot be matched. build_tree_python's
    _draw_chance forces a card to the top by popping it out of its own stockpile
    LIST and appending it, so a chance node deeper in that subtree enumerates over
    an already-permuted list; the Go builder's order comes from its deck array.
    Both enumerate the same cards with the same weights, in a different order. The
    invariants that matter -- counts, partition size, and NashConv -- are asserted
    separately.
    """
    kinds = {"T": 0, "C": 0, "D": 0}
    acting = {}
    nA = {}
    fan = {}
    utils = []
    stack = [root]
    while stack:
        nd = stack.pop()
        kinds[nd.kind] += 1
        if nd.kind == "T":
            utils.append(nd.util)
            continue
        if nd.kind == "C":
            fan[len(nd.children)] = fan.get(len(nd.children), 0) + 1
        else:
            acting[nd.player] = acting.get(nd.player, 0) + 1
            nA[len(nd.actions)] = nA.get(len(nd.actions), 0) + 1
        stack.extend(nd.children)
    return kinds, acting, nA, fan, sorted(utils)


# ---------------------------------------------------------------------------
# 1. The engine-free deal generator IS the engine's deal.
# ---------------------------------------------------------------------------


@requires_lib
@pytest.mark.parametrize("config", [CONTROL_CONFIG, A6_CONFIG])
def test_go_deal_decks_match_the_python_engine_deal(config):
    """go_deal_decks == extract_deck_from_python_game on the engine's own deal.

    The Go tree can only reproduce the recorded trees if it starts from the same
    deals, and build_tree_python gets those from CambiaGameState's internal
    shuffle. go_deal_decks re-derives them from stdlib random + create_standard_deck
    so the Go path imports no engine; this asserts the re-derivation is exact.
    """
    from src.ffi.bridge import extract_deck_from_python_game
    from src.game.engine import CambiaGameState

    cfg = load_config(config)
    got = go_deal_decks(cfg, 5, 0)
    assert len(got) == 5
    for d, (deck, start) in enumerate(got):
        game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(0 + d))
        want_deck, want_start = extract_deck_from_python_game(game)
        assert deck == want_deck, f"deal {d} deck differs"
        assert start == want_start, f"deal {d} starting seat differs"


@requires_lib
def test_go_deal_decks_reach_the_engines_dealt_hands(config=CONTROL_CONFIG):
    """from_deck on a go_deal_decks deck reproduces the engine's dealt state."""
    from src.ffi.bridge import GoEngine
    from src.game.engine import CambiaGameState

    cfg = load_config(config)
    for d, (deck, start) in enumerate(go_deal_decks(cfg, 5, 0)):
        py = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(0 + d))
        eng = GoEngine.from_deck(deck, start, cfg.cambia_rules)
        try:
            for seat in (0, 1):
                got = [(c.rank, c.suit) for c in eng.get_player_hand(seat)]
                want = [(c.rank, c.suit) for c in py.players[seat].hand]
                assert got == want, f"deal {d} seat {seat} hand differs"
            got_top = eng.get_discard_top()
            want_top = py.get_discard_top()
            assert (got_top.rank, got_top.suit) == (want_top.rank, want_top.suit)
            assert eng.acting_player() == py.current_player_index
        finally:
            eng.close()


def test_action_inverse_table_inverts_action_to_index():
    """The index -> GameAction table is a true inverse over the whole space.

    A Go Decision node must carry the same GameAction objects a Python one does:
    Decision.actions is read by the PRT-CFR trainer, the tiny worker, prtcfr_net
    and the X2 scorer, all of which route actions through action_to_index. This
    pins the inverse rather than the forward map, since that is the direction the
    Go builder depends on.
    """
    table = _go_action_table()
    assert len(table) == NUM_ACTIONS
    covered = 0
    for idx, action in enumerate(table):
        if action is None:
            continue
        covered += 1
        assert action_to_index(action) == idx
    assert covered == NUM_ACTIONS, f"{NUM_ACTIONS - covered} indices unnamed"
    # And the draw-stockpile round trip the draw enumeration turns on.
    draw_idx = action_to_index(ActionDrawStockpile())
    assert isinstance(table[draw_idx], ActionDrawStockpile)


def test_actions_from_mask_sorts_like_the_python_builder():
    """_go_actions_from_mask returns build_tree_python's repr order.

    Matching the order is what makes the two backends' trees agree to the bit
    rather than to float64 summation noise: _cfr, _policy_value and _br_eval all
    reduce over a node's children in list order.
    """
    idxs = [action_to_index(ActionDrawStockpile()), 40, 12, 7]
    got = _go_actions_from_mask(idxs)
    assert [repr(a) for a in got] == sorted(repr(a) for a in got)
    assert len(got) == len(idxs)
    assert sorted(action_to_index(a) for a in got) == sorted(idxs)


# ---------------------------------------------------------------------------
# 2. Cross-engine tree parity on the reshuffle-free control config.
# ---------------------------------------------------------------------------


@requires_lib
def test_control_config_tree_counts_and_profile_match_across_backends():
    """Same node count, infoset count and aggregate profile from both engines.

    Not a node-for-node identity: see _profile for why a nested chance node's
    child ORDER cannot be matched, and test_control_config_nashconv_matches_across_backends
    for the invariant that does pin the trees' agreement numerically.

    config/tiny_norecall.yaml never exhausts the stockpile, so nothing in its
    enumerated tree depends on a reshuffle. That makes it the clean cross-engine
    check: any divergence here is an engine or builder defect, not the chance-point
    collapse the {A,6} divergence test documents.
    """
    go_root, go_isets, go_n, go_ab = _build(CONTROL_CONFIG, "go", stats={})
    py_root, py_isets, py_n, py_ab = _build(CONTROL_CONFIG, "python")

    assert (go_n, len(go_isets), go_ab) == (CONTROL_NODES, CONTROL_ISETS, 0)
    assert (py_n, len(py_isets), py_ab) == (CONTROL_NODES, CONTROL_ISETS, 0)
    assert _profile(go_root) == _profile(py_root)


@requires_lib
def test_control_config_go_build_enumerates_every_draw():
    """No chance point in the control tree is collapsed to a sampled outcome."""
    stats = {}
    _build(CONTROL_CONFIG, "go", stats=stats)
    assert stats["unenumerated_draws"] == 0
    assert stats["reshuffle_draws"] == 0
    assert stats["unchecked_draws"] == 0
    # And the engine never refused an action its own legal mask offered, which
    # would have stubbed a whole subtree out as a zero-utility Terminal.
    assert stats["rejected_actions"] == 0


@requires_lib
def test_control_config_nashconv_matches_across_backends():
    """Tabular CFR+ over either backend's control tree gives the same NashConv.

    The AC2 check the {A,6} tree cannot support: on the config where the two trees
    ARE the same tree, the solver and the exploitability certifier must agree.

    Agreement is asserted to a tolerance, not bit for bit. Both builders sort a
    decision node's legal actions by repr(action), but a nested chance node's child
    order cannot be matched: build_tree_python forces a card to the top by popping
    it out of its own stockpile LIST and appending it, so deeper chance nodes in
    that subtree enumerate an already-permuted list. Same cards, same weights,
    different order, so the reductions in _cfr / _policy_value / _br_eval sum the
    same terms in a different order. The measured gap is at or below ~1e-13
    relative -- ten orders of magnitude under the X2 bar's decision scale -- and at
    60 iterations the two happen to land bit-identical.
    """
    results = {}
    for backend in ("go", "python"):
        root, isets, _n, _ab = _build(CONTROL_CONFIG, backend)
        solver = TabularCFR(isets)
        for it in range(1, 61):
            solver.iterate(root, it)
        nc, parts = exploitability(root, solver.average_strategy())
        results[backend] = (float(nc), tuple(float(x) for x in parts))
    go_nc, go_parts = results["go"]
    py_nc, py_parts = results["python"]
    assert go_nc == pytest.approx(py_nc, rel=1e-10, abs=1e-15)
    for g, p in zip(go_parts, py_parts):
        assert g == pytest.approx(p, rel=1e-10, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. The suit-collapse trap in the key components.
# ---------------------------------------------------------------------------


def test_card_equality_drops_the_suit():
    """The trap _go_card_key exists for: Card hashes by rank alone.

    If this ever changes, _go_card_key becomes redundant rather than wrong -- but
    while it holds, putting Card objects in a perfect-recall key silently merges
    all four suits of a rank and coarsens the infoset partition ~4x per card
    position, leaving the tree structurally perfect. Pinned because the failure is
    invisible to every structural check.
    """
    from src.card import Card

    a, b = Card(rank="6", suit="D"), Card(rank="6", suit="C")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    # _go_card_key separates them, and keeps None passthrough.
    assert _go_card_key(a) != _go_card_key(b)
    assert len({_go_card_key(a), _go_card_key(b)}) == 2
    assert _go_card_key(None) is None


@requires_lib
def test_go_key_components_are_suit_exact():
    """Every card identity in a Go pkey carries its suit.

    pkey = ("PR", priv_init, priv_draw, pub_path). Walks the control tree and
    asserts the card slots are (rank, suit) pairs, and that more than one suit of
    the same rank actually shows up (which a collapsed key could not produce).
    """
    root, isets, _n, _ab = _build(CONTROL_CONFIG, "go")
    suits_by_rank = {}
    for pkey, _nA in isets:
        assert pkey[0] == "PR"
        for _slot, card in pkey[1]:
            assert isinstance(card, tuple) and len(card) == 2
            suits_by_rank.setdefault(card[0], set()).add(card[1])
        for card in pkey[2]:
            assert isinstance(card, tuple) and len(card) == 2
            suits_by_rank.setdefault(card[0], set()).add(card[1])
        for _acting, action, top in pkey[3]:
            assert isinstance(action, str)
            if top is not None:
                assert isinstance(top, tuple) and len(top) == 2
                suits_by_rank.setdefault(top[0], set()).add(top[1])
    assert suits_by_rank, "no card identities found in any pkey"
    assert any(len(v) > 1 for v in suits_by_rank.values()), (
        f"every rank appears with a single suit ({suits_by_rank}): the key "
        f"components have collapsed onto rank"
    )


@requires_lib
def test_go_pkey_determines_the_legal_action_set():
    """Perfect-recall keying's load-bearing property, on the Go tree.

    The X1 keystone rests on the key determining the legal-action count (and set);
    prtcfr_eval.enumerate_infosets asserts it while collapsing nodes per pkey.
    """
    from src.cfr.prtcfr_eval import enumerate_infosets

    root, isets, _n, _ab = _build(CONTROL_CONFIG, "go")
    by_key = {}
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.kind == "T":
            continue
        if nd.kind == "D":
            prev = by_key.setdefault(nd.pkey, list(nd.actions))
            assert prev == list(nd.actions), f"pkey {nd.pkey!r} has two legal sets"
        stack.extend(nd.children)
    assert len(by_key) == len(isets)
    # enumerate_infosets keeps one representative per pkey and raises otherwise.
    assert len(enumerate_infosets(root)) == len(by_key)


# ---------------------------------------------------------------------------
# 4. The {A,6} divergence, recorded with its mechanism.
# ---------------------------------------------------------------------------


@requires_lib
@pytest.mark.slow
def test_a6_tree_divergence_is_diagnosed():
    """The recorded 230,206 / 69,636 {A,6} counts are not reachable on Go.

    Mechanism. The {A,6} enumerated tree EXHAUSTS the 8-card deck: 30,019 node
    visits in the Python build trigger a reshuffle, which recycles the discard
    pile and shuffles it with the engine's own RNG. Two consequences:

      - At an exhausted-stockpile draw, build_tree_python does not enumerate at
        all (its guard is ``and game.stockpile``); it takes the single card the
        Python random.Random(seed0 + d) stream happens to put on top. That stream
        is consumed in DFS order and is never rewound by undo, so the outcome
        depends on how many reshuffles the traversal did earlier.
      - Downstream of a reshuffle the Python builder DOES enumerate, because it
        can reorder its own stockpile list. The Go builder cannot: the FFI exposes
        no stockpile accessor and no way to order it, and the deck-order channel
        build_tree_go enumerates through (cambia_game_new_with_deck) only reaches
        cards that have not been recycled.

    So the recorded counts are a property of one random.Random stream, not of the
    game or of either engine. Holding the Python engine fixed and only replacing
    the reshuffle's shuffle with a deterministic order moves both counts:
    random.Random 230,206 / 69,636; sorted 234,185 / 69,873; reverse-sorted
    229,835 / 68,522.

    This test pins the Go counts and the diagnostic that explains them, so the
    divergence stays visible and a later FFI stockpile-ordering export can be
    measured against it rather than rediscovered.
    """
    stats = {}
    _root, isets, n, ab = _build(A6_CONFIG, "go", stats=stats)
    assert ab == 0
    assert (n, len(isets)) != (A6_RECORDED_NODES, A6_RECORDED_ISETS)
    assert (n, len(isets)) == (208844, 64383)
    # The reason, not just the symptom: draw points the deck channel could not
    # enumerate, all of them downstream of an engine reshuffle.
    assert stats["unenumerated_draws"] == 4594
    assert stats["reshuffle_draws"] == 14518
    assert stats["unchecked_draws"] == 0
    # The gap is the enumeration channel, not engine disagreement: nothing the
    # legal mask offered was refused by the apply path.
    assert stats["rejected_actions"] == 0


# ---------------------------------------------------------------------------
# AC3: the exact-tree consumers hold no Python-engine dependency.
# ---------------------------------------------------------------------------


@requires_lib
def test_importing_the_tree_consumers_does_not_import_the_engine():
    """tiny_solver / tiny_exact / prtcfr_eval import without src.game.engine.

    Run in a subprocess: this session has almost certainly imported the engine
    already through another test, so an in-process sys.modules check would pass
    for the wrong reason.
    """
    import subprocess

    code = (
        "import sys\n"
        "import tools.tiny_solver, tools.tiny_exact\n"
        "import src.cfr.prtcfr_eval\n"
        "leaked = [m for m in sys.modules if m.startswith('src.game')"
        " or m == 'src.analysis_tools' or m == 'src.cfr.worker']\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, f"stdout={out.stdout} stderr={out.stderr}"
    assert "clean" in out.stdout


@requires_lib
def test_go_tree_build_does_not_import_the_engine():
    """A full Go tree build pulls in no engine module either."""
    import subprocess

    code = (
        "import sys\n"
        "from src.config import load_config\n"
        "from tools.tiny_solver import build_tree\n"
        f"cfg = load_config({CONTROL_CONFIG!r})\n"
        "root, isets, n, ab = build_tree(cfg, n_deals=1, seed0=0,\n"
        "    max_nodes_per_deal=2_000_000, enumerate_draws=True,\n"
        "    perfect_recall=True, backend='go')\n"
        "leaked = [m for m in sys.modules if m.startswith('src.game')"
        " or m == 'src.analysis_tools' or m == 'src.cfr.worker']\n"
        "assert not leaked, leaked\n"
        "assert n > 0 and len(isets) > 0\n"
        "print('clean', n, len(isets))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, f"stdout={out.stdout} stderr={out.stderr}"
    assert "clean" in out.stdout


@requires_lib
def test_residual_python_reference_import_is_agent_state_via_encoding():
    """The one Python-reference module the Go path still pulls in, and its route.

    The game engine and the observation machinery layered on it (src.game.engine,
    src.analysis_tools, src.cfr.worker) are gone from the Go path. src.agent_state
    still arrives, because src/encoding.py imports AgentState at module scope for
    its belief-tensor helpers and prtcfr_eval needs NUM_ACTIONS / action_to_index
    from it. src/encoding.py is outside cambia-1429's scope; this pins the residual
    and its route so the follow-up has a target and so a NEW leak still fails a
    test.
    """
    import subprocess

    code = (
        "import sys\n"
        "import src.encoding\n"
        "assert 'src.agent_state' in sys.modules\n"
        "engine = [m for m in sys.modules if m.startswith('src.game')]\n"
        "assert not engine, engine\n"
        "print('agent_state-via-encoding')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, f"stdout={out.stdout} stderr={out.stderr}"
    assert "agent_state-via-encoding" in out.stdout


@requires_lib
def test_exact_certifier_runs_on_the_go_tree():
    """tools.tiny_exact certifies a Go-built tree (AC3), agreeing with float64.

    Uses the control config so the certifier's exact-rational arithmetic stays
    cheap; the point is that the certifier reads a Go tree at all, since it needs
    Chance.wfrac from the builder and only len(node.actions)/node.pkey from the
    decision nodes.
    """
    from fractions import Fraction

    from tools import tiny_exact

    cfg = load_config(CONTROL_CONFIG)
    root, isets, _n, _ab = build_tree(
        cfg,
        n_deals=5,
        seed0=0,
        max_nodes_per_deal=2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        exact_weights=True,
        backend="go",
    )
    assert root.wfrac is not None and root.wfrac[0] == Fraction(1, 5)
    policy = {pkey: np.ones(nA, dtype=np.float64) / nA for (pkey, nA) in isets.items()}
    nc_f, _ = exploitability(root, policy)
    nc_e, _ = tiny_exact.exploitability_exact(root, policy)
    assert isinstance(nc_e, Fraction)
    assert abs(float(nc_f) - float(nc_e)) < 1e-12


@requires_lib
def test_go_nodes_carry_the_same_action_contract_as_python_nodes():
    """Decision.actions holds GameActions the scorer's own encoders accept.

    The whole reason the Go builder decodes its index-native legal mask back into
    GameAction NamedTuples: prtcfr_eval, prtcfr_net, the trainer and the tiny
    worker all read Decision.actions through action_to_index / encode_action_mask.
    """
    from src.encoding import encode_action_mask
    from src.cfr.prtcfr_eval import enumerate_infosets

    root, _isets, _n, _ab = _build(CONTROL_CONFIG, "go", deals=1)
    nodes = enumerate_infosets(root)
    assert nodes
    for node in nodes[:200]:
        assert not any(isinstance(a, (int, np.integer)) for a in node.actions)
        idx = [action_to_index(a) for a in node.actions]
        mask = encode_action_mask(node.actions)
        assert mask.dtype == np.bool_ and mask.shape == (NUM_ACTIONS,)
        assert int(mask.sum()) == len(set(idx)) == len(node.actions)
        assert all(mask[i] for i in idx)


# ---------------------------------------------------------------------------
# Backend guards.
# ---------------------------------------------------------------------------


def test_unknown_backend_rejected():
    cfg = load_config(CONTROL_CONFIG)
    with pytest.raises(ValueError, match="unknown backend"):
        build_tree(cfg, 1, 0, 1000, backend="rust")


@requires_lib
def test_go_backend_refuses_imperfect_recall():
    """The belief key has no FFI export, so the Go backend will not fake one."""
    cfg = load_config(CONTROL_CONFIG)
    with pytest.raises(NotImplementedError, match="perfect_recall=True only"):
        build_tree(cfg, 1, 0, 1000, perfect_recall=False, backend="go")


@requires_lib
def test_go_backend_keeps_the_snaprace_fence():
    """Race-ON stays refused: its winner draw is an unenumerable RNG draw."""
    cfg = load_config(CONTROL_CONFIG)
    cfg.cambia_rules.snapRace = True
    with pytest.raises(ValueError, match="snapRace"):
        build_tree(cfg, 1, 0, 1000, perfect_recall=True, backend="go")


@requires_lib
def test_go_builder_frees_its_handles():
    """Repeated builds do not leak game/agent handles from the finite pools."""
    from src.ffi.bridge import get_handle_pool_stats

    before = get_handle_pool_stats()
    for _ in range(3):
        _build(CONTROL_CONFIG, "go", deals=1)
    after = get_handle_pool_stats()
    for key in ("games", "agents", "snapshots"):
        assert after[key] <= before[key], f"{key} grew: {before} -> {after}"
