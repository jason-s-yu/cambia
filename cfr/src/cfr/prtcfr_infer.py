"""src/cfr/prtcfr_infer.py

Batched incremental GRU inference service for PRT-CFR (v0.4 Phase 2, S1W3
stage 2). Pinned interface (sprint-1-plan.md, "Inference service" section):

    register(stream_id, tokens) -> hidden
    step(stream_ids, new_tokens) -> batched hidden
    advantages(stream_ids, masks) -> [n, 146]

bf16, single CUDA stream (or CPU for scoped/dev use -- device is a
constructor argument), batching across all live rollouts.

Window semantics (v0.4 Phase 2 decision note, option A): the service carries
the RAW per-layer GRU hidden state (shape (num_layers, hidden) per stream, pre
LayerNorm/head) across calls. Appending a token and stepping the GRU once is
mathematically the SAME recurrence as re-encoding the full prefix from scratch
in one pack_padded pass: h_n after processing tokens[0:k] then tokens[k:n] with
that h_n as h_0 equals h_n from processing tokens[0:n] in one call. There is no
approximation and no window: carry conditions on the exact full prefix, by
construction, as long as callers never skip appending a token (register once,
then step for every subsequent token/frame -- never re-register mid-stream).
The carry-vs-batch equivalence is enforced by
tests/test_prtcfr_infer.py::test_incremental_carry_matches_full_batch_reencode_bf16.

Dropout: the encoder's GRU dropout must be OFF for the carry-vs-batch identity
to hold (dropout draws would otherwise differ between a one-shot forward and a
register+step split). The service calls ``net.eval()`` and never trains through
it; the encoder is a frozen snapshot for the lifetime of a service instance
(a new snapshot means a new service instance, matching the SD-CFR per-iteration
regret-net turnover).

Serving-throughput (cambia-472, X3 fix ladder step b): the carried hidden state
lives in ONE device-resident packed tensor (``_DeviceHiddenStore``), keyed by a
slot index per stream, so every per-tick gather/scatter is a single batched
``index_select``/``index_copy_`` kernel rather than a Python per-stream tensor
list-build + N ``.contiguous()`` writes. Token batches are padded with one
vectorized numpy scatter and a single host-to-device copy (``_pad_batch``), and
the production per-tick call fuses the persist-advance and the transient policy
query into ONE GRU forward (``advance_query``). These remove the per-stream
Python overhead the X3 microbench isolates as the inference-wall dominator; they
do not touch the recurrence math (all bit-identical in fp32 / within-dtype in
bf16 to the pre-fix advance()+query_transient() path).
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..sequence_encoding import PAD_ID
from .prtcfr_net import PRTCFRNet, _regret_match


class _DeviceHiddenStore:
    """Device-resident packed carry for the live streams (cambia-472).

    One ``(num_layers, capacity, hidden)`` tensor holds every stream's raw GRU
    hidden state; a Python ``stream_id -> row`` slot map plus a free-list assign
    rows. The hot path reads and writes MANY streams at once through a single
    ``index_select`` (gather) / ``index_copy_`` (scatter) kernel keyed by a row
    tensor, so no per-stream ``torch.stack`` list-build or per-stream
    ``.contiguous()`` write is on the critical path.

    A dict-compatible surface (``__getitem__``/``__setitem__``/``get``/``pop``/
    ``__contains__``/``__len__``) keeps the primitive ``register``/``step`` API
    and the existing scoped tests working unchanged; ``__getitem__``/``get``
    return a VIEW into the packed tensor (read-once or clone-immediately callers
    only -- the hot path uses ``rows``/``gather``/``scatter`` and never holds a
    view across a mutation).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        capacity: int = 1024,
    ):
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self.dtype = dtype
        self._cap = int(capacity)
        self._H = torch.zeros(
            (self.num_layers, self._cap, self.hidden_dim), device=device, dtype=dtype
        )
        self._slot: Dict[object, int] = {}
        # Free rows, popped from the end so allocation is ascending + deterministic.
        self._free: List[int] = list(range(self._cap - 1, -1, -1))

    # -- capacity -----------------------------------------------------------

    def _ensure_free(self, need: int) -> None:
        if len(self._free) >= need:
            return
        old = self._cap
        new_cap = old
        while new_cap - len(self._slot) < need:
            new_cap *= 2
        new_h = torch.zeros(
            (self.num_layers, new_cap, self.hidden_dim),
            device=self.device,
            dtype=self.dtype,
        )
        new_h[:, :old, :] = self._H
        self._H = new_h
        self._free.extend(range(new_cap - 1, old - 1, -1))
        self._cap = new_cap

    # -- batched row helpers (hot path) -------------------------------------

    def rows(self, ids: Sequence[object]) -> torch.Tensor:
        """(len(ids),) int64 device tensor of the slot rows for ``ids`` (all
        must be present). The only per-stream Python left on the hot path is
        this dict lookup -- O(100 ns)/stream, not a tensor op."""
        slot = self._slot
        return torch.tensor([slot[i] for i in ids], dtype=torch.long, device=self.device)

    def alloc_zeroed(self, ids: Sequence[object]) -> None:
        """Assign a fresh, ZEROED row to each currently-absent id in ``ids`` (one
        batched ``index_fill_``). A zeroed row is the ``h0 = 0`` an unregistered
        stream's first advance conditions on; rows returned to the free-list by
        ``pop`` carry stale data, so zeroing at alloc (not read) is required."""
        if not ids:
            return
        self._ensure_free(len(ids))
        rws: List[int] = []
        for i in ids:
            r = self._free.pop()
            self._slot[i] = r
            rws.append(r)
        idx = torch.tensor(rws, dtype=torch.long, device=self.device)
        self._H.index_fill_(1, idx, 0.0)

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """(num_layers, len(rows), hidden) COPY of the carried hidden at ``rows``
        (index_select copies, so a later scatter never aliases the returned h0)."""
        return self._H.index_select(1, rows)

    def scatter(self, rows: torch.Tensor, values: torch.Tensor) -> None:
        """Write ``values`` (num_layers, len(rows), hidden) into ``rows`` in one
        ``index_copy_`` (rows are unique per call)."""
        self._H.index_copy_(1, rows, values.to(self.dtype))

    # -- dict-compatible surface (primitive API + tests) --------------------

    def __contains__(self, sid) -> bool:
        return sid in self._slot

    def __len__(self) -> int:
        return len(self._slot)

    def get(self, sid, default=None):
        r = self._slot.get(sid)
        if r is None:
            return default
        return self._H[:, r, :]

    def __getitem__(self, sid):
        return self._H[:, self._slot[sid], :]

    def __setitem__(self, sid, value: torch.Tensor) -> None:
        r = self._slot.get(sid)
        if r is None:
            self._ensure_free(1)
            r = self._free.pop()
            self._slot[sid] = r
        self._H[:, r, :] = value.to(self.dtype)

    def pop(self, sid, default=None):
        r = self._slot.pop(sid, None)
        if r is None:
            return default
        self._free.append(r)  # stale data left in the row; alloc_zeroed clears on reuse
        return None


class PRTCFRInferenceService:
    """Batched incremental GRU inference over live PRT-CFR rollout streams.

    One instance wraps ONE frozen ``PRTCFRNet`` snapshot (encoder + head).
    Each live rollout/trajectory registers a ``stream_id`` once at its start
    and calls ``step`` as new tokens (typically one observation frame at a
    time) arrive; the service carries that stream's raw GRU hidden state
    between calls so a policy query never re-encodes the full prefix.

    The service holds a DEDICATED ``dtype``-cast copy of ``net`` (embed, GRU,
    LayerNorm, and head all cast to ``dtype``, default bf16) -- genuine
    end-to-end bf16 inference, matching the pinned interface's "bf16, single
    CUDA stream" contract (PyTorch's GRU requires the input dtype to exactly
    match its weight dtype; casting only activations while leaving fp32
    weights raises). The caller's original ``net`` (e.g. the live training
    net) is never mutated -- this is a snapshot copy, matching SD-CFR's
    per-iteration regret-net turnover (a new iterate means a new service
    instance from a fresh snapshot, not a live-mutated shared module).

    Carried hidden state lives in a device-resident ``_DeviceHiddenStore`` so
    per-tick gather/scatter over the batch is a single kernel (cambia-472).
    """

    def __init__(
        self,
        net: PRTCFRNet,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device) if device is not None else net.device
        self.dtype = dtype
        self.net = copy.deepcopy(net).eval().to(self.device).to(self.dtype)
        # Per-stream carried hidden state: packed (num_layers, capacity, hidden).
        self._hidden = _DeviceHiddenStore(
            self.net.num_layers, self.net.hidden_dim, self.device, self.dtype
        )

    # -- internal helpers ---------------------------------------------------

    def _pad_batch(
        self, token_lists: Sequence[Sequence[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-pad a ragged batch of token-id lists to a dense (B, Lmax)
        int64 tensor (PAD_ID) plus a (B,) int64 CPU tensor of real lengths.

        Vectorized (cambia-472): the dense grid is built with one numpy scatter
        (row/col index arithmetic over the flattened tokens) and moved to the
        device in a SINGLE host-to-device copy, replacing the pre-fix per-stream
        ``torch.tensor(...).to(device)`` loop that paid one small H2D per stream.
        Empty rows clamp to length 1 (a lone PAD step), preserving the pre-fix
        ``max(1, len)`` behavior the primitive ``step`` relies on."""
        n = len(token_lists)
        lengths = [len(t) for t in token_lists]
        total = sum(lengths)
        lmax = max(max(lengths, default=0), 1)
        batch_np = np.full((n, lmax), PAD_ID, dtype=np.int64)
        if total:
            flat = np.fromiter(
                itertools.chain.from_iterable(token_lists), dtype=np.int64, count=total
            )
            lengths_np = np.asarray(lengths, dtype=np.int64)
            row_idx = np.repeat(np.arange(n), lengths_np)
            starts = np.repeat(np.cumsum(lengths_np) - lengths_np, lengths_np)
            col_idx = np.arange(total, dtype=np.int64) - starts
            batch_np[row_idx, col_idx] = flat
        batch = torch.from_numpy(batch_np).to(self.device, non_blocking=True)
        real_lengths = torch.tensor([max(1, L) for L in lengths], dtype=torch.long)
        return batch, real_lengths

    def _run_gru(
        self, token_lists: Sequence[Sequence[int]], h0: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Embed + pack + run the encoder GRU from ``h0`` (or zero-init if
        None); return the new (num_layers, B, hidden) hidden state in
        ``self.dtype``."""
        batch, lengths = self._pad_batch(token_lists)
        emb = self.net.encoder["embed"](batch).to(self.dtype)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        gru = self.net.encoder["gru"]
        if h0 is None:
            _out, h_n = gru(packed)
        else:
            _out, h_n = gru(packed, h0.to(self.dtype))
        return h_n.to(self.dtype)

    def _query_h0(self, stream_ids: Sequence[object]) -> torch.Tensor:
        """(num_layers, B, hidden) query base for ``stream_ids``: the gathered
        carry for present streams, a zero column for absent ones (a transient
        query from a zero hidden -- the defensive path; production always
        advances a stream before querying it). All-present is the common case
        and takes one gather."""
        n = len(stream_ids)
        present = [s in self._hidden for s in stream_ids]
        if all(present):
            return self._hidden.gather(self._hidden.rows(stream_ids))
        h0 = torch.zeros(
            (self.net.num_layers, n, self.net.hidden_dim),
            device=self.device,
            dtype=self.dtype,
        )
        pres_pos = [i for i, p in enumerate(present) if p]
        if pres_pos:
            src = self._hidden.gather(
                self._hidden.rows([stream_ids[i] for i in pres_pos])
            )
            pos = torch.tensor(pres_pos, dtype=torch.long, device=self.device)
            h0.index_copy_(1, pos, src)
        return h0

    # -- pinned interface -----------------------------------------------

    @torch.no_grad()
    def register(self, stream_id: str, tokens: List[int]) -> torch.Tensor:
        """Encode ``tokens`` (the stream's initial prefix, e.g. BOS + peeked
        initial hand) from a zero initial hidden state; store and return the
        resulting (num_layers, hidden) raw hidden state for ``stream_id``.

        Calling ``register`` again for an already-registered ``stream_id``
        re-encodes from scratch (a fresh rollout reusing an old id), discarding
        any carried state -- this is the ONLY re-encode path; every subsequent
        token for that stream must go through ``step``.
        """
        h_n = self._run_gru([tokens], h0=None)  # (num_layers, 1, hidden)
        h = h_n[:, 0, :].contiguous()  # (num_layers, hidden)
        self._hidden[stream_id] = h
        return h

    @torch.no_grad()
    def step(self, stream_ids: List[str], new_tokens: List[List[int]]) -> torch.Tensor:
        """Advance each stream's carried hidden state by its ``new_tokens``
        (usually one observation frame, a handful of ids) via ONE batched GRU
        pass from the carried ``h0`` -- never a full re-encode. Returns the
        batched TOP-LAYER hidden state (len(stream_ids), hidden), post-step,
        pre LayerNorm/head (raw carry state; see ``advantages``/``strategy``
        for the queryable head output).
        """
        missing = [sid for sid in stream_ids if sid not in self._hidden]
        if missing:
            raise KeyError(
                f"PRTCFRInferenceService.step: stream(s) {missing} not registered; "
                f"call register(stream_id, initial_tokens) first"
            )
        rows = self._hidden.rows(stream_ids)
        h0 = self._hidden.gather(rows)
        h_n = self._run_gru(new_tokens, h0=h0)  # (num_layers, B, hidden)
        self._hidden.scatter(rows, h_n)
        return h_n[-1]  # (B, hidden) top layer

    def _top_hidden_batch(self, stream_ids: Sequence[str]) -> torch.Tensor:
        missing = [sid for sid in stream_ids if sid not in self._hidden]
        if missing:
            raise KeyError(f"PRTCFRInferenceService: stream(s) {missing} not registered")
        return self._hidden.gather(self._hidden.rows(stream_ids))[-1]  # (B, hidden)

    @torch.no_grad()
    def advantages(
        self, stream_ids: List[str], masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Raw advantage logits [len(stream_ids), 146] from each stream's
        CURRENT carried hidden state (LayerNorm -> head). ``masks`` is
        accepted for signature symmetry with ``PRTCFRNet.raw_advantages`` (and
        the pinned interface) but NOT applied here, matching that method's own
        convention; use ``strategy`` for the masked, regret-matched policy."""
        top = self._top_hidden_batch(stream_ids)
        normed = self.net.encoder["norm"](top.to(next(self.net.head.parameters()).dtype))
        return self.net.head(normed)

    @torch.no_grad()
    def strategy(self, stream_ids: List[str], masks: torch.Tensor) -> torch.Tensor:
        """Regret-matched strategy [len(stream_ids), 146] from each stream's
        current carried hidden state: ReLU(advantages) * mask normalized,
        uniform-over-legal fallback. Mirrors
        ``PRTCFRNet.strategy_from_tokens``/``src.networks.get_strategy_from_advantages``
        exactly, computed from the carried hidden state rather than a fresh
        token batch -- this IS the production policy-query path (p2-redesign
        sec 6): one batched forward per decision tick, not a re-encode."""
        adv = self.advantages(stream_ids, masks=None)
        return _regret_match(adv, masks)

    def drop(self, stream_id: str) -> None:
        """Release a finished stream's carried hidden state."""
        self._hidden.pop(stream_id, None)

    def __len__(self) -> int:
        return len(self._hidden)

    # -- production wiring surface (S1W15) -----------------------------------
    #
    # ``register``/``step`` above are the pinned S1W3 primitives. The methods
    # below are the S1W15 refinement the production generation loop
    # (prtcfr_worker.PRTCFRBatchedProductionWorker) consumes; each is covered
    # by an equivalence test in tests/test_prtcfr_infer.py /
    # tests/test_prtcfr_batched_gen.py so the carry-vs-reencode identity extends
    # to the exact call pattern production uses (batched advance, hidden fork at
    # rollout branch points, transient EOS query). None of them touch
    # ``register``/``step``.

    @torch.no_grad()
    def fork(self, src_id, dst_id) -> None:
        """Copy ``src_id``'s carried hidden state into ``dst_id`` (a rollout
        branch point: the child stream conditions on the SAME prefix as the
        parent, then advances independently -- the torch-side mirror of the Go
        ``cambia_state_clone`` fan-out, S1W12).

        No-op if ``src_id`` has no carried state (an unqueried stream): the
        child stays absent and its first ``advance`` carries from a zero hidden
        over the full prefix, which is identical to a fork of the zero state.
        The clone is a fresh tensor so subsequent ``advance``/``step`` on either
        stream never aliases the other's hidden.
        """
        h = self._hidden.get(src_id)
        if h is not None:
            self._hidden[dst_id] = h.clone()

    @torch.no_grad()
    def advance(self, stream_ids: List, new_tokens: List[List[int]]) -> None:
        """Persist-advance each stream's carried hidden by its ``new_tokens``.

        Differs from ``step`` in three ways the production carry needs: (1) a
        stream with an EMPTY ``new_tokens`` is left untouched (``step`` would
        inject a spurious PAD-token GRU step, corrupting the carry); (2) a
        stream NOT yet present is registered from a zero hidden over its
        ``new_tokens`` (no separate ``register`` call needed -- a first
        ``advance`` over ``[BOS]+body`` IS the registration); (3) present and
        absent streams are advanced in ONE batched GRU pass (absent rows use a
        zero ``h0`` column), so a decision tick with a mix of fresh and carried
        streams is a single forward.
        """
        idx = [i for i, t in enumerate(new_tokens) if len(t) > 0]
        if not idx:
            return
        sids = [stream_ids[i] for i in idx]
        toks = [new_tokens[i] for i in idx]
        absent = [s for s in sids if s not in self._hidden]
        if absent:
            self._hidden.alloc_zeroed(absent)  # zero carry -> h0 = 0 for fresh streams
        rows = self._hidden.rows(sids)
        h0 = self._hidden.gather(rows)
        h_n = self._run_gru(toks, h0=h0)
        self._hidden.scatter(rows, h_n)

    @torch.no_grad()
    def query_transient(
        self,
        stream_ids: List,
        transient_tokens: List[List[int]],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Regret-matched strategy [n, 146] from each stream's carried hidden
        advanced by a TRANSIENT ``transient_tokens`` suffix that is NOT
        persisted.

        Production usage: the carried hidden tracks ``[BOS]+body`` (no trailing
        EOS, so the carry stays a monotonic prefix of the growing observation
        stream); a policy query needs the hidden after ``[BOS]+body+[EOS]`` to
        match a full ``encode_observation_sequence`` re-encode (which appends
        EOS). ``transient_tokens=[[EOS_ID], ...]`` supplies that suffix for the
        query only, leaving the carry at ``[BOS]+body`` so the NEXT frame
        appends cleanly. Absent streams query from a zero hidden over just the
        transient tokens (defensive; the production flow always ``advance``s a
        stream before querying it, so it is present by query time).
        """
        h0 = self._query_h0(stream_ids)
        h_n = self._run_gru(transient_tokens, h0=h0)  # transient (not stored)
        top = h_n[-1]  # (B, hidden), top layer post-suffix
        head_dtype = next(self.net.head.parameters()).dtype
        normed = self.net.encoder["norm"](top.to(head_dtype))
        adv = self.net.head(normed)
        return _regret_match(adv, masks)

    @torch.no_grad()
    def advance_query(
        self,
        stream_ids: List,
        new_tokens: List[List[int]],
        transient_tokens: List[List[int]],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Fused per-tick production call (cambia-472): persist-advance every
        stream by ``new_tokens`` AND return the masked regret-matched strategy
        after a TRANSIENT ``transient_tokens`` suffix, in ONE gather, ONE GRU
        forward, ONE scatter, ONE head.

        Numerically identical to ``advance(stream_ids, new_tokens)`` followed by
        ``query_transient(stream_ids, transient_tokens, masks)``. The GRU is a
        deterministic recurrence, so h_n over ``new+transient`` from ``h0`` has
        the same top hidden as one step over ``transient`` from
        (h_n over ``new`` from ``h0``); running both the persist rows (``new``)
        and the query rows (``new+transient``) from the SAME gathered ``h0`` in a
        single batched forward therefore reproduces the two-call path bit-for-bit
        in fp32 (within-dtype in bf16). Persist membership matches ``advance``
        (streams with non-empty ``new`` only); query membership is every stream
        (empty-``new`` present streams query from their carry, absent streams
        from a zero hidden -- see ``query_transient``/``advance``).
        """
        n = len(stream_ids)
        # Persist set: streams with non-empty new frames (matches advance()).
        persist_pos = [i for i, t in enumerate(new_tokens) if len(t) > 0]
        persist_sids = [stream_ids[i] for i in persist_pos]
        absent_persist = [s for s in persist_sids if s not in self._hidden]
        if absent_persist:
            self._hidden.alloc_zeroed(absent_persist)  # zero carry, now present

        # Query base for ALL streams (present -> carry, absent -> zero). Persist
        # streams were just made present, so their column here is their carry.
        h0_q = self._query_h0(stream_ids)

        query_toks = [
            list(new_tokens[i]) + list(transient_tokens[i]) for i in range(n)
        ]
        n_persist = len(persist_pos)
        if n_persist:
            p_pos = torch.tensor(persist_pos, dtype=torch.long, device=self.device)
            p_rows = self._hidden.rows(persist_sids)
            h0_p = h0_q.index_select(1, p_pos)  # same h0 the query rows use
            combined_toks = query_toks + [list(new_tokens[i]) for i in persist_pos]
            combined_h0 = torch.cat([h0_q, h0_p], dim=1)
        else:
            combined_toks = query_toks
            combined_h0 = h0_q

        h_n = self._run_gru(combined_toks, h0=combined_h0)  # (L, n + n_persist, hidden)
        if n_persist:
            self._hidden.scatter(p_rows, h_n[:, n:, :])  # persist rows -> carry
        top = h_n[-1, :n, :]  # query rows, top layer post new+transient
        head_dtype = next(self.net.head.parameters()).dtype
        normed = self.net.encoder["norm"](top.to(head_dtype))
        adv = self.net.head(normed)
        return _regret_match(adv, masks)
