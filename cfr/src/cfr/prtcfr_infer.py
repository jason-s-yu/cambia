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
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..sequence_encoding import PAD_ID
from .prtcfr_net import PRTCFRNet, _regret_match


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
        # Per-stream carried hidden state: (num_layers, hidden), service dtype.
        self._hidden: Dict[str, torch.Tensor] = {}

    # -- internal helpers ---------------------------------------------------

    def _pad_batch(
        self, token_lists: Sequence[Sequence[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-pad a ragged batch of token-id lists to a dense (B, Lmax)
        int64 tensor (PAD_ID) plus a (B,) int64 tensor of real lengths."""
        lengths = [max(1, len(t)) for t in token_lists]
        lmax = max(lengths)
        batch = torch.full(
            (len(token_lists), lmax), PAD_ID, dtype=torch.long, device=self.device
        )
        for i, toks in enumerate(token_lists):
            if len(toks) > 0:
                batch[i, : len(toks)] = torch.tensor(
                    toks, dtype=torch.long, device=self.device
                )
        return batch, torch.tensor(lengths, dtype=torch.long)

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
        h0 = torch.stack([self._hidden[sid] for sid in stream_ids], dim=1)
        h_n = self._run_gru(new_tokens, h0=h0)  # (num_layers, B, hidden)
        for i, sid in enumerate(stream_ids):
            self._hidden[sid] = h_n[:, i, :].contiguous()
        return h_n[-1]  # (B, hidden) top layer

    def _top_hidden_batch(self, stream_ids: Sequence[str]) -> torch.Tensor:
        missing = [sid for sid in stream_ids if sid not in self._hidden]
        if missing:
            raise KeyError(f"PRTCFRInferenceService: stream(s) {missing} not registered")
        return torch.stack([self._hidden[sid][-1] for sid in stream_ids], dim=0)

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
    # ``register``/``step`` above are the pinned S1W3 primitives. The three
    # methods below are the S1W15 refinement the production generation loop
    # (prtcfr_worker.PRTCFRBatchedProductionWorker) consumes; each is covered
    # by an equivalence test in tests/test_prtcfr_infer.py so the carry-vs-
    # reencode identity extends to the exact call pattern production uses
    # (batched advance, hidden fork at rollout branch points, transient EOS
    # query). None of them touch ``register``/``step``.

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
        zero: Optional[torch.Tensor] = None
        cols: List[torch.Tensor] = []
        for s in sids:
            h = self._hidden.get(s)
            if h is None:
                if zero is None:
                    zero = torch.zeros(
                        self.net.num_layers,
                        self.net.hidden_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                cols.append(zero)
            else:
                cols.append(h)
        h0 = torch.stack(cols, dim=1)  # (num_layers, B, hidden)
        h_n = self._run_gru(toks, h0=h0)
        for j, s in enumerate(sids):
            self._hidden[s] = h_n[:, j, :].contiguous()

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
        zero: Optional[torch.Tensor] = None
        cols: List[torch.Tensor] = []
        for s in stream_ids:
            h = self._hidden.get(s)
            if h is None:
                if zero is None:
                    zero = torch.zeros(
                        self.net.num_layers,
                        self.net.hidden_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                cols.append(zero)
            else:
                cols.append(h)
        h0 = torch.stack(cols, dim=1)
        h_n = self._run_gru(transient_tokens, h0=h0)  # transient (not stored)
        top = h_n[-1]  # (B, hidden), top layer post-suffix
        head_dtype = next(self.net.head.parameters()).dtype
        normed = self.net.encoder["norm"](top.to(head_dtype))
        adv = self.net.head(normed)
        return _regret_match(adv, masks)
