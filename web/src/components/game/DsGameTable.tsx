// src/components/game/DsGameTable.tsx
// The live table (cambia-484, restyled for the flat card-room language in
// cambia-848). Composes the design-system game primitives (PlayerSeat,
// PlayingCard, ScorePill, TimerBar) on the felt tokens. Interaction and every
// outgoing WS action are the same as the legacy GameBoard: clicking the
// stockpile, discard pile, own cards and opponent cards drives draw / discard /
// replace / snap / special / Cambia via the same action constructors. Snap,
// penalty and reshuffle feedback is derived from state deltas the store already
// applies (player_snap_success, player_snap_penalty, game_reshuffle_stockpile),
// so no store or protocol change rides with this file.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ObfCard, ObfGameState, ObfPlayerState, ClientGameAction } from '@/types/game';
import {
  drawStockpileAction,
  drawDiscardPileAction,
  discardAction,
  replaceAction,
  snapAction,
  peekSelfAction,
  peekOtherAction,
  blindSwapAction,
  kingPeekAction,
  kingSwapConfirmAction,
  callCambiaAction,
  skipSpecialAction
} from '@/types/game';
import {
  useGameStore,
  selectPendingAction,
  selectIsSelfTurn,
  selectIsProcessingAction,
  selectDisplayedDrawnCard,
  selectServerClockOffsetMs,
  selectAbilityReveal,
  selectDroppedActionNonce
} from '@/stores/gameStore';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import PlayingCard from '@/components/ds/game/PlayingCard';
import PlayerSeat, { type PlayerSeatState } from '@/components/ds/game/PlayerSeat';
import ScorePill from '@/components/ds/game/ScorePill';
import TimerBar from '@/components/ds/game/TimerBar';
import { toDsCardFace } from './dsCardMap';

interface DsGameTableProps {
  gameState: ObfGameState;
  phase: LobbyPhase;
  sendMessage: (msg: ClientGameAction) => void;
  onLeave: () => void;
  /**
   * Socket state from the page. While false the table stays mounted with every control
   * locked and a reconnect notice in the top strip; the hook reconnects on its own.
   */
  connected?: boolean;
  /** The hook's last connection error, used to tell a live retry from a dead socket. */
  connectionError?: string | null;
}

/** Ability name by the discarded rank, for seat notes and prompts. */
function abilityName(rank: string | undefined): string | null {
  switch (rank) {
    case '7':
    case '8':
      return 'Peek own card';
    case '9':
    case 'T':
      return 'Peek a card';
    case 'J':
    case 'Q':
      return 'Blind swap';
    case 'K':
      return 'Look and swap';
    default:
      return null;
  }
}

function seatStateFor(p: ObfPlayerState, currentPlayerId: string | null): PlayerSeatState | undefined {
  if (!p.connected) return 'disconnected';
  if (p.hasCalledCambia) return 'cambia';
  if (p.playerId === currentPlayerId) return 'turn';
  return undefined;
}

/** The pair sent with the King look, kept so the follow-up swap names the same cards. */
interface KingPair {
  myId: string;
  myIdx: number;
  oppId: string;
  oppIdx: number;
  oppOwner: string;
}

/** How long a peeked opponent face stays up once the ability itself has resolved. */
const REVEAL_HOLD_MS = 6000;

interface TableNotice {
  id: number;
  tone: 'success' | 'danger' | 'info' | 'warning';
  text: string;
}

interface PileSnapshot {
  gameId: string;
  stock: number;
  discardTopId: string | null;
  hands: Record<string, number>;
}

/**
 * Transient table notice derived from state deltas. A hand that grows is a snap
 * penalty (nothing else adds a card to a hand mid-game; the drawn card is held
 * apart from the hand), a hand that shrinks as the discard top changes is a
 * successful snap, and a stockpile that grows is a reshuffle. Each notice
 * clears itself after a few seconds. Penalty cards are drawn unseen, so this
 * never names a face (cambia-820).
 */
function useTableNotice(gs: ObfGameState, selfId: string | undefined, names: Map<string, string>): TableNotice | null {
  const [notice, setNotice] = useState<TableNotice | null>(null);
  const prev = useRef<PileSnapshot | null>(null);

  // The one notice that is not a state delta: an action the hub discarded on its staleness
  // gate that useSocket could not safely resend, because the board moved under it (cambia-891).
  // The socket only counts them; the copy lives here with the rest of the table's voice.
  const droppedNonce = useGameStore(selectDroppedActionNonce);
  const seenDrop = useRef(droppedNonce);
  useEffect(() => {
    if (droppedNonce === seenDrop.current) return;
    seenDrop.current = droppedNonce;
    setNotice({ id: Date.now(), tone: 'warning', text: 'That did not go through.' });
  }, [droppedNonce]);

  useEffect(() => {
    const snap: PileSnapshot = {
      gameId: gs.gameId,
      stock: gs.stockpileSize,
      discardTopId: gs.discardTop?.id ?? null,
      hands: Object.fromEntries(gs.players.map((p) => [p.playerId, p.handSize]))
    };
    const before = prev.current;
    prev.current = snap;
    if (!before || before.gameId !== snap.gameId || !gs.started || gs.gameOver) return;

    let next: Omit<TableNotice, 'id'> | null = null;
    for (const p of gs.players) {
      const was = before.hands[p.playerId];
      if (was === undefined) continue;
      const you = p.playerId === selfId;
      if (p.handSize > was) {
        const who = names.get(p.playerId) ?? 'Opponent';
        next = { tone: 'danger', text: you ? 'Snap missed. A penalty card joins your hand.' : `Snap missed. ${who} draws a penalty card.` };
      } else if (p.handSize < was && snap.discardTopId !== before.discardTopId) {
        const who = names.get(p.playerId) ?? 'Opponent';
        next = { tone: 'success', text: you ? 'Snap. Your card matched the discard.' : `Snap. ${who} matched the discard.` };
      }
    }
    if (snap.stock > before.stock) {
      next = { tone: 'info', text: 'Discard pile reshuffled into the stock.' };
    }
    if (next) setNotice({ id: Date.now(), ...next });
  }, [gs, selfId, names]);

  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(null), 4500);
    return () => window.clearTimeout(t);
  }, [notice]);

  return notice;
}

const NOTICE_TONES: Record<TableNotice['tone'], { color: string; border: string }> = {
  success: { color: 'var(--status-success)', border: 'var(--status-success-border)' },
  danger: { color: 'var(--status-danger)', border: 'var(--status-danger-border)' },
  info: { color: 'var(--status-info)', border: 'var(--status-info-border)' },
  warning: { color: 'var(--status-warning)', border: 'var(--status-warning-border)' }
};

/**
 * Status pill placed on the felt. Badge's tinted fill is built for a panel: on green the
 * light-theme foreground drops below 3:1, so a pill on the felt sits on an opaque
 * surface-1 chip with the status color on top, like the notice line.
 */
const FeltChip: React.FC<{ tone: 'warning' | 'danger' | 'info' | 'success'; children: React.ReactNode }> = ({ tone, children }) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '2px 10px',
      background: 'var(--surface-1)',
      border: '1px solid ' + NOTICE_TONES[tone].border,
      borderRadius: 'var(--radius-pill)',
      color: NOTICE_TONES[tone].color,
      fontSize: 'var(--ds-text-xs)',
      fontWeight: 'var(--weight-bold)',
      lineHeight: 1.5,
      whiteSpace: 'nowrap'
    }}
  >
    <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor', flex: 'none' }}></span>
    {children}
  </span>
);

const FELT_LABEL: React.CSSProperties = {
  ...EYEBROW,
  marginTop: 8,
  color: 'var(--text-on-felt-muted)',
  fontVariantNumeric: 'tabular-nums',
  whiteSpace: 'nowrap'
};

/** Outlined empty pile slot: same footprint as a md card, hairline on the felt. */
const EmptySlot: React.FC<{ onClick?: () => void; highlight?: boolean; label?: string }> = ({ onClick, highlight, label }) => (
  <div
    role={onClick ? 'button' : undefined}
    tabIndex={onClick ? 0 : undefined}
    aria-label={label}
    onClick={onClick}
    onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } } : undefined}
    style={{
      width: 'var(--card-w-md)',
      height: 'var(--card-h-md)',
      boxSizing: 'border-box',
      borderRadius: 'var(--radius-playing-card)',
      border: '1px dashed ' + (highlight ? 'var(--border-accent)' : 'var(--border-on-felt)'),
      cursor: onClick ? 'pointer' : 'default'
    }}
  />
);

const DsGameTable: React.FC<DsGameTableProps> = ({ gameState, phase, sendMessage, onLeave, connected = true, connectionError = null }) => {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [kingPair, setKingPair] = useState<KingPair | null>(null);

  const selfId = useAuthStore((s) => s.user?.id);
  const authName = useAuthStore((s) => s.user?.username);
  const pendingAction = useGameStore(selectPendingAction);
  const isMyTurn = useGameStore(selectIsSelfTurn);
  const isProcessing = useGameStore(selectIsProcessingAction);
  // Every interaction gate reads `busy`: an action in flight or a dropped socket both lock
  // the felt. The hook retries a dropped socket by itself, so the notice says so unless it
  // reported that it stopped (cambia-848 F1).
  const offline = !connected;
  const gaveUp = offline && !!connectionError && /stopped|after \d+ retries/i.test(connectionError);
  const busy = isProcessing || offline;
  const displayedDrawnCard = useGameStore(selectDisplayedDrawnCard);
  const serverClockOffsetMs = useGameStore(selectServerClockOffsetMs);
  const abilityReveal = useGameStore(selectAbilityReveal);
  const matchState = useCurrentLobbyStore((s) => s.matchState);
  const lobbyPlayers = useCurrentLobbyStore((s) => s.lobbyDetails?.lobby_status?.users);

  const selfState = gameState.players.find((p) => p.playerId === selfId);
  const opponents = gameState.players.filter((p) => p.playerId !== selfId);
  const specialAction = gameState.specialAction;
  const specialRank = pendingAction === 'special_action' && specialAction ? specialAction.cardRank : null;
  const turnTimerSec = gameState.houseRules?.turnTimerSec ?? 0;
  const currentPlayer = gameState.players.find((p) => p.playerId === gameState.currentPlayerId);
  // turnId is 0-based at both sources (the adapter's g.TurnID on game_player_turn,
  // the engine's TurnNumber on a sync), so the opening turn arrives as 0 and the
  // old `> 0` guard hid it. Display the ordinal (cambia-876, DL-4 review F7).
  const turnNo = gameState.started && typeof gameState.turnId === 'number' && gameState.turnId >= 0 ? gameState.turnId + 1 : null;
  const preGame = !!gameState.preGameActive && !gameState.started;

  // Display names: the game snapshot's username, then the lobby roster, then the signed-in
  // user's own name, then a seat number. The snapshot's username can arrive empty.
  const names = useMemo(() => {
    const m = new Map<string, string>();
    gameState.players.forEach((p, i) => {
      const fromLobby = (lobbyPlayers ?? []).find((u) => u.id === p.playerId)?.username;
      const own = p.playerId === selfId ? authName : undefined;
      m.set(p.playerId, p.username || fromLobby || own || `Player ${i + 1}`);
    });
    return m;
  }, [gameState.players, lobbyPlayers, selfId, authName]);
  const nameOf = useCallback((id: string | null | undefined) => (id ? names.get(id) : undefined) ?? 'Player', [names]);
  const notice = useTableNotice(gameState, selfId, names);

  // --- Interaction handlers (semantics unchanged from GameBoard) ---

  const handlePlayerCardClick = useCallback((card: ObfCard, idx: number) => {
    if (busy) return;
    if (pendingAction === 'discard_replace') {
      sendMessage(replaceAction(card.id, idx));
      setSelectedIdx(null);
      return;
    }
    if (pendingAction === 'special_action' && specialAction) {
      const rank = specialAction.cardRank;
      if (rank === '7' || rank === '8') {
        sendMessage(peekSelfAction(card.id, idx));
        setSelectedIdx(null);
        return;
      }
      if ((rank === 'J' || rank === 'Q') && selectedIdx === null) {
        setSelectedIdx(idx);
        return;
      }
      if (rank === 'K' && selectedIdx === null) {
        setSelectedIdx(idx);
        return;
      }
    }
    if (pendingAction === null) {
      setSelectedIdx((prev) => (prev === idx ? null : idx));
    }
  }, [busy, pendingAction, specialAction, selectedIdx, sendMessage]);

  const handleDeckClick = useCallback(() => {
    if (!isMyTurn || pendingAction !== null || busy) return;
    sendMessage(drawStockpileAction());
  }, [isMyTurn, pendingAction, busy, sendMessage]);

  const handleDiscardClick = useCallback(() => {
    if (busy) return;
    if (isMyTurn && pendingAction === null) {
      if (gameState.houseRules.allowDrawFromDiscardPile && gameState.discardTop) {
        sendMessage(drawDiscardPileAction());
        return;
      }
    }
    if (pendingAction === 'discard_replace') {
      if (selfState?.drawnCard) {
        sendMessage(discardAction(selfState.drawnCard.id));
        setSelectedIdx(null);
        return;
      }
    }
    if (selectedIdx !== null && pendingAction === null) {
      const selectedCard = selfState?.revealedHand?.[selectedIdx];
      if (selectedCard) {
        sendMessage(snapAction(selectedCard.id));
        setSelectedIdx(null);
      }
    }
  }, [isMyTurn, busy, pendingAction, selectedIdx, selfState, gameState, sendMessage]);

  const handleOpponentCardClick = useCallback((playerId: string, card: ObfCard, idx: number) => {
    if (busy) return;
    // Target opponent cards by their real server-assigned UUID (card.id), sourced from the
    // opponent's revealedHand slot (hidden id references, cambia-509).
    if (pendingAction === 'special_action' && specialAction) {
      const rank = specialAction.cardRank;
      if (rank === '9' || rank === 'T') {
        sendMessage(peekOtherAction(card.id, idx, playerId));
        setSelectedIdx(null);
        return;
      }
      if ((rank === 'J' || rank === 'Q') && selectedIdx !== null) {
        const myCard = selfState?.revealedHand?.[selectedIdx];
        if (myCard && selfId) {
          sendMessage(blindSwapAction(myCard.id, selectedIdx, selfId, card.id, idx, playerId));
          setSelectedIdx(null);
        }
        return;
      }
      if (rank === 'K' && selectedIdx !== null) {
        const myCard = selfState?.revealedHand?.[selectedIdx];
        if (myCard && selfId) {
          sendMessage(kingPeekAction(myCard.id, selectedIdx, selfId, card.id, idx, playerId));
          // The King is two steps on the wire: swap_peek reveals both cards, then
          // swap_peek_swap or skip settles them. Remember the pair for the second step.
          setKingPair({ myId: myCard.id, myIdx: selectedIdx, oppId: card.id, oppIdx: idx, oppOwner: playerId });
          setSelectedIdx(null);
        }
        return;
      }
    }
    if (selectedIdx !== null && pendingAction === null) {
      const allowOpponentSnapping = gameState.houseRules.allowOpponentSnapping ?? true;
      if (allowOpponentSnapping) {
        const selectedCard = selfState?.revealedHand?.[selectedIdx];
        if (selectedCard) {
          sendMessage(snapAction(selectedCard.id));
          setSelectedIdx(null);
        }
      }
    }
  }, [busy, pendingAction, specialAction, selectedIdx, selfState, selfId, gameState, sendMessage]);

  const snapSelected = useCallback(() => {
    if (busy || selectedIdx === null || pendingAction !== null) return;
    const selectedCard = selfState?.revealedHand?.[selectedIdx];
    if (selectedCard) {
      sendMessage(snapAction(selectedCard.id));
      setSelectedIdx(null);
    }
  }, [busy, selectedIdx, pendingAction, selfState, sendMessage]);

  // The King's second step is over once the ability resolves or the turn moves on.
  useEffect(() => {
    if (pendingAction !== 'special_action' || specialAction?.cardRank !== 'K') setKingPair(null);
  }, [pendingAction, specialAction]);

  const confirmKingSwap = useCallback((swap: boolean) => {
    if (!kingPair || !selfId || busy) return;
    sendMessage(swap
      ? kingSwapConfirmAction(kingPair.myId, kingPair.myIdx, selfId, kingPair.oppId, kingPair.oppIdx, kingPair.oppOwner)
      : skipSpecialAction());
    setKingPair(null);
  }, [kingPair, selfId, busy, sendMessage]);

  // --- Derived flags ---

  const roundOver = phase === 'round_end' || gameState.gameOver;
  const canTakeDiscard = isMyTurn && pendingAction === null && !busy && !!gameState.houseRules.allowDrawFromDiscardPile && !!gameState.discardTop;
  const deckInteractive = isMyTurn && pendingAction === null && !busy && gameState.stockpileSize > 0;
  const discardInteractive =
    canTakeDiscard ||
    (pendingAction === 'discard_replace' && !!selfState?.drawnCard) ||
    (selectedIdx !== null && pendingAction === null);
  const canSnap = selectedIdx !== null && pendingAction === null && !busy;
  const canCallCambia = isMyTurn && pendingAction === null && !busy && !gameState.cambiaCalled && gameState.started && !gameState.gameOver;
  const kingConfirm = !!kingPair && specialRank === 'K' && isMyTurn && !busy;
  const canSkipSpecial = isMyTurn && pendingAction === 'special_action' && !busy && !kingConfirm;
  const allowOpponentSnapping = gameState.houseRules.allowOpponentSnapping ?? true;

  // Ability reveals (cambia-848 F3). Own faces are durable in revealedHand (the store folds
  // them in), so the table only has to show an opponent face: for the whole King confirm,
  // and for a short hold after a 9/T peek or a settled King. A tick re-renders once the hold
  // ends so the face goes back down without another store event.
  const [, setRevealTick] = useState(0);
  useEffect(() => {
    if (!abilityReveal) return;
    const left = REVEAL_HOLD_MS - (Date.now() - abilityReveal.at);
    if (left <= 0) return;
    const t = window.setTimeout(() => setRevealTick((n) => n + 1), left + 30);
    return () => window.clearTimeout(t);
  }, [abilityReveal]);
  const revealShown = !!abilityReveal && !roundOver && (kingConfirm || Date.now() - abilityReveal.at < REVEAL_HOLD_MS);
  const revealById = useMemo(() => {
    const m = new Map<string, ObfCard>();
    if (!revealShown || !abilityReveal) return m;
    for (const c of abilityReveal.cards) m.set(c.id, { id: c.id, known: true, rank: c.rank, suit: c.suit, value: c.value, idx: c.idx });
    return m;
  }, [revealShown, abilityReveal]);

  // Legal-target highlighting. The click handlers above already no-op outside these
  // cases; this only decides what the felt shows as a target.
  const opponentTargetable = (() => {
    if (busy || kingConfirm) return false;
    if (specialRank) {
      if (specialRank === '9' || specialRank === 'T') return true;
      if (specialRank === 'J' || specialRank === 'Q' || specialRank === 'K') return selectedIdx !== null;
      return false;
    }
    return selectedIdx !== null && pendingAction === null && allowOpponentSnapping;
  })();
  const ownTargetable = (() => {
    if (busy || kingConfirm) return false;
    if (pendingAction === 'discard_replace') return true;
    if (specialRank === '7' || specialRank === '8') return true;
    if (specialRank === 'J' || specialRank === 'Q' || specialRank === 'K') return selectedIdx === null;
    return false;
  })();

  const hint = useMemo(() => {
    if (gaveUp) return 'Connection lost. Leave the table and rejoin from the dashboard.';
    if (offline) return 'Connection lost. Reconnecting.';
    if (roundOver) return phase === 'round_end' ? 'Round over. Waiting for the next round.' : 'Game over.';
    // No pre-game deadline reaches the client, so the table shows no countdown
    // during the peek window and the copy must not point at one (cambia-876,
    // DL-4 review F5).
    if (preGame) return 'Memorize your peeked cards. Play starts in a moment.';
    // A snap selection is actionable out of turn (snapping is), so it outranks
    // the whose-turn line, which otherwise sat above the Snap button that the
    // selection had just enabled (cambia-876, DL-4 review F6).
    if (selectedIdx !== null && pendingAction === null) return 'Snap the selected card onto the discard, or pick another card.';
    if (!isMyTurn) {
      if (specialAction?.active && currentPlayer && specialAction.playerId === currentPlayer.playerId) {
        return `${nameOf(currentPlayer.playerId)} is choosing a target for ${abilityName(specialAction.cardRank)?.toLowerCase() ?? 'an ability'}.`;
      }
      return currentPlayer ? `Waiting for ${nameOf(currentPlayer.playerId)}.` : 'Waiting for the next turn.';
    }
    if (specialRank === '7' || specialRank === '8') return 'Peek: choose one of your cards to look at.';
    if (specialRank === '9' || specialRank === 'T') return 'Peek: choose an opponent card to look at.';
    if (specialRank === 'J' || specialRank === 'Q') return selectedIdx === null ? 'Blind swap: choose one of your cards.' : 'Blind swap: now choose the opponent card.';
    if (kingConfirm) return 'King: both cards are face up. Swap them, or keep them where they are.';
    if (specialRank === 'K') return selectedIdx === null ? 'King: choose one of your cards.' : 'King: now choose the opponent card to look at.';
    if (pendingAction === 'discard_replace') return 'Swap the drawn card into a slot, or discard it.';
    if (gameState.cambiaCalled) return canTakeDiscard ? 'Last turn. Draw from the stock or take the discard.' : 'Last turn. Draw from the stock.';
    return canTakeDiscard ? 'Your turn. Draw from the stock or take the discard.' : 'Your turn. Draw from the stock.';
  }, [gaveUp, offline, roundOver, phase, preGame, isMyTurn, specialAction, currentPlayer, nameOf, specialRank, kingConfirm, selectedIdx, pendingAction, gameState.cambiaCalled, canTakeDiscard]);

  const discardFace = toDsCardFace(gameState.discardTop);
  const drawnCard = selfState?.drawnCard ?? displayedDrawnCard;
  const drawnFace = toDsCardFace(drawnCard);

  // Standings: circuit cumulative totals when present, else live hand counts.
  const cumulative = matchState?.cumulativeScores;
  const hasTotals = !!cumulative && Object.keys(cumulative).length > 0;
  const standings = useMemo(() => {
    if (cumulative && Object.keys(cumulative).length > 0) {
      return Object.keys(cumulative)
        .map((id) => ({ id, name: names.get(id) ?? (lobbyPlayers ?? []).find((u) => u.id === id)?.username ?? id.substring(0, 6), score: cumulative[id] }))
        .sort((a, b) => a.score - b.score);
    }
    return gameState.players.map((p) => ({ id: p.playerId, name: names.get(p.playerId) ?? 'Player', score: p.handSize }));
  }, [cumulative, lobbyPlayers, gameState.players, names]);

  const cambiaCaller = gameState.cambiaCalled
    ? gameState.players.find((p) => p.playerId === gameState.cambiaCallerId)
    : undefined;

  const renderHand = () => {
    const hand = selfState?.revealedHand ?? [];
    const known = hand.map((card, i) => {
      const face = toDsCardFace(card);
      return (
        <PlayingCard
          key={card.id || i}
          faceDown={!face}
          rank={face?.rank}
          suit={face?.suit}
          size='md'
          selected={selectedIdx === i || (kingConfirm && kingPair?.myIdx === i)}
          highlight={ownTargetable && selectedIdx !== i}
          label={face ? `Your card ${i + 1}: ${face.rank}${face.suit ? ' of ' + face.suit : ''}` : `Your card ${i + 1}, face down`}
          onClick={() => handlePlayerCardClick(card, i)}
        />
      );
    });
    // handSize is authoritative between syncs: a penalty card drawn unseen (cambia-820) can
    // grow the hand before its slot reference is applied, so pad with backs that carry no id
    // and take no click until the next sync fills them in.
    const extra = Math.max(0, (selfState?.handSize ?? 0) - hand.length);
    const padding = Array.from({ length: extra }).map((_, j) => (
      <PlayingCard key={`pad-${j}`} faceDown size='md' label={`Your card ${hand.length + j + 1}, face down`} />
    ));
    return [...known, ...padding];
  };

  return (
    <div className='grid w-full max-w-[1320px] gap-4 p-4 mx-auto lg:grid-cols-[minmax(0,1fr)_300px]' style={{ flex: 1, minHeight: 0, alignItems: 'stretch' }}>
      {/* Rail + felt. The rail is a solid ring of the deep felt; both edges carry a 1px line. */}
      <div style={{ background: 'var(--surface-felt-deep)', border: '1px solid var(--border-default)', borderRadius: 'var(--ds-radius-xl)', padding: 10, display: 'flex', minWidth: 0 }}>
        <div
          style={{
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: 12,
            background: 'var(--surface-felt)',
            border: '1px solid var(--border-on-felt)',
            borderRadius: 'var(--ds-radius-lg)',
            padding: '14px 16px 16px',
            minHeight: 520,
            color: 'var(--text-on-green)'
          }}
        >
          {/* Top strip: turn readout, Cambia call. */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', minHeight: 24 }}>
            <span style={{ ...FELT_LABEL, marginTop: 0 }}>
              {[matchState ? `Round ${matchState.currentRound}/${matchState.totalRounds}` : null, turnNo !== null ? `Turn ${turnNo}` : null].filter(Boolean).join(' · ')}
            </span>
            {/* Shared eyebrow for the caps run: the inline copy dropped wordSpacing
                and the chip read CAMBIACALLED (cambia-892, DL-7 F1). Size and
                colour stay the chip's own. */}
            {gameState.cambiaCalled && (
              <span
                style={{
                  ...EYEBROW,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '3px 12px',
                  background: 'var(--accent-danger)',
                  color: 'var(--text-on-danger)',
                  border: '1px solid var(--accent-danger)',
                  borderRadius: 'var(--radius-pill)',
                  fontSize: 'var(--ds-text-xs)',
                  whiteSpace: 'nowrap'
                }}
              >
                Cambia called{cambiaCaller && <span style={{ textTransform: 'none', letterSpacing: 0, wordSpacing: 'normal', fontWeight: 'var(--weight-medium)' }}>by {nameOf(cambiaCaller.playerId)}</span>}
              </span>
            )}
            {offline && <FeltChip tone={gaveUp ? 'danger' : 'warning'}>{gaveUp ? 'Disconnected' : 'Reconnecting'}</FeltChip>}
            {roundOver && <FeltChip tone='warning'>{phase === 'round_end' ? 'Round over' : 'Game over'}</FeltChip>}
          </div>

          {/* Opponent seats and hand backs. */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', gap: 28, flexWrap: 'wrap' }}>
            {opponents.map((opp) => {
              // The store keeps specialAction after game_end, so an ungated note left
              // a stale 'Look and swap' under the seat behind the results overlay
              // (cambia-876, DL-4 review F12).
              const acting = !roundOver && !!specialAction?.active && specialAction.playerId === opp.playerId;
              const note = acting ? abilityName(specialAction?.cardRank) ?? undefined : undefined;
              return (
                <div key={opp.playerId} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                  <PlayerSeat username={nameOf(opp.playerId)} compact handSize={opp.handSize} note={note} state={seatStateFor(opp, gameState.currentPlayerId)} />
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 5 }}>
                    {Array.from({ length: opp.handSize }).map((_, i) => {
                      // handSize drives the slot count (authoritative between syncs); the real card
                      // UUID for targeting comes from the matching revealedHand slot. A slot is only
                      // clickable once its real id is known (cambia-509).
                      const card = opp.revealedHand?.[i];
                      const targetable = opponentTargetable && !!card;
                      const shown = card ? toDsCardFace(revealById.get(card.id)) : null;
                      const who = nameOf(opp.playerId);
                      return (
                        <PlayingCard
                          key={card?.id ?? i}
                          faceDown={!shown}
                          rank={shown?.rank}
                          suit={shown?.suit}
                          size='sm'
                          selected={!!shown}
                          highlight={targetable}
                          dimmed={!!specialRank && !targetable && !shown}
                          label={shown ? `${who} card ${i + 1}, revealed: ${shown.rank}${shown.suit ? ' of ' + shown.suit : ''}` : `${who} card ${i + 1}`}
                          onClick={targetable ? () => handleOpponentCardClick(opp.playerId, card!, i) : undefined}
                        />
                      );
                    })}
                  </div>
                </div>
              );
            })}
            {opponents.length === 0 && (
              <div style={{ color: 'var(--text-on-felt-muted)', fontSize: 'var(--ds-text-sm)' }}>No opponents seated.</div>
            )}
          </div>

          {/* Notice line: snap, penalty, reshuffle, dropped action. Space is reserved so the piles do not jump. */}
          <div style={{ display: 'flex', justifyContent: 'center', minHeight: 30 }} aria-live='polite'>
            {notice && (
              <span
                key={notice.id}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '4px 12px',
                  background: 'var(--surface-1)',
                  border: '1px solid ' + NOTICE_TONES[notice.tone].border,
                  borderRadius: 'var(--radius-pill)',
                  color: NOTICE_TONES[notice.tone].color,
                  fontSize: 'var(--ds-text-sm)',
                  fontWeight: 'var(--weight-medium)',
                  fontVariantNumeric: 'tabular-nums'
                }}
              >
                {notice.text}
              </span>
            )}
          </div>

          {/* Piles. */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', gap: 28, flexWrap: 'wrap' }}>
            <div style={{ textAlign: 'center' }}>
              {gameState.stockpileSize > 0 ? (
                <PlayingCard faceDown size='md' highlight={deckInteractive} label='Stockpile' onClick={deckInteractive ? handleDeckClick : undefined} />
              ) : (
                <EmptySlot label='Stockpile, empty' />
              )}
              <div style={FELT_LABEL}>Stock · {gameState.stockpileSize}</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              {discardFace ? (
                <PlayingCard
                  rank={discardFace.rank}
                  suit={discardFace.suit}
                  size='md'
                  highlight={discardInteractive}
                  label={`Discard pile, top card ${discardFace.rank}`}
                  onClick={discardInteractive ? handleDiscardClick : undefined}
                />
              ) : (
                <EmptySlot label='Discard pile, empty' highlight={discardInteractive} onClick={discardInteractive ? handleDiscardClick : undefined} />
              )}
              <div style={FELT_LABEL}>Discard · {gameState.discardSize}</div>
            </div>
            {drawnCard && (
              <div style={{ textAlign: 'center' }}>
                <PlayingCard faceDown={!drawnFace} rank={drawnFace?.rank} suit={drawnFace?.suit} size='md' selected label='Drawn card' />
                <div style={{ ...FELT_LABEL, color: 'var(--text-on-green)' }}>Drawn</div>
              </div>
            )}
          </div>

          {/* Own hand and the action column. */}
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 6, paddingTop: 6 }}>
                {renderHand()}
              </div>
              <PlayerSeat
                username={nameOf(selfId)}
                isYou
                compact
                handSize={selfState?.handSize}
                state={offline ? 'disconnected' : seatStateFor(selfState ?? ({ playerId: selfId ?? '', connected: true, hasCalledCambia: false } as ObfPlayerState), gameState.currentPlayerId)}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: '1 1 220px', maxWidth: 280, paddingBottom: 2 }}>
              <p style={{ margin: 0, minHeight: 20, textAlign: 'center', fontSize: 'var(--ds-text-sm)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-on-green)' }}>{hint}</p>
              {deckInteractive && <Button onClick={handleDeckClick}>Draw from stock</Button>}
              {canTakeDiscard && <Button variant='secondary' onClick={handleDiscardClick}>Take discard</Button>}
              {pendingAction === 'discard_replace' && selfState?.drawnCard && (
                <Button variant='secondary' onClick={() => { sendMessage(discardAction(selfState.drawnCard!.id)); setSelectedIdx(null); }}>Discard drawn card</Button>
              )}
              {canSnap && <Button onClick={snapSelected}>Snap selected card</Button>}
              {canSnap && <Button variant='ghost' onClick={() => setSelectedIdx(null)}>Cancel</Button>}
              {kingConfirm && <Button onClick={() => confirmKingSwap(true)}>Swap cards</Button>}
              {kingConfirm && <Button variant='secondary' onClick={() => confirmKingSwap(false)}>Keep cards</Button>}
              {canSkipSpecial && <Button variant='secondary' onClick={() => sendMessage(skipSpecialAction())}>Skip ability</Button>}
              {canCallCambia && <Button variant='cambia' onClick={() => sendMessage(callCambiaAction())}>Call Cambia</Button>}
              {turnTimerSec > 0 && !roundOver && !preGame && (
                <TimerBar
                  label='Turn'
                  totalSec={turnTimerSec}
                  remainingSec={turnTimerSec}
                  deadlineMs={gameState.turnDeadline ?? null}
                  clockOffsetMs={serverClockOffsetMs}
                  onFelt
                  style={{ marginTop: 4 }}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Side column: standings and table facts. */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
        <Panel title={hasTotals ? 'Standings' : 'Players'} action={matchState ? <Badge tone='info'>Round {matchState.currentRound}/{matchState.totalRounds}</Badge> : undefined}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {standings.map((row, i) => (
              <div key={row.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 16, fontVariantNumeric: 'tabular-nums' }}>{i + 1}</span>
                <span style={{ fontWeight: 'var(--weight-medium)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: row.id === selfId ? 'var(--accent-gold)' : 'var(--text-primary)' }}>
                  {row.name}{row.id === selfId ? ' (you)' : ''}
                </span>
                <span style={{ fontWeight: 'var(--weight-bold)', fontVariantNumeric: 'tabular-nums' }}>{row.score}</span>
              </div>
            ))}
          </div>
          <div style={{ ...EYEBROW, marginTop: 10, fontWeight: 'var(--weight-regular)' }}>
            {hasTotals ? 'Total score, lower wins' : 'Cards in hand'}
          </div>
        </Panel>
        <Panel title='Table' style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <ScorePill label='Stock' value={gameState.stockpileSize} />
              <ScorePill label='Discard' value={gameState.discardSize} />
              {turnNo !== null && <ScorePill label='Turn' value={turnNo} />}
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {turnTimerSec > 0 ? <Badge>{turnTimerSec}s turns</Badge> : <Badge>No turn timer</Badge>}
              <Badge>{gameState.houseRules.penaltyDrawCount} card penalty</Badge>
              {gameState.houseRules.allowDrawFromDiscardPile && <Badge>Discard draws</Badge>}
              {gameState.houseRules.snapRace && <Badge>Snap race</Badge>}
            </div>
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
              {offline ? (gaveUp ? 'Disconnected.' : 'Reconnecting.') : roundOver ? (phase === 'round_end' ? 'Round over.' : 'Game over.') : preGame ? 'Pre-game peek.' : isMyTurn ? 'Your turn.' : currentPlayer ? `${nameOf(currentPlayer.playerId)} to act.` : 'Waiting for the next turn.'}
            </div>
          </div>
          <Button size='sm' variant='ghost' onClick={onLeave} style={{ marginTop: 12 }}>Leave table</Button>
        </Panel>
      </div>
    </div>
  );
};

export default DsGameTable;
