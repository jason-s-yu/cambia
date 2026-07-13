// src/components/game/DsGameTable.tsx
// DS-styled live table (cambia-484). Re-skins the legacy GameBoard/ActionControls
// onto the design-system GameScreen primitives (PlayerSeat, PlayingCard,
// ScorePill, TimerBar). Interaction + every outgoing WS action is copied from
// GameBoard so the wire protocol is unchanged: clicking the stockpile, discard
// pile, own cards and opponent cards drives draw / discard / replace / snap /
// special / Cambia via the same action constructors. This is a re-skin, not a
// protocol change.
import React, { useCallback, useMemo, useState } from 'react';
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
  callCambiaAction,
  skipSpecialAction
} from '@/types/game';
import {
  useGameStore,
  selectPendingAction,
  selectIsSelfTurn,
  selectIsProcessingAction,
  selectDisplayedDrawnCard
} from '@/stores/gameStore';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Panel from '@/components/ds/chrome/Panel';
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
}

const HINTS = {
  waiting: 'Waiting for other players…',
  yourTurn: 'Your turn. Draw from the stockpile or discard pile.',
  selectReplace: 'Click a hand card to swap it in, or discard the drawn card.',
  selectSnap: 'Click the discard pile to snap the selected card.',
  peekSelf: 'Peek: click one of your own cards.',
  peekOther: 'Peek: click an opponent card.',
  swapBlind: 'Blind swap: pick one of your cards, then an opponent card.',
  swapPeek: 'King: pick one of your cards, then an opponent card to look and swap.',
  cambiaCalled: 'Cambia called - everyone gets one last turn.'
};

const CAP_LABEL: React.CSSProperties = {
  marginTop: 7,
  fontSize: 10,
  fontWeight: 'var(--weight-black)',
  letterSpacing: '0.09em',
  color: 'rgba(243,236,218,0.75)'
};

function seatStateFor(p: ObfPlayerState, currentPlayerId: string | null): PlayerSeatState | undefined {
  if (!p.connected) return 'disconnected';
  if (p.hasCalledCambia) return 'cambia';
  if (p.playerId === currentPlayerId) return 'turn';
  return undefined;
}

const DsGameTable: React.FC<DsGameTableProps> = ({ gameState, phase, sendMessage, onLeave }) => {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const selfId = useAuthStore((s) => s.user?.id);
  const pendingAction = useGameStore(selectPendingAction);
  const isMyTurn = useGameStore(selectIsSelfTurn);
  const isProcessing = useGameStore(selectIsProcessingAction);
  const displayedDrawnCard = useGameStore(selectDisplayedDrawnCard);
  const matchState = useCurrentLobbyStore((s) => s.matchState);
  const lobbyPlayers = useCurrentLobbyStore((s) => s.lobbyDetails?.lobby_status?.users);

  const selfState = gameState.players.find((p) => p.playerId === selfId);
  const opponents = gameState.players.filter((p) => p.playerId !== selfId);
  const specialAction = gameState.specialAction;
  const turnTimerSec = gameState.houseRules?.turnTimerSec ?? 0;

  // --- Interaction handlers (semantics copied verbatim from GameBoard) ---

  const handlePlayerCardClick = useCallback((card: ObfCard, idx: number) => {
    if (isProcessing) return;
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
  }, [isProcessing, pendingAction, specialAction, selectedIdx, sendMessage]);

  const handleDeckClick = useCallback(() => {
    if (!isMyTurn || pendingAction !== null || isProcessing) return;
    sendMessage(drawStockpileAction());
  }, [isMyTurn, pendingAction, isProcessing, sendMessage]);

  const handleDiscardClick = useCallback(() => {
    if (isProcessing) return;
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
  }, [isMyTurn, isProcessing, pendingAction, selectedIdx, selfState, gameState, sendMessage]);

  const handleOpponentCardClick = useCallback((playerId: string, idx: number) => {
    if (isProcessing) return;
    const placeholderId = `${playerId}-card-${idx}`;
    if (pendingAction === 'special_action' && specialAction) {
      const rank = specialAction.cardRank;
      if (rank === '9' || rank === 'T') {
        sendMessage(peekOtherAction(placeholderId, idx, playerId));
        setSelectedIdx(null);
        return;
      }
      if ((rank === 'J' || rank === 'Q') && selectedIdx !== null) {
        const myCard = selfState?.revealedHand?.[selectedIdx];
        if (myCard && selfId) {
          sendMessage(blindSwapAction(myCard.id, selectedIdx, selfId, placeholderId, idx, playerId));
          setSelectedIdx(null);
        }
        return;
      }
      if (rank === 'K' && selectedIdx !== null) {
        const myCard = selfState?.revealedHand?.[selectedIdx];
        if (myCard && selfId) {
          sendMessage(kingPeekAction(myCard.id, selectedIdx, selfId, placeholderId, idx, playerId));
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
  }, [isProcessing, pendingAction, specialAction, selectedIdx, selfState, selfId, gameState, sendMessage]);

  // --- Derived flags (copied from GameBoard) ---

  const hint = useMemo(() => {
    if (gameState.cambiaCalled) return HINTS.cambiaCalled;
    if (!isMyTurn) return HINTS.waiting;
    if (pendingAction === 'special_action' && specialAction) {
      const rank = specialAction.cardRank;
      if (rank === '7' || rank === '8') return HINTS.peekSelf;
      if (rank === '9' || rank === 'T') return HINTS.peekOther;
      if (rank === 'J' || rank === 'Q') return HINTS.swapBlind;
      if (rank === 'K') return HINTS.swapPeek;
    }
    if (pendingAction === 'discard_replace') return HINTS.selectReplace;
    if (selectedIdx !== null) return HINTS.selectSnap;
    return HINTS.yourTurn;
  }, [isMyTurn, pendingAction, selectedIdx, specialAction, gameState.cambiaCalled]);

  const canCallCambia = isMyTurn && pendingAction === null && !isProcessing && !gameState.cambiaCalled && gameState.started && !gameState.gameOver;
  const canSkipSpecial = isMyTurn && pendingAction === 'special_action' && !isProcessing;
  const opponentTargetable = pendingAction === 'special_action' || (selectedIdx !== null && pendingAction === null);
  const deckInteractive = isMyTurn && pendingAction === null && !isProcessing && gameState.stockpileSize > 0;
  const discardInteractive =
    (isMyTurn && pendingAction === null && !!gameState.houseRules.allowDrawFromDiscardPile && !!gameState.discardTop) ||
    pendingAction === 'discard_replace' ||
    (selectedIdx !== null && pendingAction === null);
  const canTakeDiscard = isMyTurn && pendingAction === null && !isProcessing && !!gameState.houseRules.allowDrawFromDiscardPile && !!gameState.discardTop;

  const discardFace = toDsCardFace(gameState.discardTop);
  const drawnCard = selfState?.drawnCard ?? displayedDrawnCard;
  const drawnFace = toDsCardFace(drawnCard);
  const roundOver = phase === 'round_end' || gameState.gameOver;

  // Standings: prefer circuit cumulative scores, else the live seat order.
  const cumulative = matchState?.cumulativeScores;
  const standings = useMemo(() => {
    const names = new Map<string, string>();
    (lobbyPlayers ?? []).forEach((u) => names.set(u.id, u.username));
    gameState.players.forEach((p) => { if (!names.has(p.playerId)) names.set(p.playerId, p.username); });
    if (cumulative && Object.keys(cumulative).length > 0) {
      return Object.keys(cumulative)
        .map((id) => ({ id, name: names.get(id) ?? id.substring(0, 6), score: cumulative[id] }))
        .sort((a, b) => a.score - b.score);
    }
    return gameState.players.map((p) => ({ id: p.playerId, name: p.username, score: p.handSize }));
  }, [cumulative, lobbyPlayers, gameState.players]);
  const scoreLabel = cumulative && Object.keys(cumulative).length > 0 ? 'TOTAL' : 'CARDS';

  const renderHand = () => {
    const hand = selfState?.revealedHand;
    if (hand && hand.length > 0) {
      return hand.map((card, i) => {
        const face = toDsCardFace(card);
        return (
          <PlayingCard
            key={card.id || i}
            faceDown={!face}
            rank={face?.rank}
            suit={face?.suit}
            size='md'
            selected={selectedIdx === i}
            onClick={() => handlePlayerCardClick(card, i)}
          />
        );
      });
    }
    const count = selfState?.handSize ?? 0;
    return Array.from({ length: count }).map((_, i) => <PlayingCard key={i} faceDown size='md' />);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 300px', gap: 18, padding: 18, width: '100%', maxWidth: 1320, margin: '0 auto', alignItems: 'stretch', flex: 1, minHeight: 0 }}>
      <div
        style={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          background: 'var(--surface-table)',
          border: '3px solid var(--outline-ink)',
          borderRadius: 'var(--ds-radius-xl)',
          boxShadow: 'var(--inset-table)',
          padding: '16px 20px',
          minHeight: 620
        }}
      >
        {gameState.cambiaCalled && (
          <div
            style={{
              position: 'absolute',
              top: 12,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 5,
              padding: '8px 18px',
              background: 'var(--berry-500)',
              color: 'var(--text-on-ember)',
              border: '2px solid var(--outline-ink)',
              borderRadius: 'var(--radius-pill)',
              boxShadow: 'var(--shadow-piece)',
              fontWeight: 'var(--weight-black)',
              whiteSpace: 'nowrap'
            }}
          >Cambia called - everyone gets one last turn.</div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', paddingTop: gameState.cambiaCalled ? 34 : 4 }}>
          {opponents.map((opp) => (
            <div key={opp.playerId} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <PlayerSeat username={opp.username} compact handSize={opp.handSize} state={seatStateFor(opp, gameState.currentPlayerId)} />
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 5 }}>
                {Array.from({ length: opp.handSize }).map((_, i) => (
                  <PlayingCard
                    key={i}
                    faceDown
                    size='sm'
                    selected={opponentTargetable}
                    onClick={opponentTargetable ? () => handleOpponentCardClick(opp.playerId, i) : undefined}
                  />
                ))}
              </div>
            </div>
          ))}
          {opponents.length === 0 && (
            <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>Waiting for opponents…</div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-end', gap: 34, margin: '10px 0' }}>
          <div style={{ textAlign: 'center' }}>
            <PlayingCard faceDown size='md' onClick={deckInteractive ? handleDeckClick : undefined} selected={deckInteractive} />
            <div style={CAP_LABEL}>STOCKPILE · <span style={{ fontFamily: 'var(--ds-font-mono)' }}>{gameState.stockpileSize}</span></div>
          </div>
          <div style={{ textAlign: 'center' }}>
            {discardFace ? (
              <PlayingCard rank={discardFace.rank} suit={discardFace.suit} size='md' selected={discardInteractive} onClick={discardInteractive ? handleDiscardClick : undefined} />
            ) : (
              <PlayingCard faceDown size='md' selected={discardInteractive} onClick={discardInteractive ? handleDiscardClick : undefined} />
            )}
            <div style={CAP_LABEL}>DISCARD · <span style={{ fontFamily: 'var(--ds-font-mono)' }}>{gameState.discardSize}</span></div>
          </div>
          {drawnCard && (
            <div style={{ textAlign: 'center' }}>
              <PlayingCard faceDown={!drawnFace} rank={drawnFace?.rank} suit={drawnFace?.suit} size='md' selected />
              <div style={{ ...CAP_LABEL, color: 'var(--honey-300)' }}>DRAWN</div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: 26, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', gap: 6 }}>
              {renderHand()}
            </div>
            <PlayerSeat username={selfState?.username ?? 'You'} isYou compact state={seatStateFor(selfState ?? ({ playerId: selfId ?? '', connected: true, hasCalledCambia: false } as ObfPlayerState), gameState.currentPlayerId)} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: 240, paddingBottom: 4 }}>
            <p style={{ margin: 0, minHeight: 20, textAlign: 'center', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>{hint}</p>
            {deckInteractive && <Button onClick={handleDeckClick}>Draw from stockpile</Button>}
            {canTakeDiscard && <Button variant='secondary' onClick={handleDiscardClick}>Take discard</Button>}
            {pendingAction === 'discard_replace' && selfState?.drawnCard && (
              <Button variant='secondary' onClick={() => { sendMessage(discardAction(selfState.drawnCard!.id)); setSelectedIdx(null); }}>Discard drawn card</Button>
            )}
            {canSkipSpecial && <Button variant='secondary' onClick={() => sendMessage(skipSpecialAction())}>Skip ability</Button>}
            {canCallCambia && <Button variant='cambia' onClick={() => sendMessage(callCambiaAction())}>Call Cambia</Button>}
            {turnTimerSec > 0 && <TimerBar label='TURN' totalSec={turnTimerSec} remainingSec={turnTimerSec} />}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
        <Panel title={matchState ? `Circuit · round ${matchState.currentRound}/${matchState.totalRounds}` : 'Standings'}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {standings.map((row, i) => (
              <div key={row.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 16 }}>{i + 1}</span>
                <span style={{ fontWeight: 'var(--weight-bold)', flex: 1, color: row.id === selfId ? 'var(--honey-400)' : 'var(--text-primary)' }}>
                  {row.name}{row.id === selfId ? ' (you)' : ''}
                </span>
                <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)' }}>{row.score}</span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
            <ScorePill label={scoreLabel} value={standings.length} />
            {roundOver && <Badge tone='warning'>round over</Badge>}
          </div>
        </Panel>
        <Panel title='Table' style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <ScorePill label='STOCK' value={gameState.stockpileSize} />
              <ScorePill label='DISCARD' value={gameState.discardSize} />
            </div>
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
              {isMyTurn ? 'Your turn.' : gameState.gameOver ? 'Game over.' : 'Waiting on other players.'}
            </div>
          </div>
          <Button size='sm' variant='ghost' onClick={onLeave} style={{ marginTop: 12 }}>Leave table</Button>
        </Panel>
      </div>
    </div>
  );
};

export default DsGameTable;
