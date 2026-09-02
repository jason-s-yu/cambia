// src/pages/LobbyPage.tsx
// Single /lobby/:lobbyId route, single WS connection (useSocket -> /ws/{lobbyId}).
// Phase-conditional render of the design-system screens (cambia-484):
//   open | searching | ready_check | countdown -> DsLobbyView
//   in_game | round_end                        -> DsGameTable
//   post_game | match_end                      -> DsResultsView
// The WS lifecycle, URL<->store sync, and redirect logic are carried over from
// the legacy page unchanged; only the rendered surface is re-skinned onto the DS.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { LeaveRefusedError, leaveLobby as apiLeaveLobby } from '@/services/lobbyService';
import { useSocket } from '@/hooks/useSocket';
import { useGameStore, selectGameState } from '@/stores/gameStore';
import ErrorBoundary from '@/components/ErrorBoundary';
import Button from '@/components/ds/core/Button';
import Modal from '@/components/ds/core/Modal';
import DsLobbyView from '@/components/lobby/DsLobbyView';
import DsLobbyConnectState from '@/components/lobby/DsLobbyConnectState';
import DsResultsView from '@/components/lobby/DsResultsView';
import DsGameTable from '@/components/game/DsGameTable';
import LoadingSpinner from '@/components/common/LoadingSpinner';

/** What the leave flow is asking the player, if anything. */
type LeavePrompt =
  | { kind: 'confirm' }
  | { kind: 'refused'; reason: string };

const LobbyPage: React.FC = () => {
  const { lobbyId: urlLobbyId } = useParams<{ lobbyId: string }>();
  const navigate = useNavigate();
  const hasAttemptedConnectionRef = useRef(false);

  const {
    currentLobbyId: storeLobbyId,
    lobbyDetails,
    isLoading: isStoreLoading,
    error: storeError,
    isConnected,
    setCurrentLobbyId,
    leaveLobby,
    clearError: clearStoreError,
    phase
  } = useCurrentLobbyStore();

  const gameState = useGameStore(selectGameState);
  const selfId = useAuthStore((s) => s.user?.id) ?? null;
  const [leavePrompt, setLeavePrompt] = useState<LeavePrompt | null>(null);

  const isValidLobbyId = useMemo(() => urlLobbyId && typeof urlLobbyId === 'string' && urlLobbyId.length > 5, [urlLobbyId]);

  // `gaveUp` is the hook's own record of why it stopped dialing, handed to the table as a state.
  // The table used to rebuild it from the error copy in the store, which read a clean close from
  // the hub as an ongoing reconnect (cambia-1239 review).
  const { sendMessage, closeSocket, reopenSocket, gaveUp } = useSocket(isValidLobbyId ? urlLobbyId : null);

  // Synchronize URL id with the store id and trigger connection.
  useEffect(() => {
    if (isValidLobbyId && urlLobbyId) {
      if (urlLobbyId !== storeLobbyId) {
        hasAttemptedConnectionRef.current = false;
        setCurrentLobbyId(urlLobbyId);
      } else if (!isConnected && !isStoreLoading && !hasAttemptedConnectionRef.current) {
        hasAttemptedConnectionRef.current = true;
      }
    } else {
      if (storeLobbyId !== null) {
        setCurrentLobbyId(null);
      }
    }
  }, [urlLobbyId, isValidLobbyId, storeLobbyId, isConnected, isStoreLoading, setCurrentLobbyId]);

  const lobbyShortId = useMemo(() => (urlLobbyId ? urlLobbyId.substring(0, 8) : '...'), [urlLobbyId]);

  // A table in hand: a dropped socket mid-game must not bounce the player to the dashboard.
  // The hook keeps retrying, the table renders its reconnect state, and a dead socket still
  // has the table's own leave control (cambia-848 F1).
  const gameLive = (phase === 'in_game' || phase === 'round_end') && !!gameState;

  // The seat the leave flow is deciding about. A forfeited seat is not one: it is out of the
  // round already, so leaving takes nothing from anybody and the server lets it go without a
  // question (cambia-1237's exit affordance is that player). A player with no seat at all -
  // watching a game they were not dealt into - is in the same position.
  const selfSeat = useMemo(
    () => (selfId ? gameState?.players?.find((p) => p.playerId === selfId) ?? null : null),
    [gameState, selfId]
  );

  // A round that has finished holds nobody's seat, so there is nothing to confirm giving up. The
  // same two signals the table reads for its own round-over line (DsGameTable): game_end and
  // game_results flip gameOver before the hub announces round_end, and the server stops counting
  // the seat as live at that same moment (IsGameOver in LeaveLobbyHandler). Asking here would put
  // a sentence in front of the player - the game is still running, leaving forfeits your seat -
  // that the scoreboard behind the dialog has already contradicted (cambia-1520).
  const roundOver = phase === 'round_end' || !!gameState?.gameOver;
  const seatAtRisk = gameLive && !roundOver && !!selfSeat && !selfSeat.forfeited;

  // Deliberate leave: release the server-side membership as well as the local state, so the
  // lobby stops being offered by GET /lobby/active and is torn down once its last member goes
  // (cambia-807). The socket closes first, because connecting to a lobby joins it and the
  // reconnect logic would otherwise re-add the membership this call just released.
  //
  // A refusal is the one answer that does not end at the dashboard (cambia-1520). It means the
  // membership is still held and the seat is still in the round, so navigating away left the
  // player looking at the dashboard while their seat sat on the table until the disconnect grace
  // ran out and forfeited it for them. The connection the leave closed is taken back and the
  // server's own sentence is put in front of them instead. Every other failure stays best-effort:
  // a lobby that is already gone, or a request that never arrived, must not strand somebody on a
  // table they asked to leave.
  const performLeave = useCallback(async (forfeit: boolean) => {
    setLeavePrompt(null);
    closeSocket();
    if (urlLobbyId) {
      try {
        await apiLeaveLobby(urlLobbyId, { forfeit });
      } catch (error) {
        if (error instanceof LeaveRefusedError) {
          reopenSocket();
          setLeavePrompt({ kind: 'refused', reason: error.reason });
          return;
        }
        // Already reported by the service layer; navigating away is the point.
      }
    }
    leaveLobby();
    navigate('/dashboard', { replace: true });
  }, [urlLobbyId, closeSocket, reopenSocket, leaveLobby, navigate]);

  // What every Leave control on the page calls. Giving up a live seat decides a round for
  // everybody at the table, so it is asked about first and the forfeit is sent as the player's
  // own answer; leaving anything else is the plain leave it always was (cambia-1520).
  const requestLeave = useCallback(() => {
    if (seatAtRisk) {
      setLeavePrompt({ kind: 'confirm' });
      return;
    }
    void performLeave(false);
  }, [seatAtRisk, performLeave]);

  // Back to lobby: the hub owns the transition, and this asks it for one (cambia-1238). The
  // button used to flip the phase locally, which left this client rendering an open lobby the
  // server did not agree it was in: the hub was still in post_game, where it admits chat and this
  // message and drops everything else, so every ready frame went in the bin until the results
  // timer fired. The results stay up until the server's phase_change lands.
  const handleReturnToLobby = () => {
    sendMessage({ type: 'return_to_lobby' });
  };

  const shouldRedirectToDash = !isValidLobbyId || (!!storeError && !isStoreLoading && !isConnected && !gameLive);

  // Bounced off a broken or unreachable lobby. Local state only: this is not a deliberate
  // leave, so the membership stays and the lobby remains resumable once it is reachable again.
  useEffect(() => {
    if (shouldRedirectToDash) {
      if (storeError) clearStoreError();
      closeSocket();
      leaveLobby();
      navigate('/dashboard', { replace: true });
    }
  }, [shouldRedirectToDash, isValidLobbyId, storeError, isStoreLoading, isConnected, navigate, clearStoreError, closeSocket, leaveLobby]);

  const isLoading = isStoreLoading || (storeLobbyId === urlLobbyId && isValidLobbyId && !isConnected && !storeError);

  // The screen under the leave dialogs. Written as one function rather than as the page's own
  // returns so that a dialog outlives every branch: the answer to "are you sure" has to survive
  // the phase changing under it, and a refusal has to be readable from the table it kept the
  // player on (cambia-1520).
  const renderContent = (): React.ReactNode => {
    // --- Pre-connection / error states (DS chrome, cambia-847) ---
    if (!isValidLobbyId) return <DsLobbyConnectState />;

    // --- Phase: live game (in_game / round_end) with a table in hand ---
    // Ahead of the connection chrome so a socket drop keeps the table mounted; the table shows
    // the reconnect state itself and locks its controls until the hook is back (cambia-848 F1).
    if (gameLive && gameState) {
      return (
        // Boundary scoped to the table itself, nested inside the app shell's own boundary
        // (App.tsx): a render throw here is caught before it reaches the shell, so the shell,
        // its nav and this same leave path stay reachable even when the table cannot (cambia-1235).
        <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: requestLeave }}>
          <DsGameTable
            gameState={gameState}
            phase={phase}
            sendMessage={sendMessage}
            onLeave={requestLeave}
            connected={isConnected}
            gaveUp={gaveUp}
          />
        </ErrorBoundary>
      );
    }
    if (isLoading) return (
      <DsLobbyConnectState message={isStoreLoading ? 'Joining lobby' : `Connecting to lobby ${lobbyShortId}`} />
    );
    if (storeError) return (
      <DsLobbyConnectState
        error={storeError}
        onClearError={clearStoreError}
        action={{ label: 'Back to home', onClick: requestLeave }}
      />
    );
    if (isConnected && !lobbyDetails) return (
      <DsLobbyConnectState
        message='Waiting for lobby state'
        action={{ label: 'Leave lobby', onClick: requestLeave, variant: 'ghost' }}
      />
    );
    if (!isConnected && !isLoading && !lobbyDetails) return <DsLobbyConnectState message='Reconnecting to lobby' />;
    if (!lobbyDetails) return (
      <DsLobbyConnectState
        error='Lobby state is missing. Rejoin from the home screen.'
        action={{ label: 'Back to home', onClick: requestLeave }}
      />
    );

    // --- Phase: results (post_game / match_end) ---
    // The finished table stays in gameStore until the next game_started, so the results
    // render as an overlay above it (cambia-848); DsResultsView falls back to a bare card
    // when no table is available (a reload straight into post_game).
    if (phase === 'post_game' || phase === 'match_end') {
      return (
        <DsResultsView
          phase={phase}
          onReturnToLobby={handleReturnToLobby}
          onLeave={requestLeave}
          gameState={gameState}
          sendMessage={sendMessage}
        />
      );
    }

    // --- Phase: live game (in_game / round_end) before the first sync ---
    // The table itself renders above, ahead of the connection chrome; this is the gap between
    // game_started and the first private_sync_state.
    if (phase === 'in_game' || phase === 'round_end') {
      return (
        <div className='flex flex-col items-center justify-center h-full pt-10'>
          <LoadingSpinner />
          <p className='mt-2' style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>Setting the table.</p>
        </div>
      );
    }

    // --- Phase: lobby (open / searching / ready_check / countdown) ---
    return (
      <DsLobbyView
        lobbyId={urlLobbyId as string}
        phase={phase}
        sendMessage={sendMessage}
        onLeave={requestLeave}
      />
    );
  };

  const dismissPrompt = () => setLeavePrompt(null);

  return (
    <>
      {renderContent()}

      {/* Giving up a live seat is asked about, never assumed. Leaving used to close the socket
          and go, and the seat then sat in the round being played by the turn clock until the
          disconnect grace forfeited it, which is not something a player should discover after
          the fact (cambia-1520). Confirming sends the forfeit as their own answer, so the seat
          goes now instead of a minute later. The destructive control carries the cambia variant,
          which is what keeps it out of the dialog's opening focus. */}
      <Modal
        open={leavePrompt?.kind === 'confirm'}
        title='Leave the table?'
        onClose={dismissPrompt}
        footer={(
          <>
            <Button variant='secondary' testId='leave-cancel' onClick={dismissPrompt}>Stay at the table</Button>
            <Button variant='cambia' testId='leave-forfeit' onClick={() => void performLeave(true)}>Leave and forfeit</Button>
          </>
        )}
      >
        <p style={{ margin: 0, fontSize: 'var(--text-md)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-secondary)' }}>
          The game is still running. Leaving forfeits your seat: your score stops counting in this
          round and the table plays on without you.
        </p>
      </Modal>

      {/* The server refused the leave, so nothing has changed: the membership is still held and
          the seat is still in the round. The reason is the server's own sentence. */}
      <Modal
        open={leavePrompt?.kind === 'refused'}
        title='Still at the table'
        onClose={dismissPrompt}
        footer={<Button variant='secondary' testId='leave-refused-dismiss' onClick={dismissPrompt}>Back to the game</Button>}
      >
        <p style={{ margin: 0, fontSize: 'var(--text-md)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-secondary)' }}>
          {leavePrompt?.kind === 'refused' ? leavePrompt.reason : ''}
        </p>
      </Modal>
    </>
  );
};

export default LobbyPage;
