// src/pages/LobbyPage.tsx
// Single /lobby/:lobbyId route, single WS connection (useSocket -> /ws/{lobbyId}).
// Phase-conditional render of the design-system screens (cambia-484):
//   open | searching | ready_check | countdown -> DsLobbyView
//   in_game | round_end                        -> DsGameTable
//   post_game | match_end                      -> DsResultsView
// The WS lifecycle, URL<->store sync, and redirect logic are carried over from
// the legacy page unchanged; only the rendered surface is re-skinned onto the DS.
import React, { useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { leaveLobby as apiLeaveLobby } from '@/services/lobbyService';
import { useSocket } from '@/hooks/useSocket';
import { useGameStore, selectGameState } from '@/stores/gameStore';
import ErrorBoundary from '@/components/ErrorBoundary';
import DsLobbyView from '@/components/lobby/DsLobbyView';
import DsLobbyConnectState from '@/components/lobby/DsLobbyConnectState';
import DsResultsView from '@/components/lobby/DsResultsView';
import DsGameTable from '@/components/game/DsGameTable';
import LoadingSpinner from '@/components/common/LoadingSpinner';

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

  const isValidLobbyId = useMemo(() => urlLobbyId && typeof urlLobbyId === 'string' && urlLobbyId.length > 5, [urlLobbyId]);

  const { sendMessage, closeSocket } = useSocket(isValidLobbyId ? urlLobbyId : null);

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

  // Deliberate leave: release the server-side membership as well as the local state, so the
  // lobby stops being offered by GET /lobby/active and is torn down once its last member goes
  // (cambia-807). The socket closes first, because connecting to a lobby joins it and the
  // reconnect logic would otherwise re-add the membership this call just released. The request
  // is best-effort: a 404 (lobby already gone) or 409 (game in progress, where leaving is a
  // disconnect rather than a membership change) must still take the user back to the dashboard.
  const handleLeaveLobby = async () => {
    closeSocket();
    if (urlLobbyId) {
      try {
        await apiLeaveLobby(urlLobbyId);
      } catch {
        // Already reported by the service layer; navigating away is the point.
      }
    }
    leaveLobby();
    navigate('/dashboard', { replace: true });
  };

  // Back to lobby: the hub owns the transition, and this asks it for one (cambia-1238). The
  // button used to flip the phase locally, which left this client rendering an open lobby the
  // server did not agree it was in: the hub was still in post_game, where it admits chat and this
  // message and drops everything else, so every ready frame went in the bin until the results
  // timer fired. The results stay up until the server's phase_change lands.
  const handleReturnToLobby = () => {
    sendMessage({ type: 'return_to_lobby' });
  };

  // A table in hand: a dropped socket mid-game must not bounce the player to the dashboard.
  // The hook keeps retrying, the table renders its reconnect state, and a dead socket still
  // has the table's own leave control (cambia-848 F1).
  const gameLive = (phase === 'in_game' || phase === 'round_end') && !!gameState;

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
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: handleLeaveLobby }}>
        <DsGameTable
          gameState={gameState}
          phase={phase}
          sendMessage={sendMessage}
          onLeave={handleLeaveLobby}
          connected={isConnected}
          connectionError={storeError}
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
      action={{ label: 'Back to home', onClick: handleLeaveLobby }}
    />
  );
  if (isConnected && !lobbyDetails) return (
    <DsLobbyConnectState
      message='Waiting for lobby state'
      action={{ label: 'Leave lobby', onClick: handleLeaveLobby, variant: 'ghost' }}
    />
  );
  if (!isConnected && !isLoading && !lobbyDetails) return <DsLobbyConnectState message='Reconnecting to lobby' />;
  if (!lobbyDetails) return (
    <DsLobbyConnectState
      error='Lobby state is missing. Rejoin from the home screen.'
      action={{ label: 'Back to home', onClick: handleLeaveLobby }}
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
        onLeave={handleLeaveLobby}
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
      onLeave={handleLeaveLobby}
    />
  );
};

export default LobbyPage;
