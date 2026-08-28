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
import DsLobbyView from '@/components/lobby/DsLobbyView';
import DsResultsView from '@/components/lobby/DsResultsView';
import DsGameTable from '@/components/game/DsGameTable';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import Button from '@/components/common/Button';

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

  const handleReturnToLobby = () => {
    useCurrentLobbyStore.getState().setPhase('open');
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

  // --- Pre-connection / error states (Tailwind chrome) ---
  if (!isValidLobbyId) return <div className='flex items-center justify-center h-screen'><LoadingSpinner /></div>;

  // --- Phase: live game (in_game / round_end) with a table in hand ---
  // Ahead of the connection chrome so a socket drop keeps the table mounted; the table shows
  // the reconnect state itself and locks its controls until the hook is back (cambia-848 F1).
  if (gameLive && gameState) {
    return (
      <DsGameTable
        gameState={gameState}
        phase={phase}
        sendMessage={sendMessage}
        onLeave={handleLeaveLobby}
        connected={isConnected}
        connectionError={storeError}
      />
    );
  }
  if (isLoading) return (
    <div className='flex flex-col items-center justify-center h-full pt-10'>
      <LoadingSpinner />
      <p className='mt-2 text-gray-600 dark:text-gray-400'>
        {isStoreLoading ? 'Processing...' : `Connecting to lobby ${lobbyShortId}...`}
      </p>
    </div>
  );
  if (storeError) return (
    <div className='text-center pt-10'>
      <ErrorMessage message={storeError} onClear={clearStoreError} />
      <Button onClick={handleLeaveLobby} className='mt-4'>Back to Dashboard</Button>
    </div>
  );
  if (isConnected && !lobbyDetails) return (
    <div className='flex flex-col items-center justify-center h-full pt-10'>
      <LoadingSpinner />
      <p className='mt-2 text-gray-600 dark:text-gray-400'>Waiting for lobby data...</p>
      <Button onClick={handleLeaveLobby} className='mt-4' variant='secondary'>Leave Lobby</Button>
    </div>
  );
  if (!isConnected && !isLoading && !lobbyDetails) return (
    <div className='flex flex-col items-center justify-center h-full pt-10'>
      <p className='text-yellow-600 dark:text-yellow-400 mb-4'>Attempting to connect to lobby...</p>
      <LoadingSpinner size='sm' />
    </div>
  );
  if (!lobbyDetails) return (
    <div className='flex flex-col items-center justify-center h-full pt-10'>
      <ErrorMessage message='Lobby data is missing. Please try rejoining.' />
      <Button onClick={handleLeaveLobby} className='mt-4'>Back to Dashboard</Button>
    </div>
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
