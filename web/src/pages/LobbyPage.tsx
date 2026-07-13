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
import { useSocket } from '@/hooks/useSocket';
import { useGameStore, selectGameState } from '@/stores/gameStore';
import { useDsPreviewTheme } from '@/hooks/useDsPreviewTheme';
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

  // Keep the DS design tokens (which key light mode off data-theme, not the
  // Tailwind .dark class) in sync with the app theme while this page is mounted.
  useDsPreviewTheme();

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

  const handleLeaveLobby = () => {
    closeSocket();
    leaveLobby();
    navigate('/dashboard', { replace: true });
  };

  const handleReturnToLobby = () => {
    useCurrentLobbyStore.getState().setPhase('open');
  };

  const shouldRedirectToDash = !isValidLobbyId || (!!storeError && !isStoreLoading && !isConnected);

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
  if (phase === 'post_game' || phase === 'match_end') {
    return <DsResultsView phase={phase} onReturnToLobby={handleReturnToLobby} onLeave={handleLeaveLobby} />;
  }

  // --- Phase: live game (in_game / round_end) ---
  if (phase === 'in_game' || phase === 'round_end') {
    if (!gameState) {
      return (
        <div className='flex flex-col items-center justify-center h-full pt-10'>
          <LoadingSpinner />
          <p className='mt-2 text-gray-600 dark:text-gray-400'>Loading game...</p>
        </div>
      );
    }
    return <DsGameTable gameState={gameState} phase={phase} sendMessage={sendMessage} onLeave={handleLeaveLobby} />;
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
