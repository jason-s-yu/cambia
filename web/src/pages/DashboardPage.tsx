import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AxiosError } from 'axios';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Modal from '@/components/ds/core/Modal';
import Select from '@/components/ds/core/Select';
import QueueCard from '@/components/ds/data/QueueCard';
import TierBadge from '@/components/ds/data/TierBadge';
import StatRow from '@/components/ds/data/StatRow';
import Panel from '@/components/ds/chrome/Panel';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore, useLobbyListStore } from '@/stores/lobbyStore';
import { useQueueStore } from '@/stores/queueStore';
import { useFriendsStore } from '@/stores/friendsStore';
import { joinLobby as apiJoinPublicLobby } from '@/services/lobbyService';
import type { QueueInfo } from '@/services/matchmakingService';
import type { ApiErrorResponse, LobbyState } from '@/types';
import { gameModeLabel } from '@/utils/gameMode';

/** Queues considered "flagship" for the primary/highlighted card treatment. */
const PRIMARY_QUEUE_IDS = new Set(['h2h_rapid', 'ffa4_standard']);

/**
 * Rough estimated match length in minutes from queue shape, since the
 * matchmaking queues endpoint does not return one directly. FFA queues
 * (>2 players) run longer per round than head-to-head.
 */
function estimateMinutes(queue: QueueInfo): number {
  const perRound = queue.players > 2 ? 7 : 5;
  return Math.max(perRound, Math.round(queue.rounds * perRound));
}

/** Short fallback label for a lobby without a display name. */
function lobbyFallbackName(lobbyId: string): string {
  return `Lobby ${lobbyId.substring(0, 6)}`;
}

/**
 * Design-system home screen (cambia-483), wired to real data in place of the
 * pages/ds/HomeScreen.tsx preview mocks: matchmaking queues + join/search
 * flow (queueStore), public lobbies (useLobbyListStore), friends
 * (friendsStore) and the authenticated user's ratings (authStore).
 */
const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const authUser = useAuthStore((state) => state.user);

  const queues = useQueueStore((state) => state.queues);
  const queuesLoading = useQueueStore((state) => state.isLoading);
  const queuesError = useQueueStore((state) => state.error);
  const fetchQueues = useQueueStore((state) => state.fetchQueues);
  const searchingQueueId = useQueueStore((state) => state.searchingQueueId);
  const searchStartTime = useQueueStore((state) => state.searchStartTime);
  const joinQueue = useQueueStore((state) => state.joinQueue);
  const cancelQueueSearch = useQueueStore((state) => state.cancelSearch);

  const lobbies = useLobbyListStore((state) => state.lobbies);
  const lobbiesLoading = useLobbyListStore((state) => state.isLoading);
  const lobbiesError = useLobbyListStore((state) => state.error);
  const fetchLobbies = useLobbyListStore((state) => state.fetchLobbies);

  const friends = useFriendsStore((state) => state.friends);
  const friendsLoading = useFriendsStore((state) => state.isLoading);
  const friendsError = useFriendsStore((state) => state.error);
  const fetchFriends = useFriendsStore((state) => state.fetchFriends);

  const createAndJoinLobby = useCurrentLobbyStore((state) => state.createAndJoinLobby);
  const currentLobbyId = useCurrentLobbyStore((state) => state.currentLobbyId);

  const [searchElapsed, setSearchElapsed] = useState(0);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [joinLobbyError, setJoinLobbyError] = useState<string | null>(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [createLobbyType, setCreateLobbyType] = useState<'private' | 'public'>('public');
  const [createGameMode, setCreateGameMode] = useState('head_to_head');
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const previousLobbyIdRef = useRef<string | null>(null);

  useEffect(() => {
    fetchQueues();
    fetchLobbies();
    fetchFriends();
  }, [fetchQueues, fetchLobbies, fetchFriends]);

  // Elapsed timer for an active matchmaking search (ported from PlayPage.tsx).
  useEffect(() => {
    if (searchingQueueId) {
      elapsedTimerRef.current = setInterval(() => {
        setSearchElapsed(Math.floor((Date.now() - searchStartTime) / 1000));
      }, 1000);
    } else {
      setSearchElapsed(0);
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    }
    return () => {
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    };
  }, [searchingQueueId, searchStartTime]);

  // Navigate into the lobby once creation succeeds while the create modal is open.
  useEffect(() => {
    const previousLobbyId = previousLobbyIdRef.current;
    previousLobbyIdRef.current = currentLobbyId;
    if (createOpen && currentLobbyId && currentLobbyId !== previousLobbyId) {
      setCreateOpen(false);
      navigate(`/lobby/${currentLobbyId}`);
    }
  }, [currentLobbyId, createOpen, navigate]);

  const handleOpenCreate = () => {
    setCreateError(null);
    setCreateOpen(true);
  };

  const handleCloseCreate = () => {
    if (creating) return;
    setCreateOpen(false);
    setCreateError(null);
  };

  const handleCreateLobby = useCallback(async () => {
    setCreateError(null);
    setCreating(true);
    try {
      const settings: Partial<LobbyState> = { type: createLobbyType, gameMode: createGameMode };
      const lobbyId = await createAndJoinLobby(settings);
      if (!lobbyId) {
        setCreateError('Failed to create lobby.');
      }
      // Navigation on success is handled by the effect above.
    } finally {
      setCreating(false);
    }
  }, [createAndJoinLobby, createLobbyType, createGameMode]);

  const handlePlayQueue = useCallback((queue: QueueInfo) => {
    joinQueue(queue);
  }, [joinQueue]);

  const handleCancelSearch = useCallback(() => {
    cancelQueueSearch();
  }, [cancelQueueSearch]);

  const handleJoinPublicLobby = useCallback(async (lobbyId: string) => {
    setJoinLobbyError(null);
    try {
      await apiJoinPublicLobby(lobbyId);
      navigate(`/lobby/${lobbyId}`);
    } catch (err) {
      const error = err as AxiosError<ApiErrorResponse>;
      setJoinLobbyError(error.response?.data?.message || error.message || 'Failed to join lobby.');
    }
  }, [navigate]);

  const publicLobbies = Object.entries(lobbies).filter(([, entry]) => entry.lobby?.type === 'public');

  const ratingValue = authUser?.elo !== undefined ? `${Math.round(authUser.elo)}` : '1520';
  const ratingBand = authUser?.rd !== undefined ? `± ${Math.round(authUser.rd)}` : '± 140';
  const ffaValue = authUser?.open_skill_mu !== undefined ? authUser.open_skill_mu.toFixed(1) : 'Placement';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, padding: 22, maxWidth: 1240, margin: '0 auto', width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-3xl)', fontWeight: 'var(--weight-regular)', lineHeight: 'var(--ds-leading-tight)' }}>Lowest score wins.</h1>
          <p style={{ margin: '6px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>Pick a queue — the forest remembers your discards.</p>
        </div>

        {queuesError && <p style={{ margin: 0, color: 'var(--berry-400)', fontSize: 'var(--ds-text-sm)' }}>{queuesError}</p>}
        {queuesLoading && queues.length === 0 && <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>Loading queues...</p>}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 14 }}>
          {queues.map((queue) => {
            const isSearchingThis = searchingQueueId === queue.queueId;
            const disabled = !!searchingQueueId && !isSearchingThis;
            if (isSearchingThis) {
              return (
                <div
                  key={queue.queueId}
                  style={{
                    background: 'var(--surface-card)',
                    border: '1.5px solid var(--border-default)',
                    borderRadius: 'var(--ds-radius-lg)',
                    boxShadow: 'var(--shadow-card)',
                    padding: '16px 18px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
                    <div style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-xl)', lineHeight: 1.15 }}>{queue.name}</div>
                    <Badge tone='info'>Searching</Badge>
                  </div>
                  <div style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)' }}>
                    {Math.floor(searchElapsed / 60)}:{String(searchElapsed % 60).padStart(2, '0')} elapsed
                  </div>
                  <Button variant='cambia' fullWidth onClick={handleCancelSearch}>
                    Cancel
                  </Button>
                </div>
              );
            }
            return (
              <div key={queue.queueId} style={disabled ? { opacity: 0.45, pointerEvents: 'none' } : undefined}>
                <QueueCard
                  name={queue.name}
                  tagline={`${queue.players} players, ${queue.rounds} ${queue.rounds === 1 ? 'round' : 'rounds'}`}
                  players={queue.players}
                  rounds={queue.rounds}
                  minutes={estimateMinutes(queue)}
                  pool={queue.ratingPool}
                  primary={PRIMARY_QUEUE_IDS.has(queue.queueId)}
                  ranked={queue.ranked}
                  onPlay={() => handlePlayQueue(queue)}
                />
              </div>
            );
          })}
        </div>

        <Panel title='Public lobbies' action={<Button size='sm' onClick={handleOpenCreate}>Create lobby</Button>}>
          {lobbiesError && <p style={{ margin: '0 0 8px', color: 'var(--berry-400)', fontSize: 'var(--ds-text-sm)' }}>{lobbiesError}</p>}
          {joinLobbyError && <p style={{ margin: '0 0 8px', color: 'var(--berry-400)', fontSize: 'var(--ds-text-sm)' }}>{joinLobbyError}</p>}
          {lobbiesLoading && publicLobbies.length === 0 && <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>Loading lobbies...</p>}
          {!lobbiesLoading && publicLobbies.length === 0 && !lobbiesError && (
            <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>No public lobbies right now. Start one.</p>
          )}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {publicLobbies.map(([lobbyId, entry], i) => {
              const lobby = entry.lobby;
              const displayName = entry.name || lobbyFallbackName(lobbyId);
              const full = entry.playerCount >= entry.maxPlayers;
              return (
                <div key={lobbyId} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '10px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ fontWeight: 'var(--weight-bold)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName}</span>
                  <Badge>{gameModeLabel(lobby?.gameMode)}</Badge>
                  <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)', width: 42, textAlign: 'right' }}>
                    {entry.playerCount}/{entry.maxPlayers}
                  </span>
                  <Button size='sm' variant='secondary' disabled={full} onClick={() => handleJoinPublicLobby(lobbyId)}>
                    {full ? 'Full' : 'Join'}
                  </Button>
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <Panel title='Your ratings'>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <TierBadge tier='gold' />
            <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-lg)' }}>{ratingValue} <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-xs)' }}>{ratingBand}</span></span>
          </div>
          <StatRow label='H2H Ranked · Glicko-2' value={ratingValue} delta='+12' />
          <StatRow label='FFA-4 Ranked · OpenSkill' value={ffaValue} unit='7/10 games' />
          <StatRow label='Season peak' value='Gold II' />
          <p style={{ margin: '10px 0 0', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>Season 4 ends in 23 days. Peak tier is kept as a badge.</p>
        </Panel>

        <Panel title='Friends'>
          {friendsError && <p style={{ margin: '0 0 8px', color: 'var(--berry-400)', fontSize: 'var(--ds-text-sm)' }}>{friendsError}</p>}
          {friendsLoading && friends.length === 0 && <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>Loading friends...</p>}
          {!friendsLoading && friends.length === 0 && !friendsError && (
            <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)' }}>No friends yet.</p>
          )}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {friends.map((f, i) => {
              const dotColor = f.online === true ? 'var(--moss-500)' : f.online === false ? 'var(--border-strong)' : 'var(--text-tertiary)';
              return (
                <div key={f.userId} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', flex: 'none', background: dotColor }}></span>
                  <span style={{ lineHeight: 1.2, flex: 1 }}>
                    <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)' }}>{f.username}</span>
                    <span style={{ display: 'block', fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{f.status}</span>
                  </span>
                  {f.online === true && <Button size='sm' variant='ghost'>Invite</Button>}
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      <Modal open={createOpen} title='Create New Lobby' onClose={handleCloseCreate} footer={(
        <>
          <Button variant='secondary' onClick={handleCloseCreate} disabled={creating}>Cancel</Button>
          <Button variant='primary' onClick={handleCreateLobby} disabled={creating}>{creating ? 'Creating...' : 'Create Lobby'}</Button>
        </>
      )}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {createError && <p style={{ margin: 0, color: 'var(--berry-400)', fontSize: 'var(--ds-text-sm)' }}>{createError}</p>}
          <Select
            label='Lobby type'
            value={createLobbyType}
            disabled={creating}
            onChange={(e) => setCreateLobbyType(e.target.value as 'private' | 'public')}
            options={[
              { value: 'public', label: 'Public' },
              { value: 'private', label: 'Private' },
            ]}
          />
          <Select
            label='Game mode'
            value={createGameMode}
            disabled={creating}
            onChange={(e) => setCreateGameMode(e.target.value)}
            options={[
              { value: 'head_to_head', label: gameModeLabel('head_to_head') },
              { value: 'group_of_4', label: gameModeLabel('group_of_4') },
            ]}
          />
        </div>
      </Modal>
    </div>
  );
};

export default DashboardPage;
