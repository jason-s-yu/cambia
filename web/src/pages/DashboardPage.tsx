import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AxiosError } from 'axios';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Modal from '@/components/ds/core/Modal';
import Select from '@/components/ds/core/Select';
import Spinner from '@/components/ds/core/Spinner';
import QueueCard from '@/components/ds/data/QueueCard';
import TierBadge from '@/components/ds/data/TierBadge';
import StatRow from '@/components/ds/data/StatRow';
import Panel from '@/components/ds/chrome/Panel';
import DsResumeBanner from '@/components/dashboard/DsResumeBanner';
import { useCurrentLobbyStore, useLobbyListStore } from '@/stores/lobbyStore';
import { useQueueStore } from '@/stores/queueStore';
import { useFriendsStore } from '@/stores/friendsStore';
import { useHistoryStore } from '@/stores/historyStore';
import { joinLobby as apiJoinPublicLobby, getActiveSession } from '@/services/lobbyService';
import type { QueueInfo } from '@/services/matchmakingService';
import type { ActiveSession, ApiErrorResponse, LobbyState } from '@/types';
import { gameModeLabel } from '@/utils/gameMode';
import { ratingPoolLabel, tierFromRating } from '@/utils/ratingPool';

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

/** Secondary-tier inline notice (loading, empty) under a panel title. */
const Note: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-sm)', ...style }}>{children}</p>
);

/** Inline error line in the danger status color. */
const ErrorLine: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <p role='alert' style={{ margin: 0, color: 'var(--status-danger)', fontSize: 'var(--ds-text-sm)', ...style }}>{children}</p>
);

/**
 * Home screen (cambia-483), wired to real data: matchmaking queues +
 * join/search flow (queueStore), public lobbies (useLobbyListStore), friends
 * (friendsStore) and the authenticated user's ratings (historyStore).
 */
const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  // Ratings are fetched once per session by AppLayout (which wraps this page
  // for the whole authenticated app); this reads that shared state rather
  // than fetching again.
  const ratings = useHistoryStore((state) => state.ratings);
  const ratingsError = useHistoryStore((state) => state.ratingsError);

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

  const [activeSession, setActiveSession] = useState<ActiveSession | null>(null);
  const [resumeDismissed, setResumeDismissed] = useState(false);

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

  // Resume affordance (cambia-783): ask the server whether this user is still counted into a
  // live lobby or game. Nothing to resume, or an unreachable endpoint, leaves the banner off:
  // this is a convenience surface and must never take over the home screen with an error.
  useEffect(() => {
    let cancelled = false;
    getActiveSession()
      .then((session) => {
        if (!cancelled) setActiveSession(session);
      })
      .catch(() => {
        if (!cancelled) setActiveSession(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
        setCreateError('Could not create the lobby.');
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
      setJoinLobbyError(error.response?.data?.message || error.message || 'Could not join the lobby.');
    }
  }, [navigate]);

  const handleResumeSession = useCallback(() => {
    if (activeSession) navigate(`/lobby/${activeSession.lobbyId}`);
  }, [activeSession, navigate]);

  const publicLobbies = Object.entries(lobbies).filter(([, entry]) => entry.lobby?.type === 'public');

  // Headline rating: the 1v1 (head-to-head) pool, the same pool AppLayout's
  // header chip shows, formatted with the shared utils/ratingPool helpers so
  // the number agrees with the profile page everywhere it appears. Zero games
  // in the pool shows 'Unrated' rather than a fake number.
  const headlinePool = ratings?.pools.find((p) => p.pool === '1v1') ?? null;
  const hasHeadline = !!headlinePool && headlinePool.games > 0;

  // OpenSkill is circuit-wide (not per-pool); gate on lifetime record games,
  // matching DsRatingSummary's neverPlayed check, and format to 2 decimals
  // the same way the profile does.
  const neverPlayed = !ratings || ratings.record.games === 0;

  // record.games (lifetime, rated or not) and a pool's games (rated only) come from
  // separate queries and can disagree, e.g. a rated pool seeded without matching
  // game_results rows. hasRatedPool checks the pools directly so the hint below
  // never claims no rating exists while the headline is printing one (cambia-929).
  const hasRatedPool = !!ratings && ratings.pools.some((pool) => pool.games > 0);

  // The headline number is always the 1v1 pool, so it only earns the slot when
  // that pool has games, or when nothing has been played at all and the panel
  // would otherwise be one hint line. FFA games with no 1v1 games used to print
  // an 'Unrated' headline over a 'Head to Head  Unrated' row saying the same
  // thing; there the rows carry the panel on their own. Headlining whichever
  // pool has games is the wrong repair: the headline shows no pool label, and
  // the header chip (AppLayout) reads 1v1 (cambia-892, DL-7 F3).
  const showHeadline = hasHeadline || neverPlayed;

  return (
    <div className='grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-5 p-4 sm:p-[22px] max-w-[1240px] mx-auto w-full'>
      {activeSession && !resumeDismissed && (
        <div style={{ gridColumn: '1 / -1' }}>
          <DsResumeBanner
            session={activeSession}
            onResume={handleResumeSession}
            onDismiss={() => setResumeDismissed(true)}
          />
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 'var(--ds-text-3xl)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>
            Quick play
          </h1>
        </div>

        {queuesError && <ErrorLine>{queuesError}</ErrorLine>}
        {queuesLoading && queues.length === 0 && <Spinner label='Loading queues' />}

        {/* 3-up only above ~1200px: with the lg sidebar in place, md:grid-cols-3 left
            ~200px queue cards whose titles wrapped (cambia-876, DL-2 review F8). */}
        <div className='grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3.5'>
          {queues.map((queue) => {
            const isSearchingThis = searchingQueueId === queue.queueId;
            const disabled = !!searchingQueueId && !isSearchingThis;
            if (isSearchingThis) {
              return (
                <div
                  key={queue.queueId}
                  style={{
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-strong)',
                    borderRadius: 'var(--ds-radius-lg)',
                    padding: 'var(--space-4) var(--space-5)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
                    <div style={{ fontSize: 'var(--ds-text-lg)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>{queue.name}</div>
                    <Badge tone='info' dot>Searching</Badge>
                  </div>
                  <div style={{ fontSize: 'var(--ds-text-xs)', fontVariantNumeric: 'tabular-nums', color: 'var(--text-secondary)' }}>
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
          {lobbiesError && <ErrorLine style={{ marginBottom: 8 }}>{lobbiesError}</ErrorLine>}
          {joinLobbyError && <ErrorLine style={{ marginBottom: 8 }}>{joinLobbyError}</ErrorLine>}
          {lobbiesLoading && publicLobbies.length === 0 && <Note>Loading lobbies</Note>}
          {!lobbiesLoading && publicLobbies.length === 0 && !lobbiesError && (
            <Note>No public lobbies right now. Start one.</Note>
          )}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {publicLobbies.map(([lobbyId, entry], i) => {
              const lobby = entry.lobby;
              const displayName = entry.name || lobbyFallbackName(lobbyId);
              const full = entry.playerCount >= entry.maxPlayers;
              return (
                <div key={lobbyId} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ fontWeight: 'var(--weight-bold)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName}</span>
                  <Badge>{gameModeLabel(lobby?.gameMode)}</Badge>
                  <span style={{ fontSize: 'var(--ds-text-xs)', fontVariantNumeric: 'tabular-nums', color: 'var(--text-secondary)', minWidth: 30, textAlign: 'right', whiteSpace: 'nowrap' }}>
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

      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <Panel title='Your ratings'>
          {/* A failed fetch is an error, not a note: every other panel on this page
              reports one through ErrorLine (cambia-876, DL-2 review F3). */}
          {!ratings && ratingsError && <ErrorLine>{ratingsError}</ErrorLine>}
          {!ratings && !ratingsError && (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '12px 0' }}>
              <Spinner label='Loading ratings' />
            </div>
          )}
          {ratings && (
            <>
              {showHeadline && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 6 }}>
                  {hasHeadline
                    ? <TierBadge tier={tierFromRating(headlinePool!.rating)} />
                    : <Badge tone='neutral'>Unranked</Badge>}
                  <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6, fontVariantNumeric: 'tabular-nums' }}>
                    <span style={{ fontWeight: 'var(--weight-black)', fontSize: 'var(--ds-text-2xl)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 1 }}>
                      {hasHeadline ? Math.round(headlinePool!.rating) : 'Unrated'}
                    </span>
                    {hasHeadline && (
                      <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-xs)' }}>± {Math.round(headlinePool!.rd)}</span>
                    )}
                  </span>
                </div>
              )}
              {/* Nothing rated: the headline already says Unrated, so the rows would
                  be four more copies of it. One line explains the state instead
                  (cambia-876, DL-2 review F6). Gated on hasRatedPool too, so the
                  hint never sits under a headline that is printing a real rating
                  (cambia-929). */}
              {neverPlayed && !hasRatedPool ? (
                <Note style={{ fontSize: 'var(--ds-text-xs)' }}>Play a ranked game to start a rating.</Note>
              ) : (
                <>
                  {/* The 1v1 row is dropped only when the headline rendered its number. */}
                  {ratings.pools.filter((pool) => !(showHeadline && pool.pool === '1v1')).map((pool) => (
                    <StatRow
                      key={pool.pool}
                      label={ratingPoolLabel(pool.pool)}
                      value={pool.games > 0 ? Math.round(pool.rating) : 'Unrated'}
                      unit={pool.games > 0 ? `± ${Math.round(pool.rd)}` : undefined}
                    />
                  ))}
                  <StatRow label='OpenSkill' value={ratings.openSkill.mu.toFixed(2)} unit={`± ${ratings.openSkill.sigma.toFixed(2)}`} />
                  <StatRow
                    label='Record'
                    value={`${ratings.record.wins}W ${ratings.record.games - ratings.record.wins}L`}
                    style={{ borderBottom: 'none' }}
                  />
                </>
              )}
            </>
          )}
        </Panel>

        <Panel title='Friends'>
          {friendsError && <ErrorLine style={{ marginBottom: 8 }}>{friendsError}</ErrorLine>}
          {friendsLoading && friends.length === 0 && <Note>Loading friends</Note>}
          {!friendsLoading && friends.length === 0 && !friendsError && (
            <Note>No friends yet.</Note>
          )}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {friends.map((f, i) => {
              const dotColor = f.online === true ? 'var(--status-success)' : f.online === false ? 'var(--border-strong)' : 'var(--text-tertiary)';
              return (
                <div key={f.userId} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', flex: 'none', background: dotColor }}></span>
                  <span style={{ lineHeight: 'var(--ds-leading-snug)', flex: 1, minWidth: 0 }}>
                    <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.username}</span>
                    <span style={{ display: 'block', fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{f.status}</span>
                  </span>
                  {/* No invite control here: the button had no handler on any branch,
                      and the real invite is the lobby link (cambia-876). */}
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      <Modal open={createOpen} title='New lobby' onClose={handleCloseCreate} footer={(
        <>
          <Button variant='secondary' onClick={handleCloseCreate} disabled={creating}>Cancel</Button>
          <Button variant='primary' onClick={handleCreateLobby} disabled={creating}>{creating ? 'Creating' : 'Create'}</Button>
        </>
      )}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {createError && <ErrorLine>{createError}</ErrorLine>}
          <Select
            label='Lobby type'
            value={createLobbyType}
            disabled={creating}
            onChange={(e) => setCreateLobbyType(e.target.value as 'private' | 'public')}
            options={[
              { value: 'public', label: 'Public' },
              { value: 'private', label: 'Private' }
            ]}
          />
          <Select
            label='Game mode'
            value={createGameMode}
            disabled={creating}
            onChange={(e) => setCreateGameMode(e.target.value)}
            options={[
              { value: 'head_to_head', label: gameModeLabel('head_to_head') },
              { value: 'group_of_4', label: gameModeLabel('group_of_4') }
            ]}
          />
        </div>
      </Modal>
    </div>
  );
};

export default DashboardPage;
