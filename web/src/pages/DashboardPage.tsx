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
import { useSocket } from '@/hooks/useSocket';
import { useQueueStore } from '@/stores/queueStore';
import { useFriendsStore } from '@/stores/friendsStore';
import { useHistoryStore } from '@/stores/historyStore';
import { joinLobby as apiJoinPublicLobby, getActiveSession, getLobbyPresets, type CreateLobbyRequest } from '@/services/lobbyService';
import type { QueueInfo } from '@/services/matchmakingService';
import type { ActiveSession, ApiErrorResponse, LobbyPreset } from '@/types';
import { DEFAULT_PRESET_ID } from '@/types';
import { gameModeLabel } from '@/utils/gameMode';
import { queuePoolLabel, ratingPoolLabel, tierFromRating } from '@/utils/ratingPool';

/**
 * Queues considered "flagship" for the primary/highlighted card treatment. This is a display
 * flag only - it does not move a card earlier in the grid. Card order comes straight from
 * GET /matchmaking/queues (queues.map below, no client-side sort) and is deterministic
 * server-side as of cambia-957 (QueueConfig.Order); do not add a sort here that reads this set,
 * or the two flagship cards would stop appearing where the server placed them relative to the
 * other four.
 */
const PRIMARY_QUEUE_IDS = new Set(['h2h_rapid', 'ffa4_standard']);

/**
 * Rough estimated match length in minutes from queue shape, since the
 * matchmaking queues endpoint does not return one directly. Grounded on the
 * default 15s turn timer: a head-to-head round runs ~2-3 minutes, an FFA
 * round (>2 players) ~4-5.
 */
function estimateMinutes(queue: QueueInfo): number {
  const perRound = queue.players > 2 ? 5 : 2.5;
  return Math.max(2, Math.round(queue.rounds * perRound));
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

  const searchLobbyId = useQueueStore((state) => state.searchLobbyId);
  const finishSearch = useQueueStore((state) => state.finishSearch);

  const createAndJoinLobby = useCurrentLobbyStore((state) => state.createAndJoinLobby);
  const currentLobbyId = useCurrentLobbyStore((state) => state.currentLobbyId);
  const matchedLobbyId = useCurrentLobbyStore((state) => state.matchState?.lobbyId);

  // A search runs from a real lobby, and the hub announces the match over that lobby's socket:
  // without a connection the player waits on the dashboard forever while the match they were
  // put in starts without them (cambia-933). The socket lives for the search only.
  const { closeSocket: closeSearchSocket } = useSocket(searchLobbyId);

  const [searchElapsed, setSearchElapsed] = useState(0);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [joinLobbyError, setJoinLobbyError] = useState<string | null>(null);

  const [activeSession, setActiveSession] = useState<ActiveSession | null>(null);
  const [resumeDismissed, setResumeDismissed] = useState(false);

  const [createOpen, setCreateOpen] = useState(false);
  const [createLobbyType, setCreateLobbyType] = useState<'private' | 'public'>('public');
  const [createGameMode, setCreateGameMode] = useState('head_to_head');
  const [createPresetId, setCreatePresetId] = useState(DEFAULT_PRESET_ID);
  const [presets, setPresets] = useState<LobbyPreset[]>([]);
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const previousLobbyIdRef = useRef<string | null>(null);

  // The ruleset the dialog is currently on, and the game mode it fixes. A preset built from a
  // queue carries that queue's player count, so it decides the mode and the mode control goes
  // read-only; the default preset fixes none and leaves the choice to the host.
  const selectedPreset = presets.find((p) => p.id === createPresetId);
  const presetGameMode = selectedPreset?.gameMode ?? '';
  const effectiveGameMode = presetGameMode || createGameMode;

  useEffect(() => {
    fetchQueues();
    fetchLobbies();
    fetchFriends();
  }, [fetchQueues, fetchLobbies, fetchFriends]);

  // Selectable rulesets for the New lobby dialog (cambia-1088). An unreachable endpoint leaves
  // the list empty, which hides the Ruleset control and creates on the service's own defaults:
  // the dialog exists to make a lobby, and no field in it may become a reason it cannot.
  useEffect(() => {
    let cancelled = false;
    getLobbyPresets()
      .then((list) => {
        if (!cancelled) setPresets(list);
      })
      .catch(() => {
        if (!cancelled) setPresets([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  // A found match names the lobby it is played in, which is the searching lobby only for the
  // party that hosts the match; everyone else moves there. The socket is closed first and by
  // hand: the lobby page opens its own, and a second connection for the same user would take
  // the live one down with it when the hub evicts it.
  useEffect(() => {
    if (!searchLobbyId || !matchedLobbyId) return;
    closeSearchSocket();
    void finishSearch(matchedLobbyId);
    navigate(`/lobby/${matchedLobbyId}`);
  }, [searchLobbyId, matchedLobbyId, closeSearchSocket, finishSearch, navigate]);

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
      // presetId goes only when the service offered one, so a dialog that could not load the
      // list still creates a lobby on the service's own defaults rather than 400ing on an id
      // this client made up.
      const settings: CreateLobbyRequest = { type: createLobbyType, gameMode: effectiveGameMode };
      if (selectedPreset) settings.presetId = selectedPreset.id;
      const lobbyId = await createAndJoinLobby(settings);
      if (!lobbyId) {
        setCreateError('Could not create the lobby.');
      }
      // Navigation on success is handled by the effect above.
    } finally {
      setCreating(false);
    }
  }, [createAndJoinLobby, createLobbyType, effectiveGameMode, selectedPreset]);

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
                  <div style={{ fontSize: 'var(--ds-text-sm)', fontVariantNumeric: 'tabular-nums', color: 'var(--text-secondary)' }}>
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
                  players={queue.players}
                  rounds={queue.rounds}
                  minutes={estimateMinutes(queue)}
                  pool={queuePoolLabel(queue.ratingPool)}
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
                  <span style={{ fontSize: 'var(--ds-text-sm)', fontVariantNumeric: 'tabular-nums', color: 'var(--text-secondary)', minWidth: 34, textAlign: 'right', whiteSpace: 'nowrap' }}>
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
              {/* Nothing to show at all: the headline already says Unrated, so the rows
                  would be four more copies of it. One line explains the state instead
                  (cambia-876, DL-2 review F6). The gate is neverPlayed && !hasRatedPool,
                  deliberately not !hasRatedPool alone: a player with real casual-game
                  history (record.games > 0) but no rated pool still has a real Record
                  row worth showing, so the hint is reserved for an account with nothing
                  recorded at all (cambia-949 CP6). hasRatedPool alone covers the
                  opposite mismatch: a rated pool seeded without a matching game_results
                  row must not sit under a hint claiming no rating exists (cambia-929). */}
              {neverPlayed && !hasRatedPool ? (
                <Note>Play a ranked game to start a rating.</Note>
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
                    <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--text-md)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.username}</span>
                    <span style={{ display: 'block', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>{f.status}</span>
                  </span>
                  {/* No invite control here: the button had no handler on any branch,
                      and the real invite is the lobby link (cambia-876). */}
                </div>
              );
            })}
          </div>
        </Panel>
      </div>

      {/* Opens on Create: the dialog exists to make a lobby, and its fields all
          carry defaults (cambia-914 DL-8 R9, said explicitly in cambia-935 F6). */}
      <Modal open={createOpen} title='New lobby' onClose={handleCloseCreate} initialFocus='confirm' footer={(
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
            value={effectiveGameMode}
            disabled={creating || !!presetGameMode}
            onChange={(e) => setCreateGameMode(e.target.value)}
            options={[
              { value: 'head_to_head', label: gameModeLabel('head_to_head') },
              { value: 'group_of_4', label: gameModeLabel('group_of_4') }
            ]}
          />
          {/* Ruleset third, under the two fields it can override, so the host reads the shape
              of the lobby before the rules it plays by (cambia-1088). Hidden entirely when the
              preset list did not load: an empty dropdown is worse than none. */}
          {presets.length > 0 && (
            <div>
              <Select
                label='Ruleset'
                value={createPresetId}
                disabled={creating}
                onChange={(e) => setCreatePresetId(e.target.value)}
                options={presets.map((p) => ({ value: p.id, label: p.name }))}
              />
              {selectedPreset && (
                <p style={{ margin: '6px 0 0', fontSize: 'var(--ds-text-sm)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-tertiary)' }}>
                  {selectedPreset.description}
                  {presetGameMode ? ` Game mode is fixed at ${gameModeLabel(presetGameMode)}.` : ''}
                </p>
              )}
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
};

export default DashboardPage;
