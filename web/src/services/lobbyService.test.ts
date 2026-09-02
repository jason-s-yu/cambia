// src/services/lobbyService.test.ts
// The leave seam (cambia-1520): which HTTP answer becomes which error.
//
// The split is load-bearing and both ends of it were already pinned while the join between them
// was not (cambia-1239 review). A 409 means the membership is still held and the seat is still in
// the round, so LobbyPage keeps the player at the table and shows the server's own sentence; every
// other failure is best-effort and ends at the dashboard, because a lobby that is already gone
// must not strand somebody on a table they asked to leave. Getting that mapping wrong is the
// production bug the ticket fixed: a swallowed 409 let the seat sit in the round until the
// disconnect grace forfeited it.
//
// The axios instance is replaced wholesale; nothing here tests axios, only what leaveLobby does
// with what it throws.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LeaveRefusedError, leaveLobby } from '@/services/lobbyService';

const { post } = vi.hoisted(() => ({ post: vi.fn() }));

vi.mock('@/lib/axios', () => ({ default: { post } }));

const LOBBY_ID = '11111111-1111-4111-8111-111111111111';
/** Verbatim from handlers.leaveInProgressMessage: http.Error writes it as a bare text body. */
const REFUSAL = 'Cannot leave a lobby while its game is in progress';
/** Mirrors LEAVE_REFUSED_FALLBACK in the module under test. */
const FALLBACK = 'This lobby has a game in progress.';

/** An axios rejection carrying a status and a body. */
function httpError(status: number, data: unknown): Error & { response: { status: number; data: unknown } } {
  return Object.assign(new Error(`Request failed with status code ${status}`), { response: { status, data } });
}

beforeEach(() => {
  post.mockReset();
  // The service logs every non-refusal failure; the tests below drive several on purpose.
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('leaveLobby', () => {
  it('sends the forfeit flag as the player answered it, and never implies consent', async () => {
    post.mockResolvedValue({ data: null });

    await leaveLobby(LOBBY_ID);
    await leaveLobby(LOBBY_ID, {});
    await leaveLobby(LOBBY_ID, { forfeit: true });

    // A live seat is only released when the player said so, so an absent flag is sent as false
    // rather than left off the body for the server to interpret.
    expect(post.mock.calls.map((c) => c[1])).toEqual([
      { forfeit: false },
      { forfeit: false },
      { forfeit: true }
    ]);
    expect(post.mock.calls[0][0]).toBe(`/lobby/${LOBBY_ID}/leave`);
  });

  it('turns a 409 into a LeaveRefusedError carrying the server sentence', async () => {
    post.mockRejectedValue(httpError(409, REFUSAL));

    await expect(leaveLobby(LOBBY_ID)).rejects.toBeInstanceOf(LeaveRefusedError);
    await expect(leaveLobby(LOBBY_ID)).rejects.toMatchObject({ reason: REFUSAL, name: 'LeaveRefusedError' });
  });

  it('falls back to its own sentence when the refusal body is not text', async () => {
    // A refusal with nothing readable on it is still a refusal: the player has to be told
    // something, and a blank dialog would read as a leave that worked.
    for (const body of [undefined, null, '   ', { message: 'nope' }]) {
      post.mockRejectedValue(httpError(409, body));
      await expect(leaveLobby(LOBBY_ID)).rejects.toMatchObject({ reason: FALLBACK });
    }
  });

  it('rethrows every other failure unchanged, so the caller leaves anyway', async () => {
    // A lobby that is already gone, and a request that never arrived. Neither means the player is
    // still seated, and neither may be mistaken for a refusal.
    post.mockRejectedValue(httpError(404, 'lobby not found'));
    await expect(leaveLobby(LOBBY_ID)).rejects.not.toBeInstanceOf(LeaveRefusedError);

    const network = new Error('Network Error');
    post.mockRejectedValue(network);
    await expect(leaveLobby(LOBBY_ID)).rejects.toBe(network);
  });
});
