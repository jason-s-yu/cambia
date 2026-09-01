// src/pages/DashboardPage.test.tsx
// Render-level coverage for the create-lobby dialog's preset filter (cambia-1126 AC4, F3 of the
// cambia-1239 sprint review). presetFitsGameMode itself has unit coverage (web/scripts/test-
// lobby-preset.mjs); nothing exercised the dialog wiring that calls it: the Ruleset select
// offers only the presets the chosen game mode can play, and changing the game mode away from a
// picked preset resets the pick to the default rather than leaving a now-invalid id selected
// (DashboardPage.tsx chooseGameMode).
//
// The page also fetches queues, lobbies and friends on mount (its own useEffect, unrelated to
// this test). Rather than reach into those stores, every service call they end up making is
// mocked to resolve harmlessly: that keeps the dialog's own dependencies (presets, active
// session) as the only interesting mock surface here.
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DashboardPage from '@/pages/DashboardPage';
import { DEFAULT_PRESET_ID, type HouseRules, type LobbyPreset } from '@/types';

const { getLobbyPresetsMock, getActiveSessionMock } = vi.hoisted(() => ({
  getLobbyPresetsMock: vi.fn(),
  getActiveSessionMock: vi.fn()
}));

vi.mock('@/services/lobbyService', () => ({
  getLobbyPresets: getLobbyPresetsMock,
  getActiveSession: getActiveSessionMock,
  joinLobby: vi.fn(),
  leaveLobby: vi.fn(),
  listLobbies: vi.fn().mockResolvedValue({}),
  createLobby: vi.fn()
}));

vi.mock('@/services/matchmakingService', () => ({
  getQueues: vi.fn().mockResolvedValue([]),
  startSearch: vi.fn(),
  cancelSearch: vi.fn()
}));

vi.mock('@/services/friendsService', () => ({
  getFriends: vi.fn().mockResolvedValue([])
}));

// The search socket is not under test here; DashboardPage only reads closeSocket off it.
vi.mock('@/hooks/useSocket', () => ({
  useSocket: () => ({ sendMessage: vi.fn(), closeSocket: vi.fn(), isConnected: false, isLoading: false, error: null })
}));

function houseRules(): HouseRules {
  return {
    allowDrawFromDiscardPile: false,
    allowReplaceAbilities: false,
    allowOpponentSnapping: true,
    snapRace: false,
    lockCallerHand: true,
    forfeitOnDisconnect: true,
    disconnectGraceSec: 90,
    penaltyDrawCount: 2,
    turnTimerSec: 15,
    maxGameTurns: 46,
    cardsPerPlayer: 4,
    cambiaAllowedRound: 0,
    numJokers: 2,
    numDecks: 1,
    initialViewCount: 2
  };
}

function preset(overrides: Partial<LobbyPreset>): LobbyPreset {
  return {
    id: 'default',
    name: 'Default',
    description: 'Standard rules.',
    gameMode: '',
    players: 0,
    rounds: 0,
    ranked: false,
    houseRules: houseRules(),
    settings: { autoStart: true },
    ...overrides
  };
}

// The default preset fixes no game mode; the rest mirror the real queue presets
// (service/internal/matchmaking/validation.go), two head-to-head and one group-of-4.
const PRESETS: LobbyPreset[] = [
  preset({ id: 'default', name: 'Default', gameMode: '' }),
  preset({ id: 'h2h_rapid', name: 'H2H Rapid', gameMode: 'head_to_head', players: 2, ranked: true }),
  preset({ id: 'h2h_quickplay', name: 'H2H Quick', gameMode: 'head_to_head', players: 2, ranked: true }),
  preset({ id: 'ffa4_standard', name: 'FFA-4 Standard', gameMode: 'group_of_4', players: 4, ranked: true })
];

function renderDashboard() {
  return render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>
  );
}

afterEach(() => {
  getLobbyPresetsMock.mockReset();
  getActiveSessionMock.mockReset();
});

describe('DashboardPage create-lobby dialog preset filter', () => {
  it('offers only the rulesets the chosen game mode can play', async () => {
    getLobbyPresetsMock.mockResolvedValue(PRESETS);
    getActiveSessionMock.mockResolvedValue(null);
    renderDashboard();

    await userEvent.click(screen.getByRole('button', { name: 'Create lobby' }));

    // The accessible name of a select wrapped in a <label> includes the control's own current
    // value (e.g. "Ruleset Default ▼"), so the query anchors on the eyebrow text rather than
    // matching it exactly.
    const rulesetSelect = (await screen.findByRole('combobox', { name: /^Ruleset/ })) as HTMLSelectElement;
    const optionLabels = () => Array.from(rulesetSelect.options).map((o) => o.textContent);

    // head_to_head is the dialog's default game mode.
    expect(optionLabels()).toEqual(['Default', 'H2H Rapid', 'H2H Quick']);

    await userEvent.selectOptions(screen.getByRole('combobox', { name: /^Game mode/ }), 'group_of_4');

    expect(optionLabels()).toEqual(['Default', 'FFA-4 Standard']);
  });

  it('resets a ruleset pick the new game mode cannot play', async () => {
    getLobbyPresetsMock.mockResolvedValue(PRESETS);
    getActiveSessionMock.mockResolvedValue(null);
    renderDashboard();

    await userEvent.click(screen.getByRole('button', { name: 'Create lobby' }));

    const rulesetSelect = (await screen.findByRole('combobox', { name: /^Ruleset/ })) as HTMLSelectElement;
    await userEvent.selectOptions(rulesetSelect, 'h2h_rapid');
    expect(rulesetSelect.value).toBe('h2h_rapid');

    // group_of_4 cannot play a head-to-head preset, so the pick falls back to the default rather
    // than staying selected on an id the Ruleset list no longer offers.
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /^Game mode/ }), 'group_of_4');

    expect(rulesetSelect.value).toBe(DEFAULT_PRESET_ID);
  });
});
