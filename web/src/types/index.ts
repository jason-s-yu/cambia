// src/types/index.ts

/** Represents basic user information, often retrieved from /user/me or lobby/game state */
export interface User {
  id: string;
  username: string; // Now mandatory based on usage
  is_ephemeral: boolean;
  is_admin?: boolean; // Optional, might not always be present
  email?: string;
  created_at?: string;
  last_login?: string;
}

/** Represents the state of a user within a lobby */
export interface LobbyUser extends User {
  is_host: boolean;
  is_ready: boolean;
}

/** Represents the overall status of a lobby */
export interface LobbyStatus {
  users: LobbyUser[]; // Ensures users have usernames
}

/**
 * Structure for House Rules - based on lobby_actions.md and internal/game/rules.go.
 * The numeric rules are range-checked by the server; the ranges in the comments are the ones
 * update_rules accepts, and a value outside them rejects the whole settings save.
 */
export interface HouseRules {
  allowDrawFromDiscardPile: boolean;
  allowReplaceAbilities: boolean;
  allowOpponentSnapping?: boolean;
  snapRace: boolean;
  lockCallerHand?: boolean;
  forfeitOnDisconnect: boolean;
  penaltyDrawCount: number; // 0-6
  turnTimerSec: number; // 0-86400, 0 disables the turn timer
  maxGameTurns?: number; // 0-65535, 0 means unlimited
  cardsPerPlayer?: number; // 1-6
  cambiaAllowedRound?: number; // 0-255
  numJokers?: number; // 0-2
  numDecks?: number; // 1-4
  initialViewCount?: number; // 0-2
}

/** Nested rules structure within CircuitSettings */
export interface CircuitRules {
	targetScore: number;
	winBonus: number;
	falseCambiaPenalty: number;
	freezeUserOnDisconnect: boolean;
}

/** Structure for Circuit settings - based on lobby_actions.md and internal/game/game.go */
export interface CircuitSettings {
	enabled: boolean;
	mode: string; // e.g., "circuit_4p" - Corresponds to LobbyState.gameMode when enabled? Consider consolidation if modes overlap.
	rules: CircuitRules;
}

/** Structure for Lobby settings - based on lobby_actions.md and internal/lobby/lobby.go */
export interface LobbySettings {
  autoStart: boolean;
}


/**
 * Represents the detailed state of a lobby, received via WS (lobby_state) or REST (/lobby/create).
 * Includes consolidated fields from both sources.
 */
export interface LobbyState {
  id: string; // Present in both REST and WS (lobby_id)
  hostUserID: string; // From REST /lobby/create response (camelCase)
  host_id?: string;   // From WS lobby_state message (snake_case)
  type: 'private' | 'public' | 'matchmaking';
  gameMode: string;             // e.g., "head_to_head"
  inGame: boolean;
  game_id?: string | null;      // Present if game has started
  houseRules: HouseRules;       // Present in both REST and WS
  circuit: CircuitSettings;     // Present in both REST and WS
  lobbySettings: LobbySettings; // From REST payload
  settings?: LobbySettings;     // From WS lobby_state message (nested under root)
  lobby_status?: LobbyStatus;   // From WS lobby_state message
  // WS specific convenience fields
  lobby_id?: string;      // From WS, should match 'id'
  your_id?: string;       // From WS
  your_is_host?: boolean; // From WS
  // Matchmaking fields (from WS or REST)
  queueId?: string;
  visibility?: 'private' | 'public';
  mode?: 'casual' | 'ranked';
  isRanked?: boolean;
  matchState?: MatchState;
}

/** Entry in the /lobby/list response: lobby summary plus player counts (see service ListLobbiesResponse). */
export interface LobbyListEntry {
  lobby: LobbyState;
  playerCount: number;
  maxPlayers: number;
  /** Display name for the lobby. Optional: a parallel branch adds this field server-side. */
  name?: string;
}


/**
 * The lobby (and in-progress game, if any) the signed-in user can return to after a refresh
 * or a lost tab. Mirrors the service ActiveSession in
 * service/internal/handlers/active_session.go (GET /lobby/active).
 */
export interface ActiveSession {
  lobbyId: string;
  lobbyType: string;
  gameMode: string;
  /** Host-supplied lobby name. Absent when the lobby was created without one. */
  name?: string;
  /** Derived from lobby state, not the hub's own finer-grained phase. */
  phase: 'open' | 'searching' | 'in_game';
  /** Set only while a live game is registered for that lobby. */
  gameId?: string;
  /** True when the user holds a seat in the running game, false for a lobby-only member. */
  seated: boolean;
  /** Seated players while in game, otherwise joined lobby members. */
  playerCount: number;
}

/** Represents a chat message */
export interface ChatMessage {
  user_id: string;  // ID of the sender
  username: string; // Username is mandatory (from server or derived client-side)
  msg: string;      // The message content
  ts: number;       // Timestamp (Unix seconds)
}

/** Represents a friend relationship, potentially including usernames for display */
// TODO: finish
export interface FriendRelationship {
    user1_id: string;
    user2_id: string;
    status: 'pending' | 'accepted';
    // Optional usernames - Frontend might need to fetch these separately or backend API needs update
    user1_username?: string;
    user2_username?: string;
}


/** Match state for multi-round ranked matches */
export interface MatchState {
  queueId: string;
  isRanked: boolean;
  totalRounds: number;
  currentRound: number;
  roundScores: Record<string, number>[];  // per-round scores (player UUID string → score)
  cumulativeScores: Record<string, number>;
  subsidies?: Record<string, number>;      // last round's subsidies
  ratingChanges?: Record<string, { before: number; after: number }>;
}

/** Generic type for API error responses */
export interface ApiErrorResponse {
  message: string;
}

/** Represents the available theme options */
export type Theme = 'light' | 'dark' | 'system';