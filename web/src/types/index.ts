// src/types/index.ts

/** Represents basic user information, often retrieved from /user/me or lobby/game state */
export interface User {
  id: string;
  username: string; // Now mandatory based on usage
  is_ephemeral: boolean;
  is_admin?: boolean; // Optional, might not always be present
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

/** Structure for House Rules - based on lobby_actions.md and internal/game/rules.go */
export interface HouseRules {
  allowDrawFromDiscardPile: boolean;
  allowReplaceAbilities: boolean;
  snapRace: boolean;
  forfeitOnDisconnect: boolean;
  penaltyDrawCount: number;
  autoKickTurnCount: number;
  turnTimerSec: number;
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


/** Generic type for API error responses */
export interface ApiErrorResponse {
  message: string;
}

/** Represents the available theme options */
export type Theme = 'light' | 'dark' | 'system';