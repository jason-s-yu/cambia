// Add common type definitions as needed, e.g.:
export interface User {
  id: string;
  username: string;
  is_ephemeral: boolean;
  is_admin: boolean;
  // Add other fields like Elo if needed later
}

export interface Lobby {
  id: string;
  hostUserID: string;
  type: 'private' | 'public' | 'matchmaking';
  gameMode: string; // e.g., "head_to_head", "group_of_4"
  inGame: boolean;
  game_id?: string; // Present when game starts
  houseRules: Record<string, any>; // Define more strictly if possible
  circuit: Record<string, any>;     // Define more strictly if possible
  lobbySettings: Record<string, any>; // Define more strictly if possible
}

export interface LobbyUser {
    id: string;
    is_host: boolean;
    is_ready: boolean;
    // Add username if available/needed
}

export interface LobbyStatus {
    users: LobbyUser[];
}

export interface LobbyChatMessage {
    user_id: string;
    username?: string; // Add username if you plan to display it
    msg: string;
    ts: number; // UNIX timestamp (seconds)
}

// Add more types based on API/WebSocket responses as you implement features
// e.g., Friend, GameState, Card, etc.
export interface FriendRelationship {
    user1_id: string;
    user2_id: string;
    status: 'pending' | 'accepted';
}

export type Theme = 'light' | 'dark' | 'system';