# Platform UI kit

Interactive click-through of the Cambia game platform, composed from the design-system components (\`window.Cambia_cc8727\`).

Flow: **Home** (queues, public lobbies, ratings, friends) → **Lobby** (players, host-only match settings with real house-rule keys, chat) → **Game table** (draw → ability/swap, snap, Call Cambia, circuit scoreboard, game log) → **Leaderboard** (pool toggle, tiers).

- Theme toggle in the top bar demonstrates \`data-theme="light"\`.
- All queue/rule/tier vocabulary comes from the repo: \`MATCHMAKING.md\`, \`service/internal/game/rules.go\`, \`web/src/types/\`.
- Screens: \`HomeScreen.jsx\`, \`LobbyScreen.jsx\`, \`GameScreen.jsx\`, \`LeaderboardScreen.jsx\`; chrome in \`shared.jsx\`.
