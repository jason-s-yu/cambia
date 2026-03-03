// src/components/lobby/MatchScoreboard.tsx
import type { MatchState } from '@/types/index';

interface MatchScoreboardProps {
  matchState: MatchState;
  players: Array<{ id: string; username: string }>;
  phase: 'round_end' | 'match_end';
}

export function MatchScoreboard({ matchState, players, phase }: MatchScoreboardProps) {
  const { roundScores, cumulativeScores, subsidies, ratingChanges, currentRound } = matchState;
  const numRounds = roundScores.length;

  // Sort players ascending by cumulative score (lower is better in Cambia)
  const sortedPlayers = [...players].sort((a, b) => {
    const aScore = cumulativeScores[a.id] ?? 0;
    const bScore = cumulativeScores[b.id] ?? 0;
    return aScore - bScore;
  });

  const leaderId = sortedPlayers[0]?.id;

  const title = phase === 'match_end' ? 'Final Standings' : 'Match Scoreboard';

  return (
    <div className="rounded border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-900 dark:text-gray-100">{title}</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
              <th className="py-2 pr-4 font-medium">Player</th>
              {Array.from({ length: numRounds }, (_, i) => (
                <th
                  key={i}
                  className={`py-2 px-2 font-medium text-center ${
                    i + 1 === currentRound
                      ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : ''
                  }`}
                >
                  R{i + 1}
                </th>
              ))}
              {subsidies && Object.keys(subsidies).length > 0 && (
                <th className="py-2 px-2 font-medium text-center text-amber-600 dark:text-amber-400">Sub</th>
              )}
              <th className="py-2 pl-2 font-medium text-right">Total</th>
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((player) => {
              const isLeader = player.id === leaderId;
              const total = cumulativeScores[player.id] ?? 0;
              const subsidy = subsidies?.[player.id];

              return (
                <tr
                  key={player.id}
                  className={`border-b border-gray-100 dark:border-gray-700 last:border-0 ${
                    isLeader ? 'font-bold text-gray-900 dark:text-white' : 'text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <td className="py-2 pr-4">{player.username}</td>
                  {roundScores.map((rs, i) => (
                    <td
                      key={i}
                      className={`py-2 px-2 text-center ${
                        i + 1 === currentRound
                          ? 'bg-blue-50 dark:bg-blue-900/20'
                          : ''
                      }`}
                    >
                      {rs[player.id] !== undefined ? rs[player.id] : '-'}
                    </td>
                  ))}
                  {subsidies && Object.keys(subsidies).length > 0 && (
                    <td className="py-2 px-2 text-center text-amber-600 dark:text-amber-400">
                      {subsidy !== undefined ? (subsidy > 0 ? `-${subsidy}` : subsidy === 0 ? '0' : `${subsidy}`) : '-'}
                    </td>
                  )}
                  <td className="py-2 pl-2 text-right">{total}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {phase === 'match_end' && ratingChanges && Object.keys(ratingChanges).length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Rating Changes:</h3>
          <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
            {Object.entries(ratingChanges).map(([playerId, change]) => {
              const player = players.find((p) => p.id === playerId);
              const name = player?.username ?? playerId.substring(0, 8);
              const diff = Math.round(change.after - change.before);
              const sign = diff >= 0 ? '+' : '';
              return (
                <li key={playerId} className="flex justify-between">
                  <span>{name}</span>
                  <span className={diff >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                    {Math.round(change.before)} → {Math.round(change.after)} ({sign}{diff})
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default MatchScoreboard;
