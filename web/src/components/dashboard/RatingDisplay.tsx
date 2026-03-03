import React from 'react';

export interface RatingInfo {
  mu: number;
  rd: number;
  gamesPlayed: number;
  tier?: string;
}

interface RatingsProps {
  ratings: {
    h2h_qp?: RatingInfo;
    h2h_ranked?: RatingInfo;
    ffa4?: RatingInfo;
  };
}

const TIER_COLORS: Record<string, string> = {
  Bronze: 'bg-amber-700 text-white',
  Silver: 'bg-gray-400 text-white',
  Gold: 'bg-yellow-500 text-black',
  Platinum: 'bg-cyan-400 text-black',
  Diamond: 'bg-blue-400 text-white',
  Master: 'bg-purple-500 text-white',
  Grandmaster: 'bg-red-600 text-white',
};

function getH2HTier(mu: number): string {
  if (mu < 1400) return 'Bronze';
  if (mu < 1500) return 'Silver';
  if (mu < 1600) return 'Gold';
  if (mu < 1720) return 'Platinum';
  if (mu < 1850) return 'Diamond';
  if (mu < 2000) return 'Master';
  return 'Grandmaster';
}

function getFFA4Tier(mu: number): string {
  if (mu < 22) return 'Bronze';
  if (mu < 25) return 'Silver';
  if (mu < 28) return 'Gold';
  if (mu < 32) return 'Platinum';
  if (mu < 37) return 'Diamond';
  if (mu < 42) return 'Master';
  return 'Grandmaster';
}

const TierBadge: React.FC<{ tier: string }> = ({ tier }) => {
  const colorClass = TIER_COLORS[tier] ?? 'bg-gray-200 text-gray-800';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${colorClass}`}>
      {tier}
    </span>
  );
};

const RatingDisplay: React.FC<RatingsProps> = ({ ratings }) => {
  const { h2h_qp, h2h_ranked, ffa4 } = ratings;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {/* H2H Quick Play */}
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">H2H Quick Play</p>
        {!h2h_qp || h2h_qp.gamesPlayed === 0 ? (
          <span className="text-sm text-gray-400 dark:text-gray-500">Unplaced</span>
        ) : (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300">
            Active
          </span>
        )}
      </div>

      {/* H2H Ranked */}
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">H2H Ranked</p>
        {!h2h_ranked || h2h_ranked.gamesPlayed === 0 ? (
          <span className="text-sm text-gray-400 dark:text-gray-500">Unplaced</span>
        ) : h2h_ranked.gamesPlayed < 15 ? (
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Placing: {h2h_ranked.gamesPlayed}/15
          </span>
        ) : (
          <div className="flex flex-col gap-1">
            <TierBadge tier={h2h_ranked.tier ?? getH2HTier(h2h_ranked.mu)} />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {Math.round(h2h_ranked.mu)} ± {Math.round(2 * h2h_ranked.rd)}
            </span>
          </div>
        )}
      </div>

      {/* FFA-4 */}
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">FFA-4</p>
        {!ffa4 || ffa4.gamesPlayed === 0 ? (
          <span className="text-sm text-gray-400 dark:text-gray-500">Unplaced</span>
        ) : ffa4.gamesPlayed < 10 ? (
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Placing: {ffa4.gamesPlayed}/10
          </span>
        ) : (
          <div className="flex flex-col gap-1">
            <TierBadge tier={ffa4.tier ?? getFFA4Tier(ffa4.mu)} />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {ffa4.mu.toFixed(1)} ± {(2 * ffa4.rd).toFixed(1)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default RatingDisplay;
