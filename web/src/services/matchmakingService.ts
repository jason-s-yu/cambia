import api from '@/lib/axios';

export interface QueueInfo {
  queueId: string;
  name: string;
  players: number;
  rounds: number;
  ratingPool: string;
  ranked: boolean;
  hiddenRating: boolean;
  playerCount: number;
  avgWaitSec: number;
}

export const startSearch = async (lobbyId: string): Promise<void> => {
  await api.post(`/lobby/${lobbyId}/search`);
};

export const cancelSearch = async (lobbyId: string): Promise<void> => {
  await api.delete(`/lobby/${lobbyId}/search`);
};

export const getQueues = async (): Promise<QueueInfo[]> => {
  const res = await api.get<QueueInfo[]>('/matchmaking/queues');
  return res.data;
};
