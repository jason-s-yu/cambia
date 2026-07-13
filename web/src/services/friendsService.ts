// src/services/friendsService.ts
import api from '@/lib/axios';
import type { AxiosError } from 'axios';
import type { ApiErrorResponse } from '@/types';

/**
 * Friend entry as returned by GET /friends/list for the design-system home
 * screen (cambia-483). This is a flatter contract than the pending/accepted
 * relationship shape `services/friendService.ts` consumes elsewhere; the
 * server side of this contract lands in a parallel branch.
 */
export interface FriendEntry {
  userId: string;
  username: string;
  status: string;
  online?: boolean;
}

/**
 * Fetches the authenticated user's friends list for home screen display.
 * @returns A promise resolving to an array of FriendEntry objects.
 * @throws {Error} If the API request fails.
 */
export const getFriends = async (): Promise<FriendEntry[]> => {
  try {
    const response = await api.get<FriendEntry[]>('/friends/list');
    return response.data;
  } catch (error) {
    const err = error as AxiosError<ApiErrorResponse>;
    console.error('Get Friends API call failed:', err.response?.data || err.message);
    throw error;
  }
};
