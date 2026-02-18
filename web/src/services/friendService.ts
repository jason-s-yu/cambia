import api from '@/lib/axios';
import type { FriendRelationship } from '@/types'; // Import from central types

/**
 * Fetches the list of friend relationships for the authenticated user.
 * @returns A promise resolving to an array of FriendRelationship objects.
 * @throws {Error} If the API request fails.
 */
export const listFriends = async (): Promise<FriendRelationship[]> => {
	try {
		// Replace with actual API call when backend is ready
		// const response = await api.get<FriendRelationship[]>('/friends/list');
		// return response.data;
		await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay
		// Example data:
		return [
			{ user1_id: 'me', user2_id: 'friend1', status: 'accepted' },
			{ user1_id: 'friend2', user2_id: 'me', status: 'pending' },
		];
	} catch (error: any) {
		console.error('List Friends API call failed:', error.response?.data || error.message);
		throw error; // Re-throw for UI error handling
	}
};

/**
 * Sends a friend request to the specified user ID.
 * @param friendId The UUID string of the user to send the request to.
 * @throws {Error} If the API request fails.
 */
export const addFriend = async (friendId: string): Promise<void> => {
	try {
		// TODO: Replace with actual API call when backend is ready
		// await api.post('/friends/add', { friend_id: friendId });
		await new Promise(resolve => setTimeout(resolve, 300)); // Simulate network delay
	} catch (error: any) {
		console.error('Add Friend API call failed:', error.response?.data || error.message);
		throw error; // Re-throw for UI error handling
	}
};

/**
 * Accepts a friend request from the specified user ID.
 * @param friendId The UUID string of the user whose request is being accepted.
 * @throws {Error} If the API request fails.
 */
export const acceptFriend = async (friendId: string): Promise<void> => {
	try {
		// TODO: Replace with actual API call when backend is ready
		// await api.post('/friends/accept', { friend_id: friendId });
		await new Promise(resolve => setTimeout(resolve, 300)); // Simulate network delay
	} catch (error: any) {
		console.error('Accept Friend API call failed:', error.response?.data || error.message);
		throw error; // Re-throw for UI error handling
	}
};

/**
 * Removes a friend relationship or declines/cancels a pending request.
 * @param friendId The UUID string of the user to remove/decline.
 * @throws {Error} If the API request fails.
 */
export const removeFriend = async (friendId: string): Promise<void> => {
	try {
		// TODO: Replace with actual API call when backend is ready
		// await api.post('/friends/remove', { friend_id: friendId });
		await new Promise(resolve => setTimeout(resolve, 300)); // Simulate network delay
	} catch (error: any) {
		console.error('Remove Friend API call failed:', error.response?.data || error.message);
		throw error; // Re-throw for UI error handling
	}
};