import React, { useEffect, useState } from 'react';
import { listFriends, addFriend, acceptFriend, removeFriend } from '@/services/friendService';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import Button from '../common/Button';
import Input from '../common/Input';
import type { FriendRelationship } from '@/types';

const FriendsPanel: React.FC = () => {
  const [friends, setFriends] = useState<FriendRelationship[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addFriendId, setAddFriendId] = useState('');
  const currentUserId = 'me'; // TODO: Replace with actual user ID from authStore

  const fetchFriendsList = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listFriends();
      // TODO: Map usernames based on currentUserId and friend IDs if needed
      setFriends(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load friends list.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchFriendsList();
  }, []);

  const handleAddFriend = async (e: React.FormEvent) => {
     e.preventDefault();
     if (!addFriendId) return;
     setError(null);
     try {
         await addFriend(addFriendId);
         setAddFriendId('');
         fetchFriendsList(); // Refetch for now
     } catch (err: any) {
         setError(err.message || 'Failed to send friend request.');
     }
  };

  const handleAcceptFriend = async (friendId: string) => {
     setError(null);
     try {
         await acceptFriend(friendId);
         fetchFriendsList();
     } catch (err: any) {
         setError(err.message || 'Failed to accept friend request.');
     }
  };

   const handleRemoveFriend = async (friendId: string) => {
     setError(null);
     try {
         await removeFriend(friendId);
         fetchFriendsList();
     } catch (err: any) {
         setError(err.message || 'Failed to remove friend.');
     }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 space-y-4">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 border-b pb-2 dark:border-gray-700">Friends</h3>
      <ErrorMessage message={error} onClear={() => setError(null)} />

      <form onSubmit={handleAddFriend} className="flex space-x-2">
         <Input
            id="add-friend"
            type="text"
            value={addFriendId}
            onChange={(e) => setAddFriendId(e.target.value)}
            placeholder="Enter friend's ID/Username"
            className="flex-grow !mb-0"
            required
         />
         <Button type="submit" size="md" variant="primary" className='shrink-0'>Add</Button>
      </form>

      {isLoading && <div className="flex justify-center p-4"><LoadingSpinner size="sm" /></div>}
      {!isLoading && friends.length === 0 && <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-2">No friends yet.</p>}
      {!isLoading && friends.length > 0 && (
        <ul className="space-y-2 max-h-60 overflow-y-auto">
          {friends.map((rel) => {
            const friendId = rel.user1_id === currentUserId ? rel.user2_id : rel.user1_id;
            // TODO: Replace placeholder username logic
            const friendUsername = rel.user1_id === currentUserId ? rel.user2_username : rel.user1_username;
            const isPending = rel.status === 'pending';
            const requestSentByOther = isPending && rel.user2_id === currentUserId;

            return (
              <li key={friendId} className="flex justify-between items-center text-sm p-2 bg-gray-50 dark:bg-gray-700 rounded">
                 <span className='truncate mr-2 dark:text-gray-200'>{friendUsername || friendId}</span>
                {rel.status === 'accepted' && (
                   <Button size="sm" variant="danger" onClick={() => handleRemoveFriend(friendId)}>Remove</Button>
                )}
                {isPending && requestSentByOther && (
                    <div className='flex space-x-1'>
                        <Button size="sm" variant="primary" onClick={() => handleAcceptFriend(friendId)}>Accept</Button>
                        <Button size="sm" variant="secondary" onClick={() => handleRemoveFriend(friendId)}>Decline</Button>
                     </div>
                )}
                 {isPending && !requestSentByOther && (
                    <span className="text-xs text-yellow-600 dark:text-yellow-400 italic">Pending</span>
                 )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default FriendsPanel;