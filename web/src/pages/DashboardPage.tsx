import React from 'react';
import LobbyList from '@/components/dashboard/LobbyList';
import CreateLobbyButton from '@/components/dashboard/CreateLobbyButton';
import FriendsPanel from '@/components/dashboard/FriendsPanel';

/**
 * The main dashboard page displayed after a user logs in.
 * Shows available public lobbies, a button to create a new lobby,
 * and the user's friends list/management panel.
 */
const DashboardPage: React.FC = () => {
	return (
		<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
			{/* Lobby Section */}
			<div className="md:col-span-2 space-y-6">
				<div className='flex justify-between items-center'>
					<h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">Available Lobbies</h2>
					<CreateLobbyButton />
				</div>
				<LobbyList />
			</div>
			{/* Friends Section */}
			<div className="md:col-span-1">
				<FriendsPanel />
			</div>
		</div>
	);
};

export default DashboardPage;