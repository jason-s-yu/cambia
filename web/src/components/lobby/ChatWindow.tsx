// src/components/lobby/ChatWindow.tsx
import React, { useState, useRef, useEffect } from 'react';
import type { ChatMessage } from '@/types';
import Input from '@/components/common/Input';
import Button from '@/components/common/Button';
import { format } from 'date-fns';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';

interface ChatWindowProps {
	messages: ChatMessage[];
	sendMessage: (message: { type: string; msg: string; }) => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, sendMessage }) => {
	const [newMessage, setNewMessage] = useState('');
	const messagesEndRef = useRef<HTMLDivElement>(null);
	const inputRef = useRef<HTMLInputElement>(null);
	const currentUserId = useAuthStore((state) => state.user?.id);
	const hostId = useCurrentLobbyStore((state) => state.lobbyDetails?.hostUserID);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	const handleSendMessage = (e: React.FormEvent) => {
		e.preventDefault();
		const trimmedMessage = newMessage.trim();
		if (trimmedMessage) {
			sendMessage({ type: 'chat', msg: trimmedMessage });
			setNewMessage('');
		}
	};

	return (
		<div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 flex flex-col h-full">
			<h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-100 border-b pb-2 dark:border-gray-700">Lobby Chat</h3>
			<div className="flex-grow overflow-y-auto mb-4 pr-2 space-y-2">
				{messages.map((msg, index) => {
					const isSender = msg.user_id === currentUserId;
					const isHost = msg.user_id === hostId;
					const usernameColor = isSender
						? 'text-blue-600 dark:text-blue-400'
						: 'text-green-600 dark:text-green-400';

					return (
						<div key={`${msg.user_id}-${msg.ts}-${index}`} className="text-sm break-words">
							<span className={`font-semibold ${usernameColor} mr-1`}>
								{msg.username || `User...${msg.user_id.substring(msg.user_id.length - 4)}`}
								{isHost && <span className="text-gray-500 dark:text-gray-400 font-normal ml-1">(Host)</span>}
								:
							</span>
							<span className='text-gray-800 dark:text-gray-200'>{msg.msg}</span>
							<span className="text-xs text-gray-400 dark:text-gray-500 ml-2 whitespace-nowrap">
								{format(new Date(msg.ts * 1000), 'HH:mm')}
							</span>
						</div>
					);
				})}
				{messages.length === 0 && (
					<div className="text-center text-sm text-gray-500 dark:text-gray-400 py-4">No messages yet.</div>
				)}
				<div ref={messagesEndRef} />
			</div>
			<form onSubmit={handleSendMessage} className="flex items-end space-x-2 mt-auto border-t pt-3 dark:border-gray-700">
				<div className="flex-grow">
					<Input
						ref={inputRef}
						id="chat-message"
						type="text"
						value={newMessage}
						onChange={(e) => setNewMessage(e.target.value)}
						placeholder="Type your message..."
						className="!mb-0 w-full rounded-r-none"
						autoComplete="off"
						aria-label="Chat message input"
					/>
				</div>
				<Button type="submit" variant="primary" className='shrink-0 rounded-l-none h-[calc(2.25rem+2px)]'>
					Send
				</Button>
			</form>
		</div>
	);
};

export default ChatWindow;