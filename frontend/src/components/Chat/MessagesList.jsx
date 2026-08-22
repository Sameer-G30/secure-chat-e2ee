// src/components/Chat/MessagesList.jsx
import { forwardRef } from 'react';
import MessageBubble from './MessageBubble';

const MessagesList = forwardRef(({ 
    isLoadingMessages, 
    chatMessages, 
    selectedContact, 
    openMessageActions 
}, ref) => {
    return (
        <div className="messages" id="messages">
            {isLoadingMessages && <div className="empty-chat">Loading messages...</div>}
            {!isLoadingMessages && chatMessages.length === 0 && (
                <div className="empty-chat">No messages yet. Start a conversation.</div>
            )}
            {chatMessages.map((msg, index) => (
                <MessageBubble
                    key={msg.id || `msg-${index}`}
                    msg={msg}
                    index={index}
                    chatMessages={chatMessages}
                    selectedContact={selectedContact}
                    openMessageActions={openMessageActions}
                />
            ))}
            <div ref={ref} />
        </div>
    );
});

export default MessagesList;