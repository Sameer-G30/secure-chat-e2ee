// src/components/Chat/MessagesList.jsx
import { forwardRef } from 'react';
import MessageBubble from './MessageBubble';

const MessagesList = forwardRef(({ 
    isLoadingMessages, 
    chatMessages, 
    selectedContact, 
    openMessageActions,
    isBlocked = false
}, ref) => {

    const formatDateLabel = (timestamp) => {
        if (!timestamp) return 'Today';
        const date = new Date(timestamp);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        today.setHours(0, 0, 0, 0);
        yesterday.setHours(0, 0, 0, 0);
        const msgDate = new Date(date);
        msgDate.setHours(0, 0, 0, 0);

        if (msgDate.getTime() === today.getTime()) return 'Today';
        else if (msgDate.getTime() === yesterday.getTime()) return 'Yesterday';
        else {
            const diffDays = Math.floor((today - msgDate) / (1000 * 60 * 60 * 24));
            if (diffDays < 7) {
                const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
                return days[date.getDay()];
            } else {
                return date.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
            }
        }
    };

    const shouldShowDate = (currentMsg, prevMsg) => {
        if (!prevMsg) return true;
        if (!currentMsg || !currentMsg.timestamp) return false;
        const currentDate = new Date(currentMsg.timestamp);
        const prevDate = new Date(prevMsg.timestamp);
        currentDate.setHours(0, 0, 0, 0);
        prevDate.setHours(0, 0, 0, 0);
        return currentDate.getTime() !== prevDate.getTime();
    };

    return (
        <div className="messages" id="messages">
            {isLoadingMessages && <div className="empty-chat">Loading messages...</div>}
            {!isLoadingMessages && chatMessages.length === 0 && (
                <div className="empty-chat">No messages yet. Start a conversation.</div>
            )}
            {chatMessages.map((msg, index) => {
                const showDate = shouldShowDate(msg, chatMessages[index - 1]);

                return (
                    <div key={msg.id || `msg-${index}`}>
                        {showDate && (
                            <div className="date-separator">
                                <span className="date-label">{formatDateLabel(msg.timestamp)}</span>
                            </div>
                        )}
                        <div className="message-wrapper">
                            <MessageBubble
                                msg={msg}
                                index={index}
                                chatMessages={chatMessages}
                                selectedContact={selectedContact}
                                openMessageActions={openMessageActions}
                            />
                        </div>
                    </div>
                );
            })}
            <div ref={ref} />
        </div>
    );
});

export default MessagesList;