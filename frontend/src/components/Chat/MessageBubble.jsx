// src/components/Chat/MessageBubble.jsx
export default function MessageBubble({ 
    msg, 
    index, 
    chatMessages, 
    selectedContact, 
    openMessageActions 
}) {
    const prevMsg = index > 0 ? chatMessages[index - 1] : null;
    const showSender = !msg.isSent && (!prevMsg || prevMsg.sender !== msg.sender || prevMsg.isSent);
    const isSentByMe = msg.isSent;
    
    return (
        <div
            className={`message ${isSentByMe ? 'sent' : 'received'} ${msg.isDeleted ? 'deleted' : ''}`}
            onClick={() => openMessageActions(msg)}
            style={{ cursor: 'pointer' }}
        >
            {!isSentByMe && showSender && (
                <div className="message-sender">{selectedContact?.username}</div>
            )}
            <div className="message-text">
                {msg.isDeleted ? 'Message deleted' : msg.text || 'Message'}
                {msg.isEdited && !msg.isDeleted && (
                    <span className="edited-indicator"> (edited)</span>
                )}
            </div>
            <div className="message-time">
                {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                {isSentByMe && !msg.isDeleted && ' ✓'}
            </div>
        </div>
    );
}