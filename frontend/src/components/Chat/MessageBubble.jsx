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

    const getStatusIcon = () => {
        if (!isSentByMe) return null;
        const status = msg.status || { sent: true, delivered: false, read: false };
        if (status.read) return <span className="status-read">✓✓</span>;
        else if (status.delivered) return <span className="status-delivered">✓✓</span>;
        else return <span className="status-sent">✓</span>;
    };

    if (msg.isDeleted) {
        return (
            <div className="message deleted" style={{ cursor: 'default' }}>
                <div className="message-text">This message was deleted</div>
                <div className="message-time">
                    {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                </div>
            </div>
        );
    }

    // Check if it's a short message (1-2 words)
    const isShortMessage = msg.text && msg.text.split(' ').length <= 3 && msg.text.length < 20;

    return (
        <div
            className={`message ${isSentByMe ? 'sent' : 'received'} ${isShortMessage ? 'short' : ''}`}
            onClick={() => openMessageActions(msg)}
            style={{ 
                cursor: 'pointer',
                maxWidth: isShortMessage ? 'auto' : '65%',
                padding: isShortMessage ? '4px 12px' : '6px 14px',
            }}
        >
            {!isSentByMe && showSender && (
                <div className="message-sender">{selectedContact?.username}</div>
            )}
            <div className="message-text">
                {msg.text || 'Message'}
                {msg.isEdited && !msg.isDeleted && (
                    <span className="edited-indicator"> (edited)</span>
                )}
            </div>
            <div className="message-time">
                {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                {getStatusIcon()}
            </div>
        </div>
    );
}