// src/components/Chat/MessageActions.jsx
export default function MessageActions({
    selectedMessage,
    closeMessageActions,
    handleDeleteMessage,
    startEditing,
    showClearChatConfirmation,
    setShowClearChatConfirmation,
    handleClearChat,
    selectedContact
}) {
    // If clear chat confirmation
    if (showClearChatConfirmation) {
        return (
            <div className="message-actions-overlay" onClick={() => setShowClearChatConfirmation(false)}>
                <div className="message-actions-modal" onClick={(e) => e.stopPropagation()}>
                    <div className="message-actions-header">
                        <span>Clear Chat</span>
                        <button className="icon-btn" onClick={() => setShowClearChatConfirmation(false)}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>
                    </div>
                    <div className="message-actions-content">
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '16px' }}>
                            Are you sure you want to clear all messages with {selectedContact?.username}?
                            <br/>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>This action cannot be undone.</span>
                        </p>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <button 
                                className="btn-primary" 
                                onClick={() => setShowClearChatConfirmation(false)}
                                style={{ flex: 1, background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                            >
                                Cancel
                            </button>
                            <button 
                                className="btn-primary" 
                                onClick={handleClearChat}
                                style={{ flex: 1, background: '#ef4444' }}
                            >
                                Clear All
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Default: Message actions
    if (!selectedMessage) return null;

    return (
        <div className="message-actions-overlay" onClick={closeMessageActions}>
            <div className="message-actions-modal" onClick={(e) => e.stopPropagation()}>
                <div className="message-actions-header">
                    <span>Message Actions</span>
                    <button className="icon-btn" onClick={closeMessageActions}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div className="message-actions-content">
                    {selectedMessage.isSent && (
                        <>
                            <button className="message-action-btn" onClick={startEditing}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                                </svg>
                                Edit
                            </button>
                            <button className="message-action-btn" onClick={() => handleDeleteMessage(false)}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M3 6h18"/>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                </svg>
                                Delete for me
                            </button>
                            <button className="message-action-btn" onClick={() => handleDeleteMessage(true)}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M3 6h18"/>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                    <path d="M9 11v6"/>
                                    <path d="M15 11v6"/>
                                </svg>
                                Delete for everyone
                            </button>
                        </>
                    )}
                    {!selectedMessage.isSent && (
                        <button className="message-action-btn" onClick={() => handleDeleteMessage(false)}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 6h18"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            </svg>
                            Delete for me
                        </button>
                    )}
                    <button 
                        className="message-action-btn" 
                        onClick={() => {
                            const text = selectedMessage.text;
                            navigator.clipboard.writeText(text).then(() => {
                                alert('Message copied to clipboard!');
                            }).catch(() => {
                                // Fallback
                                const textarea = document.createElement('textarea');
                                textarea.value = text;
                                document.body.appendChild(textarea);
                                textarea.select();
                                document.execCommand('copy');
                                document.body.removeChild(textarea);
                                alert('Message copied to clipboard!');
                            });
                            closeMessageActions();
                        }}
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                        Copy
                    </button>
                </div>
            </div>
        </div>
    );
}