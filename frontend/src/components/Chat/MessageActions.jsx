// src/components/Chat/MessageActions.jsx
import { useState } from 'react';

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
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [deleteForEveryone, setDeleteForEveryone] = useState(false);

    if (showClearChatConfirmation) {
        return (
            <div className="message-actions-overlay" onClick={() => setShowClearChatConfirmation(false)}>
                <div className="message-actions-modal" onClick={(e) => e.stopPropagation()}>
                    <div className="message-actions-header">
                        <span>Clear Chat</span>
                        <button className="icon-btn" onClick={() => setShowClearChatConfirmation(false)}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
                                className="modal-btn modal-btn-cancel" 
                                onClick={() => setShowClearChatConfirmation(false)}
                            >
                                Cancel
                            </button>
                            <button 
                                className="modal-btn modal-btn-confirm danger" 
                                onClick={handleClearChat}
                            >
                                Clear All
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Delete Confirmation Modal
    if (showDeleteConfirm) {
        return (
            <div className="message-actions-overlay" onClick={() => setShowDeleteConfirm(false)}>
                <div className="message-actions-modal" onClick={(e) => e.stopPropagation()}>
                    <div className="message-actions-header">
                        <span>Delete Message</span>
                        <button className="icon-btn" onClick={() => setShowDeleteConfirm(false)}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>
                    </div>
                    <div className="message-actions-content">
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '16px' }}>
                            {deleteForEveryone ? 
                                'Delete this message for everyone?' : 
                                'Delete this message only for you?'}
                            <br/>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                                {deleteForEveryone ? 
                                    'This will be deleted for everyone in the chat.' : 
                                    'Others will still see this message.'}
                            </span>
                        </p>
                        <div style={{ display: 'flex', gap: '12px' }}>
                            <button 
                                className="modal-btn modal-btn-cancel" 
                                onClick={() => setShowDeleteConfirm(false)}
                            >
                                Cancel
                            </button>
                            <button 
                                className="modal-btn modal-btn-confirm danger" 
                                onClick={() => {
                                    handleDeleteMessage(deleteForEveryone);
                                    setShowDeleteConfirm(false);
                                }}
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (!selectedMessage) return null;

    const isSentByMe = selectedMessage.isSent || false;
    const isDeleted = selectedMessage.isDeleted || false;

    return (
        <div className="message-actions-overlay" onClick={closeMessageActions}>
            <div className="message-actions-modal" onClick={(e) => e.stopPropagation()}>
                <div className="message-actions-header">
                    <span>Message Actions</span>
                    <button className="icon-btn" onClick={closeMessageActions}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div className="message-actions-content">
                    {/* Edit - Only for own messages that are not deleted */}
                    {isSentByMe && !isDeleted && (
                        <button className="message-action-btn" onClick={startEditing}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                            Edit
                        </button>
                    )}

                    {/* Delete for me - Only for own messages */}
                    {isSentByMe && !isDeleted && (
                        <button className="message-action-btn" onClick={() => {
                            setDeleteForEveryone(false);
                            setShowDeleteConfirm(true);
                        }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 6h18"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            </svg>
                            Delete for me
                        </button>
                    )}

                    {/* Delete for everyone - Only for own messages */}
                    {isSentByMe && !isDeleted && (
                        <button className="message-action-btn" onClick={() => {
                            setDeleteForEveryone(true);
                            setShowDeleteConfirm(true);
                        }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 6h18"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                <path d="M9 11v6"/>
                                <path d="M15 11v6"/>
                            </svg>
                            Delete for everyone
                        </button>
                    )}

                    {/* Copy - Always visible */}
                    <button 
                        className="message-action-btn" 
                        onClick={() => {
                            const text = selectedMessage.text;
                            navigator.clipboard.writeText(text).then(() => {
                                alert('Message copied to clipboard!');
                            }).catch(() => {
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