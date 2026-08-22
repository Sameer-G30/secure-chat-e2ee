// src/components/Chat/ChatWindow.jsx
import { useState } from 'react';
import ChatHeader from './ChatHeader';
import MessagesList from './MessagesList';
import ChatInput from './ChatInput';
import MessageActions from './MessageActions';
import Modal from '../UI/Modal';
import { submitReport } from '../../utils/reports';

export default function ChatWindow({
    selectedContact,
    chatMessages,
    isLoadingMessages,
    messageInput,
    setMessageInput,
    handleSendMessage,
    showMessageActions,
    selectedMessage,
    closeMessageActions,
    handleDeleteMessage,
    startEditing,
    isEditing,
    editText,
    setEditText,
    saveEdit,
    showClearChatConfirmation,
    setShowClearChatConfirmation,
    handleClearChat,
    messagesEndRef,
    openMessageActions,
    isBlocked,
    onBlockUser,
    onUnblockUser,
    currentUser
}) {
    // ===== MODAL STATES =====
    const [showBlockModal, setShowBlockModal] = useState(false);
    const [showUnblockModal, setShowUnblockModal] = useState(false);
    const [showReportModal, setShowReportModal] = useState(false);
    const [showReportSuccess, setShowReportSuccess] = useState(false);
    const [showReportError, setShowReportError] = useState(false);
    const [showSearchModal, setShowSearchModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [reportReason, setReportReason] = useState('');
    const [blockOnReport, setBlockOnReport] = useState(false);

    // ===== EXPORT CHAT =====
    const exportChat = () => {
        if (chatMessages.length === 0) {
            return;
        }

        let text = `Chat with ${selectedContact?.username}\n`;
        text += `Exported on: ${new Date().toLocaleString()}\n`;
        text += `${'='.repeat(50)}\n\n`;

        chatMessages.forEach(msg => {
            const sender = msg.isSent ? 'You' : selectedContact?.username;
            const time = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : '';
            text += `[${time}] ${sender}: ${msg.text}\n`;
        });

        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_${selectedContact?.username}_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // ===== BLOCK USER =====
    const handleBlockUser = () => {
        setShowBlockModal(true);
    };

    const confirmBlockUser = () => {
        setShowBlockModal(false);
        onBlockUser(selectedContact.uid);
    };

    // ===== UNBLOCK USER =====
    const handleUnblockUser = () => {
        setShowUnblockModal(true);
    };

    const confirmUnblockUser = () => {
        setShowUnblockModal(false);
        onUnblockUser(selectedContact.uid);
    };

    // ===== REPORT USER =====
    const handleReportUser = () => {
        setReportReason('');
        setBlockOnReport(false);
        setShowReportModal(true);
    };

    const confirmReportUser = async () => {
        console.log('📝 Report data:', {
            reporterUid: currentUser?.uid,
            reportedUid: selectedContact.uid,
            reason: reportReason,
            blockOnReport: blockOnReport
        });

        if (!reportReason.trim()) {
            console.warn('⚠️ No reason provided');
            return;
        }
        
        setShowReportModal(false);
        
        try {
            const success = await submitReport(
                currentUser?.uid,
                selectedContact.uid,
                reportReason,
                blockOnReport
            );
            
            console.log('📤 Submit result:', success);
            
            if (blockOnReport) {
                onBlockUser(selectedContact.uid);
            }
            
            if (success) {
                setShowReportSuccess(true);
            } else {
                setShowReportError(true);
            }
        } catch (error) {
            console.error('❌ Report error:', error);
            setShowReportError(true);
        }
    };

    // ===== SEARCH MESSAGES =====
    const handleSearchMessages = () => {
        setSearchQuery('');
        setShowSearchModal(true);
    };

    const confirmSearch = () => {
        if (searchQuery.trim()) {
            const results = chatMessages.filter(msg => 
                msg.text && msg.text.toLowerCase().includes(searchQuery.toLowerCase())
            );
            if (results.length > 0) {
                // You can implement scrolling to results here
            }
            setShowSearchModal(false);
        }
    };

    if (!selectedContact) {
        return (
            <div className="empty-chat-state">
                <div className="empty-chat-icon">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                    </svg>
                </div>
                <h3>Select a contact</h3>
                <p>Choose someone from your contacts to start chatting</p>
            </div>
        );
    }

    return (
        <>
            <ChatHeader 
                selectedContact={selectedContact}
                setShowClearChatConfirmation={setShowClearChatConfirmation}
                onSearchMessages={handleSearchMessages}
                onExportChat={exportChat}
                onBlockUser={handleBlockUser}
                onReportUser={handleReportUser}
                isBlocked={isBlocked}
                onUnblockUser={handleUnblockUser}
            />

            {/* Blocked Banner */}
            {isBlocked && (
                <div className="blocked-banner">
                    <div className="blocked-banner-content">
                        <span className="blocked-icon">🔒</span>
                        <div>
                            <div className="blocked-title">You have blocked {selectedContact?.username}</div>
                            <div className="blocked-subtitle">They can't message or call you.</div>
                        </div>
                        <button className="unblock-btn" onClick={handleUnblockUser}>
                            Unblock
                        </button>
                    </div>
                </div>
            )}

            <MessagesList
                isLoadingMessages={isLoadingMessages}
                chatMessages={chatMessages}
                selectedContact={selectedContact}
                openMessageActions={openMessageActions}
                ref={messagesEndRef}
                isBlocked={isBlocked}
            />

            {/* Show blocked message instead of input */}
            {isBlocked ? (
                <div className="blocked-input-banner">
                    <span>You can't send messages to a blocked user.</span>
                    <button className="unblock-link" onClick={handleUnblockUser}>
                        Unblock to chat
                    </button>
                </div>
            ) : (
                <ChatInput
                    messageInput={messageInput}
                    setMessageInput={setMessageInput}
                    handleSendMessage={handleSendMessage}
                />
            )}

            {/* ===== MODALS ===== */}

            {/* Clear Chat Confirmation */}
            <Modal
                isOpen={showClearChatConfirmation}
                onClose={() => setShowClearChatConfirmation(false)}
                onConfirm={handleClearChat}
                title="Clear Chat"
                message={`Are you sure you want to clear all messages with ${selectedContact?.username}? This action cannot be undone.`}
                confirmText="Clear All"
                cancelText="Cancel"
            />

            {/* Block User Confirmation */}
            <Modal
                isOpen={showBlockModal}
                onClose={() => setShowBlockModal(false)}
                onConfirm={confirmBlockUser}
                title={`Block ${selectedContact?.username}?`}
                message="This person won't be able to message or call you. They won't know you blocked them."
                confirmText="Block"
                cancelText="Cancel"
            />

            {/* Unblock User Confirmation */}
            <Modal
                isOpen={showUnblockModal}
                onClose={() => setShowUnblockModal(false)}
                onConfirm={confirmUnblockUser}
                title={`Unblock ${selectedContact?.username}?`}
                message="This person will be able to message and call you again."
                confirmText="Unblock"
                cancelText="Cancel"
            />

            {/* Report User Modal */}
            <Modal
                isOpen={showReportModal}
                onClose={() => setShowReportModal(false)}
                onConfirm={confirmReportUser}
                title="Report User"
                message="Please describe why you are reporting this user:"
                confirmText="Submit Report"
                cancelText="Cancel"
                type="report"
                inputValue={reportReason}
                setInputValue={setReportReason}
                inputPlaceholder="Describe the issue..."
                showBlockOption={true}
                isBlockChecked={blockOnReport}
                setBlockChecked={setBlockOnReport}
            />

            {/* Report Success Modal */}
            <Modal
                isOpen={showReportSuccess}
                onClose={() => setShowReportSuccess(false)}
                onConfirm={() => setShowReportSuccess(false)}
                title="Report Submitted"
                message="Thank you for your report. We will review it and take appropriate action."
                confirmText="OK"
                cancelText=""
            />

            {/* Report Error Modal */}
            <Modal
                isOpen={showReportError}
                onClose={() => setShowReportError(false)}
                onConfirm={() => setShowReportError(false)}
                title="Error"
                message="Failed to submit report. Please try again later."
                confirmText="OK"
                cancelText=""
            />

            {/* Search Messages Modal */}
            <Modal
                isOpen={showSearchModal}
                onClose={() => setShowSearchModal(false)}
                onConfirm={confirmSearch}
                title="Search Messages"
                message="Enter a search term"
                confirmText="Search"
                cancelText="Cancel"
                type="input"
                inputValue={searchQuery}
                setInputValue={setSearchQuery}
                inputPlaceholder="Type to search..."
                inputLabel="Search"
            />

            {/* Message Actions Modal */}
            {showMessageActions && selectedMessage && (
                <MessageActions
                    selectedMessage={selectedMessage}
                    closeMessageActions={closeMessageActions}
                    handleDeleteMessage={handleDeleteMessage}
                    startEditing={startEditing}
                    isEditing={isEditing}
                    editText={editText}
                    setEditText={setEditText}
                    saveEdit={saveEdit}
                    showClearChatConfirmation={showClearChatConfirmation}
                    setShowClearChatConfirmation={setShowClearChatConfirmation}
                    handleClearChat={handleClearChat}
                    selectedContact={selectedContact}
                />
            )}

            {/* Edit Message Input */}
            {isEditing && (
                <div className="edit-message-container">
                    <div className="edit-message-input">
                        <input
                            type="text"
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            placeholder="Edit message..."
                            onKeyPress={(e) => e.key === 'Enter' && saveEdit()}
                            autoFocus
                        />
                        <button className="btn-send" onClick={saveEdit}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                            </svg>
                        </button>
                        <button 
                            className="icon-btn" 
                            onClick={() => { 
                                setIsEditing(false); 
                                setEditText('');
                                setSelectedMessage(null);
                            }}
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}