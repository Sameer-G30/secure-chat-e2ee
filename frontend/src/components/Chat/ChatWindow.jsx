// src/components/Chat/ChatWindow.jsx
import { useState, useEffect } from 'react';
import ChatHeader from './ChatHeader';
import MessagesList from './MessagesList';
import ChatInput from './ChatInput';
import MessageActions from './MessageActions';
import Modal from '../UI/Modal';
import ContactProfile from './ContactProfile';
import { submitReport } from '../../utils/reports';
import { ref, set, get, remove } from 'firebase/database';
import { db } from '../../utils/encryption';

const DEFAULT_WALLPAPERS = [
    'https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=600&h=1200&fit=crop&q=80',
    'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&h=1200&fit=crop&q=80',
    'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=600&h=1200&fit=crop&q=80',
    'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&h=1200&fit=crop&q=80',
    'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=600&h=1200&fit=crop&q=80',
    'https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=600&h=1200&fit=crop&q=80',
];

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
    currentUser,
    chatWallpaper,
    onSetWallpaper,
    profilePicture
}) {
    const [showBlockModal, setShowBlockModal] = useState(false);
    const [showUnblockModal, setShowUnblockModal] = useState(false);
    const [showReportModal, setShowReportModal] = useState(false);
    const [showReportSuccess, setShowReportSuccess] = useState(false);
    const [showReportError, setShowReportError] = useState(false);
    const [showSearchModal, setShowSearchModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [reportReason, setReportReason] = useState('');
    const [blockOnReport, setBlockOnReport] = useState(false);
    const [showWallpaperOptions, setShowWallpaperOptions] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const [isBlocked, setIsBlocked] = useState(false);

    const reportOptions = [
        { value: 'spam', label: 'Spam or promotional content' },
        { value: 'harassment', label: 'Harassment or bullying' },
        { value: 'hate', label: 'Hate speech or discrimination' },
        { value: 'impersonation', label: 'Impersonating someone else' },
        { value: 'inappropriate', label: 'Inappropriate content' },
        { value: 'other', label: 'Other' },
    ];

    useEffect(() => {
        if (!selectedContact || !currentUser) return;
        const checkBlockStatus = async () => {
            try {
                const blockRef = ref(db, `blocks/${currentUser.uid}/${selectedContact.uid}`);
                const snapshot = await get(blockRef);
                setIsBlocked(snapshot.exists());
            } catch (error) {
                console.error('Error checking block status:', error);
            }
        };
        checkBlockStatus();
    }, [selectedContact, currentUser]);

    const exportChat = () => {
        if (chatMessages.length === 0) return;
        let text = `Chat with ${selectedContact?.username}\n`;
        text += `Exported on: ${new Date().toLocaleString()}\n`;
        text += `${'='.repeat(50)}\n\n`;
        chatMessages.forEach((msg) => {
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

    const handleProfileClick = () => setShowProfile(true);
    const handleBlockUser = () => setShowBlockModal(true);

    const confirmBlockUser = async () => {
        try {
            const blockRef = ref(db, `blocks/${currentUser.uid}/${selectedContact.uid}`);
            await set(blockRef, {
                blockedAt: Date.now(),
                blockedUser: selectedContact.uid,
                blockedUsername: selectedContact.username
            });
            setIsBlocked(true);
            setShowBlockModal(false);
        } catch (error) {
            console.error('Error blocking user:', error);
            alert('Failed to block user');
        }
    };

    const handleUnblockUser = () => setShowUnblockModal(true);

    const confirmUnblockUser = async () => {
        try {
            const blockRef = ref(db, `blocks/${currentUser.uid}/${selectedContact.uid}`);
            await remove(blockRef);
            setIsBlocked(false);
            setShowUnblockModal(false);
        } catch (error) {
            console.error('Error unblocking user:', error);
            alert('Failed to unblock user');
        }
    };

    const handleReportUser = () => {
        setReportReason('');
        setBlockOnReport(false);
        setShowReportModal(true);
    };

    const confirmReportUser = async () => {
        if (!reportReason.trim()) return;
        setShowReportModal(false);
        const success = await submitReport(
            currentUser?.uid,
            selectedContact.uid,
            reportReason,
            blockOnReport
        );
        if (success) setShowReportSuccess(true);
        else setShowReportError(true);
    };

    const handleSearchMessages = () => {
        setSearchQuery('');
        setShowSearchModal(true);
    };

    const confirmSearch = () => {
        if (searchQuery.trim()) setShowSearchModal(false);
    };

    const handleWallpaperUpload = (event) => {
        const file = event.target.files[0];
        if (!file || !selectedContact) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            onSetWallpaper(selectedContact.uid, e.target.result);
            setShowWallpaperOptions(false);
        };
        reader.readAsDataURL(file);
    };

    const handleDefaultWallpaper = (wallpaper) => {
        if (!selectedContact) return;
        onSetWallpaper(selectedContact.uid, wallpaper);
        setShowWallpaperOptions(false);
    };

    const removeWallpaper = () => {
        if (!selectedContact) return;
        onSetWallpaper(selectedContact.uid, null);
        setShowWallpaperOptions(false);
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
        <div 
            className={`chat-area ${chatWallpaper ? 'with-wallpaper' : ''}`}
            style={{
                backgroundImage: chatWallpaper ? `url(${chatWallpaper})` : 'none',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
            }}
        >
            <ChatHeader 
                selectedContact={selectedContact}
                setShowClearChatConfirmation={setShowClearChatConfirmation}
                onSearchMessages={handleSearchMessages}
                onExportChat={exportChat}
                onBlockUser={handleBlockUser}
                onReportUser={handleReportUser}
                onSetWallpaper={() => setShowWallpaperOptions(true)}
                onProfileClick={handleProfileClick}
                isBlocked={isBlocked}
                onUnblockUser={handleUnblockUser}
            />

            {isBlocked && (
                <div className="blocked-banner">
                    <div className="blocked-banner-content">
                        <span className="blocked-icon">🔒</span>
                        <div>
                            <div className="blocked-title">You have blocked {selectedContact?.username}</div>
                            <div className="blocked-subtitle">They can't message or call you.</div>
                        </div>
                    </div>
                    <div className="blocked-actions">
                        <button className="unblock-btn" onClick={handleUnblockUser}>Unblock</button>
                    </div>
                </div>
            )}

            {showWallpaperOptions && (
                <div className="wallpaper-options-overlay">
                    <div className="wallpaper-options-panel">
                        <div className="wallpaper-options-header">
                            <span>Choose Wallpaper</span>
                            <button className="icon-btn" onClick={() => setShowWallpaperOptions(false)}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <line x1="18" y1="6" x2="6" y2="18"/>
                                    <line x1="6" y1="6" x2="18" y2="18"/>
                                </svg>
                            </button>
                        </div>
                        <div className="wallpaper-options-grid">
                            {DEFAULT_WALLPAPERS.map((wp, i) => (
                                <div 
                                    key={i}
                                    className="wallpaper-option"
                                    style={{ backgroundImage: `url(${wp})`, backgroundSize: 'cover' }}
                                    onClick={() => handleDefaultWallpaper(wp)}
                                />
                            ))}
                            <div 
                                className="wallpaper-option upload-option"
                                onClick={() => document.getElementById('wallpaperInput').click()}
                            >
                                <span>+</span>
                                <span style={{ fontSize: '10px' }}>Upload</span>
                            </div>
                            <div 
                                className="wallpaper-option remove-option"
                                onClick={removeWallpaper}
                            >
                                <span>✕</span>
                                <span style={{ fontSize: '10px' }}>Remove</span>
                            </div>
                        </div>
                        <input 
                            type="file" 
                            id="wallpaperInput" 
                            accept="image/*" 
                            style={{ display: 'none' }} 
                            onChange={handleWallpaperUpload} 
                        />
                    </div>
                </div>
            )}

            <ContactProfile
                isOpen={showProfile}
                onClose={() => setShowProfile(false)}
                contact={selectedContact}
                currentUser={currentUser}
                profilePicture={profilePicture}
            />

            <MessagesList
                isLoadingMessages={isLoadingMessages}
                chatMessages={chatMessages}
                selectedContact={selectedContact}
                openMessageActions={openMessageActions}
                ref={messagesEndRef}
                isBlocked={isBlocked}
            />

            {isBlocked ? (
                <div className="blocked-input-banner">
                    <span>You can't send messages to a blocked user.</span>
                    <button className="unblock-link" onClick={handleUnblockUser}>Unblock to chat</button>
                </div>
            ) : (
                <ChatInput
                    messageInput={messageInput}
                    setMessageInput={setMessageInput}
                    handleSendMessage={handleSendMessage}
                />
            )}

            <Modal isOpen={showClearChatConfirmation} onClose={() => setShowClearChatConfirmation(false)} onConfirm={handleClearChat} title="Clear Chat" message={`Are you sure you want to clear all messages with ${selectedContact?.username}? This action cannot be undone.`} confirmText="Clear All" cancelText="Cancel" />
            <Modal isOpen={showBlockModal} onClose={() => setShowBlockModal(false)} onConfirm={confirmBlockUser} title={`Block ${selectedContact?.username}?`} message="This person won't be able to message or call you. They won't know you blocked them." confirmText="Block" cancelText="Cancel" />
            <Modal isOpen={showUnblockModal} onClose={() => setShowUnblockModal(false)} onConfirm={confirmUnblockUser} title={`Unblock ${selectedContact?.username}?`} message="This person will be able to message and call you again." confirmText="Unblock" cancelText="Cancel" />
            <Modal isOpen={showReportModal} onClose={() => setShowReportModal(false)} onConfirm={confirmReportUser} title="Report User" message="Please describe why you are reporting this user:" confirmText="Submit Report" cancelText="Cancel" type="report" inputValue={reportReason} setInputValue={setReportReason} inputPlaceholder="Describe the issue..." showBlockOption={true} isBlockChecked={blockOnReport} setBlockChecked={setBlockOnReport} />
            <Modal isOpen={showReportSuccess} onClose={() => setShowReportSuccess(false)} onConfirm={() => setShowReportSuccess(false)} title="Report Submitted" message="Thank you for your report. We will review it and take appropriate action." confirmText="OK" cancelText="" />
            <Modal isOpen={showReportError} onClose={() => setShowReportError(false)} onConfirm={() => setShowReportError(false)} title="Error" message="Failed to submit report. Please try again later." confirmText="OK" cancelText="" />
            <Modal isOpen={showSearchModal} onClose={() => setShowSearchModal(false)} onConfirm={confirmSearch} title="Search Messages" message="Enter a search term" confirmText="Search" cancelText="Cancel" type="input" inputValue={searchQuery} setInputValue={setSearchQuery} inputPlaceholder="Type to search..." inputLabel="Search" />

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

            {isEditing && (
                <div className="edit-message-container">
                    <div className="edit-message-input">
                        <input type="text" value={editText} onChange={(e) => setEditText(e.target.value)} placeholder="Edit message..." onKeyPress={(e) => e.key === 'Enter' && saveEdit()} autoFocus />
                        <button className="btn-send" onClick={saveEdit}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                            </svg>
                        </button>
                        <button className="icon-btn" onClick={() => { setIsEditing(false); setEditText(''); setSelectedMessage(null); }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="18" y1="6" x2="6" y2="18"/>
                                <line x1="6" y1="6" x2="18" y2="18"/>
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}