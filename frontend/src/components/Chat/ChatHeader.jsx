// src/components/Chat/ChatHeader.jsx
import { useState } from 'react';
import MoreOptionsDropdown from './MoreOptionsDropdown';

export default function ChatHeader({ 
    selectedContact, 
    setShowClearChatConfirmation,
    onSearchMessages,
    onExportChat,
    onBlockUser,
    onReportUser,
    isBlocked,
    onUnblockUser
}) {
    const [showDropdown, setShowDropdown] = useState(false);

    return (
        <>
            <div className="chat-header">
                <div className="chat-header-left">
                    <div className="chat-contact-avatar">
                        {selectedContact.username.charAt(0).toUpperCase()}
                    </div>
                    <div>
                        <div className="chat-contact-name">{selectedContact.username}</div>
                        <div className="chat-contact-status">
                            {isBlocked ? 'Blocked' : 'End-to-end encrypted'}
                        </div>
                    </div>
                </div>
                <div className="chat-header-actions">
                    <button 
                        className="icon-btn" 
                        title="Search messages"
                        onClick={onSearchMessages}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                    </button>
                    <button 
                        className="icon-btn" 
                        title="More options"
                        onClick={() => setShowDropdown(!showDropdown)}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="5" r="1"/>
                            <circle cx="12" cy="12" r="1"/>
                            <circle cx="12" cy="19" r="1"/>
                        </svg>
                    </button>
                </div>
            </div>

            {/* More Options Dropdown */}
            <MoreOptionsDropdown
                isOpen={showDropdown}
                onClose={() => setShowDropdown(false)}
                onClearChat={() => setShowClearChatConfirmation(true)}
                onExportChat={onExportChat}
                onBlockUser={onBlockUser}
                onReportUser={onReportUser}
                onUnblockUser={onUnblockUser}
                selectedContact={selectedContact}
                isBlocked={isBlocked}
            />
        </>
    );
}