// src/components/Chat/ChatHeader.jsx
import { useState, useRef, useEffect } from 'react';

export default function ChatHeader({ 
    selectedContact, 
    setShowClearChatConfirmation,
    onSearchMessages,
    onExportChat,
    onBlockUser,
    onReportUser,
    onSetWallpaper,
    onProfileClick,
    isBlocked,
    onUnblockUser
}) {
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef(null);
    const buttonRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target) && 
                buttonRef.current && !buttonRef.current.contains(event.target)) {
                setShowDropdown(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <>
            <div className="chat-header">
                <div className="chat-header-left" onClick={onProfileClick} style={{ cursor: 'pointer' }}>
                    <div className="chat-contact-avatar">
                        {selectedContact.username.charAt(0).toUpperCase()}
                    </div>
                    <div>
                        <div className="chat-contact-name">{selectedContact.username}</div>
                        <div className="chat-contact-status">
                            {isBlocked ? (
                                <span className="blocked-status">Blocked</span>
                            ) : (
                                'End-to-end encrypted'
                            )}
                        </div>
                    </div>
                </div>
                <div className="chat-header-actions">
                    <button className="icon-btn" title="Search messages" onClick={onSearchMessages}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                    </button>
                    <button className="icon-btn" title="Change wallpaper" onClick={onSetWallpaper}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="2" y="2" width="20" height="20" rx="2"/>
                            <circle cx="8.5" cy="8.5" r="1.5"/>
                            <path d="M21 15l-5-5L5 21"/>
                        </svg>
                    </button>
                    <button 
                        ref={buttonRef}
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

            {showDropdown && (
                <div className="dropdown-popup" ref={dropdownRef}>
                    <div className="dropdown-menu">
                        <div className="dropdown-header">
                            <span>Chat Options</span>
                        </div>
                        <div className="dropdown-items">
                            <button className="dropdown-item" onClick={() => { setShowDropdown(false); onExportChat(); }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7 10 12 15 17 10"/>
                                    <line x1="12" y1="15" x2="12" y2="3"/>
                                </svg>
                                Export Chat
                            </button>
                            <button className="dropdown-item" onClick={() => { setShowDropdown(false); setShowClearChatConfirmation(true); }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M3 6h18"/>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                    <line x1="10" y1="11" x2="10" y2="17"/>
                                    <line x1="14" y1="11" x2="14" y2="17"/>
                                </svg>
                                Clear Chat
                            </button>
                            {isBlocked ? (
                                <button className="dropdown-item" onClick={() => { setShowDropdown(false); onUnblockUser(); }}>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <circle cx="12" cy="12" r="10"/>
                                        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
                                    </svg>
                                    Unblock User
                                </button>
                            ) : (
                                <button className="dropdown-item" onClick={() => { setShowDropdown(false); onBlockUser(); }}>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <circle cx="12" cy="12" r="10"/>
                                        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
                                    </svg>
                                    Block User
                                </button>
                            )}
                            <button className="dropdown-item" onClick={() => { setShowDropdown(false); onReportUser(); }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    <line x1="12" y1="8" x2="12" y2="12"/>
                                    <circle cx="12" cy="16" r="0.5" fill="currentColor"/>
                                </svg>
                                Report User
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}