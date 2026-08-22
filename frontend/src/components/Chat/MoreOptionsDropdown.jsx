// src/components/Chat/MoreOptionsDropdown.jsx
export default function MoreOptionsDropdown({
    isOpen,
    onClose,
    onClearChat,
    onExportChat,
    onBlockUser,
    onReportUser,
    onUnblockUser,
    selectedContact,
    isBlocked
}) {
    if (!isOpen) return null;

    return (
        <div className="dropdown-overlay" onClick={onClose}>
            <div className="dropdown-menu" onClick={(e) => e.stopPropagation()}>
                <div className="dropdown-header">
                    <span>Chat Options</span>
                    <button className="icon-btn" onClick={onClose}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div className="dropdown-items">
                    <button className="dropdown-item" onClick={() => { onClose(); onExportChat(); }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Export Chat
                    </button>
                    <button className="dropdown-item" onClick={() => { onClose(); onClearChat(); }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M3 6h18"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            <line x1="10" y1="11" x2="10" y2="17"/>
                            <line x1="14" y1="11" x2="14" y2="17"/>
                        </svg>
                        Clear Chat
                    </button>
                    
                    {isBlocked ? (
                        <>
                            <button className="dropdown-item" onClick={() => { onClose(); onUnblockUser(); }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10"/>
                                    <path d="M4.93 4.93l14.14 14.14"/>
                                    <path d="M19.07 4.93l-14.14 14.14" stroke="currentColor"/>
                                </svg>
                                Unblock User
                            </button>
                            <button className="dropdown-item" onClick={() => { onClose(); onReportUser(); }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    <line x1="12" y1="8" x2="12" y2="12"/>
                                    <circle cx="12" cy="16" r="0.5" fill="currentColor"/>
                                </svg>
                                Report User
                            </button>
                        </>
                    ) : (
                        <>
                            <button className="dropdown-item" onClick={() => { onClose(); onBlockUser(); }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10"/>
                                    <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
                                </svg>
                                Block User
                            </button>
                            <button className="dropdown-item" onClick={() => { onClose(); onReportUser(); }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    <line x1="12" y1="8" x2="12" y2="12"/>
                                    <circle cx="12" cy="16" r="0.5" fill="currentColor"/>
                                </svg>
                                Report User
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}