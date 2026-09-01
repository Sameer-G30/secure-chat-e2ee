// src/components/Settings/SettingsGeneral.jsx
export default function SettingsGeneral({ goBackToMain, goToTheme, closeSettings, selectedTheme }) {
    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>General</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="settings-item" onClick={goToTheme}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="5"/>
                                <line x1="12" y1="1" x2="12" y2="3"/>
                                <line x1="12" y1="21" x2="12" y2="23"/>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                                <line x1="1" y1="12" x2="3" y2="12"/>
                                <line x1="21" y1="12" x2="23" y2="12"/>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">App Theme</div>
                            <div className="settings-item-desc">{selectedTheme === 'dark' ? 'Dark' : selectedTheme === 'light' ? 'Light' : 'System Default'}</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={() => alert('Font size coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="4 7 4 4 20 4 20 7"/>
                                <line x1="9" y1="20" x2="15" y2="20"/>
                                <line x1="12" y1="4" x2="12" y2="20"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Font Size</div>
                            <div className="settings-item-desc">Medium</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={() => alert('Language coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M4 4h16v16H4z"/>
                                <line x1="8" y1="8" x2="16" y2="16"/>
                                <line x1="8" y1="16" x2="16" y2="8"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Language</div>
                            <div className="settings-item-desc">English</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>
            </div>
        </>
    );
}