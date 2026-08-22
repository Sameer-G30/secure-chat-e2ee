// src/components/Settings/SettingsTheme.jsx
export default function SettingsTheme({
    goBackToGeneral,
    closeSettings,
    selectedTheme,
    applyTheme
}) {
    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToGeneral}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>App Theme</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div 
                    className={`theme-option ${selectedTheme === 'light' ? 'active' : ''}`}
                    onClick={() => applyTheme('light')}
                >
                    <div className="theme-option-content">
                        <div className="theme-preview light-preview">
                            <div className="theme-preview-header"></div>
                            <div className="theme-preview-body"></div>
                        </div>
                        <div className="theme-option-label">Light</div>
                    </div>
                </div>

                <div 
                    className={`theme-option ${selectedTheme === 'dark' ? 'active' : ''}`}
                    onClick={() => applyTheme('dark')}
                >
                    <div className="theme-option-content">
                        <div className="theme-preview dark-preview">
                            <div className="theme-preview-header"></div>
                            <div className="theme-preview-body"></div>
                        </div>
                        <div className="theme-option-label">Dark</div>
                    </div>
                </div>

                <div 
                    className={`theme-option ${selectedTheme === 'system' ? 'active' : ''}`}
                    onClick={() => applyTheme('system')}
                >
                    <div className="theme-option-content">
                        <div className="theme-preview system-preview">
                            <div className="theme-preview-header"></div>
                            <div className="theme-preview-body"></div>
                        </div>
                        <div className="theme-option-label">System Default</div>
                    </div>
                </div>
            </div>
        </>
    );
}