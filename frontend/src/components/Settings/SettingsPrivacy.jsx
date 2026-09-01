// src/components/Settings/SettingsPrivacy.jsx
import { useState, useEffect } from 'react';

export default function SettingsPrivacy({ goBackToMain, closeSettings }) {
    const [privacySettings, setPrivacySettings] = useState(() => {
        const saved = localStorage.getItem('privacySettings');
        return saved ? JSON.parse(saved) : {
            lastSeen: 'everyone',
            profilePhoto: 'everyone',
            readReceipts: true,
            typingIndicator: true
        };
    });

    useEffect(() => {
        localStorage.setItem('privacySettings', JSON.stringify(privacySettings));
    }, [privacySettings]);

    const updateSetting = (key, value) => {
        setPrivacySettings(prev => ({ ...prev, [key]: value }));
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Privacy</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Last Seen</div>
                            <div className="settings-item-desc">Who can see your last seen</div>
                        </div>
                    </div>
                    <select className="settings-select" value={privacySettings.lastSeen} onChange={(e) => updateSetting('lastSeen', e.target.value)}>
                        <option value="everyone">Everyone</option>
                        <option value="contacts">My Contacts</option>
                        <option value="none">Nobody</option>
                    </select>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Profile Photo</div>
                            <div className="settings-item-desc">Who can see your photo</div>
                        </div>
                    </div>
                    <select className="settings-select" value={privacySettings.profilePhoto} onChange={(e) => updateSetting('profilePhoto', e.target.value)}>
                        <option value="everyone">Everyone</option>
                        <option value="contacts">My Contacts</option>
                        <option value="none">Nobody</option>
                    </select>
                </div>

                <div className="settings-divider"></div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Read Receipts</div>
                            <div className="settings-item-desc">Show when you've read messages</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={privacySettings.readReceipts} onChange={(e) => updateSetting('readReceipts', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Typing Indicator</div>
                            <div className="settings-item-desc">Show when you're typing</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={privacySettings.typingIndicator} onChange={(e) => updateSetting('typingIndicator', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>
            </div>
        </>
    );
}