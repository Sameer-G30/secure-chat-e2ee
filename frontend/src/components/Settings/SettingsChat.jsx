// src/components/Settings/SettingsChat.jsx
import { useState, useEffect } from 'react';

export default function SettingsChat({ goBackToMain, closeSettings }) {
    const [chatSettings, setChatSettings] = useState(() => {
        const saved = localStorage.getItem('chatSettings');
        return saved ? JSON.parse(saved) : {
            fontSize: 'medium',
            enterKey: 'send',
            messageReactions: true,
            deleteConfirmation: true
        };
    });

    useEffect(() => {
        localStorage.setItem('chatSettings', JSON.stringify(chatSettings));
    }, [chatSettings]);

    const updateSetting = (key, value) => {
        setChatSettings(prev => ({ ...prev, [key]: value }));
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Chat Settings</h2>
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
                            <div className="settings-item-title">Font Size</div>
                            <div className="settings-item-desc">Text size in chats</div>
                        </div>
                    </div>
                    <select className="settings-select" value={chatSettings.fontSize} onChange={(e) => updateSetting('fontSize', e.target.value)}>
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                        <option value="large">Large</option>
                        <option value="xlarge">Extra Large</option>
                    </select>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Enter Key</div>
                            <div className="settings-item-desc">Action when pressing Enter</div>
                        </div>
                    </div>
                    <select className="settings-select" value={chatSettings.enterKey} onChange={(e) => updateSetting('enterKey', e.target.value)}>
                        <option value="send">Send</option>
                        <option value="newline">New Line</option>
                    </select>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Message Reactions</div>
                            <div className="settings-item-desc">Enable reactions on messages</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={chatSettings.messageReactions} onChange={(e) => updateSetting('messageReactions', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Delete Confirmation</div>
                            <div className="settings-item-desc">Confirm before deleting messages</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={chatSettings.deleteConfirmation} onChange={(e) => updateSetting('deleteConfirmation', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>
            </div>
        </>
    );
}