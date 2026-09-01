// src/components/Settings/SettingsNotifications.jsx
import { useState, useEffect } from 'react';

export default function SettingsNotifications({ goBackToMain, closeSettings }) {
    const [notificationSettings, setNotificationSettings] = useState(() => {
        const saved = localStorage.getItem('notificationSettings');
        return saved ? JSON.parse(saved) : {
            messageNotifications: true,
            sound: true,
            vibration: true,
            previewMessage: true
        };
    });

    useEffect(() => {
        localStorage.setItem('notificationSettings', JSON.stringify(notificationSettings));
    }, [notificationSettings]);

    const updateSetting = (key, value) => {
        setNotificationSettings(prev => ({ ...prev, [key]: value }));
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Notifications</h2>
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
                            <div className="settings-item-title">Message Notifications</div>
                            <div className="settings-item-desc">Show notifications for new messages</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={notificationSettings.messageNotifications} onChange={(e) => updateSetting('messageNotifications', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Sound</div>
                            <div className="settings-item-desc">Play sound for notifications</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={notificationSettings.sound} onChange={(e) => updateSetting('sound', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Vibration</div>
                            <div className="settings-item-desc">Vibrate for notifications</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={notificationSettings.vibration} onChange={(e) => updateSetting('vibration', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Message Preview</div>
                            <div className="settings-item-desc">Show message content in notification</div>
                        </div>
                    </div>
                    <label className="settings-toggle-switch">
                        <input type="checkbox" checked={notificationSettings.previewMessage} onChange={(e) => updateSetting('previewMessage', e.target.checked)} />
                        <span className="settings-toggle-slider"></span>
                    </label>
                </div>

                <div className="settings-divider"></div>

                <div className="settings-item">
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Notification Tone</div>
                            <div className="settings-item-desc">Select notification sound</div>
                        </div>
                    </div>
                    <select className="settings-select" defaultValue="default">
                        <option value="default">Default</option>
                        <option value="chime">Chime</option>
                        <option value="bell">Bell</option>
                        <option value="soft">Soft</option>
                    </select>
                </div>
            </div>
        </>
    );
}