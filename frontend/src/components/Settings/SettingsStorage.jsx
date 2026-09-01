// src/components/Settings/SettingsStorage.jsx
import { useState, useEffect } from 'react';

export default function SettingsStorage({ goBackToMain, closeSettings }) {
    const [storageData, setStorageData] = useState({
        messages: 0,
        images: 0,
        files: 0,
        voiceNotes: 0,
        total: 0
    });

    useEffect(() => {
        const calculateStorage = () => {
            let total = 0;
            let messages = 0;
            let images = 0;
            let files = 0;
            let voiceNotes = 0;

            const chatMessages = localStorage.getItem('chatMessages');
            if (chatMessages) {
                try {
                    const msgs = JSON.parse(chatMessages);
                    messages = msgs.length || 0;
                    total += messages;
                } catch (e) {}
            }

            const imagesData = localStorage.getItem('chatImages');
            if (imagesData) {
                try {
                    const imgs = JSON.parse(imagesData);
                    images = imgs.length || 0;
                    total += images;
                } catch (e) {}
            }

            setStorageData({ messages, images, files, voiceNotes, total });
        };
        calculateStorage();
    }, []);

    const clearAllData = () => {
        if (confirm('Are you sure you want to clear all chat data? This cannot be undone.')) {
            try {
                localStorage.removeItem('chatMessages');
                localStorage.removeItem('chatImages');
                localStorage.removeItem('chatFiles');
                localStorage.removeItem('chatVoiceNotes');
                setStorageData({ messages: 0, images: 0, files: 0, voiceNotes: 0, total: 0 });
                alert('All data cleared successfully!');
            } catch (error) {
                console.error('Error clearing data:', error);
                alert('Failed to clear data');
            }
        }
    };

    const clearCache = () => {
        if (confirm('Clear app cache?')) {
            try {
                const keysToKeep = ['secureChatUser', 'secureChatTheme', 'profilePicture_', 'contacts_'];
                const allKeys = Object.keys(localStorage);
                for (const key of allKeys) {
                    if (!keysToKeep.some(k => key.startsWith(k))) {
                        localStorage.removeItem(key);
                    }
                }
                alert('Cache cleared successfully!');
            } catch (error) {
                console.error('Error clearing cache:', error);
                alert('Failed to clear cache');
            }
        }
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Storage & Data</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="storage-stats">
                    <div className="storage-item">
                        <span className="storage-label">Messages</span>
                        <span className="storage-value">{storageData.messages}</span>
                    </div>
                    <div className="storage-item">
                        <span className="storage-label">Images</span>
                        <span className="storage-value">{storageData.images}</span>
                    </div>
                    <div className="storage-item">
                        <span className="storage-label">Files</span>
                        <span className="storage-value">{storageData.files}</span>
                    </div>
                    <div className="storage-item">
                        <span className="storage-label">Voice Notes</span>
                        <span className="storage-value">{storageData.voiceNotes}</span>
                    </div>
                    <div className="storage-item total">
                        <span className="storage-label">Total Items</span>
                        <span className="storage-value">{storageData.total}</span>
                    </div>
                </div>

                <div className="settings-divider"></div>

                <div className="settings-item" onClick={clearCache}>
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Clear Cache</div>
                            <div className="settings-item-desc">Remove temporary files</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={clearAllData} style={{ color: '#ef4444' }}>
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Clear All Data</div>
                            <div className="settings-item-desc">Delete all messages and media</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>
            </div>
        </>
    );
}