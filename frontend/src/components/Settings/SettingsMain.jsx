// src/components/Settings/SettingsMain.jsx
export default function SettingsMain({
    closeSettings,
    goToGeneral,
    handleLogout,
    setSettingsView,
    profilePicture,
    onProfilePictureUpload,
    goToPrivacy,
    goToAccount,
    goToNotifications,
    goToChatSettings,
    goToStorage
}) {
    return (
        <>
            <div className="settings-header">
                <h2>Settings</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="settings-item" onClick={() => document.getElementById('profilePicInput').click()}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            {profilePicture ? (
                                <img src={profilePicture} alt="Profile" style={{ width: '32px', height: '32px', borderRadius: '50%', objectFit: 'cover' }} />
                            ) : (
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                                    <circle cx="12" cy="7" r="4"/>
                                </svg>
                            )}
                        </span>
                        <div>
                            <div className="settings-item-title">Profile Picture</div>
                            <div className="settings-item-desc">Change your profile photo</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>
                <input type="file" id="profilePicInput" accept="image/*" style={{ display: 'none' }} onChange={onProfilePictureUpload} />

                <div className="settings-item" onClick={() => setSettingsView('profile')}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                                <circle cx="12" cy="7" r="4"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Profile</div>
                            <div className="settings-item-desc">Name, bio, about</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToPrivacy}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                                <path d="M7 11V7a5 5 0 0110 0v4"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Privacy</div>
                            <div className="settings-item-desc">Last seen, profile photo, read receipts</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToNotifications}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                                <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Notifications</div>
                            <div className="settings-item-desc">Message alerts, sound, vibration</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToChatSettings}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Chat Settings</div>
                            <div className="settings-item-desc">Wallpaper, font size, enter key</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToGeneral}>
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
                            <div className="settings-item-title">General</div>
                            <div className="settings-item-desc">App theme, font size, language</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToAccount}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                <polyline points="9 12 11 14 15 10"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Account</div>
                            <div className="settings-item-desc">Security, change password, delete account</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={goToStorage}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <ellipse cx="12" cy="5" rx="9" ry="3"/>
                                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Storage & Data</div>
                            <div className="settings-item-desc">Manage storage, clear files</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={() => alert('Help & Support coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Help & Support</div>
                            <div className="settings-item-desc">FAQ, contact support</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-divider"></div>

                <div className="settings-item" onClick={handleLogout} style={{ color: '#ef4444' }}>
                    <div className="settings-item-left">
                        <span className="settings-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                                <polyline points="16 17 21 12 16 7"/>
                                <line x1="21" y1="12" x2="9" y2="12"/>
                            </svg>
                        </span>
                        <div>
                            <div className="settings-item-title">Logout</div>
                            <div className="settings-item-desc">Sign out of your account</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>
            </div>
        </>
    );
}