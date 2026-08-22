// src/components/Settings/SettingsMain.jsx
import { 
    IconUser, IconLock, IconPalette, IconSettings, 
    IconStorage, IconHelp, IconLogout, IconArrow 
} from '../../icons';

export default function SettingsMain({
    closeSettings,
    goToGeneral,
    handleLogout,
    setSettingsView
}) {
    return (
        <>
            <div className="settings-header">
                <h2>Settings</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="settings-item" onClick={() => alert('Profile coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconUser /></span>
                        <div>
                            <div className="settings-item-title">Profile</div>
                            <div className="settings-item-desc">Name, bio, profile picture</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-item" onClick={() => alert('Privacy coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconLock /></span>
                        <div>
                            <div className="settings-item-title">Privacy</div>
                            <div className="settings-item-desc">Last seen, profile photo</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-item" onClick={goToGeneral}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconPalette /></span>
                        <div>
                            <div className="settings-item-title">General</div>
                            <div className="settings-item-desc">App theme, font size, language</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-item" onClick={() => alert('Account coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconSettings /></span>
                        <div>
                            <div className="settings-item-title">Account</div>
                            <div className="settings-item-desc">Two-factor authentication, security</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-item" onClick={() => alert('Storage coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconStorage /></span>
                        <div>
                            <div className="settings-item-title">Storage & Data</div>
                            <div className="settings-item-desc">Manage storage, clear files</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-item" onClick={() => alert('Help coming soon')}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconHelp /></span>
                        <div>
                            <div className="settings-item-title">Help & Support</div>
                            <div className="settings-item-desc">FAQ, contact support</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>

                <div className="settings-divider"></div>

                <div className="settings-item" onClick={handleLogout} style={{ color: '#ef4444' }}>
                    <div className="settings-item-left">
                        <span className="settings-icon"><IconLogout /></span>
                        <div>
                            <div className="settings-item-title">Logout</div>
                            <div className="settings-item-desc">Sign out of your account</div>
                        </div>
                    </div>
                    <span className="settings-arrow"><IconArrow /></span>
                </div>
            </div>
        </>
    );
}