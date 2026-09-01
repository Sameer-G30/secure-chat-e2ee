// src/components/Settings/SettingsPanel.jsx
import SettingsMain from './SettingsMain';
import SettingsGeneral from './SettingsGeneral';
import SettingsTheme from './SettingsTheme';
import SettingsPrivacy from './SettingsPrivacy';
import SettingsAccount from './SettingsAccount';
import SettingsNotifications from './SettingsNotifications';
import SettingsChat from './SettingsChat';
import SettingsProfile from './SettingsProfile';
import SettingsStorage from './SettingsStorage';

export default function SettingsPanel({
    settingsView,
    setSettingsView,
    closeSettings,
    handleLogout,
    goToGeneral,
    goBackToMain,
    goBackToGeneral,
    goToTheme,
    selectedTheme,
    applyTheme,
    profilePicture,
    onProfilePictureUpload,
    currentUser
}) {
    const goToPrivacy = () => setSettingsView('privacy');
    const goToAccount = () => setSettingsView('account');
    const goToNotifications = () => setSettingsView('notifications');
    const goToChatSettings = () => setSettingsView('chat');
    const goToStorage = () => setSettingsView('storage');

    const renderView = () => {
        switch (settingsView) {
            case 'general':
                return <SettingsGeneral goBackToMain={goBackToMain} goToTheme={goToTheme} closeSettings={closeSettings} selectedTheme={selectedTheme} />;
            case 'theme':
                return <SettingsTheme goBackToGeneral={goBackToGeneral} closeSettings={closeSettings} selectedTheme={selectedTheme} applyTheme={applyTheme} />;
            case 'privacy':
                return <SettingsPrivacy goBackToMain={goBackToMain} closeSettings={closeSettings} />;
            case 'account':
                return <SettingsAccount goBackToMain={goBackToMain} closeSettings={closeSettings} currentUser={currentUser} />;
            case 'notifications':
                return <SettingsNotifications goBackToMain={goBackToMain} closeSettings={closeSettings} />;
            case 'chat':
                return <SettingsChat goBackToMain={goBackToMain} closeSettings={closeSettings} />;
            case 'profile':
                return <SettingsProfile goBackToMain={goBackToMain} closeSettings={closeSettings} currentUser={currentUser} profilePicture={profilePicture} onProfilePictureUpload={onProfilePictureUpload} />;
            case 'storage':
                return <SettingsStorage goBackToMain={goBackToMain} closeSettings={closeSettings} />;
            default:
                return <SettingsMain
                    closeSettings={closeSettings}
                    goToGeneral={goToGeneral}
                    handleLogout={handleLogout}
                    setSettingsView={setSettingsView}
                    profilePicture={profilePicture}
                    onProfilePictureUpload={onProfilePictureUpload}
                    currentUser={currentUser}
                    goToPrivacy={goToPrivacy}
                    goToAccount={goToAccount}
                    goToNotifications={goToNotifications}
                    goToChatSettings={goToChatSettings}
                    goToStorage={goToStorage}
                />;
        }
    };

    return (
        <div className="settings-overlay" onClick={closeSettings}>
            <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
                {renderView()}
            </div>
        </div>
    );
}