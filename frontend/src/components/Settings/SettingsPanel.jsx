// src/components/Settings/SettingsPanel.jsx
import SettingsMain from './SettingsMain';
import SettingsGeneral from './SettingsGeneral';
import SettingsTheme from './SettingsTheme';

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
    applyTheme
}) {
    const renderView = () => {
        switch (settingsView) {
            case 'general':
                return (
                    <SettingsGeneral
                        goBackToMain={goBackToMain}
                        goToTheme={goToTheme}
                        closeSettings={closeSettings}
                        selectedTheme={selectedTheme}
                    />
                );
            case 'theme':
                return (
                    <SettingsTheme
                        goBackToGeneral={goBackToGeneral}
                        closeSettings={closeSettings}
                        selectedTheme={selectedTheme}
                        applyTheme={applyTheme}
                    />
                );
            default:
                return (
                    <SettingsMain
                        closeSettings={closeSettings}
                        goToGeneral={goToGeneral}
                        handleLogout={handleLogout}
                        setSettingsView={setSettingsView}
                    />
                );
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