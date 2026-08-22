// src/hooks/useSettings.js
import { useState } from 'react';

export function useSettings() {
    const [showSettings, setShowSettings] = useState(false);
    const [settingsView, setSettingsView] = useState('main');

    const openSettings = () => {
        setShowSettings(true);
        setSettingsView('main');
    };

    const closeSettings = () => {
        setShowSettings(false);
        setSettingsView('main');
    };

    const goToGeneral = () => setSettingsView('general');
    const goToTheme = () => setSettingsView('theme');
    const goBackToMain = () => setSettingsView('main');
    const goBackToGeneral = () => setSettingsView('general');

    return {
        showSettings,
        settingsView,
        openSettings,
        closeSettings,
        goToGeneral,
        goToTheme,
        goBackToMain,
        goBackToGeneral
    };
}