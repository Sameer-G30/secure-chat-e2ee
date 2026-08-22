// src/hooks/useTheme.js
import { useState, useEffect } from 'react';

export function useTheme() {
    const [isDarkMode, setIsDarkMode] = useState(true);
    const [selectedTheme, setSelectedTheme] = useState(() => {
        return localStorage.getItem('secureChatTheme') || 'dark';
    });

    useEffect(() => {
        const savedTheme = localStorage.getItem('secureChatTheme');
        if (savedTheme === 'light') {
            setIsDarkMode(false);
            document.body.classList.add('light-mode');
        } else {
            setIsDarkMode(true);
            document.body.classList.remove('light-mode');
        }
    }, []);

    const applyTheme = (theme) => {
        setSelectedTheme(theme);
        localStorage.setItem('secureChatTheme', theme);
        
        if (theme === 'light') {
            setIsDarkMode(false);
            document.body.classList.add('light-mode');
        } else if (theme === 'dark') {
            setIsDarkMode(true);
            document.body.classList.remove('light-mode');
        } else if (theme === 'system') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            setIsDarkMode(prefersDark);
            if (prefersDark) {
                document.body.classList.remove('light-mode');
            } else {
                document.body.classList.add('light-mode');
            }
        }
    };

    return {
        isDarkMode,
        selectedTheme,
        applyTheme
    };
}
