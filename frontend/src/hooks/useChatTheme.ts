// Extracted from ChatScreen.tsx during the pre-deployment refactor (Phase 3a), then
// extended during Phase 1 (legacy feature port) with the legacy React prototype's
// "system" theme option (`SettingsTheme.jsx`), which the current stack did not have
// until now. The quick header toggle keeps its original light<->dark-only behavior
// unchanged; "system" is only reachable from the new Settings panel.

// Import React's state/effect hooks.
import { useEffect, useState } from 'react'

// Import the per-username theme helpers; only the preference name (never a token) is
// ever persisted in localStorage, keyed by username.
import {
  applyTheme,
  clearTheme,
  loadThemePreference,
  resolveThemeName,
  saveTheme,
} from '../theme'
import type { ThemeName, ThemePreference } from '../theme'

// Describe what this hook hands back to ChatScreen.
export interface ChatThemeApi {
  // Carry the currently applied, concrete light/dark theme.
  theme: ThemeName
  // Carry the stored preference, which may be 'system'.
  themePreference: ThemePreference
  // Flip the resolved theme between light and dark and persist that explicit choice,
  // overriding any 'system' preference. Matches the original header quick-toggle.
  toggleTheme: (username: string) => void
  // Set an explicit preference (including 'system') from the Settings panel.
  setThemePreference: (username: string, preference: ThemePreference) => void
}

// Load, apply, and toggle the per-username chat theme, including "system".
export function useChatTheme(username: string | undefined): ChatThemeApi {
  // Hold the stored preference, which may be 'system'.
  const [themePreference, setPreferenceState] = useState<ThemePreference>('light')
  // Hold the concrete, currently applied light/dark theme.
  const [theme, setTheme] = useState<ThemeName>('light')

  // Load this username's preference, apply the resolved theme, then restore the
  // default palette when it goes away (logout, or this screen unmounting before a
  // session exists).
  useEffect(() => {
    if (!username) {
      clearTheme()
      return
    }
    const preference = loadThemePreference(username)
    const resolved = resolveThemeName(preference)
    setPreferenceState(preference)
    setTheme(resolved)
    applyTheme(resolved)
    return () => {
      clearTheme()
    }
  }, [username])

  // Re-resolve live if the OS preference changes while "system" is selected, so the
  // chat shell does not require a reload to pick up a day/night switch.
  useEffect(() => {
    if (themePreference !== 'system' || typeof window.matchMedia !== 'function') {
      return
    }
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      const resolved = resolveThemeName('system')
      setTheme(resolved)
      applyTheme(resolved)
    }
    query.addEventListener('change', handleChange)
    return () => {
      query.removeEventListener('change', handleChange)
    }
  }, [themePreference])

  // Persist the theme for this username only and apply it to the document.
  function toggleTheme(forUsername: string) {
    const next: ThemeName = theme === 'dark' ? 'light' : 'dark'
    setPreferenceState(next)
    setTheme(next)
    applyTheme(next)
    saveTheme(forUsername, next)
  }

  // Set an explicit preference, including "system", from the Settings panel.
  function setThemePreference(forUsername: string, preference: ThemePreference) {
    const resolved = resolveThemeName(preference)
    setPreferenceState(preference)
    setTheme(resolved)
    applyTheme(resolved)
    saveTheme(forUsername, preference)
  }

  return { theme, themePreference, toggleTheme, setThemePreference }
}
