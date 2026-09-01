// Prefix localStorage keys so theme is stored per signed-in username, not shared.
const THEME_STORAGE_PREFIX = 'secure-chat-theme:'

// Name the two visual themes the chat shell can actually render.
export type ThemeName = 'light' | 'dark'

// Name every value a user can choose to have stored, including the legacy React
// prototype's "system" option (`SettingsTheme.jsx`), which the current stack did not
// have until this port. "system" is resolved to a concrete ThemeName at apply time,
// via resolveThemeName below, and re-resolved live if the OS preference changes.
export type ThemePreference = ThemeName | 'system'

// Build the per-user localStorage key for the chat theme.
export function themeStorageKey(username: string): string {
  return `${THEME_STORAGE_PREFIX}${username}`
}

// Read the saved theme preference for this username, defaulting to light.
export function loadThemePreference(username: string): ThemePreference {
  try {
    const stored = window.localStorage.getItem(themeStorageKey(username))
    if (stored === 'dark' || stored === 'light' || stored === 'system') {
      return stored
    }
  } catch {
    // Private mode or blocked storage must not break chat.
  }
  return 'light'
}

// Read the saved theme for this username, defaulting to light. Kept as a thin
// backward-compatible wrapper: any earlier code (or test) that only ever expected
// 'light' | 'dark' back still gets a concrete, renderable value.
export function loadTheme(username: string): ThemeName {
  return resolveThemeName(loadThemePreference(username))
}

// Persist the theme preference for this username only; tokens stay out of localStorage.
export function saveTheme(username: string, theme: ThemePreference): void {
  try {
    window.localStorage.setItem(themeStorageKey(username), theme)
  } catch {
    // A storage failure leaves the in-memory theme in place for this session.
  }
}

// Resolve "system" to a concrete light/dark value using the OS-level media query;
// an explicit light/dark preference passes through unchanged.
export function resolveThemeName(preference: ThemePreference): ThemeName {
  if (preference !== 'system') {
    return preference
  }
  try {
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    // A browser without matchMedia (or a locked-down test environment) defaults to light.
    return 'light'
  }
}

// Apply the theme to the document so CSS variables can switch palettes.
export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme
}

// Remove the theme attribute so the signed-out auth screen stays on the default palette.
export function clearTheme(): void {
  delete document.documentElement.dataset.theme
}
