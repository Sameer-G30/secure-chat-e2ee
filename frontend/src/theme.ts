// Prefix localStorage keys so theme is stored per signed-in username, not shared.
const THEME_STORAGE_PREFIX = 'secure-chat-theme:'

// Name the two visual themes the chat shell can apply.
export type ThemeName = 'light' | 'dark'

// Build the per-user localStorage key for the chat theme.
export function themeStorageKey(username: string): string {
  return `${THEME_STORAGE_PREFIX}${username}`
}

// Read the saved theme for this username, defaulting to light.
export function loadTheme(username: string): ThemeName {
  try {
    const stored = window.localStorage.getItem(themeStorageKey(username))
    if (stored === 'dark' || stored === 'light') {
      return stored
    }
  } catch {
    // Private mode or blocked storage must not break chat.
  }
  return 'light'
}

// Persist the theme for this username only; tokens stay out of localStorage.
export function saveTheme(username: string, theme: ThemeName): void {
  try {
    window.localStorage.setItem(themeStorageKey(username), theme)
  } catch {
    // A storage failure leaves the in-memory theme in place for this session.
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
