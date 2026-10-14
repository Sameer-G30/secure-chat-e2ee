// Store a per-conversation wallpaper data URL on this device only.
// Wallpapers are never uploaded; the server has no wallpaper column.

// Prefix used by every wallpaper key so a device clear can find them.
const WALLPAPER_PREFIX = 'secure-chat-wallpaper:'

// Build the localStorage key for one owner's wallpaper in one conversation.
export function wallpaperStorageKey(username: string, conversationId: string): string {
  // Namespace by handle so two accounts on one browser keep separate wallpapers.
  return `${WALLPAPER_PREFIX}${username}:${conversationId}`
}

// Read the stored wallpaper data URL, or null when the user has not set one.
export function readConversationWallpaper(
  username: string,
  conversationId: string,
): string | null {
  if (!username || !conversationId) {
    return null
  }
  try {
    return window.localStorage.getItem(wallpaperStorageKey(username, conversationId))
  } catch {
    return null
  }
}

// Persist a wallpaper data URL chosen on this device.
export function writeConversationWallpaper(
  username: string,
  conversationId: string,
  dataUrl: string,
): void {
  if (!username || !conversationId) {
    return
  }
  window.localStorage.setItem(wallpaperStorageKey(username, conversationId), dataUrl)
}

// Remove a wallpaper so the chat area falls back to the theme background.
export function clearConversationWallpaper(username: string, conversationId: string): void {
  if (!username || !conversationId) {
    return
  }
  window.localStorage.removeItem(wallpaperStorageKey(username, conversationId))
}

// Remove every wallpaper stored on this device. Messages and the key vault stay.
export function clearAllConversationWallpapers(): void {
  // Collect keys first so removing them does not skip indexes.
  const keys: string[] = []
  // Walk localStorage. Theme and banner keys use different prefixes.
  for (let index = 0; index < window.localStorage.length; index += 1) {
    // Read one key. A null key is skipped.
    const key = window.localStorage.key(index)
    // Keep only wallpaper entries.
    if (key?.startsWith(WALLPAPER_PREFIX)) {
      keys.push(key)
    }
  }
  // Remove each wallpaper. This does not call the server.
  for (const key of keys) {
    window.localStorage.removeItem(key)
  }
}

// Cap wallpaper data URLs so localStorage cannot be filled by a huge image.
export const MAX_WALLPAPER_BYTES = 400 * 1024
