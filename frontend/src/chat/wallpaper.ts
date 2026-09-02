// Store a per-conversation wallpaper data URL on this device only.
// Wallpapers are never uploaded; the server has no wallpaper column.

// Build the localStorage key for one owner's wallpaper in one conversation.
export function wallpaperStorageKey(username: string, conversationId: string): string {
  // Namespace by handle so two accounts on one browser keep separate wallpapers.
  return `secure-chat-wallpaper:${username}:${conversationId}`
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

// Cap wallpaper data URLs so localStorage cannot be filled by a huge image.
export const MAX_WALLPAPER_BYTES = 400 * 1024
