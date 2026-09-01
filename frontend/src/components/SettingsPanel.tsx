// Settings overlay ported from the legacy React prototype's Settings panel, minus
// Firebase and minus features this stack does not implement (email verification,
// password reset). Theme includes the legacy "system" option that the current
// stack was missing until Phase 1. Logout already lives on the header; it is
// repeated here so the panel is a complete account surface.

// Import the reusable overlay this panel renders inside.
import { Modal } from './Modal'
// Import the stored preference type the theme hook already understands.
import type { ThemePreference } from '../theme'

// Describe the small surface ChatScreen wires into this panel.
export interface SettingsPanelProps {
  // Carry the stored preference, which may be 'system'.
  themePreference: ThemePreference
  // Persist a new preference, including 'system'.
  onThemePreferenceChange: (preference: ThemePreference) => void
  // Close the overlay without changing anything else.
  onClose: () => void
  // Sign out from the panel as well as from the header.
  onLogout: () => void
}

// Name the three theme choices with labels the Settings panel displays.
const THEME_OPTIONS: Array<{ value: ThemePreference; label: string }> = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' },
]

// Render account settings that this stack actually implements.
export function SettingsPanel({
  themePreference,
  onThemePreferenceChange,
  onClose,
  onLogout,
}: SettingsPanelProps) {
  return (
    <Modal title="Settings" onClose={onClose}>
      <section className="chat-settings-section">
        <h3>Theme</h3>
        <p className="chat-settings-hint">
          System follows this device&apos;s light/dark preference and updates live.
        </p>
        <div className="chat-settings-theme" role="radiogroup" aria-label="Theme">
          {THEME_OPTIONS.map((option) => (
            <label key={option.value} className="chat-settings-choice">
              <input
                type="radio"
                name="chat-theme"
                value={option.value}
                checked={themePreference === option.value}
                onChange={() => onThemePreferenceChange(option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
      </section>
      <section className="chat-settings-section">
        <h3>Account</h3>
        <p className="chat-settings-hint">
          Email verification and password reset are not implemented (no SMTP in this
          deployment). Sign-out is available here and on the header.
        </p>
        <button type="button" className="text-button" onClick={onLogout}>
          Log out
        </button>
      </section>
    </Modal>
  )
}
