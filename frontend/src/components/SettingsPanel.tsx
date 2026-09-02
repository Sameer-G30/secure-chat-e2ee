// Settings overlay copied from the frontend/vrati slide-over: list rows, theme
// cards, and logout. Profile/Privacy/Account/Storage/Help are visual stubs
// because this stack has no SMTP, avatars, or 2FA. Theme includes the "system"
// option tests select immediately after opening Settings.

// Import state so the panel can switch between the main list and stub pages.
import { useState } from 'react'
// Import the stored preference type the theme hook already understands.
import type { ThemePreference } from '../theme'
// Import the avatar glyph used on the working profile form.
import { Avatar } from './Avatar'
// Import the vrati icon set used on each settings row.
import {
  IconArrow,
  IconClose,
  IconHelp,
  IconLock,
  IconLogout,
  IconPalette,
  IconShield,
  IconStorage,
  IconUser,
} from '../icons'

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
  // Carry the signed-in account's editable public profile (not E2EE).
  profileUsername: string
  profileAccessToken: string
  profileDisplayName: string
  profileBio: string
  profileHasAvatar: boolean
  onSaveProfile: (displayName: string, bio: string) => Promise<void>
  onUploadAvatar: (file: File) => Promise<void>
}

// Name the three theme choices with labels the Settings panel displays.
const THEME_OPTIONS: Array<{ value: ThemePreference; label: string; preview: string }> = [
  { value: 'light', label: 'Light', preview: 'light-preview' },
  { value: 'dark', label: 'Dark', preview: 'dark-preview' },
  { value: 'system', label: 'System', preview: 'system-preview' },
]

// Name the unimplemented rows that still need to look like vrati's list.
const STUB_ROWS: Array<{ id: string; title: string; desc: string; icon: 'user' | 'lock' | 'shield' | 'storage' | 'help' }> =
  [
    { id: 'profile', title: 'Profile', desc: 'Name, bio, profile picture', icon: 'user' },
    { id: 'privacy', title: 'Privacy', desc: 'Last seen, profile photo', icon: 'lock' },
    { id: 'account', title: 'Account', desc: 'Two-factor authentication, security', icon: 'shield' },
    { id: 'storage', title: 'Storage & Data', desc: 'Manage storage, clear files', icon: 'storage' },
    { id: 'help', title: 'Help & Support', desc: 'FAQ, contact support', icon: 'help' },
  ]

// Map a stub id onto the matching lucide-style icon.
function StubIcon({ name }: { name: (typeof STUB_ROWS)[number]['icon'] }) {
  // Choose the glyph that vrati used for this row.
  if (name === 'user') {
    return <IconUser />
  }
  if (name === 'lock') {
    return <IconLock />
  }
  if (name === 'shield') {
    return <IconShield />
  }
  if (name === 'storage') {
    return <IconStorage />
  }
  return <IconHelp />
}

// Render account settings that this stack actually implements, in vrati chrome.
export function SettingsPanel({
  themePreference,
  onThemePreferenceChange,
  onClose,
  onLogout,
  profileUsername,
  profileAccessToken,
  profileDisplayName,
  profileBio,
  profileHasAvatar,
  onSaveProfile,
  onUploadAvatar,
}: SettingsPanelProps) {
  // Hold which slide is visible: the main list, or one nested page.
  const [view, setView] = useState<'main' | string>('main')
  // Hold the in-progress display name until Save.
  const [displayName, setDisplayName] = useState(profileDisplayName)
  // Hold the in-progress bio until Save.
  const [bio, setBio] = useState(profileBio)
  // Hold a profile-form status line distinct from the chat status.
  const [profileStatus, setProfileStatus] = useState<string | null>(null)
  // Look up the open stub so the nested header can reuse its title.
  const stub = STUB_ROWS.find((row) => row.id === view) ?? null

  return (
    <div
      className="settings-overlay"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose()
        }
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          onClose()
        }
      }}
    >
      <div className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        {view === 'main' ? (
          <>
            <div className="settings-header">
              <h2 id="settings-title">Settings</h2>
              <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
                <IconClose />
              </button>
            </div>
            <div className="settings-content">
              <div className="settings-item" aria-hidden="true">
                <div className="settings-item-left">
                  <span className="settings-icon">
                    <IconPalette />
                  </span>
                  <div>
                    <div className="settings-item-title">Theme</div>
                    <div className="settings-item-desc">App theme, including this device&apos;s system setting</div>
                  </div>
                </div>
              </div>
              <div className="chat-settings-theme" role="radiogroup" aria-label="Theme">
                {THEME_OPTIONS.map((option) => (
                  <label
                    key={option.value}
                    className={themePreference === option.value ? 'theme-option active' : 'theme-option'}
                  >
                    <input
                      type="radio"
                      name="chat-theme"
                      value={option.value}
                      checked={themePreference === option.value}
                      aria-label={option.label}
                      onChange={() => onThemePreferenceChange(option.value)}
                    />
                    <div className="theme-option-content">
                      <div className={`theme-preview ${option.preview}`}>
                        <div className="theme-preview-header" />
                        <div className="theme-preview-body" />
                      </div>
                      <div className="theme-option-label">{option.label}</div>
                    </div>
                  </label>
                ))}
              </div>

              {STUB_ROWS.map((row) => (
                <button
                  key={row.id}
                  type="button"
                  className="settings-item"
                  onClick={() => setView(row.id)}
                >
                  <div className="settings-item-left">
                    <span className="settings-icon">
                      <StubIcon name={row.icon} />
                    </span>
                    <div>
                      <div className="settings-item-title">{row.title}</div>
                      <div className="settings-item-desc">{row.desc}</div>
                    </div>
                  </div>
                  <span className="settings-arrow">
                    <IconArrow />
                  </span>
                </button>
              ))}

              <div className="settings-divider" />

              <button type="button" className="settings-item settings-item-logout" onClick={onLogout}>
                <div className="settings-item-left">
                  <span className="settings-icon">
                    <IconLogout />
                  </span>
                  <div>
                    <div className="settings-item-title">Logout</div>
                    <div className="settings-item-desc">Sign out of your account</div>
                  </div>
                </div>
                <span className="settings-arrow">
                  <IconArrow />
                </span>
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="settings-header">
              <button type="button" className="icon-btn" aria-label="Back to settings" onClick={() => setView('main')}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <polyline points="15 18 9 12 15 6" />
                </svg>
              </button>
              <h2 id="settings-title">{stub?.title ?? 'Settings'}</h2>
              <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
                <IconClose />
              </button>
            </div>
            <div className="settings-content">
              {view === 'profile' ? (
                <form
                  className="profile-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    void onSaveProfile(displayName, bio).then(() => {
                      setProfileStatus('Profile saved.')
                    })
                  }}
                >
                  <Avatar
                    username={profileUsername}
                    accessToken={profileAccessToken}
                    className="contact-profile-avatar"
                    initials={profileUsername.slice(0, 1).toUpperCase()}
                  />
                  {profileHasAvatar ? (
                    <p className="settings-stub">A public photo is on this account.</p>
                  ) : (
                    <p className="settings-stub">No photo yet. JPEG, PNG, or WebP up to 200KB.</p>
                  )}
                  <label className="profile-file-label">
                    Change photo
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      className="visually-hidden"
                      onChange={(event) => {
                        const file = event.target.files?.[0]
                        if (file) {
                          void onUploadAvatar(file).then(() => {
                            setProfileStatus('Photo updated.')
                          })
                        }
                        event.target.value = ''
                      }}
                    />
                  </label>
                  <label htmlFor="profile-display-name">Display name</label>
                  <input
                    id="profile-display-name"
                    type="text"
                    maxLength={64}
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
                  />
                  <label htmlFor="profile-bio">Bio</label>
                  <textarea
                    id="profile-bio"
                    maxLength={280}
                    rows={3}
                    value={bio}
                    onChange={(event) => setBio(event.target.value)}
                  />
                  <button type="submit" className="primary-button">
                    Save profile
                  </button>
                  {profileStatus ? <p className="chat-status">{profileStatus}</p> : null}
                </form>
              ) : (
                <p className="settings-stub">
                  {stub?.title} is not implemented on this stack. There is no SMTP or
                  two-factor flow in this deployment. Theme, profile, and logout work from
                  Settings.
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
