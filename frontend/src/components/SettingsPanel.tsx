// Settings overlay copied from the frontend/vrati slide-over: list rows, theme
// cards, and logout. Profile edits the public name, bio, and photo. Privacy,
// Account, Storage, and Help explain what this stack actually does.

// Import state and effect so nested pages can load blocks.
import { useEffect, useState } from 'react'
import { Modal } from './Modal'
// Import the stored preference type the theme hook already understands.
import type { ThemePreference } from '../theme'
// Import the two on-device scam models Settings can keep resident.
import type { BrowserScamModelId } from '../ml/types'
// Import the block row the Privacy page lists and unblocks.
import type { BlockedAccount } from '../api/blocksClient'
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

// Name the pages this panel can open. Profile is one of them, not a second screen.
export type SettingsView = 'main' | 'profile' | 'privacy' | 'account' | 'storage' | 'help'

// Describe the small surface ChatScreen wires into this panel.
export interface SettingsPanelProps {
  // Open on the main list, or already on Profile when the sidebar photo was clicked.
  initialView?: SettingsView
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
  profileEmail: string
  profileAccessToken: string
  profileDisplayName: string
  profileBio: string
  profileHasAvatar: boolean
  // Bumped after upload or remove so the avatar image refetches.
  avatarRevision: number
  onSaveProfile: (displayName: string, bio: string) => Promise<void>
  onUploadAvatar: (file: File) => Promise<void>
  onRemoveAvatar: () => Promise<void>
  // Load and change this account's server-side block list.
  onLoadBlocks: () => Promise<BlockedAccount[]>
  onUnblock: (username: string) => Promise<void>
  // Clear this device's previews, wallpapers, banner cache, and DistilBERT cache.
  onClearDeviceData: () => Promise<void>
  // Open the public photo in the shared lightbox. There is no forward action.
  onViewAvatar: (objectUrl: string) => void
  // The on-device model this account has selected.
  scamModel: BrowserScamModelId
  // Unload the current graph and load this one.
  onScamModelChange: (modelId: BrowserScamModelId) => void
}

// Name the three theme choices with labels the Settings panel displays.
const THEME_OPTIONS: Array<{ value: ThemePreference; label: string; preview: string }> = [
  { value: 'light', label: 'Light', preview: 'light-preview' },
  { value: 'dark', label: 'Dark', preview: 'dark-preview' },
  { value: 'system', label: 'System', preview: 'system-preview' },
]

// The two graphs the chat will load, one at a time.
const SCAM_MODEL_OPTIONS: Array<{ id: BrowserScamModelId; label: string; desc: string }> = [
  {
    id: 'distilbert_student',
    label: 'MiniLM student',
    desc: 'Default. Smaller model, 128 tokens.',
  },
  {
    id: 'distilbert_best',
    label: 'DistilBERT best',
    desc: 'Larger model, 512 tokens.',
  },
]

// Name the nested rows. Each one has a real page below.
const SETTINGS_ROWS: Array<{
  id: Exclude<SettingsView, 'main'>
  title: string
  desc: string
  icon: 'user' | 'lock' | 'shield' | 'storage' | 'help'
}> = [
  { id: 'profile', title: 'Profile', desc: 'Name, bio, profile picture', icon: 'user' },
  { id: 'privacy', title: 'Privacy', desc: 'Presence, typing, blocks', icon: 'lock' },
  { id: 'account', title: 'Account', desc: 'Username, email, logout', icon: 'shield' },
  { id: 'storage', title: 'Storage & Data', desc: 'Clear data on this device', icon: 'storage' },
  { id: 'help', title: 'Help & Support', desc: 'How encryption and clear-chat work', icon: 'help' },
]

// Map a row id onto the matching lucide-style icon.
function RowIcon({ name }: { name: (typeof SETTINGS_ROWS)[number]['icon'] }) {
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
  initialView = 'main',
  themePreference,
  onThemePreferenceChange,
  onClose,
  onLogout,
  profileUsername,
  profileEmail,
  profileAccessToken,
  profileDisplayName,
  profileBio,
  profileHasAvatar,
  avatarRevision,
  onSaveProfile,
  onUploadAvatar,
  onRemoveAvatar,
  onLoadBlocks,
  onUnblock,
  onClearDeviceData,
  onViewAvatar,
  scamModel,
  onScamModelChange,
}: SettingsPanelProps) {
  // Hold which slide is visible. The sidebar photo passes profile as the initial view.
  const [view, setView] = useState<SettingsView>(initialView)
  // Hold the in-progress display name until Save.
  const [displayName, setDisplayName] = useState(profileDisplayName)
  // Hold the in-progress bio until Save.
  const [bio, setBio] = useState(profileBio)
  // Hold a profile-form status line distinct from the chat status.
  const [profileStatus, setProfileStatus] = useState<string | null>(null)
  // True while the remove-photo confirmation is open.
  const [confirmRemovePhoto, setConfirmRemovePhoto] = useState(false)
  // Hold the block list while Privacy is open.
  const [blocks, setBlocks] = useState<BlockedAccount[]>([])
  // Hold a short Privacy error. Empty means the list loaded.
  const [blockStatus, setBlockStatus] = useState<string | null>(null)
  // True while the device-data confirmation is open.
  const [confirmClearData, setConfirmClearData] = useState(false)
  // Hold the result of a device clear.
  const [storageStatus, setStorageStatus] = useState<string | null>(null)
  // Look up the open row so the nested header can reuse its title.
  const row = SETTINGS_ROWS.find((item) => item.id === view) ?? null

  // Load blocks when Privacy opens. Other pages do not need the request.
  useEffect(() => {
    if (view !== 'privacy') {
      return
    }
    let cancelled = false
    void onLoadBlocks()
      .then((rows) => {
        if (!cancelled) {
          setBlocks(rows)
          setBlockStatus(null)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBlockStatus('Could not load your block list.')
        }
      })
    return () => {
      cancelled = true
    }
  }, [view, onLoadBlocks])

  // Unblock one account and drop it from the list on success.
  async function handleUnblock(username: string) {
    try {
      await onUnblock(username)
      setBlocks((existing) => existing.filter((block) => block.username !== username))
      setBlockStatus(null)
    } catch {
      setBlockStatus('Could not unblock that account.')
    }
  }

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

              <div className="settings-divider" />
              <div className="settings-model-heading">
                <div className="settings-item-title">Scam detection</div>
                <div className="settings-item-desc">
                  On this device only. Switching unloads the current model.
                </div>
              </div>
              <div className="chat-settings-theme" role="radiogroup" aria-label="Scam detection model">
                {SCAM_MODEL_OPTIONS.map((option) => (
                  <label
                    key={option.id}
                    className={scamModel === option.id ? 'theme-option active' : 'theme-option'}
                  >
                    <input
                      type="radio"
                      name="scam-model"
                      value={option.id}
                      checked={scamModel === option.id}
                      aria-label={option.label}
                      onChange={() => onScamModelChange(option.id)}
                    />
                    <div>
                      <div className="theme-option-label">{option.label}</div>
                      <div className="settings-item-desc">{option.desc}</div>
                    </div>
                  </label>
                ))}
              </div>

              {SETTINGS_ROWS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="settings-item"
                  onClick={() => setView(item.id)}
                >
                  <div className="settings-item-left">
                    <span className="settings-icon">
                      <RowIcon name={item.icon} />
                    </span>
                    <div>
                      <div className="settings-item-title">{item.title}</div>
                      <div className="settings-item-desc">{item.desc}</div>
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
              <h2 id="settings-title">{row?.title ?? 'Settings'}</h2>
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
                    reloadKey={avatarRevision}
                    onView={onViewAvatar}
                  />
                  {profileHasAvatar ? (
                    <p className="settings-copy">A public photo is on this account.</p>
                  ) : (
                    <p className="settings-copy">No photo yet. JPEG, PNG, or WebP up to 200KB.</p>
                  )}
                  {profileHasAvatar ? (
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => setConfirmRemovePhoto(true)}
                    >
                      Remove photo
                    </button>
                  ) : null}
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
              ) : null}

              {view === 'privacy' ? (
                <div className="settings-page">
                  <p className="settings-copy">
                    Online and offline presence, typing indicators, and profile photos are
                    visible the way they are today. This app does not offer a last-seen time
                    or a switch to hide your profile photo.
                  </p>
                  <p className="settings-copy">Blocked accounts cannot message you. The server enforces the block.</p>
                  {blockStatus ? <p className="chat-status">{blockStatus}</p> : null}
                  {blocks.length === 0 ? (
                    <p className="settings-copy">No blocked accounts.</p>
                  ) : (
                    <ul className="settings-block-list" aria-label="Blocked accounts">
                      {blocks.map((block) => (
                        <li key={block.id} className="settings-block-row">
                          <span>{block.username}</span>
                          <button
                            type="button"
                            className="text-button"
                            onClick={() => void handleUnblock(block.username)}
                          >
                            Unblock {block.username}
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : null}

              {view === 'account' ? (
                <div className="settings-page">
                  <p className="settings-account-line">Username: {profileUsername}</p>
                  <p className="settings-account-line">Email: {profileEmail || 'Loading…'}</p>
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
                  </button>
                </div>
              ) : null}

              {view === 'storage' ? (
                <div className="settings-page">
                  <p className="settings-copy">
                    Chat photos stay sealed in the server database. Clearing data here removes
                    this device&apos;s sidebar previews, chat wallpapers, scam-banner cache, and
                    the downloaded DistilBERT file. It does not delete messages, sealed photos,
                    or the key vault.
                  </p>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => setConfirmClearData(true)}
                  >
                    Clear data on this device
                  </button>
                  {storageStatus ? <p className="chat-status">{storageStatus}</p> : null}
                </div>
              ) : null}

              {view === 'help' ? (
                <div className="settings-page">
                  <h3>End-to-end encryption</h3>
                  <p className="settings-copy">
                    Messages are encrypted on this device before they are sent. The server
                    stores and relays ciphertext, a nonce, and a key epoch. It cannot read
                    the message.
                  </p>
                  <h3>On-device scam banner</h3>
                  <p className="settings-copy">
                    After a message decrypts, this browser scores it with the model selected
                    in Settings (MiniLM student by default, or DistilBERT best). Only one model
                    is loaded. The score stays on this device. A banner does not block or delete
                    the message. A link in a flagged message asks before it opens.
                  </p>
                  <h3>Clear chat and delete</h3>
                  <p className="settings-copy">
                    Clear chat for me hides the conversation on your account only. The other
                    person keeps their copy. Delete for everyone removes a message you sent
                    for both people.
                  </p>
                  <h3>Staying signed in</h3>
                  <p className="settings-copy">
                    This browser stays signed in across a reload until you tap Logout. Logout
                    removes the saved session from this device. The account password is not
                    stored.
                  </p>
                </div>
              ) : null}
            </div>
          </>
        )}
      </div>
      {confirmRemovePhoto ? (
        <Modal title="Remove photo" onClose={() => setConfirmRemovePhoto(false)}>
          <p>
            This photo is public. Removing it takes it off the account for everyone.
          </p>
          <div className="chat-modal-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => {
                setConfirmRemovePhoto(false)
                void onRemoveAvatar().then(() => {
                  setProfileStatus('Photo removed.')
                })
              }}
            >
              Remove photo
            </button>
            <button type="button" className="text-button" onClick={() => setConfirmRemovePhoto(false)}>
              Cancel
            </button>
          </div>
        </Modal>
      ) : null}
      {confirmClearData ? (
        <Modal title="Clear data on this device" onClose={() => setConfirmClearData(false)}>
          <p>
            This removes local previews, wallpapers, scam-banner flags, and the downloaded
            DistilBERT file. Messages and the key vault stay.
          </p>
          <div className="chat-modal-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => {
                setConfirmClearData(false)
                void onClearDeviceData()
                  .then(() => {
                    setStorageStatus('Cleared data on this device.')
                  })
                  .catch(() => {
                    setStorageStatus('Could not clear data on this device.')
                  })
              }}
            >
              Clear data
            </button>
            <button type="button" className="text-button" onClick={() => setConfirmClearData(false)}>
              Cancel
            </button>
          </div>
        </Modal>
      ) : null}
    </div>
  )
}
