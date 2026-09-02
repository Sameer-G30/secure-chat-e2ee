// Contact profile slide-over: public metadata, online state, and chat actions.

// Import the public profile shape loaded from GET /users/{username}/profile.
import type { PublicProfile } from '../api/usersClient'
// Import the avatar glyph that fetches authenticated image bytes.
import { Avatar } from './Avatar'
// Import the close icon used on the overlay header.
import { IconClose } from '../icons'

// Describe the handlers ChatScreen wires into this panel.
export interface ContactProfilePanelProps {
  // Carry the peer handle used for avatar fetch and headings.
  username: string
  // Carry the bearer token used for the authenticated avatar GET.
  accessToken: string
  // Carry the public profile metadata, or null while it is still loading.
  profile: PublicProfile | null
  // True when the peer currently has a socket in this conversation.
  online: boolean
  // True when this account has blocked the peer.
  blocked: boolean
  // Close the overlay without changing anything else.
  onClose: () => void
  // Open in-chat search over decrypted in-memory text.
  onSearch: () => void
  // Block or unblock from the profile panel.
  onBlock: () => void
  onUnblock: () => void
  // Open the metadata-only report form.
  onReport: () => void
}

// Render public profile metadata the server is allowed to store (not E2EE).
export function ContactProfilePanel({
  username,
  accessToken,
  profile,
  online,
  blocked,
  onClose,
  onSearch,
  onBlock,
  onUnblock,
  onReport,
}: ContactProfilePanelProps) {
  const displayName = profile?.displayName?.trim() || username
  const initials = username.trim().slice(0, 1).toUpperCase() || '?'

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
      <div className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="contact-profile-title">
        <div className="settings-header">
          <h2 id="contact-profile-title">Contact info</h2>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <IconClose />
          </button>
        </div>
        <div className="settings-content contact-profile-content">
          <Avatar
            username={username}
            accessToken={accessToken}
            className="contact-profile-avatar"
            initials={initials}
          />
          <div className="contact-profile-name">{displayName}</div>
          <div className="contact-profile-handle">@{username}</div>
          <div className="contact-profile-status">{online ? 'Online' : 'Offline'}</div>
          {profile?.bio ? <p className="contact-profile-bio">{profile.bio}</p> : null}
          <div className="contact-profile-actions">
            <button type="button" className="settings-item" onClick={onSearch}>
              Search in chat
            </button>
            {blocked ? (
              <button type="button" className="settings-item" onClick={onUnblock}>
                Unblock
              </button>
            ) : (
              <button type="button" className="settings-item" onClick={onBlock}>
                Block
              </button>
            )}
            <button type="button" className="settings-item" onClick={onReport}>
              Report
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
