// Render a circular avatar from public profile bytes, falling back to initials.

// Import React's effect hook so object URLs are revoked on unmount.
import { useEffect, useState } from 'react'
// Import the authenticated avatar fetch used for both self and contacts.
import { fetchUserAvatar } from '../api/usersClient'

// Describe the small surface this glyph needs from ChatScreen.
export interface AvatarProps {
  // Identify which handle to fetch GET /users/{username}/avatar for.
  username: string
  // Carry the bearer token used for the authenticated avatar GET.
  accessToken: string
  // Apply the CSS class that sizes this circle (sidebar, header, or contact).
  className: string
  // Provide initials while the image loads or when none is stored.
  initials: string
  // Change this after an upload or remove so the image is fetched again.
  reloadKey?: number
  // Called with the object URL when a large avatar is clicked. Omitted on small glyphs.
  onView?: (objectUrl: string) => void
  // Called when this glyph is clicked, even if no photo is stored. Not a lightbox.
  onPress?: () => void
  // Accessible name for onPress. Defaults to a profile label.
  pressLabel?: string
}

// Show a public avatar image when the account has one, otherwise a letter glyph.
export function Avatar({
  username,
  accessToken,
  className,
  initials,
  reloadKey = 0,
  onView,
  onPress,
  pressLabel,
}: AvatarProps) {
  // Hold the blob: URL created from the authenticated avatar response.
  const [objectUrl, setObjectUrl] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let createdUrl: string | null = null
    void fetchUserAvatar(accessToken, username)
      .then((url) => {
        if (cancelled) {
          if (url) {
            URL.revokeObjectURL(url)
          }
          return
        }
        createdUrl = url
        setObjectUrl(url)
      })
      .catch(() => {
        if (!cancelled) {
          setObjectUrl(null)
        }
      })
    return () => {
      cancelled = true
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl)
      }
    }
  }, [username, accessToken, reloadKey])

  if (onPress) {
    return (
      <button
        type="button"
        className={className}
        aria-label={pressLabel ?? `Profile for ${username}`}
        onClick={onPress}
      >
        {objectUrl ? <img src={objectUrl} alt="" className="avatar-image" /> : initials}
      </button>
    )
  }

  if (objectUrl && onView) {
    return (
      <button
        type="button"
        className={className}
        aria-label={`View photo for ${username}`}
        onClick={() => onView(objectUrl)}
      >
        <img src={objectUrl} alt="" className="avatar-image" />
      </button>
    )
  }

  return (
    <div className={className} aria-hidden="true">
      {objectUrl ? (
        <img src={objectUrl} alt="" className="avatar-image" />
      ) : (
        initials
      )}
    </div>
  )
}
