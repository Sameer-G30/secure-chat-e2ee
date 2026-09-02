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
}

// Show a public avatar image when the account has one, otherwise a letter glyph.
export function Avatar({ username, accessToken, className, initials }: AvatarProps) {
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
  }, [username, accessToken])

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
