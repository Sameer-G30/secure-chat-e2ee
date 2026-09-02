// Search accounts by username, replacing the legacy React prototype's approach
// (`users` RTDB node `.get()` — download every account row, substring-match in the
// browser). This client calls the authenticated, server-side, prefix-only search
// added during the pre-deployment review.

// Read the API base URL from the same build-time environment variable other clients use.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe one matched account. Username only — no email, key, or hash.
export interface UserSearchResult {
  username: string
}

// Distinguish expected API rejections from unexpected network failures.
export class UsersApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'UsersApiError'
    this.status = status
  }
}

function extractErrorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') {
      return detail
    }
  }
  return fallback
}

// Search for accounts whose username starts with `query`, excluding the caller.
// Returns an empty list (rather than throwing) for a too-short query, so a caller
// can wire this directly to an input's onChange without special-casing length.
export async function searchUsers(accessToken: string, query: string): Promise<UserSearchResult[]> {
  const trimmed = query.trim()
  if (trimmed.length < 2) {
    return []
  }
  const response = await fetch(
    `${API_BASE_URL}/users/search?${new URLSearchParams({ q: trimmed }).toString()}`,
    { method: 'GET', headers: { Authorization: `Bearer ${accessToken}` } },
  )
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not search for accounts. Please try again shortly.'),
      response.status,
    )
  }
  const payload = body as { users?: Array<{ username: string }> }
  if (!Array.isArray(payload.users)) {
    return []
  }
  return payload.users.map((row) => ({ username: row.username }))
}

// Describe the signed-in account's editable public profile.
export interface MeProfile {
  username: string
  email: string
  displayName: string | null
  bio: string | null
  hasAvatar: boolean
}

// Describe another account's public profile (no email).
export interface PublicProfile {
  username: string
  displayName: string | null
  bio: string | null
  hasAvatar: boolean
}

// Load the signed-in account's public profile for the Settings form.
export async function fetchMyProfile(accessToken: string): Promise<MeProfile> {
  const response = await fetch(`${API_BASE_URL}/users/me`, {
    method: 'GET',
    headers: { Authorization: `Bearer ${accessToken}` },
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not load your profile. Please try again shortly.'),
      response.status,
    )
  }
  return parseMeProfile(body)
}

// Update the signed-in account's display name and bio.
export async function patchMyProfile(
  accessToken: string,
  payload: { displayName: string; bio: string },
): Promise<MeProfile> {
  const response = await fetch(`${API_BASE_URL}/users/me`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ display_name: payload.displayName, bio: payload.bio }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not save your profile. Please try again shortly.'),
      response.status,
    )
  }
  return parseMeProfile(body)
}

// Upload a public avatar image for the signed-in account.
export async function uploadMyAvatar(accessToken: string, file: File): Promise<MeProfile> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_BASE_URL}/users/me/avatar`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${accessToken}` },
    body: form,
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not upload that photo. Please try again shortly.'),
      response.status,
    )
  }
  return parseMeProfile(body)
}

// Load another account's public profile for the contact panel.
export async function fetchUserProfile(
  accessToken: string,
  username: string,
): Promise<PublicProfile> {
  const response = await fetch(
    `${API_BASE_URL}/users/${encodeURIComponent(username)}/profile`,
    { method: 'GET', headers: { Authorization: `Bearer ${accessToken}` } },
  )
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not load that profile. Please try again shortly.'),
      response.status,
    )
  }
  const row = body as {
    username?: unknown
    display_name?: unknown
    bio?: unknown
    has_avatar?: unknown
  }
  return {
    username: typeof row.username === 'string' ? row.username : username,
    displayName: typeof row.display_name === 'string' ? row.display_name : null,
    bio: typeof row.bio === 'string' ? row.bio : null,
    hasAvatar: row.has_avatar === true,
  }
}

// Fetch another account's avatar as an object URL, or null when none is stored.
export async function fetchUserAvatar(
  accessToken: string,
  username: string,
): Promise<string | null> {
  const response = await fetch(
    `${API_BASE_URL}/users/${encodeURIComponent(username)}/avatar`,
    { method: 'GET', headers: { Authorization: `Bearer ${accessToken}` } },
  )
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new UsersApiError(
      extractErrorDetail(body, 'Could not load that photo. Please try again shortly.'),
      response.status,
    )
  }
  const blob = await response.blob()
  return URL.createObjectURL(blob)
}

// Narrow GET/PATCH /users/me JSON into the Settings form shape.
function parseMeProfile(body: unknown): MeProfile {
  const row = body as {
    username?: unknown
    email?: unknown
    display_name?: unknown
    bio?: unknown
    has_avatar?: unknown
  }
  return {
    username: typeof row.username === 'string' ? row.username : '',
    email: typeof row.email === 'string' ? row.email : '',
    displayName: typeof row.display_name === 'string' ? row.display_name : null,
    bio: typeof row.bio === 'string' ? row.bio : null,
    hasAvatar: row.has_avatar === true,
  }
}
