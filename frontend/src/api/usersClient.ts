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
