// Wrap an authenticated API call so an expired 15-minute access token is refreshed
// exactly once and the original call retried, instead of surfacing a raw 401 to the
// user or silently breaking a session that has been open longer than the token TTL.
//
// This closes a real pre-deployment gap: `refreshTokens()` and `AuthContext.updateTokens`
// already existed but nothing ever called them, so any tab left open past 15 minutes
// started failing every REST call with no recovery path.

// Import the real refresh call this helper wraps around every authenticated request.
import { refreshTokens } from './authClient'

// Describe the minimal session fields this helper needs from AuthContext's session.
export interface RefreshableSession {
  // Carry the short-lived token every wrapped request tries first.
  accessToken: string
  // Carry the single-use token exchanged for a new pair on exactly one 401.
  refreshToken: string
}

// Narrow the shared `.status` field every typed API error class in this app exposes
// (AuthApiError, KeysApiError, ContactsApiError, ConversationsApiError) without
// importing all four just to check one number.
interface HasStatus {
  status: number
}

// Return whether an unknown thrown value looks like one of this app's typed API errors.
function hasStatus(value: unknown): value is HasStatus {
  return (
    typeof value === 'object' && value !== null && typeof (value as HasStatus).status === 'number'
  )
}

// De-duplicate concurrent refreshes. Several REST calls can each hit a 401 in the same
// tick (contacts, key lookup, and conversation start can all fire close together), and
// refresh tokens are single-use with rotation: calling POST /auth/refresh twice with the
// same token would let the first call succeed and force every other caller to fail with
// a *second*, unrecoverable 401. Every 401 seen while one refresh is already in flight
// waits on that same promise instead of starting its own.
let inFlightRefresh: Promise<{ accessToken: string; refreshToken: string }> | null = null

// Perform (or join) the single in-flight token refresh for this tab. Exported so a
// closed WebSocket — which cannot "retry a request" the way a REST call can, since
// the token only travels once at connect time — can share the exact same dedupe
// latch a concurrent REST 401 might already be using, then open a new socket itself.
export function refreshSessionOnce(
  refreshToken: string,
): Promise<{ accessToken: string; refreshToken: string }> {
  if (inFlightRefresh === null) {
    inFlightRefresh = refreshTokens(refreshToken)
      .then((pair) => ({ accessToken: pair.accessToken, refreshToken: pair.refreshToken }))
      .finally(() => {
        // Clear the latch once settled so a later, independent expiry can refresh again.
        inFlightRefresh = null
      })
  }
  return inFlightRefresh
}

// Run one authenticated request, refreshing the access token and retrying exactly once
// if the server reports it as expired or otherwise invalid (401). A second 401 against a
// freshly rotated token is treated as a real session failure and propagated to the caller
// (typically shown to the user, who must log in again).
export async function withTokenRefresh<T>(
  // Read the caller's current token pair; this value is not mutated in place.
  session: RefreshableSession,
  // Publish a rotated pair back into AuthContext so later calls use the new tokens too.
  onTokensRotated: (accessToken: string, refreshToken: string) => void,
  // Perform the actual network call with whichever access token it is given.
  request: (accessToken: string) => Promise<T>,
): Promise<T> {
  try {
    return await request(session.accessToken)
  } catch (error) {
    // Only 401 (expired/invalid access token) is recoverable this way; every other
    // failure (validation, 404, network) must reach the caller unchanged.
    if (!hasStatus(error) || error.status !== 401) {
      throw error
    }
    const rotated = await refreshSessionOnce(session.refreshToken)
    onTokensRotated(rotated.accessToken, rotated.refreshToken)
    return request(rotated.accessToken)
  }
}
