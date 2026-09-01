// Extracted from ChatScreen.tsx during the pre-deployment refactor (Phase 3a): every
// piece of this screen that talks to the REST API needs the same "retry once after a
// token refresh" behavior, so that wiring lives in exactly one hook instead of being
// copy-pasted into useContacts and useEncryptedConversation separately.

// Import React's ref/effect/callback hooks used to keep a live pointer to the session.
import { useCallback, useEffect, useRef } from 'react'
// Import the RefObject type separately; verbatimModuleSyntax requires type-only imports.
import type { RefObject } from 'react'

// Import the single-flight refresh primitives this hook wraps for component use.
import { refreshSessionOnce, withTokenRefresh } from '../api/authorizedRequest'
import type { RefreshableSession } from '../api/authorizedRequest'

// Describe what this hook hands back to a component or another hook.
export interface AuthorizedRequestApi<TSession extends RefreshableSession> {
  // Always read the latest session inside an async closure (a socket callback, a
  // .then() chain) instead of whatever was captured when that closure was created.
  sessionRef: RefObject<TSession | null>
  // Run one authenticated call, refreshing and retrying exactly once on a 401.
  request: <T>(fn: (accessToken: string) => Promise<T>) => Promise<T>
  // Force a refresh right now (used by the WebSocket close handler, which has no HTTP
  // request to retry — reopening a socket is its own kind of "retry").
  refreshAndRotate: () => Promise<{ accessToken: string; refreshToken: string }>
}

// Wire one session's worth of "refresh on 401" behavior for every caller in this screen.
export function useAuthorizedRequest<TSession extends RefreshableSession>(
  session: TSession | null,
  onTokensRotated: (accessToken: string, refreshToken: string) => void,
): AuthorizedRequestApi<TSession> {
  // Mirror the session into a ref every render so async closures never retry with a
  // token pair that was already rotated by the time they actually run.
  const sessionRef = useRef<TSession | null>(session)
  useEffect(() => {
    sessionRef.current = session
  }, [session])

  const request = useCallback(
    <T,>(fn: (accessToken: string) => Promise<T>): Promise<T> => {
      const current = sessionRef.current
      if (!current) {
        return Promise.reject(new Error('No active session to authorize this request.'))
      }
      return withTokenRefresh(current, onTokensRotated, fn)
    },
    [onTokensRotated],
  )

  const refreshAndRotate = useCallback(async (): Promise<{
    accessToken: string
    refreshToken: string
  }> => {
    const current = sessionRef.current
    if (!current) {
      throw new Error('No active session to refresh.')
    }
    // Share the module-level single-flight latch a concurrent REST 401 might already
    // be using, so a token expiring mid-conversation never triggers two refreshes.
    const rotated = await refreshSessionOnce(current.refreshToken)
    onTokensRotated(rotated.accessToken, rotated.refreshToken)
    return rotated
  }, [onTokensRotated])

  return { sessionRef, request, refreshAndRotate }
}
