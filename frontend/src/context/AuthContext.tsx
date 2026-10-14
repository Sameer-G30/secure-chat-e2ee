// Import React's context, provider-state, and consumer-hook primitives.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
// Import the ReactNode type separately; verbatimModuleSyntax requires type-only imports.
import type { ReactNode } from 'react'

// Import the real login/logout/refresh network calls this context wraps.
import { logoutAccount, refreshTokens } from '../api/authClient'
// Import the reload identity check. It throws IdentitySetupError for a missing vault.
import { assertSessionIdentity } from '../crypto/identitySetup'
// Import the local identity keypair type produced during login's key-bootstrap step.
import type { IdentityKeyPair } from '../crypto/keyExchange'
// Import the IndexedDB session row. It is not localStorage and it has no password.
import {
  deleteRestorableSession,
  readRestorableSession,
  writeRestorableSession,
} from '../crypto/sessionStore'
// Import the claim bit so a child setSession wins over a startup refresh.
import {
  hasInMemorySession,
  markInMemorySession,
  resetInMemorySessionClaim,
} from './sessionClaim'

// Describe the fields a completed, authenticated session carries in memory.
export interface Session {
  // Identify the signed-in account.
  username: string
  // Carry the short-lived JWT sent with every authenticated API request.
  accessToken: string
  // Carry the longer-lived, single-use JWT used only to request a new pair.
  refreshToken: string
  // Carry the unsealed X25519 identity keypair for this browser.
  //
  // A copy also lives in IndexedDB (secure-chat-session) so a reload can
  // reopen ChatScreen without the password. Logout deletes that copy.
  // localStorage, sessionStorage, and cookies never receive it.
  identityKeyPair: IdentityKeyPair
}

// Describe the operations components can call against the current session.
interface AuthContextValue {
  // Expose the current session, or null while signed out.
  session: Session | null
  // False until the IndexedDB restore attempt has finished.
  sessionReady: boolean
  // Replace the current session after a successful login/registration flow.
  setSession: (session: Session) => void
  // Replace only the token pair after a refresh, keeping identity keys in memory.
  updateTokens: (accessToken: string, refreshToken: string) => void
  // Revoke the refresh token server-side (best-effort) and clear local session state.
  logout: () => Promise<void>
}

// Create the context with no default value; useAuth enforces it is always provided.
const AuthContext = createContext<AuthContextValue | null>(null)

// One in-flight restore so React StrictMode cannot rotate the refresh token twice.
let inFlightRestore: Promise<Session | null> | null = null

// Copy the fields IndexedDB is allowed to keep. No password and no access token.
function persistSession(session: Session): Promise<void> {
  // Write the username, the current refresh token, and the unsealed keypair.
  return writeRestorableSession({
    username: session.username,
    refreshToken: session.refreshToken,
    publicKey: session.identityKeyPair.publicKey,
    privateKey: session.identityKeyPair.privateKey,
  })
}

// Read IndexedDB, rotate the refresh token once, and return a session or null.
async function performRestore(): Promise<Session | null> {
  // A login that already called setSession owns this page load.
  if (hasInMemorySession()) {
    return null
  }
  // Read the row Logout deletes. A miss means AuthScreen.
  const stored = await readRestorableSession()
  // setSession may have landed while the read was in flight. Do not refresh.
  if (hasInMemorySession()) {
    return null
  }
  // No row means this browser is signed out.
  if (!stored) {
    return null
  }
  // Rotate the single-use refresh token exactly once. A rejection is not retried.
  let tokens: Awaited<ReturnType<typeof refreshTokens>>
  try {
    tokens = await refreshTokens(stored.refreshToken)
  } catch {
    // The stored token is dead. Drop the row so the next load shows AuthScreen.
    await deleteRestorableSession().catch(() => {
      // A failed delete still leaves React signed out.
    })
    return null
  }
  // Build the keypair from the saved bytes before any further await.
  const identityKeyPair: IdentityKeyPair = {
    publicKey: stored.publicKey,
    privateKey: stored.privateKey,
  }
  // Replace the consumed refresh token immediately so a crash cannot present it again.
  const restored: Session = {
    username: stored.username,
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken,
    identityKeyPair,
  }
  try {
    await persistSession(restored)
  } catch {
    // Without the new refresh token on disk, the next load cannot continue.
    await deleteRestorableSession().catch(() => {
      // Ignore a second storage failure.
    })
    return null
  }
  // Reject a saved key the vault cannot support. Do not call refresh again.
  try {
    await assertSessionIdentity(
      restored.username,
      identityKeyPair,
      tokens.accessToken,
      tokens.hasPublicKey,
    )
  } catch {
    // IdentitySetupError and a failed public-key upload both end the saved session.
    await deleteRestorableSession().catch(() => {
      // The in-memory result is still null either way.
    })
    return null
  }
  // This page load now has a session. A second effect must not refresh again.
  markInMemorySession()
  return restored
}

// Join the single in-flight restore. StrictMode's second effect awaits the same promise.
function restorePersistedSession(): Promise<Session | null> {
  // Reuse the promise started by the first effect in this page load.
  if (inFlightRestore) {
    return inFlightRestore
  }
  // Start the one refresh. Clear the latch when it settles so logout can remount.
  inFlightRestore = performRestore().finally(() => {
    inFlightRestore = null
  })
  return inFlightRestore
}

// Provide session state to the whole application tree.
export function AuthProvider({ children }: { children: ReactNode }) {
  // Hold the live session. A reload fills this from IndexedDB before AuthScreen.
  const [session, setSessionState] = useState<Session | null>(null)
  // Stay false until restore finishes so AuthScreen cannot flash on a saved session.
  const [sessionReady, setSessionReady] = useState(false)

  // Restore once on mount. A rejected refresh or IdentitySetupError shows AuthScreen.
  useEffect(() => {
    // Ignore the result if this provider unmounted (StrictMode's first effect).
    let cancelled = false
    // A child effect already called setSession. Do not touch the refresh token.
    if (hasInMemorySession()) {
      setSessionReady(true)
      return
    }
    // One POST /auth/refresh when a row exists. No retry loop.
    void restorePersistedSession().then((restored) => {
      // The cleanup of an unmounted provider must not write into a new one.
      if (cancelled) {
        return
      }
      // Keep a session setSession already stored. Otherwise apply the restore.
      setSessionState((current) => current ?? restored)
      // AuthScreen or ChatScreen may render now.
      setSessionReady(true)
    })
    // Mark the first StrictMode effect so its late resolution cannot setState.
    return () => {
      cancelled = true
    }
  }, [])

  // Replace the session wholesale after login/registration completes key bootstrap.
  const setSession = useCallback((next: Session) => {
    // Claim the page load before the IndexedDB write so restore will not refresh.
    markInMemorySession()
    // Show ChatScreen immediately. The write finishes in the background.
    setSessionState(next)
    // Persist username, refresh token, and the unsealed keypair. Not the password.
    void persistSession(next)
  }, [])

  // Replace only the rotated token pair, preserving the already-unsealed identity keys.
  const updateTokens = useCallback((accessToken: string, refreshToken: string) => {
    // Update React state and write the new refresh token back to IndexedDB.
    setSessionState((current) => {
      // A refresh that arrives after logout has nothing to update.
      if (!current) {
        return current
      }
      // Keep the keypair. Only the token pair changed.
      const next = { ...current, accessToken, refreshToken }
      // The refresh token is single-use, so the stored copy must be the new one.
      void persistSession(next)
      return next
    })
  }, [])

  // Revoke the current refresh token server-side, then clear local session state either way.
  const logout = useCallback(async () => {
    // Read the latest refresh token from the render that created this callback.
    const current = session
    // Best-effort revoke. A network failure still clears this browser.
    if (current) {
      await logoutAccount(current.refreshToken)
    }
    // Drop the IndexedDB row so the next load shows AuthScreen.
    await deleteRestorableSession().catch(() => {
      // React state is cleared below even if IndexedDB is unavailable.
    })
    // Allow a later mount in this page to try restore (the row is gone).
    resetInMemorySessionClaim()
    // Return to AuthScreen.
    setSessionState(null)
  }, [session])

  // Memoize the context value so consumers do not re-render on unrelated parent updates.
  const value = useMemo(
    () => ({ session, sessionReady, setSession, updateTokens, logout }),
    [session, sessionReady, setSession, updateTokens, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Read the current authentication context from any descendant component.
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (context === null) {
    // Fail loudly if a component forgets to render inside <AuthProvider>.
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
