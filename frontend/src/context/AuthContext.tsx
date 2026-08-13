// Import React's context, provider-state, and consumer-hook primitives.
import { createContext, useCallback, useContext, useMemo, useState } from 'react'
// Import the ReactNode type separately; verbatimModuleSyntax requires type-only imports.
import type { ReactNode } from 'react'

// Import the real login/logout network calls this context wraps.
import { logoutAccount } from '../api/authClient'
// Import the local identity keypair type produced during login's key-bootstrap step.
import type { IdentityKeyPair } from '../crypto/keyExchange'

// Describe the fields a completed, authenticated session carries in memory.
export interface Session {
  // Identify the signed-in account.
  username: string
  // Carry the short-lived JWT sent with every authenticated API request.
  accessToken: string
  // Carry the longer-lived, single-use JWT used only to request a new pair.
  refreshToken: string
  // Carry the unsealed X25519 identity keypair for this browser session only.
  //
  // This lives in JS memory for the tab's lifetime and is never written to
  // localStorage/sessionStorage/cookies; only its *sealed* form persists,
  // in IndexedDB (see crypto/keyVault.ts).
  identityKeyPair: IdentityKeyPair
}

// Describe the operations components can call against the current session.
interface AuthContextValue {
  // Expose the current session, or null while signed out.
  session: Session | null
  // Replace the current session after a successful login/registration flow.
  setSession: (session: Session) => void
  // Replace only the token pair after a refresh, keeping identity keys in memory.
  updateTokens: (accessToken: string, refreshToken: string) => void
  // Revoke the refresh token server-side (best-effort) and clear local session state.
  logout: () => Promise<void>
}

// Create the context with no default value; useAuth enforces it is always provided.
const AuthContext = createContext<AuthContextValue | null>(null)

// Provide session state to the whole application tree.
export function AuthProvider({ children }: { children: ReactNode }) {
  // Hold the session purely in React state.
  //
  // Deliberate scope decision: tokens and the unsealed private key are kept
  // in memory only, not persisted across a page reload. This avoids the
  // XSS-exfiltration risk of localStorage/sessionStorage entirely; the
  // trade-off (a reload always requires logging in again) is documented in
  // the README as a known Slice 3 limitation, with httpOnly-cookie-based
  // persistence noted as future hardening once the backend supports it.
  const [session, setSessionState] = useState<Session | null>(null)

  // Replace the session wholesale after login/registration completes key bootstrap.
  const setSession = useCallback((next: Session) => {
    setSessionState(next)
  }, [])

  // Replace only the rotated token pair, preserving the already-unsealed identity keys.
  const updateTokens = useCallback((accessToken: string, refreshToken: string) => {
    setSessionState((current) => (current ? { ...current, accessToken, refreshToken } : current))
  }, [])

  // Revoke the current refresh token server-side, then clear local session state either way.
  const logout = useCallback(async () => {
    const current = session
    if (current) {
      await logoutAccount(current.refreshToken)
    }
    setSessionState(null)
  }, [session])

  // Memoize the context value so consumers do not re-render on unrelated parent updates.
  const value = useMemo(
    () => ({ session, setSession, updateTokens, logout }),
    [session, setSession, updateTokens, logout],
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
