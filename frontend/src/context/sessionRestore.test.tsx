// Prove a reload restores ChatScreen from IndexedDB, and Logout removes that row.

// Import React Testing Library helpers.
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
// Import Vitest helpers.
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Import the app shell that chooses AuthScreen or ChatScreen.
import App from '../App'
// Import the API error used when a stored refresh token is rejected.
import { AuthApiError } from '../api/authClient'
// Import the identity error a missing vault raises on restore.
import { IdentitySetupError } from '../crypto/identitySetup'
// Import the session row helpers under test.
import { readRestorableSession, writeRestorableSession } from '../crypto/sessionStore'
// Import the claim reset so each case starts signed out.
import { resetInMemorySessionClaim } from './sessionClaim'

// Replace ChatScreen with a small signed-in probe. The real screen is covered elsewhere.
vi.mock('../components/ChatScreen', async () => {
  // Load useAuth from the same module the provider uses.
  const { useAuth } = await import('./AuthContext')
  return {
    // Render the signed-in shell and a logout control.
    ChatScreen() {
      const { session, logout } = useAuth()
      return (
        <div>
          <h1>Secure Chat</h1>
          <p>Signed in as {session?.username}</p>
          <button type="button" onClick={() => void logout()}>
            Logout
          </button>
        </div>
      )
    },
  }
})

// Replace network auth calls. The real AuthApiError class stays.
vi.mock('../api/authClient', async () => {
  const actual = await vi.importActual<typeof import('../api/authClient')>('../api/authClient')
  return {
    ...actual,
    refreshTokens: vi.fn(),
    logoutAccount: vi.fn(async () => {}),
  }
})

// Replace the vault check so these cases do not need a sealed identity or libsodium.
vi.mock('../crypto/identitySetup', async () => {
  const actual = await vi.importActual<typeof import('../crypto/identitySetup')>('../crypto/identitySetup')
  return {
    ...actual,
    assertSessionIdentity: vi.fn(async () => {}),
  }
})

import { logoutAccount, refreshTokens } from '../api/authClient'
import { assertSessionIdentity } from '../crypto/identitySetup'

const mockedRefresh = vi.mocked(refreshTokens)
const mockedLogout = vi.mocked(logoutAccount)
const mockedAssert = vi.mocked(assertSessionIdentity)

// Delete the session database before seeding a case.
function deleteSessionDatabase(): Promise<void> {
  return new Promise((resolve) => {
    const request = indexedDB.deleteDatabase('secure-chat-session')
    request.onsuccess = () => {
      resolve()
    }
    request.onerror = () => {
      resolve()
    }
    request.onblocked = () => {
      resolve()
    }
  })
}

// Seed one restorable session. The password and access token are not stored.
async function seedSession(): Promise<void> {
  await writeRestorableSession({
    username: 'alice',
    refreshToken: 'refresh-1',
    publicKey: new Uint8Array(32).fill(4),
    privateKey: new Uint8Array(32).fill(5),
  })
}

describe('session restore', () => {
  beforeEach(async () => {
    // Start each case with no claim and no leftover row.
    resetInMemorySessionClaim()
    await deleteSessionDatabase()
    window.localStorage.clear()
    mockedRefresh.mockReset()
    mockedLogout.mockReset()
    mockedAssert.mockReset()
    mockedRefresh.mockResolvedValue({
      accessToken: 'access-2',
      refreshToken: 'refresh-2',
      hasPublicKey: true,
    })
    mockedLogout.mockResolvedValue(undefined)
    mockedAssert.mockResolvedValue(undefined)
  })

  it('restores a session and skips AuthScreen', async () => {
    await seedSession()
    render(<App />)
    expect(await screen.findByText('Signed in as alice')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Log in' })).not.toBeInTheDocument()
    expect(screen.queryByText('Welcome back')).not.toBeInTheDocument()
    expect(mockedRefresh).toHaveBeenCalledTimes(1)
    expect(mockedRefresh).toHaveBeenCalledWith('refresh-1')
    const stored = await readRestorableSession()
    expect(stored?.refreshToken).toBe('refresh-2')
    expect(stored?.username).toBe('alice')
    expect(stored).not.toHaveProperty('password')
    expect(stored).not.toHaveProperty('accessToken')
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index)
      const value = key ? window.localStorage.getItem(key) : ''
      expect(value ?? '').not.toContain('refresh-1')
      expect(value ?? '').not.toContain('refresh-2')
    }
  })

  it('clears the saved session on logout so the next load shows AuthScreen', async () => {
    await seedSession()
    const view = render(<App />)
    expect(await screen.findByText('Signed in as alice')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Logout' }))
    expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(screen.queryByText('Signed in as alice')).not.toBeInTheDocument()
    await waitFor(async () => {
      expect(await readRestorableSession()).toBeNull()
    })
    expect(mockedLogout).toHaveBeenCalledWith('refresh-2')
    view.unmount()
    render(<App />)
    expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(screen.queryByText('Signed in as alice')).not.toBeInTheDocument()
    expect(mockedRefresh).toHaveBeenCalledTimes(1)
  })

  it('drops a rejected refresh token and does not retry it', async () => {
    await seedSession()
    mockedRefresh.mockRejectedValue(new AuthApiError('expired', 401))
    render(<App />)
    expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(screen.queryByText('Signed in as alice')).not.toBeInTheDocument()
    expect(await readRestorableSession()).toBeNull()
    await new Promise((resolve) => {
      window.setTimeout(resolve, 30)
    })
    expect(mockedRefresh).toHaveBeenCalledTimes(1)
  })

  it('drops the saved session when identity setup throws', async () => {
    await seedSession()
    mockedAssert.mockRejectedValue(new IdentitySetupError('No local key vault'))
    render(<App />)
    expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(screen.queryByText('Signed in as alice')).not.toBeInTheDocument()
    expect(await readRestorableSession()).toBeNull()
    expect(mockedRefresh).toHaveBeenCalledTimes(1)
  })
})
