// Import React Testing Library's user-facing render, query, and async helpers.
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
// Import Vitest's grouping, assertion, and mocking helpers.
import { afterEach, describe, expect, it, vi } from 'vitest'

// Import the authentication shell under test.
import { AuthScreen } from './AuthScreen'
// Import the API error type so tests can simulate specific server responses.
import { AuthApiError } from '../api/authClient'
// Import the session provider AuthScreen requires via useAuth().
import { AuthProvider, useAuth } from '../context/AuthContext'
// Import the identity-setup error type so tests can simulate the new-device case.
import { IdentitySetupError } from '../crypto/identitySetup'

// Mock the network-calling module so tests never perform a real HTTP request.
vi.mock('../api/authClient', async () => {
  // Preserve the real AuthApiError class while replacing only the network calls.
  const actual = await vi.importActual<typeof import('../api/authClient')>('../api/authClient')
  return {
    ...actual,
    registerAccount: vi.fn(),
    loginAccount: vi.fn(),
    logoutAccount: vi.fn(),
  }
})

// Mock the client-side key-bootstrap flow so tests never touch real IndexedDB/libsodium.
vi.mock('../crypto/identitySetup', async () => {
  const actual =
    await vi.importActual<typeof import('../crypto/identitySetup')>('../crypto/identitySetup')
  return {
    ...actual,
    ensureIdentityKeys: vi.fn(),
  }
})

// Import the mocked functions with their Vitest mock typing for assertions.
import { loginAccount, registerAccount } from '../api/authClient'
import { ensureIdentityKeys } from '../crypto/identitySetup'
const mockedRegisterAccount = vi.mocked(registerAccount)
const mockedLoginAccount = vi.mocked(loginAccount)
const mockedEnsureIdentityKeys = vi.mocked(ensureIdentityKeys)

// Render AuthScreen inside the real AuthProvider, since it calls useAuth() internally.
function renderAuthScreen() {
  return render(
    <AuthProvider>
      <AuthScreen />
    </AuthProvider>,
  )
}

// Expose whether a session was established, by rendering a probe alongside AuthScreen.
function renderAuthScreenWithSessionProbe() {
  function SessionProbe() {
    const { session } = useAuth()
    return <div data-testid="session-probe">{session ? session.username : 'signed-out'}</div>
  }
  return render(
    <AuthProvider>
      <AuthScreen />
      <SessionProbe />
    </AuthProvider>,
  )
}

// Fill out and submit the registration form with the given field values.
function submitRegistration(fields: {
  username: string
  email: string
  password: string
  confirmPassword: string
}) {
  // Switch from the default login view into registration mode.
  fireEvent.click(screen.getByRole('button', { name: 'Create one' }))
  // Fill every registration field through its accessible label.
  fireEvent.change(screen.getByLabelText('Username'), { target: { value: fields.username } })
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: fields.email } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: fields.password } })
  fireEvent.change(screen.getByLabelText('Confirm password'), {
    target: { value: fields.confirmPassword },
  })
  // Submit through the real primary button rather than calling form.submit directly.
  fireEvent.click(screen.getByRole('button', { name: 'Create account' }))
}

// Fill out and submit the login form with the given field values.
function submitLogin(fields: { username: string; password: string }) {
  fireEvent.change(screen.getByLabelText('Username'), { target: { value: fields.username } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: fields.password } })
  fireEvent.click(screen.getByRole('button', { name: 'Log in' }))
}

// Group accessible authentication behavior, including live login and registration wiring.
describe('AuthScreen', () => {
  // Reset mock call history and implementations between tests.
  afterEach(() => {
    vi.resetAllMocks()
  })

  // Prove the default login view exposes labeled controls.
  it('renders an accessible login form by default', () => {
    renderAuthScreen()
    // Require one visible page heading.
    expect(screen.getByRole('heading', { name: 'Secure Chat' })).toBeInTheDocument()
    // Require a username input discoverable through its label.
    expect(screen.getByLabelText('Username')).toBeRequired()
    // Require a password input discoverable through its label.
    expect(screen.getByLabelText('Password')).toHaveAttribute('type', 'password')
    // Require login to be the initial primary action.
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument()
  })

  // Prove the local toggle reveals registration-specific controls.
  it('switches to registration fields without a network request', () => {
    renderAuthScreen()
    // Activate the semantic mode-switch button.
    fireEvent.click(screen.getByRole('button', { name: 'Create one' }))
    // Require the schema's email field in registration mode.
    expect(screen.getByLabelText('Email')).toHaveAttribute('type', 'email')
    // Require an explicit password confirmation field.
    expect(screen.getByLabelText('Confirm password')).toBeRequired()
    // Require the registration-specific primary action.
    expect(screen.getByRole('button', { name: 'Create account' })).toBeInTheDocument()
    // Require that switching modes alone never calls the backend.
    expect(mockedRegisterAccount).not.toHaveBeenCalled()
    expect(mockedLoginAccount).not.toHaveBeenCalled()
  })

  // Prove a successful registration calls the real API and returns to the login view.
  it('registers an account through the backend and switches to login', async () => {
    mockedRegisterAccount.mockResolvedValueOnce({
      id: '00000000-0000-4000-8000-000000000001',
      username: 'alice',
      email: 'alice@example.com',
      createdAt: '2026-08-12T00:00:00Z',
    })
    renderAuthScreen()
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'correct horse battery staple',
    })

    await waitFor(() =>
      expect(mockedRegisterAccount).toHaveBeenCalledWith({
        username: 'alice',
        email: 'alice@example.com',
        password: 'correct horse battery staple',
      }),
    )
    // Confirm the success message renders the confirmed username.
    expect(await screen.findByText(/Account "alice" created/)).toBeInTheDocument()
    // Confirm the screen returned to the login view (email field no longer present).
    expect(screen.queryByLabelText('Email')).not.toBeInTheDocument()
  })

  // Prove a client-side mismatch never reaches the network.
  it('rejects mismatched passwords before calling the backend', async () => {
    renderAuthScreen()
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'does not match',
    })

    expect(await screen.findByText('Passwords do not match.')).toBeInTheDocument()
    expect(mockedRegisterAccount).not.toHaveBeenCalled()
  })

  // Prove a server-reported conflict surfaces the server's specific message.
  it('shows the server conflict message when the username is already registered', async () => {
    mockedRegisterAccount.mockRejectedValueOnce(
      new AuthApiError('username is already registered', 409),
    )
    renderAuthScreen()
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'correct horse battery staple',
    })

    expect(await screen.findByText('username is already registered')).toBeInTheDocument()
  })

  // Prove a successful login bootstraps identity keys and establishes a session.
  it('logs in, bootstraps identity keys, and establishes a session', async () => {
    mockedLoginAccount.mockResolvedValueOnce({
      accessToken: 'access-token',
      refreshToken: 'refresh-token',
      hasPublicKey: false,
    })
    const fakeKeyPair = {
      publicKey: new Uint8Array(32).fill(1),
      privateKey: new Uint8Array(32).fill(2),
    }
    mockedEnsureIdentityKeys.mockResolvedValueOnce({
      publicKeyBase64: 'AQEB',
      keyPair: fakeKeyPair,
      generatedNewKeyPair: true,
    })

    renderAuthScreenWithSessionProbe()
    submitLogin({ username: 'alice', password: 'correct horse battery staple' })

    await waitFor(() =>
      expect(mockedLoginAccount).toHaveBeenCalledWith({
        username: 'alice',
        password: 'correct horse battery staple',
      }),
    )
    await waitFor(() =>
      expect(mockedEnsureIdentityKeys).toHaveBeenCalledWith(
        'alice',
        'correct horse battery staple',
        'access-token',
        false,
      ),
    )
    // The session probe proves AuthContext now holds a signed-in session.
    expect(await screen.findByTestId('session-probe')).toHaveTextContent('alice')
  })

  // Prove a wrong-password login surfaces the server's generic message without a session.
  it('shows the server error and stays signed out on invalid login credentials', async () => {
    mockedLoginAccount.mockRejectedValueOnce(
      new AuthApiError('invalid username or password', 401),
    )
    renderAuthScreenWithSessionProbe()
    submitLogin({ username: 'alice', password: 'wrong password' })

    expect(await screen.findByText('invalid username or password')).toBeInTheDocument()
    expect(screen.getByTestId('session-probe')).toHaveTextContent('signed-out')
    expect(mockedEnsureIdentityKeys).not.toHaveBeenCalled()
  })

  // Prove the unsupported new-device case surfaces its explicit error without a session.
  it('shows the identity-setup error and stays signed out on an unrecoverable device', async () => {
    mockedLoginAccount.mockResolvedValueOnce({
      accessToken: 'access-token',
      refreshToken: 'refresh-token',
      hasPublicKey: true,
    })
    mockedEnsureIdentityKeys.mockRejectedValueOnce(
      new IdentitySetupError('No local key vault was found for "alice" on this device.'),
    )

    renderAuthScreenWithSessionProbe()
    submitLogin({ username: 'alice', password: 'correct horse battery staple' })

    expect(
      await screen.findByText(/No local key vault was found for "alice"/),
    ).toBeInTheDocument()
    expect(screen.getByTestId('session-probe')).toHaveTextContent('signed-out')
  })
})
