// Import React Testing Library's user-facing render, query, and async helpers.
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
// Import Vitest's grouping, assertion, and mocking helpers.
import { afterEach, describe, expect, it, vi } from 'vitest'

// Import the authentication shell under test.
import { AuthScreen } from './AuthScreen'
// Import the API error type so tests can simulate specific server responses.
import { AuthApiError } from '../api/authClient'

// Mock the network-calling module so tests never perform a real HTTP request.
vi.mock('../api/authClient', async () => {
  // Preserve the real AuthApiError class while replacing only the network call.
  const actual = await vi.importActual<typeof import('../api/authClient')>('../api/authClient')
  return {
    ...actual,
    registerAccount: vi.fn(),
  }
})

// Import the mocked function with its Vitest mock typing for assertions.
import { registerAccount } from '../api/authClient'
const mockedRegisterAccount = vi.mocked(registerAccount)

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

// Group accessible authentication behavior, now including live registration wiring.
describe('AuthScreen', () => {
  // Reset mock call history and implementations between tests.
  afterEach(() => {
    vi.resetAllMocks()
  })

  // Prove the default login view exposes labeled controls.
  it('renders an accessible login form by default', () => {
    // Render the component into Vitest's browser-like DOM.
    render(<AuthScreen />)
    // Require one visible page heading.
    expect(screen.getByRole('heading', { name: 'Secure Chat' })).toBeInTheDocument()
    // Require a username input discoverable through its label.
    expect(screen.getByLabelText('Username')).toBeRequired()
    // Require a password input discoverable through its label.
    expect(screen.getByLabelText('Password')).toHaveAttribute('type', 'password')
    // Require login to be the initial primary action.
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument()
    // Require the slice note to clarify that login is not yet implemented.
    expect(screen.getByText(/Login arrives with JWT auth/)).toBeInTheDocument()
  })

  // Prove the local toggle reveals registration-specific controls.
  it('switches to registration fields without a network request', () => {
    // Render the component into the isolated test DOM.
    render(<AuthScreen />)
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
  })

  // Prove a successful submission calls the real API and reports success.
  it('registers an account through the backend and reports success', async () => {
    // Resolve the mocked call with a representative created-account payload.
    mockedRegisterAccount.mockResolvedValueOnce({
      id: '00000000-0000-4000-8000-000000000001',
      username: 'alice',
      email: 'alice@example.com',
      createdAt: '2026-08-12T00:00:00Z',
    })
    render(<AuthScreen />)
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'correct horse battery staple',
    })

    // Confirm the API was called with exactly the submitted fields.
    await waitFor(() =>
      expect(mockedRegisterAccount).toHaveBeenCalledWith({
        username: 'alice',
        email: 'alice@example.com',
        password: 'correct horse battery staple',
      }),
    )
    // Confirm the success message renders the confirmed username.
    expect(await screen.findByText(/Account "alice" created/)).toBeInTheDocument()
  })

  // Prove a client-side mismatch never reaches the network.
  it('rejects mismatched passwords before calling the backend', async () => {
    render(<AuthScreen />)
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'does not match',
    })

    // Confirm the mismatch message renders.
    expect(await screen.findByText('Passwords do not match.')).toBeInTheDocument()
    // Confirm no network call was attempted with mismatched passwords.
    expect(mockedRegisterAccount).not.toHaveBeenCalled()
  })

  // Prove a server-reported conflict surfaces the server's specific message.
  it('shows the server conflict message when the username is already registered', async () => {
    // Simulate the backend's 409 response for a duplicate username.
    mockedRegisterAccount.mockRejectedValueOnce(
      new AuthApiError('username is already registered', 409),
    )
    render(<AuthScreen />)
    submitRegistration({
      username: 'alice',
      email: 'alice@example.com',
      password: 'correct horse battery staple',
      confirmPassword: 'correct horse battery staple',
    })

    // Confirm the exact server-provided detail message renders to the user.
    expect(await screen.findByText('username is already registered')).toBeInTheDocument()
  })
})
