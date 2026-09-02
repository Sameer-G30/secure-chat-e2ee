// Import state so the screen can track the current mode, request, and results.
import { useState } from 'react'
// Import the form-event type separately; verbatimModuleSyntax requires type-only imports.
import type { FormEvent } from 'react'
// Import the typed backend client this screen calls for registration and login.
import { AuthApiError, loginAccount, registerAccount } from '../api/authClient'
// Import the key-bootstrap flow that runs once per successful login.
import { ensureIdentityKeys, IdentitySetupError } from '../crypto/identitySetup'
// Import the session context this screen populates after a successful login.
import { useAuth } from '../context/AuthContext'
// Import the client-only password meter shown during registration.
import { scorePassword } from '../security/passwordStrength'

// Identify the two authentication views without ambiguous booleans.
type AuthMode = 'login' | 'register'

// Identify a request lifecycle so the UI can disable controls and announce state.
type RequestStatus = 'idle' | 'submitting' | 'success' | 'error'

// Render the split-panel authentication screen from the frontend/vrati layout.
export function AuthScreen() {
  // Make the session setter available for a successful login's key-bootstrap step.
  const { setSession } = useAuth()

  // Start with the returning-user login view.
  const [mode, setMode] = useState<AuthMode>('login')
  // Derive readable display state from the current authentication mode.
  const isRegistering = mode === 'register'

  // Track the in-flight request's lifecycle, shared by both login and registration.
  const [status, setStatus] = useState<RequestStatus>('idle')
  // Hold the specific message to show for the current error or success state.
  const [message, setMessage] = useState<string | null>(null)
  // Hold the password field as React state so the registration meter can score it live.
  const [passwordValue, setPasswordValue] = useState('')
  // Hold whether the password field is shown in the clear (login and register).
  const [showPassword, setShowPassword] = useState(false)

  // Score the live password; unused on the login view.
  const passwordStrength = scorePassword(passwordValue)

  // Switch modes and clear any stale status from a previous attempt.
  function switchMode(nextMode: AuthMode) {
    // Drop the previous request lifecycle so a stale error does not linger.
    setStatus('idle')
    // Clear the visible feedback line.
    setMessage(null)
    // Empty the password so the meter does not score a leftover value.
    setPasswordValue('')
    // Hide characters again after a mode switch.
    setShowPassword(false)
    // Enter the requested authentication view.
    setMode(nextMode)
  }

  // Handle a registration submission against the real backend.
  async function handleRegister(form: HTMLFormElement, formData: FormData) {
    // Read the chosen handle from the form.
    const username = String(formData.get('username') ?? '').trim()
    // Read the email collected only during registration.
    const email = String(formData.get('email') ?? '').trim()
    // Read the password the meter already scored.
    const password = String(formData.get('password') ?? '')
    // Read the confirmation field for a client-side equality check.
    const confirmPassword = String(formData.get('confirmPassword') ?? '')

    // Validate password confirmation client-side before spending a network round trip.
    if (password !== confirmPassword) {
      // Mark the request as failed so the feedback line uses the error tint.
      setStatus('error')
      // Tell the user the two fields must match.
      setMessage('Passwords do not match.')
      // Stop before calling the API.
      return
    }

    // Disable the submit button while the request is in flight.
    setStatus('submitting')
    // Clear any previous feedback.
    setMessage(null)
    try {
      // Create the account on the FastAPI backend.
      const account = await registerAccount({ username, email, password })
      // Mark the request as succeeded so the feedback line uses the success tint.
      setStatus('success')
      // Confirm the handle that was created and send the user to login.
      setMessage(`Account "${account.username}" created. Log in below to finish setup.`)
      // Empty the registration fields.
      form.reset()
      // Hand the user to the login view; key generation happens on first login.
      setMode('login')
      // Hide the password again on the login view.
      setShowPassword(false)
      // Clear the controlled password so login starts empty.
      setPasswordValue('')
    } catch (error) {
      // Prefer the API's own message when the backend rejected the request.
      const errorMessage =
        error instanceof AuthApiError ? error.message : 'Registration failed. Please try again shortly.'
      // Mark the request as failed.
      setStatus('error')
      // Show the failure reason.
      setMessage(errorMessage)
    }
  }

  // Handle a login submission against the real backend, then bootstrap this device's identity keys.
  async function handleLogin(formData: FormData) {
    // Read the account identifier.
    const username = String(formData.get('username') ?? '').trim()
    // Read the password used both to authenticate and to unseal local keys.
    const password = String(formData.get('password') ?? '')

    // Disable the submit button while login and key setup run.
    setStatus('submitting')
    // Clear any previous feedback.
    setMessage(null)
    try {
      // Exchange the password for JWTs.
      const tokens = await loginAccount({ username, password })
      // Generate-and-upload (first login) or unseal (returning login) this device's identity key.
      const identity = await ensureIdentityKeys(
        username,
        password,
        tokens.accessToken,
        tokens.hasPublicKey,
      )
      // Only now, with verified credentials and usable local key material, start the session.
      setSession({
        username,
        accessToken: tokens.accessToken,
        refreshToken: tokens.refreshToken,
        identityKeyPair: identity.keyPair,
      })
      // No further UI state matters: App renders ChatScreen once a session exists.
    } catch (error) {
      // Prefer identity-setup errors, then API errors, then a generic fallback.
      const errorMessage =
        error instanceof IdentitySetupError
          ? error.message
          : error instanceof AuthApiError
            ? error.message
            : 'Login failed. Please try again shortly.'
      // Mark the request as failed.
      setStatus('error')
      // Show the failure reason.
      setMessage(errorMessage)
    }
  }

  // Route form submissions to the mode-specific handler.
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    // Prevent the browser's native full-page form submission.
    event.preventDefault()
    // Capture the form so registration can reset it after success.
    const form = event.currentTarget
    // Serialize named fields into a FormData object.
    const formData = new FormData(form)
    // Dispatch to registration or login based on the current view.
    if (isRegistering) {
      // Create the account, then flip to login.
      await handleRegister(form, formData)
    } else {
      // Authenticate and bootstrap keys.
      await handleLogin(formData)
    }
  }

  // Render the vrati split panel: brand on the left, form on the right.
  return (
    <main className="auth-screen">
      <section className="auth-container" aria-labelledby="auth-title">
        <div className="auth-brand">
          <div className="brand-icon">
            <svg width="52" height="52" viewBox="0 0 52 52" fill="none" aria-hidden="true">
              <rect width="52" height="52" rx="14" fill="currentColor" />
              <path d="M26 15L26 37M15 26L37 26" stroke="#fdf6f0" strokeWidth="3.5" strokeLinecap="round" />
              <circle cx="26" cy="26" r="9" stroke="#fdf6f0" strokeWidth="3.5" />
            </svg>
          </div>
          <h1 id="auth-title">
            Secure <span>Chat</span>
          </h1>
          <p>
            End-to-end encrypted messaging.
            <br />
            Privacy by design, with on-device scam detection.
          </p>
          <div className="trust-badge">
            <span>End-to-end encrypted · On-device detection</span>
          </div>
        </div>

        <div className="auth-form">
          <div className="form-box">
            {isRegistering ? (
              <button type="button" className="back-btn" onClick={() => switchMode('login')}>
                ← Back
              </button>
            ) : null}
            <h2>{isRegistering ? 'Create Account' : 'Welcome back'}</h2>
            <p className="form-subtitle">
              {isRegistering ? 'Start your secure messaging journey' : 'Sign in to continue'}
            </p>

            <form onSubmit={(event) => void handleSubmit(event)}>
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  placeholder="Enter your username"
                  required
                />
              </div>

              {isRegistering ? (
                <div className="form-group">
                  <label htmlFor="email">Email</label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    placeholder="Enter your email"
                    required
                  />
                </div>
              ) : null}

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <div className="password-toggle-wrap">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete={isRegistering ? 'new-password' : 'current-password'}
                    minLength={isRegistering ? 8 : undefined}
                    placeholder={isRegistering ? 'Create a strong password' : 'Enter your password'}
                    required
                    value={passwordValue}
                    onChange={(event) => setPasswordValue(event.target.value)}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword((current) => !current)}
                  >
                    {showPassword ? 'Hide' : 'Show'}
                  </button>
                </div>
                {isRegistering && passwordValue.length > 0 ? (
                  <div
                    className="password-strength"
                    role="status"
                    aria-label={`Password strength: ${passwordStrength.label}`}
                  >
                    <div className="password-strength-track">
                      <div
                        className={`password-strength-fill password-strength-fill-${passwordStrength.score}`}
                        style={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                      />
                    </div>
                    <p className="password-strength-label">Strength: {passwordStrength.label}</p>
                  </div>
                ) : null}
              </div>

              {isRegistering ? (
                <div className="form-group">
                  <label htmlFor="confirm-password">Confirm password</label>
                  <input
                    id="confirm-password"
                    name="confirmPassword"
                    type="password"
                    autoComplete="new-password"
                    placeholder="Confirm your password"
                    required
                  />
                </div>
              ) : null}

              <button className="btn-primary" type="submit" disabled={status === 'submitting'}>
                {isRegistering
                  ? status === 'submitting'
                    ? 'Creating account…'
                    : 'Create account'
                  : status === 'submitting'
                    ? 'Logging in…'
                    : 'Log in'}
              </button>
            </form>

            {message ? (
              <p className={`auth-feedback auth-feedback-${status}`} role="status">
                {message}
              </p>
            ) : null}

            <p className="form-footer">
              {isRegistering ? 'Already have an account? ' : "Don't have an account? "}
              <button
                type="button"
                className="text-button"
                onClick={() => switchMode(isRegistering ? 'login' : 'register')}
              >
                {isRegistering ? 'Log in' : 'Create one'}
              </button>
            </p>
          </div>
        </div>
      </section>
    </main>
  )
}
