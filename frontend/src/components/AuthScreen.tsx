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

// Render the accessible authentication screen, now wired to real login and registration.
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

  // Score the live password; unused on the login view.
  const passwordStrength = scorePassword(passwordValue)

  // Switch modes and clear any stale status from a previous attempt.
  function switchMode(nextMode: AuthMode) {
    setStatus('idle')
    setMessage(null)
    setPasswordValue('')
    setMode(nextMode)
  }

  // Handle a registration submission against the real backend.
  async function handleRegister(form: HTMLFormElement, formData: FormData) {
    const username = String(formData.get('username') ?? '').trim()
    const email = String(formData.get('email') ?? '').trim()
    const password = String(formData.get('password') ?? '')
    const confirmPassword = String(formData.get('confirmPassword') ?? '')

    // Validate password confirmation client-side before spending a network round trip.
    if (password !== confirmPassword) {
      setStatus('error')
      setMessage('Passwords do not match.')
      return
    }

    setStatus('submitting')
    setMessage(null)
    try {
      const account = await registerAccount({ username, email, password })
      setStatus('success')
      setMessage(`Account "${account.username}" created. Log in below to finish setup.`)
      form.reset()
      // Hand the user to the login view; key generation happens on first login (§6.1),
      // once an access token exists to authenticate the POST /keys/me upload.
      setMode('login')
    } catch (error) {
      const errorMessage =
        error instanceof AuthApiError ? error.message : 'Registration failed. Please try again shortly.'
      setStatus('error')
      setMessage(errorMessage)
    }
  }

  // Handle a login submission against the real backend, then bootstrap this device's identity keys.
  async function handleLogin(formData: FormData) {
    const username = String(formData.get('username') ?? '').trim()
    const password = String(formData.get('password') ?? '')

    setStatus('submitting')
    setMessage(null)
    try {
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
      const errorMessage =
        error instanceof IdentitySetupError
          ? error.message
          : error instanceof AuthApiError
            ? error.message
            : 'Login failed. Please try again shortly.'
      setStatus('error')
      setMessage(errorMessage)
    }
  }

  // Route form submissions to the mode-specific handler.
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    // Prevent the browser's native full-page form submission.
    event.preventDefault()
    const form = event.currentTarget
    const formData = new FormData(form)
    if (isRegistering) {
      await handleRegister(form, formData)
    } else {
      await handleLogin(formData)
    }
  }

  // Render the legacy visual language as semantic React elements.
  return (
    // Fill the viewport with the animated purple authentication background.
    <main className="auth-screen">
      {/* Mark the dotted layer decorative so assistive technology ignores it. */}
      <div className="floating-dots" aria-hidden="true" />
      {/* Group authentication controls inside the glass-style card. */}
      <section className="auth-card" aria-labelledby="auth-title">
        {/* Present the security-oriented product identity above the form. */}
        <header className="auth-header">
          {/* Preserve the legacy lock motif without adding redundant speech. */}
          <span className="lock-icon" role="img" aria-label="Secure">
            🔒
          </span>
          {/* Give the page one stable accessible heading. */}
          <h1 id="auth-title">Secure Chat</h1>
          {/* State the product's intended trust boundary concisely. */}
          <p>End-to-end encrypted messaging with on-device scam detection</p>
        </header>

        {/* Both login and registration now submit to the real backend. */}
        <form className="auth-form" onSubmit={(event) => void handleSubmit(event)}>
          {/* Collect the account identifier in both authentication modes. */}
          <div className="input-group">
            {/* Associate the visible label with its username field. */}
            <label htmlFor="username">Username</label>
            {/* Let password managers recognize the account identifier. */}
            <input id="username" name="username" type="text" autoComplete="username" required />
          </div>

          {/* Collect an email address only while creating an account. */}
          {isRegistering && (
            // Preserve one reusable floating-label field structure.
            <div className="input-group">
              {/* Associate the visible label with its email field. */}
              <label htmlFor="email">Email</label>
              {/* Request a validated email and advertise autocomplete semantics. */}
              <input id="email" name="email" type="email" autoComplete="email" required />
            </div>
          )}

          {/* Collect the password in both authentication modes. */}
          <div className="input-group">
            {/* Associate the visible label with its password field. */}
            <label htmlFor="password">Password</label>
            {/* Select new-password autocomplete only during registration. */}
            <input
              // Give the password control a stable accessible identifier.
              id="password"
              // Give form serialization a predictable field name.
              name="password"
              // Hide entered characters from shoulder surfing.
              type="password"
              // Help password managers distinguish login from registration.
              autoComplete={isRegistering ? 'new-password' : 'current-password'}
              // Match the server's minimum length so errors surface before submission.
              minLength={isRegistering ? 8 : undefined}
              // Require a value before future backend submission.
              required
              // Keep the field controlled so the meter can score every keystroke.
              value={passwordValue}
              // Update both the form value and the live meter score.
              onChange={(event) => setPasswordValue(event.target.value)}
            />
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

          {/* Confirm the chosen password only during account creation. */}
          {isRegistering && (
            // Preserve one reusable floating-label field structure.
            <div className="input-group">
              {/* Associate the visible label with its confirmation field. */}
              <label htmlFor="confirm-password">Confirm password</label>
              {/* Collect a second hidden password for equality validation on submit. */}
              <input
                // Give the confirmation control a stable identifier.
                id="confirm-password"
                // Give form serialization a predictable field name.
                name="confirmPassword"
                // Hide entered characters.
                type="password"
                // Tell password managers this is part of new credential creation.
                autoComplete="new-password"
                // Require confirmation before future backend submission.
                required
              />
            </div>
          )}

          {/* Show the submit action with mode-specific wording and busy state. */}
          <button className="primary-button" type="submit" disabled={status === 'submitting'}>
            {/* Explain the action performed by the current form mode and request state. */}
            {isRegistering
              ? status === 'submitting'
                ? 'Creating account…'
                : 'Create account'
              : status === 'submitting'
                ? 'Logging in…'
                : 'Log in'}
          </button>
        </form>

        {/* Announce login/registration outcomes to sighted and assistive-technology users alike. */}
        {message && (
          <p className={`auth-feedback auth-feedback-${status}`} role="status">
            {message}
          </p>
        )}

        {/* Offer an explicit button to switch authentication modes. */}
        <p className="auth-switch">
          {/* Explain which account state the alternate mode serves. */}
          {isRegistering ? 'Already have an account?' : "Don't have an account?"}{' '}
          {/* Use a button rather than a fake anchor for an in-page state change. */}
          <button
            // Avoid submitting the surrounding authentication form.
            type="button"
            // Apply link-like visual styling while retaining button semantics.
            className="text-button"
            // Switch to the opposite authentication view and clear stale status.
            onClick={() => switchMode(isRegistering ? 'login' : 'register')}
          >
            {/* Name the destination mode rather than the implementation action. */}
            {isRegistering ? 'Log in' : 'Create one'}
          </button>
        </p>
      </section>
    </main>
  )
}
