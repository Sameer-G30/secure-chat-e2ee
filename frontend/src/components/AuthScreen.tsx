// Import state so the screen can track the current mode, request, and results.
import { useState } from 'react'
// Import the form-event type separately; verbatimModuleSyntax requires type-only imports.
import type { FormEvent } from 'react'
// Import the typed backend client this screen now calls for registration.
import { AuthApiError, registerAccount } from '../api/authClient'

// Identify the two authentication views without ambiguous booleans.
type AuthMode = 'login' | 'register'

// Identify the registration request lifecycle so the UI can disable and announce state.
type RegisterStatus = 'idle' | 'submitting' | 'success' | 'error'

// Render the accessible authentication screen, now wired to real registration.
export function AuthScreen() {
  // Start with the returning-user login view.
  const [mode, setMode] = useState<AuthMode>('login')
  // Derive readable display state from the current authentication mode.
  const isRegistering = mode === 'register'

  // Track the in-flight registration request's lifecycle.
  const [registerStatus, setRegisterStatus] = useState<RegisterStatus>('idle')
  // Hold the specific message to show for the current error or success state.
  const [registerMessage, setRegisterMessage] = useState<string | null>(null)

  // Switch modes and clear any stale status from a previous registration attempt.
  function switchMode(nextMode: AuthMode) {
    // Reset the registration lifecycle so switching back doesn't show a stale result.
    setRegisterStatus('idle')
    // Clear any previously displayed message.
    setRegisterMessage(null)
    // Apply the requested view.
    setMode(nextMode)
  }

  // Handle registration submissions against the real backend; login arrives in Slice 3.
  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    // Prevent the browser's native full-page form submission.
    event.preventDefault()

    // Login is not implemented yet; keep the previous slice's explicit no-op behavior.
    if (!isRegistering) {
      return
    }

    // Read field values directly from the submitted form.
    const form = event.currentTarget
    const formData = new FormData(form)
    const username = String(formData.get('username') ?? '').trim()
    const email = String(formData.get('email') ?? '').trim()
    const password = String(formData.get('password') ?? '')
    const confirmPassword = String(formData.get('confirmPassword') ?? '')

    // Validate password confirmation client-side before spending a network round trip.
    if (password !== confirmPassword) {
      setRegisterStatus('error')
      setRegisterMessage('Passwords do not match.')
      return
    }

    // Enter the submitting state so the button disables and announces progress.
    setRegisterStatus('submitting')
    setRegisterMessage(null)

    try {
      // Call the real backend registration endpoint added in Slice 2.
      const account = await registerAccount({ username, email, password })
      // Report success with the confirmed username; no token/session yet (Slice 3).
      setRegisterStatus('success')
      setRegisterMessage(`Account "${account.username}" created. Login arrives in the next checkpoint.`)
      // Clear the form so a resubmission cannot silently double-submit the same values.
      form.reset()
    } catch (error) {
      // Distinguish the API's typed errors from unexpected failures for an honest message.
      const message =
        error instanceof AuthApiError ? error.message : 'Registration failed. Please try again shortly.'
      setRegisterStatus('error')
      setRegisterMessage(message)
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

        {/* Registration now submits to the real backend; login stays a preview. */}
        <form className="auth-form" onSubmit={handleSubmit}>
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
            />
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
          <button
            className="primary-button"
            type="submit"
            disabled={isRegistering && registerStatus === 'submitting'}
          >
            {/* Explain the action performed by the current form mode and request state. */}
            {isRegistering
              ? registerStatus === 'submitting'
                ? 'Creating account…'
                : 'Create account'
              : 'Log in'}
          </button>
        </form>

        {/* Announce registration outcomes to sighted and assistive-technology users alike. */}
        {isRegistering && registerMessage && (
          <p className={`auth-feedback auth-feedback-${registerStatus}`} role="status">
            {registerMessage}
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

        {/* Clarify that login specifically is still a preview pending Slice 3. */}
        {!isRegistering && (
          <p className="slice-note" role="status">
            Login arrives with JWT auth in the next checkpoint — registration is live.
          </p>
        )}
      </section>
    </main>
  )
}
