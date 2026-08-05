// Import state so the static shell can switch between login and registration.
import { useState } from 'react'

// Identify the two authentication views without ambiguous booleans.
type AuthMode = 'login' | 'register'

// Render the accessible first-slice authentication shell.
export function AuthScreen() {
  // Start with the returning-user login view.
  const [mode, setMode] = useState<AuthMode>('login')
  // Derive readable display state from the current authentication mode.
  const isRegistering = mode === 'register'

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

        {/* Prevent submission because backend auth begins in Slice 2. */}
        <form className="auth-form" onSubmit={(event) => event.preventDefault()}>
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
              {/* Collect a second hidden password for future equality validation. */}
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

          {/* Show the future submit action with mode-specific wording. */}
          <button className="primary-button" type="submit">
            {/* Explain the action performed by the current form mode. */}
            {isRegistering ? 'Create account' : 'Log in'}
          </button>
        </form>

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
            // Switch to the opposite static authentication view.
            onClick={() => setMode(isRegistering ? 'login' : 'register')}
          >
            {/* Name the destination mode rather than the implementation action. */}
            {isRegistering ? 'Log in' : 'Create one'}
          </button>
        </p>

        {/* Clarify that this first slice intentionally has no auth network call. */}
        <p className="slice-note" role="status">
          Slice 1 interface preview — account APIs arrive in the next checkpoint.
        </p>
      </section>
    </main>
  )
}
