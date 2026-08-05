// Import React Testing Library's user-facing render and query helpers.
import { fireEvent, render, screen } from '@testing-library/react'
// Import Vitest's grouping and assertion helpers.
import { describe, expect, it } from 'vitest'

// Import the authentication shell under test.
import { AuthScreen } from './AuthScreen'

// Group accessible first-slice authentication behavior.
describe('AuthScreen', () => {
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
  })
})
