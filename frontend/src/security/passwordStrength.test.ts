// Import Vitest grouping and assertion helpers.
import { describe, expect, it } from 'vitest'

// Import the client-only meter under test.
import { scorePassword } from './passwordStrength'

// Group the five-band heuristic the registration form displays.
describe('scorePassword', () => {
  // Prove an empty field is the weakest band.
  it('scores an empty string as very weak', () => {
    expect(scorePassword('')).toEqual({ score: 0, label: 'very weak' })
  })

  // Prove a short lowercase word is still weak even if it has letters.
  it('scores a short lowercase word as very weak', () => {
    expect(scorePassword('abc').label).toBe('very weak')
  })

  // Prove the documented 8-character floor plus mixed classes reaches "strong".
  it('scores a mixed 8+ character password as strong', () => {
    const result = scorePassword('Correct1!')
    expect(result.score).toBe(5)
    expect(result.label).toBe('strong')
  })

  // Prove a long passphrase with spaces counts the special-character check.
  it('treats spaces as a special character so passphrases score well', () => {
    const result = scorePassword('correct horse battery staple')
    expect(result.score).toBeGreaterThanOrEqual(3)
  })
})
