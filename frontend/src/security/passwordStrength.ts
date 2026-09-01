// Client-only password-strength heuristic ported from the legacy React register
// form (`useAuthForms.js`). The server still only enforces the 8-character floor
// (see RegisterRequest); this meter is UX guidance, never a second policy.

// Name the five display bands the meter can land on.
export type PasswordStrengthLabel = 'very weak' | 'weak' | 'fair' | 'good' | 'strong'

// Describe one evaluation of a candidate password.
export interface PasswordStrength {
  // Count of independent checks that passed (0 through 5).
  score: number
  // Human-readable band shown next to the meter.
  label: PasswordStrengthLabel
}

// Score a candidate password without sending it anywhere.
export function scorePassword(password: string): PasswordStrength {
  // Start from zero so an empty field is honestly "very weak".
  let score = 0
  // Award one point for meeting the server's documented length floor.
  if (password.length >= 8) {
    score += 1
  }
  // Award one point for a lowercase letter (the most common character class).
  if (/[a-z]/.test(password)) {
    score += 1
  }
  // Award one point for an uppercase letter so "password123" is not "strong".
  if (/[A-Z]/.test(password)) {
    score += 1
  }
  // Award one point for a digit.
  if (/\d/.test(password)) {
    score += 1
  }
  // Award one point for a non-alphanumeric character (spaces count; passphrases benefit).
  if (/[^A-Za-z0-9]/.test(password)) {
    score += 1
  }
  // Map the integer score onto a five-band label.
  const labels: PasswordStrengthLabel[] = ['very weak', 'weak', 'fair', 'good', 'strong']
  // Score 0 and 1 share "very weak"; 2=weak, 3=fair, 4=good, 5=strong.
  const label = score <= 1 ? labels[0] : labels[score - 1]
  return { score, label }
}
