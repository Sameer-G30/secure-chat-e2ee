// Remember when this page already has an in-memory session so a startup
// restore does not rotate the refresh token a login just stored.
//
// IndexedDB reads finish after every useEffect in the same commit has
// started. ChatScreen tests call setSession from a child effect; restore
// must see that claim and skip POST /auth/refresh. A real reload starts
// with this flag false because the module is evaluated again.

// True after setSession or a successful restore until logout.
let claimed = false

// Mark that React state already holds the signed-in session.
export function markInMemorySession(): void {
  // Later restore work in this page load must not spend the refresh token.
  claimed = true
}

// Read whether an in-memory session won the race against IndexedDB.
export function hasInMemorySession(): boolean {
  // False on a cold load, and again after logout.
  return claimed
}

// Allow the next AuthProvider mount to read IndexedDB again.
export function resetInMemorySessionClaim(): void {
  // Logout and tests both need a fresh claim bit.
  claimed = false
}
