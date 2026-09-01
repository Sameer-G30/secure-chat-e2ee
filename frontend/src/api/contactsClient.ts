// Read the API base URL from the same build-time environment variable other clients use.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe one saved contact as returned by the server-side address book.
export interface ContactRecord {
  // Identify the contact account with the UUID used as AEAD associated data.
  id: string
  // Identify the contact account with the handle the sidebar displays.
  username: string
  // Record when the owner saved this contact, as an ISO-8601 string.
  createdAt: string
}

// Distinguish expected API rejections from unexpected network failures.
export class ContactsApiError extends Error {
  // Carry the HTTP status code so callers can branch on 400/404 vs. other failures.
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ContactsApiError'
    this.status = status
  }
}

// Extract a human-readable message from FastAPI's error response shapes.
function extractErrorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') {
      return detail
    }
  }
  return fallback
}

// Build the standard bearer-authenticated JSON headers shared by these endpoints.
function authHeaders(accessToken: string): HeadersInit {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${accessToken}`,
  }
}

// Parse one contact payload into camelCase for the UI.
function parseContact(body: { id: string; username: string; created_at: string }): ContactRecord {
  return {
    id: body.id,
    username: body.username,
    createdAt: body.created_at,
  }
}

// Load the authenticated owner's server-side address book.
export async function listContacts(accessToken: string): Promise<ContactRecord[]> {
  const response = await fetch(`${API_BASE_URL}/contacts`, {
    method: 'GET',
    headers: authHeaders(accessToken),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ContactsApiError(
      extractErrorDetail(body, 'Could not load contacts. Please try again shortly.'),
      response.status,
    )
  }
  const payload = body as { contacts?: Array<{ id: string; username: string; created_at: string }> }
  if (!Array.isArray(payload.contacts)) {
    return []
  }
  return payload.contacts.map(parseContact)
}

// Save a named account on the owner's server-side address book.
export async function addContact(accessToken: string, username: string): Promise<ContactRecord> {
  const response = await fetch(`${API_BASE_URL}/contacts`, {
    method: 'POST',
    headers: authHeaders(accessToken),
    body: JSON.stringify({ username }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ContactsApiError(
      extractErrorDetail(body, 'Could not add that contact. Please try again shortly.'),
      response.status,
    )
  }
  return parseContact(body as { id: string; username: string; created_at: string })
}

// Remove a named account from the owner's address book. The legacy React prototype
// never implemented this at all (its address book was add-only).
export async function deleteContact(accessToken: string, username: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/contacts/${encodeURIComponent(username)}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new ContactsApiError(
      extractErrorDetail(body, 'Could not remove that contact. Please try again shortly.'),
      response.status,
    )
  }
}
