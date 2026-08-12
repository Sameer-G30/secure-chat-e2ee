// Read the API base URL from Vite's build-time environment, with a local default.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe the fields the registration form collects from the user.
export interface RegisterInput {
  // Carry the chosen account handle.
  username: string
  // Carry the account's contact email.
  email: string
  // Carry the plaintext password only for the duration of this HTTPS request.
  password: string
}

// Describe the non-secret account fields the API returns after registration.
export interface RegisteredAccount {
  // Carry the server-assigned account identifier.
  id: string
  // Carry the stored username back for immediate UI confirmation.
  username: string
  // Carry the stored email back for immediate UI confirmation.
  email: string
  // Carry the account creation timestamp as an ISO-8601 string.
  createdAt: string
}

// Distinguish expected API rejections (validation, conflicts, rate limits) from
// unexpected network failures, so the UI can show a specific, honest message.
export class AuthApiError extends Error {
  // Carry the HTTP status code so callers can branch on conflict vs. rate limit.
  readonly status: number

  constructor(message: string, status: number) {
    // Build the standard Error message chain.
    super(message)
    // Name this error class for clearer stack traces and instanceof checks.
    this.name = 'AuthApiError'
    // Record the HTTP status that produced this error.
    this.status = status
  }
}

// Extract a human-readable message from FastAPI's error response shapes.
function extractErrorDetail(body: unknown, fallback: string): string {
  // FastAPI's HTTPException responses carry {"detail": "..."}.
  if (body && typeof body === 'object' && 'detail' in body) {
    // Read the detail field defensively without assuming its exact shape.
    const detail = (body as { detail: unknown }).detail
    // Use the detail string directly when the server sent one.
    if (typeof detail === 'string') {
      return detail
    }
    // Pydantic validation errors carry a list of {msg, loc} objects instead.
    if (Array.isArray(detail) && detail.length > 0 && typeof detail[0]?.msg === 'string') {
      return detail[0].msg as string
    }
  }
  // Fall back to a generic message when the body has no recognizable detail.
  return fallback
}

// Register a new account against the real backend.
export async function registerAccount(input: RegisterInput): Promise<RegisteredAccount> {
  // Send the registration payload as JSON to the FastAPI endpoint.
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    // Use POST to create a new account resource.
    method: 'POST',
    // Declare the JSON content type the server expects.
    headers: { 'Content-Type': 'application/json' },
    // Serialize exactly the three fields the API validates.
    body: JSON.stringify(input),
  })

  // Parse the JSON body once; both success and error paths need it.
  const body: unknown = await response.json().catch(() => null)

  // Surface rate-limit responses with a message the UI can show verbatim.
  if (response.status === 429) {
    throw new AuthApiError('Too many attempts. Please wait a moment and try again.', 429)
  }
  // Surface username/email conflicts using the server's specific detail message.
  if (response.status === 409) {
    throw new AuthApiError(extractErrorDetail(body, 'That account already exists.'), 409)
  }
  // Surface validation failures (short password, bad email) from Pydantic.
  if (response.status === 422) {
    throw new AuthApiError(extractErrorDetail(body, 'Please check your details and try again.'), 422)
  }
  // Treat any other non-2xx status as an unexpected server-side failure.
  if (!response.ok) {
    throw new AuthApiError('Registration failed. Please try again shortly.', response.status)
  }

  // Narrow the successful JSON body into the typed shape the UI expects.
  const created = body as { id: string; username: string; email: string; created_at: string }
  return {
    id: created.id,
    username: created.username,
    email: created.email,
    createdAt: created.created_at,
  }
}
