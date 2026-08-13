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

// Describe the fields the login form collects from the user.
export interface LoginInput {
  // Carry the account handle to authenticate.
  username: string
  // Carry the plaintext password only for the duration of this HTTPS request.
  password: string
}

// Describe the token pair and key-status flag returned by login and refresh.
export interface TokenPair {
  // Carry the short-lived JWT sent as "Authorization: Bearer <accessToken>".
  accessToken: string
  // Carry the longer-lived, single-use JWT used only to request a new pair.
  refreshToken: string
  // Tell the caller whether this account still needs to upload an X25519 public key.
  hasPublicKey: boolean
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

// Shape the fallback messages one call site needs for each non-2xx status.
interface StatusFallbacks {
  // Message shown when the server enforces its rate limit (429).
  rateLimited: string
  // Message shown for a request validation failure (422).
  invalid: string
  // Message shown for any other unexpected non-2xx status.
  serverError: string
  // Optional message shown for a specific conflict/auth status (409 or 401).
  specific?: { status: number; fallback: string }
}

// Parse a fetch Response into JSON, raising a typed AuthApiError on failure statuses.
async function parseJsonOrThrow(response: Response, fallbacks: StatusFallbacks): Promise<unknown> {
  // Parse the JSON body once; both success and error paths need it.
  const body: unknown = await response.json().catch(() => null)

  // Surface rate-limit responses with a message the UI can show verbatim.
  if (response.status === 429) {
    throw new AuthApiError(fallbacks.rateLimited, 429)
  }
  // Surface the endpoint-specific status (409 conflict for register, 401 for login/refresh).
  if (fallbacks.specific && response.status === fallbacks.specific.status) {
    throw new AuthApiError(
      extractErrorDetail(body, fallbacks.specific.fallback),
      fallbacks.specific.status,
    )
  }
  // Surface validation failures (short password, bad email, malformed key) from Pydantic.
  if (response.status === 422) {
    throw new AuthApiError(extractErrorDetail(body, fallbacks.invalid), 422)
  }
  // Treat any other non-2xx status as an unexpected server-side failure.
  if (!response.ok) {
    throw new AuthApiError(fallbacks.serverError, response.status)
  }
  return body
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

  const body = await parseJsonOrThrow(response, {
    rateLimited: 'Too many attempts. Please wait a moment and try again.',
    invalid: 'Please check your details and try again.',
    serverError: 'Registration failed. Please try again shortly.',
    specific: { status: 409, fallback: 'That account already exists.' },
  })

  // Narrow the successful JSON body into the typed shape the UI expects.
  const created = body as { id: string; username: string; email: string; created_at: string }
  return {
    id: created.id,
    username: created.username,
    email: created.email,
    createdAt: created.created_at,
  }
}

// Log an existing account in against the real backend, returning a fresh token pair.
export async function loginAccount(input: LoginInput): Promise<TokenPair> {
  // Send the login payload as JSON to the FastAPI endpoint.
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })

  const body = await parseJsonOrThrow(response, {
    rateLimited: 'Too many attempts. Please wait a moment and try again.',
    invalid: 'Please check your details and try again.',
    serverError: 'Login failed. Please try again shortly.',
    // Never let the UI distinguish "unknown user" from "wrong password" (matches the API).
    specific: { status: 401, fallback: 'Invalid username or password.' },
  })

  const issued = body as {
    access_token: string
    refresh_token: string
    has_public_key: boolean
  }
  return {
    accessToken: issued.access_token,
    refreshToken: issued.refresh_token,
    hasPublicKey: issued.has_public_key,
  }
}

// Rotate a refresh token for a new token pair; the presented token becomes unusable either way.
export async function refreshTokens(refreshToken: string): Promise<TokenPair> {
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })

  const body = await parseJsonOrThrow(response, {
    rateLimited: 'Too many attempts. Please wait a moment and try again.',
    invalid: 'Your session is invalid. Please log in again.',
    serverError: 'Could not refresh your session. Please try again shortly.',
    specific: { status: 401, fallback: 'Your session has expired. Please log in again.' },
  })

  const issued = body as {
    access_token: string
    refresh_token: string
    has_public_key: boolean
  }
  return {
    accessToken: issued.access_token,
    refreshToken: issued.refresh_token,
    hasPublicKey: issued.has_public_key,
  }
}

// Revoke a refresh token on explicit logout so it cannot later be rotated by anyone.
export async function logoutAccount(refreshToken: string): Promise<void> {
  // Best-effort: a failed logout call still lets the client discard its local tokens.
  await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  }).catch(() => {
    // Network failure during logout must not block the client from clearing local state.
  })
}
