// Read the API base URL from the same build-time environment variable authClient.ts uses.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe the non-secret public-key fields the API returns from either endpoint.
export interface PublicKeyRecord {
  // Identify whose public key this is.
  username: string
  // Carry the base64 X25519 public key; this value is not secret.
  publicKey: string
}

// Distinguish expected API rejections from unexpected network failures.
export class KeysApiError extends Error {
  // Carry the HTTP status code so callers can branch on 404 (no key yet) vs. other failures.
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'KeysApiError'
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

// Build the standard bearer-authenticated JSON headers shared by both endpoints.
function authHeaders(accessToken: string): HeadersInit {
  return {
    'Content-Type': 'application/json',
    // Send the short-lived access token issued by POST /auth/login or /auth/refresh.
    Authorization: `Bearer ${accessToken}`,
  }
}

// Upload the caller's own base64 X25519 public key. The matching private key never
// leaves the browser; it is generated and sealed entirely client-side (see keyVault.ts).
export async function uploadMyPublicKey(
  accessToken: string,
  publicKeyBase64: string,
): Promise<PublicKeyRecord> {
  const response = await fetch(`${API_BASE_URL}/keys/me`, {
    method: 'POST',
    headers: authHeaders(accessToken),
    body: JSON.stringify({ public_key: publicKeyBase64 }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new KeysApiError(
      extractErrorDetail(body, 'Could not upload your public key. Please try again shortly.'),
      response.status,
    )
  }
  const record = body as { username: string; public_key: string }
  return { username: record.username, publicKey: record.public_key }
}

// Look up a peer's public key so the caller can derive shared directional session keys.
export async function fetchPublicKey(
  accessToken: string,
  username: string,
): Promise<PublicKeyRecord> {
  const response = await fetch(`${API_BASE_URL}/keys/${encodeURIComponent(username)}`, {
    method: 'GET',
    headers: authHeaders(accessToken),
  })
  const body: unknown = await response.json().catch(() => null)
  if (response.status === 404) {
    throw new KeysApiError(
      extractErrorDetail(body, `${username} has not set up encryption yet.`),
      404,
    )
  }
  if (!response.ok) {
    throw new KeysApiError(
      extractErrorDetail(body, 'Could not look up that public key. Please try again shortly.'),
      response.status,
    )
  }
  const record = body as { username: string; public_key: string }
  return { username: record.username, publicKey: record.public_key }
}
