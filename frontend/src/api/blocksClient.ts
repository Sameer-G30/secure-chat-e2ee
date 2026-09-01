// Block/unblock/list accounts, server-enforced. Replaces the legacy React
// prototype's block feature, which lived entirely in `localStorage` and was never
// synced or enforced server-side — a blocked user could still send messages.

// Read the API base URL from the same build-time environment variable other clients use.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe one blocked account as returned by the server.
export interface BlockedAccount {
  id: string
  username: string
  createdAt: string
}

// Distinguish expected API rejections from unexpected network failures.
export class BlocksApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'BlocksApiError'
    this.status = status
  }
}

function extractErrorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') {
      return detail
    }
  }
  return fallback
}

function authHeaders(accessToken: string): HeadersInit {
  return { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` }
}

// Load the caller's full block list.
export async function listBlocks(accessToken: string): Promise<BlockedAccount[]> {
  const response = await fetch(`${API_BASE_URL}/blocks`, {
    method: 'GET',
    headers: authHeaders(accessToken),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new BlocksApiError(
      extractErrorDetail(body, 'Could not load your block list. Please try again shortly.'),
      response.status,
    )
  }
  const payload = body as { blocks?: Array<{ id: string; username: string; created_at: string }> }
  if (!Array.isArray(payload.blocks)) {
    return []
  }
  return payload.blocks.map((row) => ({
    id: row.id,
    username: row.username,
    createdAt: row.created_at,
  }))
}

// Block a named account. Idempotent when the block already exists.
export async function blockUser(accessToken: string, username: string): Promise<BlockedAccount> {
  const response = await fetch(`${API_BASE_URL}/blocks`, {
    method: 'POST',
    headers: authHeaders(accessToken),
    body: JSON.stringify({ username }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new BlocksApiError(
      extractErrorDetail(body, 'Could not block that account. Please try again shortly.'),
      response.status,
    )
  }
  const record = body as { id: string; username: string; created_at: string }
  return { id: record.id, username: record.username, createdAt: record.created_at }
}

// Unblock a named account. Never errors when the account was not blocked.
export async function unblockUser(accessToken: string, username: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/blocks/${encodeURIComponent(username)}`, {
    method: 'DELETE',
    headers: authHeaders(accessToken),
  })
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new BlocksApiError(
      extractErrorDetail(body, 'Could not unblock that account. Please try again shortly.'),
      response.status,
    )
  }
}
