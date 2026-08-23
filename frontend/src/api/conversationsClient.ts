// Decode standard base64 ciphertext and nonce from conversation-history JSON.
import { decodeBase64 } from '../crypto/keyExchange'
// Import the envelope type already used by the live WebSocket parser.
import type { RelayedEnvelope } from './chatSocket'

// Read the API base URL from the same build-time environment variable other clients use.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Describe one participant returned by conversation create/fetch.
export interface ConversationParticipant {
  // Identify the account with the UUID used as AEAD associated data.
  id: string
  // Identify the account with the handle the UI displays.
  username: string
  // Carry the peer's public key when present; this value is not secret.
  publicKey: string | null
}

// Describe a 1:1 conversation the caller is a member of.
export interface ConversationRecord {
  // Identify the conversation; clients bind this UUID into AEAD associated data.
  id: string
  // Carry the non-secret epoch counter used for per-message KDF subkey ids.
  currentEpoch: number
  // Carry the row creation timestamp as an ISO-8601 string.
  createdAt: string
  // Identify the authenticated caller so the client does not decode its JWT.
  self: ConversationParticipant
  // Identify the other participant the caller asked to chat with.
  peer: ConversationParticipant
}

// Describe the non-secret epoch counter returned by either epoch path.
export interface EpochRecord {
  // Identify which conversation this counter belongs to.
  conversationId: string
  // Carry the current non-secret epoch integer.
  currentEpoch: number
}

// Distinguish expected API rejections from unexpected network failures.
export class ConversationsApiError extends Error {
  // Carry the HTTP status code so callers can branch on 403/404 vs. other failures.
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ConversationsApiError'
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

// Parse a successful conversation payload into camelCase for the UI.
function parseConversation(body: {
  id: string
  current_epoch: number
  created_at: string
  self: { id: string; username: string; public_key: string | null }
  peer: { id: string; username: string; public_key: string | null }
}): ConversationRecord {
  return {
    id: body.id,
    currentEpoch: body.current_epoch,
    createdAt: body.created_at,
    self: {
      id: body.self.id,
      username: body.self.username,
      publicKey: body.self.public_key,
    },
    peer: {
      id: body.peer.id,
      username: body.peer.username,
      publicKey: body.peer.public_key,
    },
  }
}

// Start or fetch the unique 1:1 conversation with a named peer who also has a public key.
export async function startOrFetchConversation(
  accessToken: string,
  peerUsername: string,
): Promise<ConversationRecord> {
  const response = await fetch(`${API_BASE_URL}/conversations`, {
    method: 'POST',
    headers: authHeaders(accessToken),
    body: JSON.stringify({ peer_username: peerUsername }),
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ConversationsApiError(
      extractErrorDetail(body, 'Could not start that conversation. Please try again shortly.'),
      response.status,
    )
  }
  return parseConversation(
    body as {
      id: string
      current_epoch: number
      created_at: string
      self: { id: string; username: string; public_key: string | null }
      peer: { id: string; username: string; public_key: string | null }
    },
  )
}

// Fetch the non-secret epoch counter from the spec §6.4 path.
export async function fetchConversationEpoch(
  accessToken: string,
  conversationId: string,
): Promise<EpochRecord> {
  const response = await fetch(
    `${API_BASE_URL}/keys/conversations/${encodeURIComponent(conversationId)}/epoch`,
    {
      method: 'GET',
      headers: authHeaders(accessToken),
    },
  )
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ConversationsApiError(
      extractErrorDetail(body, 'Could not read the conversation epoch. Please try again shortly.'),
      response.status,
    )
  }
  const record = body as { conversation_id: string; current_epoch: number }
  return { conversationId: record.conversation_id, currentEpoch: record.current_epoch }
}

// Fetch ciphertext-only envelopes for one conversation the caller belongs to.
export async function fetchConversationMessages(
  accessToken: string,
  conversationId: string,
): Promise<RelayedEnvelope[]> {
  const response = await fetch(
    `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: 'GET',
      headers: authHeaders(accessToken),
    },
  )
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ConversationsApiError(
      extractErrorDetail(body, 'Could not load that conversation history. Please try again shortly.'),
      response.status,
    )
  }
  const payload = body as { messages?: unknown }
  if (!Array.isArray(payload.messages)) {
    return []
  }
  const envelopes: RelayedEnvelope[] = []
  for (const item of payload.messages) {
    const parsed = parseHistoryEnvelope(item)
    if (parsed !== null) {
      envelopes.push(parsed)
    }
  }
  return envelopes
}

// Narrow one history JSON object into a relayed envelope, or skip it if it is not one.
function parseHistoryEnvelope(value: unknown): RelayedEnvelope | null {
  if (!value || typeof value !== 'object') {
    return null
  }
  const frame = value as {
    type?: unknown
    id?: unknown
    conversation_id?: unknown
    sender_id?: unknown
    ciphertext?: unknown
    nonce?: unknown
    key_epoch?: unknown
  }
  if (frame.type !== 'envelope') {
    return null
  }
  if (
    typeof frame.id !== 'string' ||
    typeof frame.conversation_id !== 'string' ||
    typeof frame.sender_id !== 'string' ||
    typeof frame.ciphertext !== 'string' ||
    typeof frame.nonce !== 'string' ||
    typeof frame.key_epoch !== 'number'
  ) {
    return null
  }
  try {
    return {
      id: frame.id,
      conversationId: frame.conversation_id,
      senderId: frame.sender_id,
      ciphertext: decodeBase64(frame.ciphertext),
      nonce: decodeBase64(frame.nonce),
      keyEpoch: frame.key_epoch,
    }
  } catch {
    return null
  }
}

// Convert the REST API origin into the WebSocket origin used by the ciphertext relay.
export function conversationWebSocketUrl(conversationId: string, accessToken: string): string {
  const wsBase = API_BASE_URL.replace(/^http/, 'ws')
  const params = new URLSearchParams({ access_token: accessToken })
  return `${wsBase}/ws/conversations/${encodeURIComponent(conversationId)}?${params.toString()}`
}
