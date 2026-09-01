// File a metadata-only abuse report. There is deliberately no field, here or on the
// server, for attaching message content: the server cannot read message ciphertext,
// so it structurally cannot verify — or even accept — reported message text without
// breaking the E2EE trust boundary documented in the architecture diagram.

// Read the API base URL from the same build-time environment variable other clients use.
const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// Distinguish expected API rejections from unexpected network failures.
export class ReportsApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ReportsApiError'
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

// File a report against a named account with a free-text reason.
export async function reportUser(accessToken: string, username: string, reason: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/reports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
    body: JSON.stringify({ username, reason }),
  })
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    throw new ReportsApiError(
      extractErrorDetail(body, 'Could not file that report. Please try again shortly.'),
      response.status,
    )
  }
}
