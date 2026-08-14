import { useEffect } from 'react'
// Import React Testing Library's user-facing render, query, and async helpers.
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
// Import Vitest's grouping, assertion, and mocking helpers.
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

// Import the chat screen under test.
import { ChatScreen } from './ChatScreen'
// Import the session provider ChatScreen requires via useAuth().
import { AuthProvider, useAuth } from '../context/AuthContext'
// Import identity helpers; AEAD encrypt/decrypt are mocked below because jsdom's
// libsodium build rejects AEAD inputs that the node environment accepts.
import {
  encodeBase64,
  generateIdentityKeyPair,
  initializeSodium,
} from '../crypto/keyExchange'
import type { IdentityKeyPair } from '../crypto/keyExchange'
// Import the socket handler type so the mock can capture ChatScreen's callbacks.
import type { ChatSocketHandlers, RelayedEnvelope } from '../api/chatSocket'

// Mock the public-key lookup so tests never perform a real HTTP request.
vi.mock('../api/keysClient', async () => {
  const actual = await vi.importActual<typeof import('../api/keysClient')>('../api/keysClient')
  return {
    ...actual,
    fetchPublicKey: vi.fn(),
  }
})

// Mock conversation create/fetch and epoch reads the same way.
vi.mock('../api/conversationsClient', async () => {
  const actual =
    await vi.importActual<typeof import('../api/conversationsClient')>('../api/conversationsClient')
  return {
    ...actual,
    startOrFetchConversation: vi.fn(),
    fetchConversationEpoch: vi.fn(),
  }
})

// Mock the WebSocket wrapper so tests can inject relayed envelopes without a server.
vi.mock('../api/chatSocket', async () => {
  const actual = await vi.importActual<typeof import('../api/chatSocket')>('../api/chatSocket')
  return {
    ...actual,
    connectChatSocket: vi.fn(),
  }
})

// Mock AEAD primitives for jsdom. Round-trip and tamper proofs live in keyExchange.test.ts.
vi.mock('../crypto/keyExchange', async () => {
  const actual = await vi.importActual<typeof import('../crypto/keyExchange')>('../crypto/keyExchange')
  return {
    ...actual,
    encryptMessage: vi.fn((plaintext: string, _key: Uint8Array, keyEpoch: number) => ({
      ciphertext: new Uint8Array(32).fill(7),
      nonce: new Uint8Array(24).fill(3),
      keyEpoch,
      __plaintext: plaintext,
    })),
    decryptMessage: vi.fn((envelope: { ciphertext: Uint8Array }) => {
      if (envelope.ciphertext[0] === 0xff) {
        throw new Error('verification failed')
      }
      return 'hello from bob'
    }),
  }
})

import { fetchPublicKey } from '../api/keysClient'
import { fetchConversationEpoch, startOrFetchConversation } from '../api/conversationsClient'
import { connectChatSocket } from '../api/chatSocket'

const mockedFetchPublicKey = vi.mocked(fetchPublicKey)
const mockedStartOrFetchConversation = vi.mocked(startOrFetchConversation)
const mockedFetchConversationEpoch = vi.mocked(fetchConversationEpoch)
const mockedConnectChatSocket = vi.mocked(connectChatSocket)

const conversationId = '00000000-0000-4000-8000-000000000001'
const aliceId = '00000000-0000-4000-8000-000000000002'
const bobId = '00000000-0000-4000-8000-000000000003'

const sendEnvelope = vi.fn()
const closeSocket = vi.fn()
let capturedHandlers: ChatSocketHandlers | null = null

let aliceKeys: IdentityKeyPair
let bobKeys: IdentityKeyPair

// Wait for libsodium before generating identity keys used by every case.
beforeAll(async () => {
  await initializeSodium()
  aliceKeys = generateIdentityKeyPair()
  bobKeys = generateIdentityKeyPair()
})

afterEach(() => {
  sendEnvelope.mockReset()
  closeSocket.mockReset()
  capturedHandlers = null
  vi.clearAllMocks()
})

// Render ChatScreen inside AuthProvider after injecting Alice's in-memory session.
function renderChatScreen() {
  function Harness() {
    const { setSession } = useAuth()
    useEffect(() => {
      setSession({
        username: 'alice',
        accessToken: 'access-token',
        refreshToken: 'refresh-token',
        identityKeyPair: aliceKeys,
      })
    }, [setSession])
    return <ChatScreen />
  }
  return render(
    <AuthProvider>
      <Harness />
    </AuthProvider>,
  )
}

// Complete the peer lookup + conversation + epoch + socket handshake with mocks.
async function startChatWithBob() {
  mockedFetchPublicKey.mockResolvedValue({
    username: 'bob',
    publicKey: encodeBase64(bobKeys.publicKey),
  })
  mockedStartOrFetchConversation.mockResolvedValue({
    id: conversationId,
    currentEpoch: 0,
    createdAt: '2026-08-14T00:00:00Z',
    self: { id: aliceId, username: 'alice', publicKey: encodeBase64(aliceKeys.publicKey) },
    peer: { id: bobId, username: 'bob', publicKey: encodeBase64(bobKeys.publicKey) },
  })
  mockedFetchConversationEpoch.mockResolvedValue({
    conversationId,
    currentEpoch: 0,
  })
  capturedHandlers = null
  mockedConnectChatSocket.mockImplementation((_id, _token, handlers) => {
    capturedHandlers = handlers
    return { sendEnvelope, close: closeSocket }
  })

  fireEvent.change(screen.getByLabelText('Peer username'), { target: { value: 'bob' } })
  fireEvent.click(screen.getByRole('button', { name: 'Start encrypted chat' }))
  await waitFor(() => {
    expect(mockedConnectChatSocket).toHaveBeenCalled()
  })
}

describe('ChatScreen', () => {
  it('encrypts outgoing plaintext so the relay payload is ciphertext only', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()

    fireEvent.change(screen.getByLabelText('Message'), {
      target: { value: 'secret handshake' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))

    expect(sendEnvelope).toHaveBeenCalledTimes(1)
    const [envelope, routing] = sendEnvelope.mock.calls[0] as [
      { ciphertext: Uint8Array; nonce: Uint8Array; keyEpoch: number },
      { conversationId: string; senderId: string },
    ]
    expect(routing.conversationId).toBe(conversationId)
    expect(routing.senderId).toBe(aliceId)
    expect(envelope.keyEpoch).toBe(0)
    expect(new TextDecoder().decode(envelope.ciphertext)).not.toContain('secret handshake')
    expect(screen.getByText('secret handshake')).toBeInTheDocument()
  })

  it('renders a received envelope after decrypt+verify succeeds', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()

    const relayed: RelayedEnvelope = {
      id: 'msg-1',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
    }
    capturedHandlers?.onEnvelope(relayed)
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(screen.queryByText('message failed verification')).not.toBeInTheDocument()
  })

  it('shows message failed verification instead of corrupted plaintext', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()

    capturedHandlers?.onEnvelope({
      id: 'msg-bad',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(0xff),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
    })

    expect(await screen.findByText('message failed verification')).toBeInTheDocument()
    expect(screen.queryByText('do not show garbage')).not.toBeInTheDocument()
    expect(screen.queryByText('hello from bob')).not.toBeInTheDocument()
  })
})
