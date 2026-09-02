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

// Mock on-device classification so ChatScreen tests never load ORT Web.
vi.mock('../ml/scamClassifier', () => ({
  ensureChatDefaultClassifier: vi.fn(async () => {}),
  classifyVerifiedPlaintext: vi.fn(async () => ({
    pScam: 0.01,
    warned: false,
    checkpointId: 'tfidf_best',
    inferenceMs: 1,
  })),
  enableDistilbertOptIn: vi.fn(async () => {}),
  disableDistilbertOptIn: vi.fn(async () => {}),
  enableLstmOptIn: vi.fn(async () => {}),
  disableLstmOptIn: vi.fn(async () => {}),
}))

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
    fetchConversationMessages: vi.fn(),
    deleteConversationMessage: vi.fn(async () => {}),
    hideConversationMessage: vi.fn(async () => {}),
  }
})

// Mock the server-side address book so tests never hit GET/POST /contacts.
vi.mock('../api/contactsClient', async () => {
  const actual =
    await vi.importActual<typeof import('../api/contactsClient')>('../api/contactsClient')
  return {
    ...actual,
    listContacts: vi.fn(async () => []),
    addContact: vi.fn(),
    deleteContact: vi.fn(async () => {}),
  }
})

// Mock prefix user search so typing two characters in Add contact never hits the network.
vi.mock('../api/usersClient', async () => {
  const actual = await vi.importActual<typeof import('../api/usersClient')>('../api/usersClient')
  return {
    ...actual,
    searchUsers: vi.fn(async () => []),
  }
})

// Mock server-side blocks so the more-options menu never hits GET/POST /blocks.
vi.mock('../api/blocksClient', async () => {
  const actual = await vi.importActual<typeof import('../api/blocksClient')>('../api/blocksClient')
  return {
    ...actual,
    listBlocks: vi.fn(async () => []),
    blockUser: vi.fn(),
    unblockUser: vi.fn(async () => {}),
  }
})

// Mock metadata-only reports so the report form never hits POST /reports.
vi.mock('../api/reportsClient', async () => {
  const actual = await vi.importActual<typeof import('../api/reportsClient')>('../api/reportsClient')
  return {
    ...actual,
    reportUser: vi.fn(async () => {}),
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
import {
  deleteConversationMessage,
  fetchConversationEpoch,
  fetchConversationMessages,
  hideConversationMessage,
  startOrFetchConversation,
} from '../api/conversationsClient'
import { connectChatSocket } from '../api/chatSocket'
import { addContact, deleteContact, listContacts } from '../api/contactsClient'
import { searchUsers } from '../api/usersClient'
import { blockUser, listBlocks } from '../api/blocksClient'
import { reportUser } from '../api/reportsClient'
import { classifyVerifiedPlaintext, enableDistilbertOptIn } from '../ml/scamClassifier'
import { writeCachedBanner } from '../ml/scamBannerCache'
import { decryptMessage } from '../crypto/keyExchange'

const mockedFetchPublicKey = vi.mocked(fetchPublicKey)
const mockedStartOrFetchConversation = vi.mocked(startOrFetchConversation)
const mockedFetchConversationEpoch = vi.mocked(fetchConversationEpoch)
const mockedFetchConversationMessages = vi.mocked(fetchConversationMessages)
const mockedConnectChatSocket = vi.mocked(connectChatSocket)
const mockedListContacts = vi.mocked(listContacts)
const mockedAddContact = vi.mocked(addContact)
const mockedDeleteContact = vi.mocked(deleteContact)
const mockedSearchUsers = vi.mocked(searchUsers)
const mockedListBlocks = vi.mocked(listBlocks)
const mockedBlockUser = vi.mocked(blockUser)
const mockedReportUser = vi.mocked(reportUser)
const mockedDeleteConversationMessage = vi.mocked(deleteConversationMessage)
const mockedHideConversationMessage = vi.mocked(hideConversationMessage)
const mockedEnableDistilbertOptIn = vi.mocked(enableDistilbertOptIn)
const mockedClassify = vi.mocked(classifyVerifiedPlaintext)

const conversationId = '00000000-0000-4000-8000-000000000001'
const aliceId = '00000000-0000-4000-8000-000000000002'
const bobId = '00000000-0000-4000-8000-000000000003'

const sendEnvelope = vi.fn()
const sendTyping = vi.fn()
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
  sendTyping.mockReset()
  closeSocket.mockReset()
  capturedHandlers = null
  window.localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
  vi.clearAllMocks()
  mockedClassify.mockReset()
  mockedClassify.mockResolvedValue({
    pScam: 0.01,
    warned: false,
    checkpointId: 'tfidf_best',
    inferenceMs: 1,
  })
  mockedEnableDistilbertOptIn.mockReset()
  mockedEnableDistilbertOptIn.mockResolvedValue(undefined)
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
async function startChatWithBob(
  history: RelayedEnvelope[] = [],
  options: { currentEpoch?: number } = {},
) {
  // Default the mock conversation to epoch 0 unless a test is proving a later counter.
  const currentEpoch = options.currentEpoch ?? 0
  mockedFetchPublicKey.mockResolvedValue({
    username: 'bob',
    publicKey: encodeBase64(bobKeys.publicKey),
  })
  mockedStartOrFetchConversation.mockResolvedValue({
    id: conversationId,
    currentEpoch,
    createdAt: '2026-08-14T00:00:00Z',
    self: { id: aliceId, username: 'alice', publicKey: encodeBase64(aliceKeys.publicKey) },
    peer: { id: bobId, username: 'bob', publicKey: encodeBase64(bobKeys.publicKey) },
  })
  mockedFetchConversationEpoch.mockResolvedValue({
    conversationId,
    currentEpoch,
  })
  mockedFetchConversationMessages.mockResolvedValue(history)
  mockedListContacts.mockResolvedValue([])
  mockedAddContact.mockResolvedValue({
    id: bobId,
    username: 'bob',
    createdAt: '2026-08-14T00:00:00Z',
  })
  capturedHandlers = null
  mockedConnectChatSocket.mockImplementation((_id, _token, handlers) => {
    capturedHandlers = handlers
    return { sendEnvelope, sendTyping, close: closeSocket }
  })

  fireEvent.change(screen.getByLabelText('Add contact'), { target: { value: 'bob' } })
  fireEvent.click(screen.getByRole('button', { name: 'Add' }))
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
    const identity = sendEnvelope.mock.calls[0][2] as { messageId: string; revision: number }
    expect(identity.revision).toBe(0)
    expect(identity.messageId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    )
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
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
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
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })

    expect(await screen.findByText('message failed verification')).toBeInTheDocument()
    expect(screen.queryByText('do not show garbage')).not.toBeInTheDocument()
    expect(screen.queryByText('hello from bob')).not.toBeInTheDocument()
    expect(screen.queryByText('This message shows signs of a scam')).not.toBeInTheDocument()
  })

  it('shows DistilBERT and Word BiLSTM Best opt-in checkboxes', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    expect(screen.getByRole('checkbox', { name: 'Use DistilBERT (large download)' })).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Use Word BiLSTM Best' })).toBeInTheDocument()
  })

  it('clears a scam banner when DistilBERT loads and no longer warns', async () => {
    mockedClassify.mockResolvedValue({
      pScam: 0.95,
      warned: true,
      checkpointId: 'tfidf_best',
      inferenceMs: 2,
    })
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    capturedHandlers?.onEnvelope({
      id: 'msg-scam',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })
    expect(await screen.findByText('This message shows signs of a scam')).toBeInTheDocument()
    mockedClassify.mockResolvedValue({
      pScam: 0.01,
      warned: false,
      checkpointId: 'distilbert_default',
      inferenceMs: 2,
    })
    fireEvent.click(screen.getByRole('checkbox', { name: 'Use DistilBERT (large download)' }))
    await waitFor(() => {
      expect(screen.queryByText('This message shows signs of a scam')).not.toBeInTheDocument()
    })
    expect(window.localStorage.getItem('secure-chat-classifier:alice')).toBe('distilbert')
  })

  it('does not score with TF-IDF while DistilBERT is still loading', async () => {
    let resolveLoad: () => void = () => {}
    mockedEnableDistilbertOptIn.mockReturnValue(
      new Promise<void>((resolve) => {
        resolveLoad = resolve
      }),
    )
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    mockedClassify.mockClear()
    fireEvent.click(screen.getByRole('checkbox', { name: 'Use DistilBERT (large download)' }))
    capturedHandlers?.onEnvelope({
      id: 'msg-hold',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(screen.queryByText('This message shows signs of a scam')).not.toBeInTheDocument()
    expect(mockedClassify).not.toHaveBeenCalled()
    mockedClassify.mockResolvedValue({
      pScam: 0.95,
      warned: true,
      checkpointId: 'distilbert_default',
      inferenceMs: 2,
    })
    resolveLoad()
    expect(await screen.findByText('This message shows signs of a scam')).toBeInTheDocument()
    expect(mockedClassify.mock.calls.some((call) => call[1] === 'distilbert')).toBe(true)
  })

  it('restores DistilBERT from localStorage after a reload', async () => {
    window.localStorage.setItem('secure-chat-classifier:alice', 'distilbert')
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await waitFor(() => {
      expect(
        screen.getByRole('checkbox', { name: 'Use DistilBERT (large download)' }),
      ).toBeChecked()
    })
    expect(mockedEnableDistilbertOptIn).toHaveBeenCalled()
  })

  it('paints a cached DistilBERT banner before the graph loads', async () => {
    window.localStorage.setItem('secure-chat-classifier:alice', 'distilbert')
    writeCachedBanner('alice', 'msg-cached', 0, 'distilbert_default', true)
    mockedEnableDistilbertOptIn.mockReturnValue(new Promise(() => {}))
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await waitFor(() => {
      expect(
        screen.getByRole('checkbox', { name: 'Use DistilBERT (large download)' }),
      ).toBeChecked()
    })
    mockedClassify.mockClear()
    await startChatWithBob([
      {
        id: 'msg-cached',
        conversationId,
        senderId: bobId,
        ciphertext: new Uint8Array(32).fill(1),
        nonce: new Uint8Array(24).fill(2),
        keyEpoch: 0,
        adVersion: 1,
        messageId: null,
        revision: 0,
        editedAt: null,
      },
    ])
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(screen.getByText('This message shows signs of a scam')).toBeInTheDocument()
    expect(mockedClassify).not.toHaveBeenCalled()
  })

  it('shows a non-blocking scam banner on verified plaintext when the classifier warns', async () => {
    const mockedClassify = vi.mocked(classifyVerifiedPlaintext)
    mockedClassify.mockResolvedValue({
      pScam: 0.95,
      warned: true,
      checkpointId: 'tfidf_best',
      inferenceMs: 2,
    })
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()

    capturedHandlers?.onEnvelope({
      id: 'msg-scam',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })

    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(await screen.findByText('This message shows signs of a scam')).toBeInTheDocument()
  })

  it('decrypts scoped history after opening a contact', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob([
      {
        id: 'hist-1',
        conversationId,
        senderId: bobId,
        ciphertext: new Uint8Array(32).fill(1),
        nonce: new Uint8Array(24).fill(2),
        keyEpoch: 0,
        adVersion: 1,
        messageId: null,
        revision: 0,
        editedAt: null,
      },
    ])
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(mockedFetchConversationMessages).toHaveBeenCalledWith('access-token', conversationId)
  })

  it('shows message failed verification for a history envelope that does not verify', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob([
      {
        id: 'hist-bad',
        conversationId,
        senderId: bobId,
        ciphertext: new Uint8Array(32).fill(0xff),
        nonce: new Uint8Array(24).fill(2),
        keyEpoch: 0,
        adVersion: 1,
        messageId: null,
        revision: 0,
        editedAt: null,
      },
    ])
    expect(await screen.findByText('message failed verification')).toBeInTheDocument()
    expect(screen.queryByText('hello from bob')).not.toBeInTheDocument()
    expect(screen.queryByText('This message shows signs of a scam')).not.toBeInTheDocument()
  })

  it('persists dark mode per signed-in username', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark mode' }))
    expect(window.localStorage.getItem('secure-chat-theme:alice')).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
  })

  it('encrypts the next send with a bumped epoch and keeps the composer draft', async () => {
    // Render Alice's chat shell with an in-memory session.
    renderChatScreen()
    // Wait until the §8 chrome is on screen.
    await screen.findByRole('heading', { name: 'Secure Chat' })
    // Open Bob at epoch 0 so the first send would have used key_epoch 0.
    await startChatWithBob()
    // Type a draft that must survive the bump (do not send yet).
    fireEvent.change(screen.getByLabelText('Message'), {
      target: { value: 'still typing through rotation' },
    })
    // Simulate the server broadcasting {type:"epoch", current_epoch:1} to both tabs.
    capturedHandlers?.onEpoch?.(1)
    // The composer must not be cleared by the metadata frame.
    expect(screen.getByLabelText('Message')).toHaveValue('still typing through rotation')
    // Send after the bump so encrypt uses the new subkey id.
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    // The relay payload must carry key_epoch 1, never the draft as plaintext.
    expect(sendEnvelope).toHaveBeenCalledTimes(1)
    const [envelope] = sendEnvelope.mock.calls[0] as [
      { ciphertext: Uint8Array; nonce: Uint8Array; keyEpoch: number },
    ]
    expect(envelope.keyEpoch).toBe(1)
    expect(new TextDecoder().decode(envelope.ciphertext)).not.toContain(
      'still typing through rotation',
    )
    // The status line should mention the public counter.
    expect(screen.getByText(/epoch 1/)).toBeInTheDocument()
  })

  it('decrypts epoch-0 history after the conversation has already bumped to 1', async () => {
    // Render Alice's chat shell with an in-memory session.
    renderChatScreen()
    // Wait until the §8 chrome is on screen.
    await screen.findByRole('heading', { name: 'Secure Chat' })
    // Open Bob at current_epoch 1 with a stored envelope that still uses key_epoch 0.
    await startChatWithBob(
      [
        {
          id: 'hist-epoch-0',
          conversationId,
          senderId: bobId,
          ciphertext: new Uint8Array(32).fill(1),
          nonce: new Uint8Array(24).fill(2),
          keyEpoch: 0,
          adVersion: 1,
          messageId: null,
          revision: 0,
          editedAt: null,
        },
      ],
      { currentEpoch: 1 },
    )
    // History must still decrypt; decrypt uses the envelope's keyEpoch, not currentEpoch.
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    expect(vi.mocked(decryptMessage)).toHaveBeenCalledWith(
      expect.objectContaining({ keyEpoch: 0 }),
      expect.any(Uint8Array),
      expect.objectContaining({ conversationId, senderId: bobId }),
    )
  })

  it('opens settings and persists the system theme preference', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    fireEvent.click(screen.getByRole('radio', { name: 'System' }))
    expect(window.localStorage.getItem('secure-chat-theme:alice')).toBe('system')
  })

  it('fills the add-contact field from a prefix search hit', async () => {
    mockedSearchUsers.mockResolvedValue([{ username: 'bobby' }])
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    fireEvent.change(screen.getByLabelText('Add contact'), { target: { value: 'bo' } })
    expect(await screen.findByRole('button', { name: 'bobby' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'bobby' }))
    expect(screen.getByLabelText('Add contact')).toHaveValue('bobby')
  })

  it('offers hide-for-me on a received message and calls the hide endpoint', async () => {
    mockedHideConversationMessage.mockResolvedValue(undefined)
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    capturedHandlers?.onEnvelope({
      id: 'msg-1',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })
    fireEvent.click(await screen.findByText('hello from bob'))
    expect(screen.queryByRole('button', { name: 'Edit' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Hide for me' }))
    await waitFor(() => {
      expect(mockedHideConversationMessage).toHaveBeenCalledWith(
        'access-token',
        conversationId,
        'msg-1',
      )
    })
  })

  it('deletes an accepted own message for everyone', async () => {
    mockedDeleteConversationMessage.mockResolvedValue(undefined)
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    fireEvent.change(screen.getByLabelText('Message'), { target: { value: 'secret handshake' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    capturedHandlers?.onAccepted?.('server-msg-1')
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'secret handshake' })).toHaveAttribute(
        'data-pending',
        'false',
      )
    })
    fireEvent.click(screen.getByRole('button', { name: 'secret handshake' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete for everyone' }))
    await waitFor(() => {
      expect(mockedDeleteConversationMessage).toHaveBeenCalledWith(
        'access-token',
        conversationId,
        'server-msg-1',
      )
    })
  })

  it('warns that export writes plaintext to disk', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    fireEvent.click(screen.getByRole('button', { name: 'More' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Export chat' }))
    expect(
      screen.getByText(/writes decrypted plaintext to a/, { exact: false }),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download plaintext' })).toBeInTheDocument()
  })

  it('blocks the open peer and tears down the local conversation', async () => {
    mockedListBlocks.mockResolvedValue([])
    mockedBlockUser.mockResolvedValue({
      id: bobId,
      username: 'bob',
      createdAt: '2026-08-14T00:00:00Z',
    })
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    fireEvent.click(screen.getByRole('button', { name: 'More' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Block' }))
    await waitFor(() => {
      expect(mockedBlockUser).toHaveBeenCalledWith('access-token', 'bob')
    })
    expect(screen.getByText(/bob is blocked/)).toBeInTheDocument()
    expect(closeSocket).toHaveBeenCalled()
  })

  it('files a metadata-only report without attaching message contents', async () => {
    mockedReportUser.mockResolvedValue(undefined)
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    fireEvent.click(screen.getByRole('button', { name: 'More' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Report' }))
    expect(screen.getByText(/Message contents are not attached/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Reason'), { target: { value: 'unsolicited spam' } })
    fireEvent.click(screen.getByRole('button', { name: 'Submit report' }))
    await waitFor(() => {
      expect(mockedReportUser).toHaveBeenCalledWith('access-token', 'bob', 'unsolicited spam')
    })
  })

  it('filters the in-memory transcript from Search in chat', async () => {
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    await startChatWithBob()
    capturedHandlers?.onEnvelope({
      id: 'msg-1',
      conversationId,
      senderId: bobId,
      ciphertext: new Uint8Array(32).fill(1),
      nonce: new Uint8Array(24).fill(2),
      keyEpoch: 0,
      adVersion: 1,
      messageId: null,
      revision: 0,
      editedAt: null,
    })
    expect(await screen.findByText('hello from bob')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'More' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Search in chat' }))
    fireEvent.change(screen.getByLabelText('Search in chat'), { target: { value: 'nope' } })
    expect(screen.getByText('No messages match that search on this device.')).toBeInTheDocument()
  })

  it('removes a contact from the server-side address book', async () => {
    mockedListContacts.mockResolvedValue([
      { id: bobId, username: 'bob', createdAt: '2026-08-14T00:00:00Z' },
    ])
    mockedDeleteContact.mockResolvedValue(undefined)
    renderChatScreen()
    await screen.findByRole('heading', { name: 'Secure Chat' })
    expect(await screen.findByRole('button', { name: 'Remove bob' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Remove bob' }))
    await waitFor(() => {
      expect(mockedDeleteContact).toHaveBeenCalledWith('access-token', 'bob')
    })
  })
})
