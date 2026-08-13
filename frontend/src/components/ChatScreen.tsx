// Import the same base64 transport encoder used everywhere else, rather than reimplementing it.
import { encodeBase64 } from '../crypto/keyExchange'
// Import the session context so this screen can show identity and offer logout.
import { useAuth } from '../context/AuthContext'

// Render a minimal authenticated placeholder shell.
//
// This intentionally stays minimal per the Slice 3 scope: it proves
// protected routing and exposes the account's derived public key for
// visual/manual verification, but the real contacts/conversation/message
// UI (§8 of the spec) and the WebSocket relay both ship in Slice 4.
export function ChatScreen() {
  const { session, logout } = useAuth()

  // This screen is only ever rendered while a session exists (see App.tsx routing),
  // but guard defensively rather than assuming a non-null session unsafely.
  if (!session) {
    return null
  }

  return (
    <main className="chat-screen-placeholder">
      <header className="chat-placeholder-header">
        <div>
          <h1>Secure Chat</h1>
          <p>
            Signed in as <strong>{session.username}</strong>
          </p>
        </div>
        <button type="button" className="text-button" onClick={() => void logout()}>
          Log out
        </button>
      </header>

      <section className="chat-placeholder-card" aria-labelledby="identity-heading">
        <h2 id="identity-heading">Your device identity</h2>
        <p>
          Your X25519 keypair was generated in this browser and its private half is sealed in
          IndexedDB — it has never left this device.
        </p>
        <dl>
          <dt>Public key (base64)</dt>
          <dd className="identity-public-key">{encodeBase64(session.identityKeyPair.publicKey)}</dd>
        </dl>
      </section>

      <p className="slice-note" role="status">
        Real-time conversations, contacts, and message history arrive in Slice 4 (WebSocket
        ciphertext relay and epoch endpoint).
      </p>
    </main>
  )
}
