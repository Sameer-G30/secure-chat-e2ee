// Extracted from ChatScreen.tsx during the pre-deployment refactor (Phase 3a). Owns the
// server-side address book (contacts never live in localStorage) and the single-flight
// token-refresh wiring for reading/writing it. Behavior is unchanged from before the
// split, with one deliberate optimization: the reload effect now keys on the signed-in
// username rather than the whole session object, so a token *refresh* (which replaces
// the session object every time, see AuthContext.updateTokens) no longer re-fetches the
// contact list — only an actual sign-in/sign-out does.

// Import React's state/effect hooks.
import { useCallback, useEffect, useState } from 'react'

// Import the server-side address book client this hook wraps.
import {
  addContact as addContactRequest,
  ContactsApiError,
  deleteContact as deleteContactRequest,
  listContacts,
} from '../api/contactsClient'
import type { ContactRecord } from '../api/contactsClient'
// Import the shared authorized-request contract this hook is given by the caller.
import type { AuthorizedRequestApi } from './useAuthorizedRequest'
// Reuse AuthContext's own Session shape so this hook's type lines up exactly with the
// same AuthorizedRequestApi instance ChatScreen also hands to useEncryptedConversation.
import type { Session } from '../context/AuthContext'

// Describe what this hook hands back to ChatScreen.
export interface ContactsApi {
  // Carry the server-side address book loaded after login.
  contacts: ContactRecord[]
  // Save a handle on the server-side address book and return the new record.
  addContact: (username: string) => Promise<ContactRecord>
  // Remove a handle from the server-side address book (legacy was add-only).
  removeContact: (username: string) => Promise<void>
  // Clear the unread badge locally after this tab marks the chat read.
  clearUnread: (username: string) => void
  // Reload the address book so unread counts stay in sync after a focus.
  reloadContacts: () => Promise<void>
}

// Load and manage the signed-in user's server-side contact list.
export function useContacts(
  // Accept only the fields needed to key the reload effect; the full session lives in
  // authorizedRequest's own ref.
  username: string | undefined,
  authorizedRequest: AuthorizedRequestApi<Session>,
  // Surface a load/add failure the same way the rest of ChatScreen reports errors.
  onError: (message: string) => void,
): ContactsApi {
  const [contacts, setContacts] = useState<ContactRecord[]>([])

  // Load the server-side address book once a session exists, and again only if the
  // signed-in identity actually changes (not on every token refresh).
  useEffect(() => {
    if (!username) {
      return
    }
    let cancelled = false
    void authorizedRequest
      .request((accessToken) => listContacts(accessToken))
      .then((rows) => {
        if (!cancelled) {
          setContacts(rows)
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          const message =
            caught instanceof ContactsApiError
              ? caught.message
              : 'Could not load contacts. Please try again shortly.'
          onError(message)
        }
      })
    return () => {
      cancelled = true
    }
    // Depend on the stable `request` function reference, not the `authorizedRequest`
    // object itself (a fresh object every render) — otherwise this would refetch on
    // every render instead of only when the signed-in identity changes.
  }, [username, authorizedRequest.request, onError])

  async function addContact(peerUsername: string): Promise<ContactRecord> {
    const saved = await authorizedRequest.request((accessToken) =>
      addContactRequest(accessToken, peerUsername),
    )
    setContacts((existing) => {
      if (existing.some((row) => row.id === saved.id)) {
        return existing
      }
      return [saved, ...existing]
    })
    return saved
  }

  async function removeContact(peerUsername: string): Promise<void> {
    await authorizedRequest.request((accessToken) =>
      deleteContactRequest(accessToken, peerUsername),
    )
    setContacts((existing) => existing.filter((row) => row.username !== peerUsername))
  }

  function clearUnread(peerUsername: string) {
    setContacts((existing) =>
      existing.map((row) => (row.username === peerUsername ? { ...row, unreadCount: 0 } : row)),
    )
  }

  const reloadContacts = useCallback(async (): Promise<void> => {
    const rows = await authorizedRequest.request((accessToken) => listContacts(accessToken))
    setContacts(rows)
  }, [authorizedRequest.request])

  return { contacts, addContact, removeContact, clearUnread, reloadContacts }
}
