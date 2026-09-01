// Debounced prefix search against GET /users/search. The legacy React prototype
// downloaded every account from Firebase and substring-matched in the browser;
// this hook never does that — it waits until the field has two characters, then
// asks the server for a prefix match that already excludes the caller.

// Import React's state/effect hooks used to debounce and store hits.
import { useEffect, useState } from 'react'

// Import the authenticated search client this hook wraps.
import { searchUsers, UsersApiError } from '../api/usersClient'
import type { UserSearchResult } from '../api/usersClient'
// Import the shared authorized-request contract this hook is given by the caller.
import type { AuthorizedRequestApi } from './useAuthorizedRequest'
// Reuse AuthContext's Session so the same AuthorizedRequestApi instance works here.
import type { Session } from '../context/AuthContext'

// Wait this long after the last keystroke before spending a network round trip.
const SEARCH_DEBOUNCE_MS = 250

// Describe what this hook hands back to the add-contact form.
export interface UserSearchApi {
  // Carry the latest prefix matches, or an empty list while idle/too-short.
  results: UserSearchResult[]
  // True only while a request is in flight after the debounce has fired.
  isSearching: boolean
}

// Search for accounts whose username starts with `query`.
export function useUserSearch(
  query: string,
  authorizedRequest: AuthorizedRequestApi<Session>,
  onError: (message: string) => void,
): UserSearchApi {
  // Hold the most recent prefix matches for the typeahead list.
  const [results, setResults] = useState<UserSearchResult[]>([])
  // Hold whether a search request is currently in flight.
  const [isSearching, setIsSearching] = useState(false)

  useEffect(() => {
    // Skip the network entirely until the server's documented two-character floor.
    const trimmed = query.trim()
    if (trimmed.length < 2) {
      setResults([])
      setIsSearching(false)
      return
    }
    // Ignore stale responses if the query changes or the screen unmounts.
    let cancelled = false
    // Mark in-flight immediately so the empty-state does not flash during debounce.
    setIsSearching(true)
    // Wait out the debounce so each keystroke is not its own HTTP request.
    const timer = window.setTimeout(() => {
      void authorizedRequest
        .request((accessToken) => searchUsers(accessToken, trimmed))
        .then((hits) => {
          if (!cancelled) {
            setResults(hits)
            setIsSearching(false)
          }
        })
        .catch((caught: unknown) => {
          if (!cancelled) {
            setResults([])
            setIsSearching(false)
            const message =
              caught instanceof UsersApiError
                ? caught.message
                : 'Could not search for accounts. Please try again shortly.'
            onError(message)
          }
        })
    }, SEARCH_DEBOUNCE_MS)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [query, authorizedRequest.request, onError])

  return { results, isSearching }
}
