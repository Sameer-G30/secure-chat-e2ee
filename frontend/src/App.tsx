// Import the login/registration screen shown while no session exists.
import { AuthScreen } from './components/AuthScreen'
// Import the authenticated chat screen shown once a session exists.
import { ChatScreen } from './components/ChatScreen'
// Import the sequential ORT Web measurement page (not part of the chat UX).
import { MlLoadCheckScreen } from './components/MlLoadCheckScreen'
// Import the session provider and reader for protected routing.
import { AuthProvider, useAuth } from './context/AuthContext'

// Render the authenticated screen when a session exists, the auth screen otherwise.
//
// This is the complete "protected routing" surface for Slice 4: a real router
// is still unnecessary while there are only two screens, but ChatScreen is now
// the live encrypted conversation UI rather than a placeholder.
function AppRoutes() {
  // Call useAuth on every render so the hook order stays stable (load-check included).
  const { session } = useAuth()
  // Slice 6 measurement page is opted into via query string, not the chat shell.
  const isLoadCheck = new URLSearchParams(window.location.search).get('mlLoadCheck') === '1'
  // Keep the load-check unauthenticated so WASM measurement does not need two accounts.
  if (isLoadCheck) {
    // Render the six-way table instead of AuthScreen/ChatScreen.
    return <MlLoadCheckScreen />
  }
  // ChatScreen requires an in-memory session; AuthScreen is shown otherwise.
  return session ? <ChatScreen /> : <AuthScreen />
}

// Render the current application entry point, wrapped in session context.
function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}

// Export the root component for Vite and component tests.
export default App
