// Import the login/registration screen shown while no session exists.
import { AuthScreen } from './components/AuthScreen'
// Import the authenticated chat screen shown once a session exists.
import { ChatScreen } from './components/ChatScreen'
// Import the session provider and reader for protected routing.
import { AuthProvider, useAuth } from './context/AuthContext'

// Render the authenticated screen when a session exists, the auth screen otherwise.
//
// This is the complete "protected routing" surface for Slice 4: a real router
// is still unnecessary while there are only two screens, but ChatScreen is now
// the live encrypted conversation UI rather than a placeholder.
function AppRoutes() {
  const { session } = useAuth()
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
