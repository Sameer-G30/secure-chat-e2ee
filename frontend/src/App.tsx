// Import the only screen implemented in the first frontend slice.
import { AuthScreen } from './components/AuthScreen'

// Render the current application entry screen.
function App() {
  // Keep routing out of Slice 1 while exposing a real component boundary.
  return <AuthScreen />
}

// Export the root component for Vite and component tests.
export default App
