// Import StrictMode to surface unsafe React behavior during development.
import { StrictMode } from 'react'
// Import React's modern concurrent-compatible DOM root API.
import { createRoot } from 'react-dom/client'

// Import global design tokens and the authentication visual system.
import './index.css'
// Import the root application component.
import App from './App.tsx'

// Look up the static Vite mount point.
const rootElement = document.getElementById('root')

// Fail clearly if the host document no longer contains the required mount point.
if (rootElement === null) {
  // Stop startup rather than hiding a broken HTML contract.
  throw new Error('Root element was not found')
}

// Create React's managed DOM root around the verified mount element.
createRoot(rootElement).render(
  // Enable additional development checks without changing production output.
  <StrictMode>
    {/* Render the current Secure Chat application tree. */}
    <App />
  </StrictMode>,
)
