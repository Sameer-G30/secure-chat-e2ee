// Register accessible DOM assertions such as toBeInTheDocument for every test.
import '@testing-library/jest-dom/vitest'
// Import cleanup so each component test starts with an empty document.
import { cleanup } from '@testing-library/react'
// Import Vitest's post-test lifecycle hook.
import { afterEach } from 'vitest'

// Remove rendered React trees after each independent test case.
afterEach(() => {
  // Unmount components and clear Testing Library's document containers.
  cleanup()
})
