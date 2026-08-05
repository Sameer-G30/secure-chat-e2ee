// Import React's Vite transform and fast-refresh integration.
import react from '@vitejs/plugin-react'
// Import Vitest-aware Vite configuration typing.
import { defineConfig } from 'vitest/config'

// Configure both the browser build and colocated component tests.
export default defineConfig({
  // Transform React 19 TypeScript and enable development fast refresh.
  plugins: [react()],
  // Run component tests in a browser-like DOM environment.
  test: {
    // Emulate the DOM APIs required by React Testing Library.
    environment: 'jsdom',
    // Register accessible DOM matchers before each test module.
    setupFiles: './src/test/setup.ts',
  },
})
