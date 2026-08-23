// Import React's Vite transform and fast-refresh integration.
import react from '@vitejs/plugin-react'
// Import Vitest-aware Vite configuration typing.
import { defineConfig } from 'vitest/config'

// Configure both the browser build and colocated component tests.
export default defineConfig({
  // Transform React 19 TypeScript and enable development fast refresh.
  plugins: [react()],
  // Treat ORT's SIMD binary as a static asset so Vite serves the correct MIME type.
  assetsInclude: ['**/*.wasm'],
  // Avoid prebundling ORT Web so its WASM assets resolve from the package dist.
  optimizeDeps: {
    // onnxruntime-web ships its own WASM; Vite's dep optimizer breaks those URLs.
    exclude: ['onnxruntime-web'],
  },
  // Cross-origin isolation lets ORT use SharedArrayBuffer WASM threads (suggestion 3).
  server: {
    // Apply isolation headers to `npm run dev`.
    headers: {
      // Same-origin opener so the tab can receive a unique process-isolated context.
      'Cross-Origin-Opener-Policy': 'same-origin',
      // Require CORP so SharedArrayBuffer is not blocked.
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Keep the same isolation headers for `vite preview` production-like serving.
  preview: {
    // Mirror the dev-server COOP/COEP pair.
    headers: {
      // Same-origin opener for the preview origin.
      'Cross-Origin-Opener-Policy': 'same-origin',
      // Same CORP requirement as the dev server.
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Run component tests in a browser-like DOM environment.
  test: {
    // Emulate the DOM APIs required by React Testing Library.
    environment: 'jsdom',
    // Register accessible DOM matchers before each test module.
    setupFiles: './src/test/setup.ts',
  },
})
