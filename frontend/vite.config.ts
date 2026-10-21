// Import React's Vite transform and fast-refresh integration.
import react from '@vitejs/plugin-react'
// Import Vitest-aware Vite configuration typing.
import { defineConfig } from 'vitest/config'
// Import Vite's env loader for the API origin.
import { loadEnv } from 'vite'
// Import the production-only CSP injector.
import { cspPlugin } from './vite.csp.ts'
// Import the plugin that serves public/ml/*.onnx with brotli/gzip Content-Encoding.
import { precompressedOnnxPlugin } from './vite.precompressedOnnx.ts'

// Configure both the browser build and colocated component tests.
export default defineConfig(({ mode }) => ({
  // Transform React 19 TypeScript and enable development fast refresh.
  plugins: [
    react(),
    precompressedOnnxPlugin(),
    // Inject a strict CSP in production builds using the configured API origin.
    cspPlugin(loadEnv(mode, process.cwd(), 'VITE_').VITE_API_BASE_URL ?? 'http://localhost:8000'),
  ],
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
}))
