// Vite plugin that injects a strict Content-Security-Policy meta tag into production builds only.
// Dev keeps no CSP because Vite HMR and React refresh rely on inline scripts.

// Import the Plugin type so the factory is typed.
import type { Plugin } from 'vite'

// Build the policy string for one API origin (for example http://localhost:8000).
export function buildCsp(apiBaseUrl: string): string {
  // Derive the matching WebSocket origin (http -> ws, https -> wss).
  const wsOrigin = apiBaseUrl.replace(/^http/, 'ws')
  // Join directives; each is deliberately minimal.
  return [
    // Everything defaults to same-origin only.
    "default-src 'self'",
    // Scripts: same origin plus WASM compilation (ORT, libsodium); blob: for ORT's proxy worker.
    "script-src 'self' 'wasm-unsafe-eval' blob:",
    // Workers: ORT proxy worker is created from a blob URL.
    "worker-src 'self' blob:",
    // Styles: React sets style attributes at runtime, so inline styles are allowed (no inline script).
    "style-src 'self' 'unsafe-inline'",
    // Network: only this app origin and the API (REST and WebSocket).
    `connect-src 'self' ${apiBaseUrl} ${wsOrigin}`,
    // Images: own assets plus decrypted blob/data previews.
    "img-src 'self' blob: data:",
    // Plugins are never needed.
    "object-src 'none'",
    // Block <base> hijacking.
    "base-uri 'none'",
    // Forms may only post to this origin.
    "form-action 'self'",
  ].join('; ')
}

// Create the plugin; apiBaseUrl comes from VITE_API_BASE_URL at build time.
export function cspPlugin(apiBaseUrl: string): Plugin {
  return {
    name: 'secure-chat-csp',
    // Only transform the HTML for `vite build`, not the dev server.
    apply: 'build',
    transformIndexHtml() {
      // Insert the meta as the first head element so it applies before any script loads.
      return [
        {
          tag: 'meta',
          attrs: { 'http-equiv': 'Content-Security-Policy', content: buildCsp(apiBaseUrl) },
          injectTo: 'head-prepend',
        },
      ]
    },
  }
}
