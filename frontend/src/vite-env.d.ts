/// <reference types="vite/client" />

// Extend Vite's environment typing with the variables this app actually reads.
interface ImportMetaEnv {
  // Declare the API base URL so callers get autocomplete and type-checking.
  readonly VITE_API_BASE_URL?: string
}

// Re-declare import.meta.env against the extended interface above.
interface ImportMeta {
  readonly env: ImportMetaEnv
}
