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

// Explicit URL import for the ORT SIMD-threaded WebAssembly binary.
declare module 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url' {
  // Vite rewrites this default export to a served asset URL string.
  const src: string
  // Export the URL so ortRuntime can assign env.wasm.wasmPaths.wasm.
  export default src
}

// Explicit URL import for the ORT Emscripten factory (not a public/ file).
declare module 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url' {
  // Vite rewrites this default export to a served asset URL string.
  const src: string
  // Export the URL so ortRuntime can assign env.wasm.wasmPaths.mjs.
  export default src
}
