// Register accessible DOM assertions such as toBeInTheDocument for every test.
import '@testing-library/jest-dom/vitest'
// Polyfill IndexedDB in jsdom, which implements no storage APIs itself; the
// key vault (frontend/src/crypto/keyVault.ts) needs a real IndexedDB to exercise.
import 'fake-indexeddb/auto'
// Import cleanup so each component test starts with an empty document.
import { cleanup } from '@testing-library/react'
// Import the session claim reset so one test's setSession cannot skip the next restore.
import { resetInMemorySessionClaim } from '../context/sessionClaim'
// Import Vitest's post-test lifecycle hook and module mock helper.
import { afterEach, vi } from 'vitest'

// Stub ONNX Runtime Web so jsdom tests never fetch WASM or .onnx graphs.
vi.mock('onnxruntime-web/wasm', () => {
  // Minimal Tensor stand-in used if a test constructs one by accident.
  class FakeTensor {
    // ORT tensor element type string (float32, int64, ...).
    type: string
    // Backing numeric buffer.
    data: ArrayLike<number>
    // Tensor shape.
    dims: number[]
    // Record the type/dims ORT would have used.
    constructor(type: string, data: ArrayLike<number>, dims: number[]) {
      // Store the dtype string.
      this.type = type
      // Store the backing buffer.
      this.data = data
      // Store the shape.
      this.dims = dims
    }
  }
  return {
    InferenceSession: { create: vi.fn() },
    Tensor: FakeTensor,
    env: {
      // Match ortRuntime.ts: proxy + SIMD + Vite-processed wasm/mjs URLs (not public/ort).
      wasm: {
        // Keep the default single-thread mock used by jsdom tests.
        numThreads: 1,
        // Mirror the production SIMD flag so assignments in ortRuntime succeed.
        simd: true,
        // Mirror the production worker flag so assignments in ortRuntime succeed.
        proxy: true,
        // Object form matches Env.WasmFilePaths; string `/ort/` is invalid under Vite 8.
        wasmPaths: {
          // Dummy .wasm URL; ortRuntime overwrites this with the Vite asset path.
          wasm: '/mock-ort.wasm',
          // Dummy .mjs URL; ortRuntime overwrites this with the Vite asset path.
          mjs: '/mock-ort.mjs',
        },
      },
    },
  }
})

// Delete one IndexedDB database and wait until the request settles.
function deleteDatabase(name: string): Promise<void> {
  // Resolve even on error so a missing database cannot hang the suite.
  return new Promise((resolve) => {
    const request = indexedDB.deleteDatabase(name)
    request.onsuccess = () => {
      resolve()
    }
    request.onerror = () => {
      resolve()
    }
    request.onblocked = () => {
      resolve()
    }
  })
}

// Remove rendered React trees after each independent test case.
afterEach(async () => {
  // Unmount components and clear Testing Library's document containers.
  cleanup()
  // Drop last-message previews so ChatScreen tests cannot leak snippets across cases.
  await deleteDatabase('secure-chat-previews')
  // Drop a restorable session so the next test does not rotate a leftover refresh token.
  await deleteDatabase('secure-chat-session')
  // Let the next AuthProvider mount try restore instead of seeing a stale claim.
  resetInMemorySessionClaim()
})
