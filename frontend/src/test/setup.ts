// Register accessible DOM assertions such as toBeInTheDocument for every test.
import '@testing-library/jest-dom/vitest'
// Polyfill IndexedDB in jsdom, which implements no storage APIs itself; the
// key vault (frontend/src/crypto/keyVault.ts) needs a real IndexedDB to exercise.
import 'fake-indexeddb/auto'
// Import cleanup so each component test starts with an empty document.
import { cleanup } from '@testing-library/react'
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

// Remove rendered React trees after each independent test case.
afterEach(() => {
  // Unmount components and clear Testing Library's document containers.
  cleanup()
})
