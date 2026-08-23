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
    env: { wasm: { numThreads: 1, simd: true } },
  }
})

// Remove rendered React trees after each independent test case.
afterEach(() => {
  // Unmount components and clear Testing Library's document containers.
  cleanup()
})
