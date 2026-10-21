// Switching graphs unloads the current session before the next one is created.

// Import Vitest helpers.
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Shared counters the ort mock updates. vi.hoisted so the mock factory can see them.
const holders = vi.hoisted(() => {
  // One resident graph, plus how many sessions are alive.
  const state: {
    heavy: { id: string; family: string; session: { release: () => Promise<void> } } | null
    live: number
    peak: number
  } = { heavy: null, live: 0, peak: 0 }
  // Reset between cases.
  return { state }
})

// Replace WASM session create/release so this test never loads onnxruntime.
vi.mock('./ortRuntime', () => ({
  OrtTensor: class OrtTensor {},
  createSessionFromBuffer: vi.fn(async () => {
    // A new graph is alive until release.
    holders.state.live += 1
    // Remember the high-water mark. Two overlapping creates push this to 2.
    holders.state.peak = Math.max(holders.state.peak, holders.state.live)
    // The session object register/unload will release.
    return {
      release: async () => {
        // Dispose one graph.
        holders.state.live -= 1
      },
    }
  }),
  getHeavySession: vi.fn(() => holders.state.heavy),
  registerHeavySession: vi.fn(async (id: string, family: string, session: { release: () => Promise<void> }) => {
    // The real helper unloads the previous heavy session before keeping the new one.
    if (holders.state.heavy) {
      // Drop the previous graph.
      await holders.state.heavy.session.release()
    }
    // family must be distilbert or the loader will not treat the graph as resident.
    holders.state.heavy = { id, family, session }
  }),
  unloadAllSessions: vi.fn(async () => {
    // Dispose the resident graph when a switch starts.
    if (holders.state.heavy) {
      // Free it before the next create.
      await holders.state.heavy.session.release()
      // The slot is empty.
      holders.state.heavy = null
    }
  }),
}))

// Sidecars are tiny fixtures. No Cache API and no model download.
vi.mock('./modelCache', () => ({
  readOrFetchBytes: vi.fn(async (_url: string, filename: string, _version: number | string, id: string) => {
    // WordPiece JSON is enough for the loader to parse. Inference is not run here.
    if (filename.endsWith('.json')) {
      // max_length follows the checkpoint so both graphs share this tokenizer shape.
      const vocab = {
        tokens: ['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        unk_token: '[UNK]',
        cls_token: '[CLS]',
        sep_token: '[SEP]',
        pad_token: '[PAD]',
        unk_id: 1,
        cls_id: 2,
        sep_id: 3,
        pad_id: 0,
        do_lower_case: true,
        max_length: id === 'distilbert_best' ? 512 : 128,
        max_input_chars_per_word: 100,
      }
      // The loader decodes these bytes as JSON.
      return new TextEncoder().encode(JSON.stringify(vocab)).buffer
    }
    // model.onnx bytes are not parsed by the mock session.
    return new ArrayBuffer(8)
  }),
}))

// Import after the mocks so the loader binds to them.
import { createSessionFromBuffer } from './ortRuntime'
import { ensureScamModel, resetAllClassifiers } from './scamClassifier'

// A manifest the loader accepts. Thresholds stay the frozen sidecar values.
function manifest(id: string, maxLength: number, threshold: number) {
  // Fields loadDistilbertCheckpoint reads.
  return {
    id,
    load_order: 1,
    family: 'distilbert',
    label: id,
    threshold,
    max_length: maxLength,
    onnx_file: 'model.onnx',
    onnx_bytes: 8,
    sha256: id,
    quantize: 'int8',
    sidecars: { wordpiece: 'wordpiece_vocab.json', fixtures: 'fixture_scores.json' },
    artifact_bytes: { onnx: 8 },
    offline: {
      reports_dir: 'reports',
      test_accuracy: 0,
      test_fn: 0,
      test_fp: 0,
      chat_accuracy: null,
      chat_fn: null,
      chat_fp: null,
      note: '',
    },
    wired_in_chatscreen_by_default: false,
  }
}

// JSON fetch response.
function jsonResponse(body: unknown) {
  // ok and json() are all the loader uses.
  return { ok: true, status: 200, json: async () => body }
}

// Install a fetch that serves the three chat manifests.
function stubManifests(missing: string[] = []) {
  // Replace global fetch for this case.
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      // The path contains the catalog id.
      const target = String(url)
      // A missing export is how the student falls back, and how best fails closed.
      if (missing.some((id) => target.includes(id))) {
        // The loader throws on a non-ok manifest.
        return { ok: false, status: 404, json: async () => ({}) }
      }
      // Student manifest.
      if (target.includes('distilbert_student')) {
        // Frozen threshold 0.25, max_length 128.
        return jsonResponse(manifest('distilbert_student', 128, 0.25))
      }
      // Best manifest.
      if (target.includes('distilbert_best')) {
        // Frozen threshold 0.20, max_length 512.
        return jsonResponse(manifest('distilbert_best', 512, 0.2))
      }
      // Fallback manifest.
      if (target.includes('distilbert_default')) {
        // The historical fallback when the student folder is absent.
        return jsonResponse(manifest('distilbert_default', 256, 0.3))
      }
      // Anything else is a missing export.
      return { ok: false, status: 404, json: async () => ({}) }
    }),
  )
}

describe('scam model switch', () => {
  // Drop any graph the previous case left behind.
  beforeEach(async () => {
    // Counters start at zero.
    holders.state.heavy = null
    holders.state.live = 0
    holders.state.peak = 0
    // Module state (vocab, serial latch) resets through the public helper.
    await resetAllClassifiers()
    // create counts are per case.
    vi.mocked(createSessionFromBuffer).mockClear()
  })

  // Student then best: the first graph is released before the second create.
  it('unloads the student before loading distilbert_best', async () => {
    // Both exports exist.
    stubManifests()
    // Load the default.
    await expect(ensureScamModel('distilbert_student')).resolves.toBe('distilbert_student')
    // One session.
    expect(holders.state.heavy?.id).toBe('distilbert_student')
    expect(holders.state.live).toBe(1)
    // Switch. This must not leave the student resident.
    await expect(ensureScamModel('distilbert_best')).resolves.toBe('distilbert_best')
    // Only best is resident.
    expect(holders.state.heavy?.id).toBe('distilbert_best')
    expect(holders.state.live).toBe(1)
    // The creates never overlapped.
    expect(holders.state.peak).toBe(1)
    // Asking for best again does not compile a second copy.
    await ensureScamModel('distilbert_best')
    expect(vi.mocked(createSessionFromBuffer)).toHaveBeenCalledTimes(2)
  })

  // A missing student export loads the fallback, still as the only session.
  it('falls back to distilbert_default when the student export is missing', async () => {
    // Student 404, default present.
    stubManifests(['distilbert_student'])
    // The choice stays "student" from Settings; the resident id is the fallback.
    await expect(ensureScamModel('distilbert_student')).resolves.toBe('distilbert_default')
    // One graph.
    expect(holders.state.heavy?.id).toBe('distilbert_default')
    expect(holders.state.peak).toBe(1)
    expect(holders.state.live).toBe(1)
  })

  // Best has no silent fallback. Failure leaves chat with no resident graph.
  it('leaves no graph resident when distilbert_best fails to load', async () => {
    // Best 404.
    stubManifests(['distilbert_best'])
    // The promise rejects. The hook turns banners off.
    await expect(ensureScamModel('distilbert_best')).rejects.toThrow(/distilbert_best/)
    // Nothing left in the heavy slot.
    expect(holders.state.heavy).toBeNull()
    expect(holders.state.live).toBe(0)
  })
})
