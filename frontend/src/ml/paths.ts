// Public URLs for the six exported ONNX Runtime Web checkpoints.

// Import the catalog ids so Vite public paths stay aligned with the Python export.
import type { CheckpointId } from './types'

// Vite serves frontend/public/ml/<id>/ as /ml/<id>/.
const PUBLIC_ML_ROOT = '/ml'

// Return the folder URL for one checkpoint (no trailing slash).
export function checkpointBaseUrl(id: CheckpointId): string {
  // Each export_dirname matches the catalog id.
  return `${PUBLIC_ML_ROOT}/${id}`
}

// Return the manifest URL the load-check fetches first.
export function manifestUrl(id: CheckpointId): string {
  // manifest.json lists onnx_file, sidecars, threshold, and offline metrics.
  return `${checkpointBaseUrl(id)}/manifest.json`
}

// Resolve a sidecar or ONNX filename against the checkpoint folder.
export function checkpointAssetUrl(id: CheckpointId, filename: string): string {
  // Filenames come from the manifest and are never concatenated from user text.
  return `${checkpointBaseUrl(id)}/${filename}`
}
