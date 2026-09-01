// Percentile helper for sequential ONNX Runtime Web load-check timings.

// Compute a linear-interpolated percentile from an unsorted sample list.
export function percentile(samples: number[], p: number): number | null {
  // An empty series cannot produce a percentile.
  if (samples.length === 0) {
    return null
  }
  // Copy before sorting so the caller can still compute a mean from the original.
  const ordered = [...samples].sort((left, right) => left - right)
  // Clamp p into [0, 100] so a typo cannot index off the end.
  const clamped = Math.min(100, Math.max(0, p))
  // Convert the percentile into a (possibly fractional) index.
  const index = (clamped / 100) * (ordered.length - 1)
  const lower = Math.floor(index)
  const upper = Math.ceil(index)
  const lowerValue = ordered[lower]
  const upperValue = ordered[upper]
  if (lowerValue === undefined || upperValue === undefined) {
    return null
  }
  if (lower === upper) {
    return lowerValue
  }
  const weight = index - lower
  return lowerValue * (1 - weight) + upperValue * weight
}
