// Import Vitest grouping and assertion helpers.
import { describe, expect, it } from 'vitest'

// Import the percentile helper used by the in-tab load-check table.
import { percentile } from './percentiles'

describe('percentile', () => {
  it('returns null for an empty series', () => {
    expect(percentile([], 50)).toBeNull()
  })

  it('returns the only value at every percentile', () => {
    expect(percentile([4], 50)).toBe(4)
    expect(percentile([4], 99)).toBe(4)
  })

  it('interpolates p50 of an even-length series', () => {
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5)
  })
})
