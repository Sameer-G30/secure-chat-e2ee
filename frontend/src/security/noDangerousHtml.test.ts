// Fail CI when product UI code uses raw HTML injection (spec §11).

// Import Vitest grouping and assertion helpers.
import { describe, expect, it } from 'vitest'

// Load every product TypeScript file as raw text (tests are excluded from the glob).
const productSources = import.meta.glob<string>(
  // Include the React/TypeScript tree under src/.
  ['../**/*.ts', '../**/*.tsx', '!**/*.test.ts', '!**/*.test.tsx'],
  {
    // Evaluate at module load so the assertion can be synchronous.
    eager: true,
    // Read file contents as strings rather than executing the modules.
    query: '?raw',
    // Use the default export of the raw loader (the file text).
    import: 'default',
  },
)

// Spec §11: never render user-controlled strings via dangerouslySetInnerHTML or innerHTML.
describe('§11 HTML injection anti-pattern', () => {
  // Scan product sources bundled as raw strings by Vite.
  it('never uses dangerouslySetInnerHTML or .innerHTML in product TypeScript', () => {
    // Collect offender paths for a readable assertion.
    const offenders: string[] = []
    // Walk every product module the glob loaded.
    for (const [path, source] of Object.entries(productSources)) {
      // Flag React's raw-HTML prop if it appears in product code.
      if (source.includes('dangerouslySetInnerHTML')) {
        // Record the glob path and continue scanning other files.
        offenders.push(`${path}: dangerouslySetInnerHTML`)
      }
      // Flag DOM innerHTML assignment/property access used for injection.
      if (/\.innerHTML\b/.test(source)) {
        // Record the glob path for the assertion message.
        offenders.push(`${path}: .innerHTML`)
      }
    }
    // Require a clean product tree; user-controlled strings stay React text.
    expect(offenders).toEqual([])
  })
})
