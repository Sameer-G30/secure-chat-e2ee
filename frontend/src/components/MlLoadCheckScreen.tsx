// Sequential six-way ONNX Runtime Web load-check UI (open with ?mlLoadCheck=1).

// Import React state for incremental table rows.
import { useEffect, useState } from 'react'

// Import the sequential checker that loads one checkpoint at a time.
import { runSequentialLoadCheck, unsubscribeLoadCheck } from '../ml/loadCheck'
import type { LoadCheckRow } from '../ml/types'

// Format bytes for the cost table.
function formatBytes(bytes: number | null): string {
  // Missing measurements render as n/a (Firefox has no performance.memory).
  if (bytes === null) {
    // Keep the table cell compact.
    return 'n/a'
  }
  // Show mebibytes for DistilBERT-sized graphs.
  if (bytes >= 1024 * 1024) {
    // One decimal is enough for a README table.
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
  }
  // Show kibibytes for TF-IDF heads.
  if (bytes >= 1024) {
    // Round to a whole kibibyte.
    return `${Math.round(bytes / 1024)} KiB`
  }
  // Show raw bytes for tiny artifacts.
  return `${bytes} B`
}

// Format milliseconds for init / inference columns.
function formatMs(ms: number | null): string {
  // Failed loads have no timing.
  if (ms === null) {
    // Keep the table cell compact.
    return 'n/a'
  }
  // One decimal millisecond is enough for the log.
  return `${ms.toFixed(1)} ms`
}

// Insert or replace a row by checkpoint id so StrictMode remounts cannot duplicate.
function upsertRow(existing: LoadCheckRow[], row: LoadCheckRow): LoadCheckRow[] {
  // Find a row already recorded for this catalog id.
  const index = existing.findIndex((item) => item.id === row.id)
  // Replace in place when the singleton replayed or re-emitted this id.
  if (index >= 0) {
    // Copy the array so React sees a new reference.
    const next = existing.slice()
    // Overwrite the stale placeholder with the finished measurement.
    next[index] = row
    // Return the updated table.
    return next
  }
  // Append a newly completed checkpoint in load order.
  return [...existing, row]
}

// Render the six-row measurement table used to pick the ChatScreen default.
export function MlLoadCheckScreen() {
  // Hold rows as they complete so a 512 OOM still shows later successes.
  const [rows, setRows] = useState<LoadCheckRow[]>([])
  // Hold whether the sequence is still running.
  const [running, setRunning] = useState(true)
  // Hold a top-level error distinct from per-row failures.
  const [error, setError] = useState<string | null>(null)

  // Subscribe to the singleton six-way check (StrictMode remounts join, not restart).
  useEffect(() => {
    // Ignore setState after this tree unmounts (StrictMode recycle or navigation).
    let cancelled = false
    // Same callback identity so cleanup can unsubscribe from the singleton runner.
    const onRow = (row: LoadCheckRow) => {
      // Skip writes after unmount; the remounted tree has its own callback.
      if (cancelled) {
        // The singleton still runs; only this React tree is gone.
        return
      }
      // Upsert by id; StrictMode remount replays completed rows.
      setRows((existing) => upsertRow(existing, row))
    }
    // Subscribe; a second StrictMode mount joins the in-flight WASM sequence.
    void runSequentialLoadCheck(onRow)
      .then(() => {
        // Skip if this tree unmounted before the shared run finished.
        if (cancelled) {
          // The remounted tree's own then() will clear the spinner.
          return
        }
        // Mark the sequence finished so the copy-JSON button is useful.
        setRunning(false)
      })
      .catch((caught: unknown) => {
        // Skip if this tree unmounted before the shared run finished.
        if (cancelled) {
          // Avoid setState on an unmounted measurement page.
          return
        }
        // Surface a runner crash (should be rare; per-row errors are in the table).
        setError(caught instanceof Error ? caught.message : String(caught))
        // Stop the spinner.
        setRunning(false)
      })
    // Drop this tree's callback on unmount so it cannot setState after StrictMode recycle.
    return () => {
      // Flip first so in-flight then/catch cannot write.
      cancelled = true
      // Unsubscribe without aborting the shared six-way WASM run.
      unsubscribeLoadCheck(onRow)
    }
  }, [])

  // Serialize the table for pasting into ml/reports/onnx_web_load_check.md.
  const json = JSON.stringify(rows, null, 2)

  return (
    <main className="load-check-screen">
      <h1>Slice 6 ONNX Runtime Web load check</h1>
      <p>
        Loads DistilBERT 512 → DistilBERT 256 → BiLSTM 8ep → BiLSTM 4ep → TF-IDF 10k → TF-IDF 50k,
        one at a time, then unloads. This is not TEST accuracy. Classification stays on-device after
        decrypt.
      </p>
      {running ? <p role="status">Running sequential check…</p> : <p role="status">Sequence finished.</p>}
      {error ? (
        <p className="auth-feedback auth-feedback-error" role="alert">
          {error}
        </p>
      ) : null}
      <table className="load-check-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Checkpoint</th>
            <th>Load</th>
            <th>Init</th>
            <th>Infer / msg</th>
            <th>p50</th>
            <th>p95</th>
            <th>p99</th>
            <th>ONNX</th>
            <th>JS heap</th>
            <th>Fixture banners</th>
            <th>Offline TEST FN/FP</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.loadOrder}</td>
              <td>{row.label}</td>
              <td>
                {row.loadSuccess
                  ? row.error
                    ? `yes (inference: ${row.error})`
                    : 'yes'
                  : `no: ${row.error ?? 'unknown'}`}
              </td>
              <td>{formatMs(row.initMs)}</td>
              <td>{formatMs(row.inferenceMsPerMessage)}</td>
              <td>{formatMs(row.inferenceMsP50)}</td>
              <td>{formatMs(row.inferenceMsP95)}</td>
              <td>{formatMs(row.inferenceMsP99)}</td>
              <td>{formatBytes(row.onnxBytes)}</td>
              <td>{formatBytes(row.jsHeapBytes)}</td>
              <td>
                {row.fixtureBannerMatch
                  ? row.fixtureBannerMatch.every(Boolean)
                    ? 'match'
                    : `mismatch ${row.fixtureBannerMatch.map((ok) => (ok ? 'y' : 'n')).join('')}`
                  : 'n/a'}
              </td>
              <td>{row.offline ? `${row.offline.test_fn} / ${row.offline.test_fp}` : 'n/a'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <label className="load-check-json-label" htmlFor="load-check-json">
        JSON log (paste into ml/reports/onnx_web_load_check.md)
      </label>
      <textarea id="load-check-json" className="load-check-json" readOnly value={json} />
    </main>
  )
}
