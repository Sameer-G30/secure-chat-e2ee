// Headless Chromium runner for the Slice 6 sequential ORT Web load-check.
// Opens /?mlLoadCheck=1, waits for all six rows, writes JSON for the README table.
// Usage (frontend/): npm run dev  (other terminal)
//   node scripts/ort_web_load_check.mjs
// Requires playwright (npx --yes playwright install chromium).

// Import filesystem helpers to persist the measurement JSON.
import fs from 'node:fs'
// Import path helpers to locate the reports directory.
import path from 'node:path'
// Import fileURLToPath so this ESM script can resolve the repo root.
import { fileURLToPath } from 'node:url'
// Import Playwright Chromium; the load-check must run in a real browser WASM heap.
import { chromium } from 'playwright'

// This script lives at frontend/scripts/, or may be copied next to a temp playwright install.
const scriptDir = path.dirname(fileURLToPath(import.meta.url))
// Repo root: env override when this file is copied to /tmp for a one-off Chromium run.
const repoRoot = process.env.SECURE_CHAT_ROOT
  ? path.resolve(process.env.SECURE_CHAT_ROOT)
  : path.resolve(scriptDir, '..', '..')
// Persist JSON beside the markdown table (do not overwrite metric JSON).
const jsonOut = path.join(repoRoot, 'ml', 'reports', 'onnx_web_load_check.json')
// Vite dev server serving public/ml artifacts.
const url = process.env.ORT_WEB_LOAD_CHECK_URL || 'http://127.0.0.1:5173/?mlLoadCheck=1'
// DistilBERT 512 init + four padded inferences can take minutes in WASM.
const timeoutMs = Number(process.env.ORT_WEB_LOAD_CHECK_TIMEOUT_MS || 10 * 60 * 1000)

// Launch Chromium, run the in-page sequential checker, write JSON.
const browser = await chromium.launch({
  // Headless is enough; the page auto-starts the six-way sequence on mount.
  headless: true,
  // Give WASM room for DistilBERT int8 graphs (~65 MiB each plus activations).
  args: ['--js-flags=--max-old-space-size=8192'],
})
// One tab, same as a reviewer opening ?mlLoadCheck=1.
const page = await browser.newPage()
// Capture page errors so a WASM abort is visible in the JSON sidecar.
const pageErrors = []
// Record Content-Encoding and Content-Length for /ml/**/*.onnx (not .gz/.br URLs).
const onnxTransfers = []
// Record uncaught exceptions without aborting the six-way sequence.
page.on('pageerror', (error) => {
  // Store the message; per-row failures are also in the table.
  pageErrors.push(error.message)
})
// Observe the DistilBERT/LSTM/TF-IDF graph fetches so download size is measured.
page.on('response', (response) => {
  // Only checkpoint graphs; sidecars are JSON.
  const responseUrl = response.url()
  // Skip if this is not an ONNX graph request.
  if (!responseUrl.includes('/ml/') || !responseUrl.includes('.onnx')) {
    // Ignore WASM, JSON, and HTML.
    return
  }
  // Ignore direct fetches of the sibling files (the tab still requests model.onnx).
  if (responseUrl.endsWith('.onnx.gz') || responseUrl.endsWith('.onnx.br')) {
    // Those URLs are not what ChatScreen loads.
    return
  }
  // Skip query-string-only matches that are not serving graphs.
  if (!/\.onnx(\?|$)/.test(responseUrl)) {
    // Not a serving graph.
    return
  }
  // Read negotiated encoding and compressed length from response headers.
  const headers = response.headers()
  // Push one transfer record for the JSON sidecar.
  onnxTransfers.push({
    // Full URL the tab requested (still /model.onnx).
    url: responseUrl,
    // br, gzip, or identity when the Vite plugin fell through.
    contentEncoding: headers['content-encoding'] || 'identity',
    // Compressed byte count when Content-Encoding is set.
    contentLength: headers['content-length'] ? Number(headers['content-length']) : null,
    // HTTP status so a 404 is visible beside a failed load row.
    status: response.status(),
  })
})
try {
  // Load the measurement page; Vite must already be serving public/ml.
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })
  // Wait until the in-page runner finishes all six ids (or records skips).
  await page.getByRole('status').filter({ hasText: 'Sequence finished' }).waitFor({
    timeout: timeoutMs,
  })
  // Read the textarea the UI fills for pasting into the markdown report.
  const raw = await page.locator('#load-check-json').inputValue()
  // Parse so a truncated log fails here instead of silently writing garbage.
  const rows = JSON.parse(raw)
  // Bundle page errors with the six rows for the markdown write-up.
  const payload = { url, pageErrors, rows, onnxTransfers }
  // Ensure reports/ exists (it already does in this repo).
  fs.mkdirSync(path.dirname(jsonOut), { recursive: true })
  // Pretty JSON matches the textarea the operator would copy by hand.
  fs.writeFileSync(jsonOut, `${JSON.stringify(payload, null, 2)}\n`)
  // Print a one-line summary per checkpoint for the terminal log.
  for (const row of rows) {
    // Show load success and latency without dumping the full JSON twice.
    console.log(
      `${row.loadOrder} ${row.id} load=${row.loadSuccess} initMs=${row.initMs} inferMs=${row.inferenceMsPerMessage} onnx=${row.onnxBytes}`,
    )
  }
  // Print negotiated encodings so the README can cite compressed download size.
  for (const transfer of onnxTransfers) {
    // One line per ONNX GET (DistilBERT int8 should be contentEncoding=br).
    console.log(
      `onnx ${transfer.url} encoding=${transfer.contentEncoding} contentLength=${transfer.contentLength} status=${transfer.status}`,
    )
  }
  // Point the operator at the file README Slice 6 will cite.
  console.log(`wrote ${jsonOut}`)
} finally {
  // Always close Chromium so a hung WASM session cannot leak the process.
  await browser.close()
}
