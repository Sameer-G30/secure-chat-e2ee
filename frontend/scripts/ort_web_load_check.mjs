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
// Record uncaught exceptions without aborting the six-way sequence.
page.on('pageerror', (error) => {
  // Store the message; per-row failures are also in the table.
  pageErrors.push(error.message)
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
  const payload = { url, pageErrors, rows }
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
  // Point the operator at the file README Slice 6 will cite.
  console.log(`wrote ${jsonOut}`)
} finally {
  // Always close Chromium so a hung WASM session cannot leak the process.
  await browser.close()
}
