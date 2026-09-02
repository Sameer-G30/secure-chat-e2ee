// Serve public/ml/*.onnx with brotli or gzip when sibling .br/.gz files exist.
// fetch('/ml/.../model.onnx') still works; the browser decompresses via Content-Encoding.

// Import filesystem helpers to read precompressed ONNX siblings.
import fs from 'node:fs'
// Import path helpers to resolve URLs against frontend/public.
import path from 'node:path'
// Import IncomingMessage typing for Accept-Encoding negotiation.
import type { IncomingMessage, ServerResponse } from 'node:http'
// Import Vite's plugin type so this file is a first-class Vite plugin.
import type { Plugin, ViteDevServer } from 'vite'

// Filename suffix of the Python-written gzip sibling (model.onnx.gz).
const GZIP_SUFFIX = '.gz'
// Filename suffix of the Python-written brotli sibling (model.onnx.br).
const BROTLI_SUFFIX = '.br'

// Parse Accept-Encoding into a set of tokens, ignoring q-values.
function acceptedEncodings(header: string | undefined): Set<string> {
  // Missing header means the client did not advertise compression.
  if (!header) {
    // Return an empty set so the middleware falls through to uncompressed bytes.
    return new Set()
  }
  // Split on commas, drop ";q=…" parameters, and lowercase each token.
  return new Set(
    header.split(',').map((part) => part.split(';')[0]?.trim().toLowerCase() ?? ''),
  )
}

// Map a /ml/.../model.onnx request to the file under frontend/public.
function publicOnnxPath(urlPath: string, publicDir: string): string | null {
  // Ignore query strings so cache-busting params still hit the same file.
  const pathname = urlPath.split('?')[0] ?? ''
  // Only intercept checkpoint graphs, never JSON sidecars.
  if (!pathname.startsWith('/ml/') || !pathname.endsWith('.onnx')) {
    // Let Vite serve every other public asset as usual.
    return null
  }
  // Reject path traversal before joining onto public/.
  if (pathname.includes('..')) {
    // A crafted URL must not escape the public directory.
    return null
  }
  // Resolve the uncompressed ONNX path the catalog still names in manifest.json.
  return path.join(publicDir, pathname.slice(1))
}

// Write a compressed body with the headers COEP-isolated ORT fetches need.
function sendCompressed(
  req: IncomingMessage,
  res: ServerResponse,
  body: Buffer,
  encoding: 'br' | 'gzip',
): void {
  // Tell the browser how to decode the body before handing bytes to ORT.
  res.setHeader('Content-Encoding', encoding)
  // Vary so caches do not mix br/gzip/identity responses.
  res.setHeader('Vary', 'Accept-Encoding')
  // ORT consumes opaque ONNX bytes after decompression.
  res.setHeader('Content-Type', 'application/octet-stream')
  // COEP require-corp needs an explicit CORP on this response.
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin')
  // Content-Length is the compressed size (the download the user pays for).
  res.setHeader('Content-Length', String(body.length))
  // HEAD must not include a body; GET sends the precompressed payload.
  res.end(req.method === 'HEAD' ? undefined : body)
}

// Handle one GET/HEAD for a public ONNX graph, or call next().
function tryServePrecompressedOnnx(
  req: IncomingMessage,
  res: ServerResponse,
  publicDir: string,
  next: () => void,
): void {
  // Only rewrite GET/HEAD; POST is unused for static graphs.
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    // Continue the Vite middleware chain.
    next()
    // Stop this handler.
    return
  }
  // Resolve the uncompressed ONNX file the manifest still points at.
  const onnxPath = publicOnnxPath(req.url ?? '', publicDir)
  // Non-ONNX URLs are Vite's problem.
  if (onnxPath === null) {
    // Continue the chain.
    next()
    // Stop this handler.
    return
  }
  // Skip missing exports so the load-check can record a 404 skip.
  if (!fs.existsSync(onnxPath)) {
    // Vite will 404 the uncompressed path the same way as before.
    next()
    // Stop this handler.
    return
  }
  // Negotiate brotli first, then gzip, matching common CDN policy.
  const accepted = acceptedEncodings(req.headers['accept-encoding'])
  // Prefer the smaller brotli sibling when the client listed "br".
  const brotliPath = `${onnxPath}${BROTLI_SUFFIX}`
  // Gzip sibling is the fallback for older clients.
  const gzipPath = `${onnxPath}${GZIP_SUFFIX}`
  // Serve brotli when both the client and the sibling exist.
  if (accepted.has('br') && fs.existsSync(brotliPath)) {
    // Read the Python-written .br file (quality 11).
    const body = fs.readFileSync(brotliPath)
    // GET or HEAD with Content-Encoding: br.
    sendCompressed(req, res, body, 'br')
    // Stop this handler.
    return
  }
  // Serve gzip when brotli was not chosen.
  if (accepted.has('gzip') && fs.existsSync(gzipPath)) {
    // Read the Python-written .gz file (level 9).
    const body = fs.readFileSync(gzipPath)
    // GET or HEAD with Content-Encoding: gzip.
    sendCompressed(req, res, body, 'gzip')
    // Stop this handler.
    return
  }
  // No matching sibling: Vite serves uncompressed model.onnx.
  next()
}

// Attach the middleware to a Vite dev or preview server.
function attach(server: ViteDevServer, publicDir: string): void {
  // Register before Vite's static handler so we can set Content-Encoding.
  server.middlewares.use((req, res, next) => {
    // Attempt a compressed ONNX serve, otherwise continue.
    tryServePrecompressedOnnx(req, res, publicDir, next)
  })
}

// Export the Vite plugin factory used from vite.config.ts.
export function precompressedOnnxPlugin(): Plugin {
  // Return a plugin object Vite 8 understands.
  return {
    // Stable name for Vite's plugin list and error messages.
    name: 'precompressed-onnx',
    // Dev server: npm run dev.
    configureServer(server) {
      // publicDir is frontend/public, where export_onnx_web copies graphs.
      attach(server, server.config.publicDir)
    },
    // Preview server: npm run preview (production-like static serve).
    configurePreviewServer(server) {
      // Preview still needs Content-Encoding; dist copies public/ml as-is.
      attach(server as unknown as ViteDevServer, server.config.publicDir)
    },
  }
}
