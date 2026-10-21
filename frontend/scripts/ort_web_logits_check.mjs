// Load an int8 ONNX in onnxruntime-web (WASM, node) and print logits for a fixed input.
import * as ort from 'onnxruntime-web/wasm'
import fs from 'node:fs'
const [,, file] = process.argv
ort.env.wasm.numThreads = 1
const buf = fs.readFileSync(file)
const t0 = performance.now()
const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] })
const t1 = performance.now()
// "hello" style ids: [CLS] 7592 2088 [SEP]
const ids = [101, 7592, 2088, 102]
const feeds = {
  input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
  attention_mask: new ort.Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]),
}
const out = await session.run(feeds)
const t2 = performance.now()
console.log(file.split('/').slice(-2).join('/'), 'load', Math.round(t1 - t0), 'ms run', Math.round(t2 - t1), 'ms logits', Array.from(out.logits.data))
