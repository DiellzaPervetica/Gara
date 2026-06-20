#!/usr/bin/env node
// Run the public Graph Route Validator from the command line without a browser.
// Usage: node tools/official_validator.mjs data/train_a.txt outputs/train_a.out

import fs from "node:fs";

if (process.argv.length !== 4) {
  console.error("Usage: node tools/official_validator.mjs <instance.txt> <submission.out>");
  process.exit(64);
}

const [inputPath, outputPath] = process.argv.slice(2);
const wasmUrl = "https://graph-route-validator.netlify.app/wasm/validator.wasm";
const wasm = await (await fetch(wasmUrl)).arrayBuffer();
const imports = {
  env: new Proxy({}, { get: () => () => 0 }),
  wasi_snapshot_preview1: new Proxy({}, { get: () => () => 0 }),
};
const { instance } = await WebAssembly.instantiate(wasm, imports);
const exports = instance.exports;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function put(text) {
  const bytes = encoder.encode(`${text}\0`);
  const pointer = exports._emscripten_stack_alloc(bytes.length);
  new Uint8Array(exports.memory.buffer).set(bytes, pointer);
  return pointer;
}

function readCString(pointer) {
  const memory = new Uint8Array(exports.memory.buffer);
  let end = pointer;
  while (memory[end] !== 0) end += 1;
  return decoder.decode(memory.slice(pointer, end));
}

const stack = exports.emscripten_stack_get_current();
try {
  const resultPointer = exports.validate(
    put(fs.readFileSync(inputPath, "utf8")),
    put(fs.readFileSync(outputPath, "utf8")),
  );
  const result = JSON.parse(readCString(resultPointer));
  console.log(JSON.stringify(result, null, 2));
  if (result.status !== "VALID") process.exitCode = 2;
} finally {
  exports._emscripten_stack_restore(stack);
}
