const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Sha256 = require('../static/sha256.js');

function hashInChunks(input, chunkSizes) {
  const hash = new Sha256();
  let offset = 0, index = 0;
  while (offset < input.length) {
    const end = Math.min(input.length, offset + chunkSizes[index++ % chunkSizes.length]);
    hash.update(input.subarray(offset, end));
    offset = end;
  }
  return hash.digestHex();
}

async function workerHash(input) {
  const messages = [];
  const sandbox = {
    importScripts() { sandbox.StereoLabSha256 = Sha256; },
    postMessage(message) { messages.push(message); },
  };
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/source_hash_worker.js'), 'utf8'), sandbox);
  const file = {
    size: input.length,
    slice(start, end) {
      const bytes = Uint8Array.from(input.subarray(start, end));
      return {arrayBuffer: async () => bytes.buffer};
    },
  };
  await sandbox.onmessage({data: {file}});
  return messages;
}

async function main() {
  assert.equal(hashInChunks(Buffer.alloc(0), [1]), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
  assert.equal(hashInChunks(Buffer.from('abc'), [1]), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(hashInChunks(Buffer.from('The quick brown fox jumps over the lazy dog'), [2, 7, 1, 13]), 'd7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592');
  const data = Buffer.alloc(2 * 1024 * 1024 + 137);
  for (let index = 0; index < data.length; index++) data[index] = (index * 31 + 17) & 255;
  const expected = crypto.createHash('sha256').update(data).digest('hex');
  assert.equal(hashInChunks(data, [1, 63, 64, 65, 4097]), expected, 'Incremental updates match Node SHA-256 across block boundaries');
  const messages = await workerHash(data);
  assert.equal(messages.filter(message => typeof message.progress === 'number').length, 3, 'Worker reads in bounded 1 MiB chunks');
  assert.equal(messages.at(-1).sha256, expected);
  assert.equal(messages.at(-1).size_bytes, data.length);
  console.log('PASS source SHA-256 vectors and bounded worker streaming');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
