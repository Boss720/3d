(function (root, factory) {
  const Sha256 = factory();
  if (typeof module === "object" && module.exports) module.exports = Sha256;
  root.StereoLabSha256 = Sha256;
})(globalThis, function () {
  "use strict";
  const K = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
  ]);
  const INITIAL = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  const rotateRight = (value, count) => (value >>> count) | (value << (32 - count));

  class Sha256 {
    constructor() {
      this.state = new Uint32Array(INITIAL);
      this.buffer = new Uint8Array(64);
      this.bufferLength = 0;
      this.bytesHashed = 0;
      this.finished = false;
    }

    update(input) {
      if (this.finished) throw new Error("SHA-256 has already been finalized.");
      const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
      if (!Number.isSafeInteger(this.bytesHashed + bytes.length)) throw new Error("SHA-256 input is too large.");
      this.bytesHashed += bytes.length;
      let offset = 0;
      if (this.bufferLength) {
        const count = Math.min(64 - this.bufferLength, bytes.length);
        this.buffer.set(bytes.subarray(0, count), this.bufferLength);
        this.bufferLength += count;
        offset += count;
        if (this.bufferLength === 64) { this._compress(this.buffer, 0); this.bufferLength = 0; }
      }
      while (offset + 64 <= bytes.length) { this._compress(bytes, offset); offset += 64; }
      if (offset < bytes.length) {
        this.buffer.set(bytes.subarray(offset), 0);
        this.bufferLength = bytes.length - offset;
      }
      return this;
    }

    _compress(block, offset) {
      const words = new Uint32Array(64);
      for (let index = 0; index < 16; index++) {
        const start = offset + index * 4;
        words[index] = ((block[start] << 24) | (block[start + 1] << 16) | (block[start + 2] << 8) | block[start + 3]) >>> 0;
      }
      for (let index = 16; index < 64; index++) {
        const left = words[index - 15], right = words[index - 2];
        const sigma0 = rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
        const sigma1 = rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
        words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
      }
      let [a,b,c,d,e,f,g,h] = this.state;
      for (let index = 0; index < 64; index++) {
        const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
        const choice = (e & f) ^ (~e & g);
        const temp1 = (h + sum1 + choice + K[index] + words[index]) >>> 0;
        const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
        const majority = (a & b) ^ (a & c) ^ (b & c);
        const temp2 = (sum0 + majority) >>> 0;
        h = g; g = f; f = e; e = (d + temp1) >>> 0;
        d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
      }
      const result = [a,b,c,d,e,f,g,h];
      for (let index = 0; index < 8; index++) this.state[index] = (this.state[index] + result[index]) >>> 0;
    }

    digestHex() {
      if (!this.finished) {
        const bitLength = this.bytesHashed * 8;
        this.buffer[this.bufferLength++] = 0x80;
        if (this.bufferLength > 56) {
          this.buffer.fill(0, this.bufferLength);
          this._compress(this.buffer, 0);
          this.bufferLength = 0;
        }
        this.buffer.fill(0, this.bufferLength, 56);
        const high = Math.floor(bitLength / 0x100000000) >>> 0, low = bitLength >>> 0;
        this.buffer[56] = high >>> 24; this.buffer[57] = high >>> 16; this.buffer[58] = high >>> 8; this.buffer[59] = high;
        this.buffer[60] = low >>> 24; this.buffer[61] = low >>> 16; this.buffer[62] = low >>> 8; this.buffer[63] = low;
        this._compress(this.buffer, 0);
        this.finished = true;
      }
      return Array.from(this.state, value => value.toString(16).padStart(8, "0")).join("");
    }
  }
  return Sha256;
});
