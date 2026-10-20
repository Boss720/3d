importScripts("/static/sha256.js?v=stereolab-projects-1");

self.onmessage = async event => {
  const file = event.data?.file;
  const chunkSize = 1024 * 1024;
  try {
    if (!file || !Number.isSafeInteger(file.size) || file.size < 0 || typeof file.slice !== "function") {
      throw new Error("Choose a readable local file.");
    }
    const digest = new self.StereoLabSha256();
    let offset = 0;
    while (offset < file.size) {
      const end = Math.min(file.size, offset + chunkSize);
      const buffer = await file.slice(offset, end).arrayBuffer();
      if (buffer.byteLength !== end - offset) throw new Error("The source file changed or could not be read completely.");
      digest.update(new Uint8Array(buffer));
      offset = end;
      self.postMessage({progress: offset, total: file.size});
    }
    self.postMessage({sha256: digest.digestHex(), size_bytes: file.size});
  } catch (error) {
    self.postMessage({error: String(error?.message || error).slice(0, 240)});
  }
};
