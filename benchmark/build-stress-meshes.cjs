#!/usr/bin/env node
// Builds the scanned meshes for the WebGPU mover stress test (webgpu.html "Stress test" scene) from npm:
//   npm i --no-save bunny@1.0.1 stanford-dragon@1.1.1
//   node benchmark/build-stress-meshes.cjs
// Writes assets/stress/stanford-dragon-202k.b64.txt and assets/stress/stanford-bunny.b64.txt.
// Format: base64(gzip([u32 vertexCount, u32 triangleCount, f32 xyz * vertexCount, u32 abc * triangleCount])),
// as .txt because static hosts (including claude.ai artifacts) do not all serve .bin/.gz files.
// The models come from the Stanford 3D Scanning Repository (Stanford Computer Graphics Laboratory);
// they are generated locally and not committed (see .gitignore).
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const root = path.resolve(__dirname, '..');
const out = path.join(root, 'assets', 'stress');
const req = (name) => require(require.resolve(name, { paths: [process.cwd(), root] }));

function pack({ positions, cells }) {
  const nv = positions.length, nt = cells.length;
  const buf = Buffer.alloc(8 + nv * 12 + nt * 12);
  buf.writeUInt32LE(nv, 0);
  buf.writeUInt32LE(nt, 4);
  let o = 8;
  for (const p of positions) for (let k = 0; k < 3; k++, o += 4) buf.writeFloatLE(p[k], o);
  for (const c of cells) for (let k = 0; k < 3; k++, o += 4) buf.writeUInt32LE(c[k], o);
  return zlib.gzipSync(buf, { level: 9 }).toString('base64');
}

fs.mkdirSync(out, { recursive: true });
for (const [file, mod] of [['stanford-dragon-202k.b64.txt', 'stanford-dragon/2'], ['stanford-bunny.b64.txt', 'bunny']]) {
  const mesh = req(mod);
  const text = pack(mesh);
  fs.writeFileSync(path.join(out, file), text);
  console.log(`${file}: ${mesh.positions.length.toLocaleString()} verts, ${mesh.cells.length.toLocaleString()} tris, ${(text.length / 1048576).toFixed(2)} MiB`);
}
