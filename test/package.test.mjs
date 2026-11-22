#!/usr/bin/env node
// Automated verification for the fast-sdf-ao package:
//   1. package.json metadata & exports
//   2. src/index.js exports & GENERIC_SDF_AO_GLSL structure
//   3. src/tsl.js syntax & exported node factories
//   4. loadSDF round-trip on assets/atrium.f16.gz
//   5. buildContactBVH on non-box curved Atrium geometry
//   6. bin/bake-sdf.mjs --help + small OBJ bake with --validate 200
//   7. WebGPU mover BLAS (src/webgpu-movers.js): exact traversal vs brute force after packing

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  QUALITY_PRESETS,
  loadSDF,
  GENERIC_SDF_AO_GLSL,
  patchMaterialWithSDFAO,
  TracedSDFAO,
  buildContactBVH,
  CONTACT_GLSL,
} from '../src/index.js';
import {
  buildMoverBLAS,
  packMoverBLAS,
  invertMat4,
  composeMat4,
  moverTraceWGSL,
  moverUniformWGSL,
} from '../src/webgpu-movers.js';
import {
  ATRIUM_SDF_DIMS,
  ATRIUM_SDF_MIN,
  ATRIUM_SDF_SPAN,
  buildAtriumGeometryData,
} from '../assets/atrium-scene.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

// Minimal Three.js mock for headless Node verification of Data3DTexture & buildContactBVH
class MockTexture {
  constructor(data, width, height, depth) {
    this.image = { data, width, height, depth };
    this.data = data;
    this.needsUpdate = false;
  }
  dispose() {}
}
class MockVector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
}
class MockMatrix4 {
  constructor() { this.elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
  copy(m) { this.elements = m.elements.slice(); return this; }
  multiplyMatrices(a, b) { this.elements = a.elements.slice(); return this; }
}
const MockTHREE = {
  DataTexture: class extends MockTexture { constructor(d, w, h) { super(d, w, h, 1); } },
  Data3DTexture: class extends MockTexture { constructor(d, w, h, z) { super(d, w, h, z); } },
  Vector3: MockVector3,
  Matrix4: MockMatrix4,
  RedFormat: 1028,
  RGBAFormat: 1023,
  RGBAIntegerFormat: 1033,
  RGIntegerFormat: 1031,
  RedIntegerFormat: 1030,
  HalfFloatType: 1016,
  FloatType: 1015,
  UnsignedIntType: 1014,
  LinearFilter: 1006,
  NearestFilter: 1003,
};

console.log('1. Checking package.json & LICENSE...');
const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8'));
assert.equal(pkg.name, 'fast-sdf-ao');
assert.equal(pkg.type, 'module');
assert.equal(pkg.license, 'MIT');
assert.equal(pkg.exports['.'], './src/index.js');
assert.equal(pkg.exports['./contact-bvh'], './contact-bvh.js');
assert.equal(pkg.exports['./tsl'], './src/tsl.js');
assert.equal(pkg.bin['bake-sdf'], './bin/bake-sdf.mjs');
assert.ok(fs.existsSync(path.join(repoRoot, 'LICENSE')));
console.log('   OK');

console.log('2. Checking src/index.js exports & GLSL...');
assert.equal(typeof QUALITY_PRESETS, 'object');
assert.deepEqual(Object.keys(QUALITY_PRESETS), ['low', 'medium', 'high', 'ultra']);
assert.equal(QUALITY_PRESETS.low.giConeMode, true);
assert.equal(QUALITY_PRESETS.low.giDenoisePasses, 0);
assert.equal(QUALITY_PRESETS.high.giScale, 0.25);
assert.equal(QUALITY_PRESETS.ultra.giScale, 0.5);
assert.equal(typeof loadSDF, 'function');
assert.equal(typeof patchMaterialWithSDFAO, 'function');
assert.equal(typeof TracedSDFAO, 'function');
assert.equal(typeof buildContactBVH, 'function');
assert.equal(typeof CONTACT_GLSL, 'string');
assert.equal(typeof GENERIC_SDF_AO_GLSL, 'string');
assert.ok(GENERIC_SDF_AO_GLSL.includes('sdfAOFourTap'));
assert.ok(GENERIC_SDF_AO_GLSL.includes('multiBounceAO'));
assert.ok(GENERIC_SDF_AO_GLSL.includes('specularOcclusion'));
assert.ok(!GENERIC_SDF_AO_GLSL.includes('wallFace'), 'Generic GLSL must not depend on wallFace');
console.log('   OK');

console.log('3. Checking src/tsl.js syntax & exports...');
const tslSrc = fs.readFileSync(path.join(repoRoot, 'src/tsl.js'), 'utf8');
execFileSync('node', ['--check', path.join(repoRoot, 'src/tsl.js')], { stdio: 'inherit' });
assert.ok(tslSrc.includes("from 'three/tsl'"));
assert.ok(tslSrc.includes('export function createSDFAONode'));
assert.ok(tslSrc.includes('export const createMultiBounceAONode'));
console.log('   OK');

console.log('4. Testing loadSDF round-trip with assets/atrium.f16.gz...');
const atriumGz = fs.readFileSync(path.join(repoRoot, 'assets/atrium.f16.gz'));
const tex = await loadSDF(atriumGz, { dims: ATRIUM_SDF_DIMS, format: 'f16', THREE: MockTHREE });
const expectedVoxels = ATRIUM_SDF_DIMS[0] * ATRIUM_SDF_DIMS[1] * ATRIUM_SDF_DIMS[2];
assert.equal(tex.data.length, expectedVoxels);
assert.equal(tex.format, MockTHREE.RedFormat);
assert.equal(tex.type, MockTHREE.HalfFloatType);
assert.equal(tex.minFilter, MockTHREE.LinearFilter);
assert.equal(tex.needsUpdate, true);
console.log(`   OK (${expectedVoxels.toLocaleString()} FP16 voxels decoded from ${(atriumGz.byteLength / 1024).toFixed(1)} KiB gz)`);

console.log('5. Testing buildContactBVH on non-box curved Atrium geometry...');
const atriumData = buildAtriumGeometryData();
const mockAtriumMesh = {
  isInstancedMesh: false,
  matrixWorld: new MockMatrix4(),
  updateWorldMatrix() {},
  geometry: {
    index: {
      count: atriumData.indices.length,
      array: atriumData.indices,
      getX: (i) => atriumData.indices[i],
    },
    attributes: {
      position: {
        count: atriumData.vertexCount,
        itemSize: 3,
        array: atriumData.positions,
        getX: (i) => atriumData.positions[i * 3],
        getY: (i) => atriumData.positions[i * 3 + 1],
        getZ: (i) => atriumData.positions[i * 3 + 2],
      },
    },
  },
};
const t0 = performance.now();
const bvh = buildContactBVH([mockAtriumMesh], MockTHREE, { gridSize: 1.0, range: 0.25 });
const bvhMs = performance.now() - t0;
assert.equal(bvh.triangles, atriumData.triangleCount);
assert.ok(bvh.nodes > 0);
assert.ok(bvh.bytes > 0);
assert.ok(bvh.uniforms.contactNodes.value);
console.log(`   OK (${bvh.triangles.toLocaleString()} tris, ${bvh.nodes.toLocaleString()} nodes, ${(bvh.bytes / 1024).toFixed(1)} KiB in ${bvhMs.toFixed(1)} ms)`);

console.log('6. Testing bin/bake-sdf.mjs --help and small OBJ bake...');
const helpOut = execFileSync('node', [path.join(repoRoot, 'bin/bake-sdf.mjs'), '--help'], { encoding: 'utf8' });
assert.ok(helpOut.includes('npx bake-sdf'));
assert.ok(helpOut.includes('--validate'));

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fast-sdf-ao-test-'));
try {
  // Unit octahedron OBJ (closed watertight non-box mesh)
  const objPath = path.join(tmpDir, 'octa.obj');
  const outPath = path.join(tmpDir, 'octa.f16');
  fs.writeFileSync(objPath, [
    'v 1 0 0', 'v -1 0 0', 'v 0 1 0', 'v 0 -1 0', 'v 0 0 1', 'v 0 0 -1',
    'f 1 3 5', 'f 3 2 5', 'f 2 4 5', 'f 4 1 5',
    'f 3 1 6', 'f 2 3 6', 'f 4 2 6', 'f 1 4 6',
  ].join('\n') + '\n');
  const bakeOut = execFileSync('node', [
    path.join(repoRoot, 'bin/bake-sdf.mjs'),
    '--obj', objPath,
    '--lo', '-2', '-2', '-2',
    '--span', '4', '4', '4',
    '--res', '32',
    '--f16',
    '--validate', '200',
    '--out', outPath,
  ], { encoding: 'utf8' });
  const stats = JSON.parse(bakeOut.trim());
  assert.equal(stats.validate.mismatches, 0);
  assert.equal(stats.validate.sign_errors, 0);
  assert.equal(fs.statSync(outPath).size, 32 * 32 * 32 * 2);
  console.log(`   OK (32^3 FP16 bake in ${stats.ms.total.toFixed(1)} ms, 200/200 brute-force exact)`);

  // --sign vote (majority of +x/+y/+z winding numbers) must equal the +z sign on a closed mesh
  const votePath = path.join(tmpDir, 'octa-vote.f16');
  const voteOut = execFileSync('node', [
    path.join(repoRoot, 'bin/bake-sdf.mjs'), '--obj', objPath, '--lo', '-2', '-2', '-2', '--span', '4', '4', '4',
    '--res', '32', '--f16', '--sign', 'vote', '--validate', '200', '--out', votePath,
  ], { encoding: 'utf8' });
  const voteStats = JSON.parse(voteOut.trim());
  assert.equal(voteStats.validate.mismatches, 0);
  assert.equal(voteStats.validate.sign_errors, 0);
  assert.ok(fs.readFileSync(votePath).equals(fs.readFileSync(outPath)), '--sign vote differs from --sign z on a closed mesh');
  console.log('   OK (--sign vote identical to the +z winding sign on a closed mesh)');
} finally {
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

console.log('7. Testing WebGPU mover BLAS exactness (CPU port of the WGSL traversal vs brute force)...');
{
  // Two meshes packed into shared buffers so index rebasing is exercised: a 2-triangle single-leaf
  // mesh followed by the curved 13,704-triangle Atrium.
  const tiny = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]), indices: null };
  const atrium = { positions: atriumData.positions, indices: atriumData.indices };
  const meshes = [tiny, atrium];
  const tb0 = performance.now();
  const blas = meshes.map(m => buildMoverBLAS(m.positions, m.indices));
  const blasMs = performance.now() - tb0;
  const { nodes, tris } = packMoverBLAS(blas);
  assert.equal(blas[0].single, true);
  assert.equal(blas[1].triCount, atriumData.triangleCount);
  assert.equal(blas[1].nodeBase, blas[0].nodeCount);
  assert.equal(blas[1].triBase, blas[0].triCount);

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const triHit = (v0, e1, e2, o, d, tMin, tMax) => {
    const h = cross(d, e2), det = dot(e1, h);
    if (Math.abs(det) < 1e-12) return -1;
    const inv = 1 / det, s = sub(o, v0), u = dot(s, h) * inv;
    if (u < 0 || u > 1) return -1;
    const q = cross(s, e1), v = dot(d, q) * inv;
    if (v < 0 || u + v > 1) return -1;
    const t = dot(e2, q) * inv;
    return t > tMin && t < tMax ? t : -1;
  };
  const triVec = (i) => [tris[i * 4], tris[i * 4 + 1], tris[i * 4 + 2]];
  const node = (i) => [nodes[i * 4], nodes[i * 4 + 1], nodes[i * 4 + 2], nodes[i * 4 + 3]];
  const box = (nd, qInv, qOff, tMax) => {
    const lo = [nd[0] & 0xFFFF, nd[0] >>> 16, nd[1] & 0xFFFF], hi = [nd[1] >>> 16, nd[2] & 0xFFFF, nd[2] >>> 16];
    let t0 = 0, t1 = tMax;
    for (let k = 0; k < 3; k++) {
      const a = lo[k] * qInv[k] + qOff[k], b = hi[k] * qInv[k] + qOff[k];
      t0 = Math.max(t0, Math.min(a, b));
      t1 = Math.min(t1, Math.max(a, b));
    }
    return t1 >= t0 ? t0 : -1;
  };
  const leaf = (info, o, d, tMin, st) => {
    const first = (info & 0x7FFFFFFF) >>> 3;
    let hit = false;
    for (let j = 0; j < (info & 7); j++) {
      const t3 = (first + j) * 3;
      const t = triHit(triVec(t3), triVec(t3 + 1), triVec(t3 + 2), o, d, tMin, st.best);
      if (t > 0) { st.best = t; hit = true; }
    }
    return hit;
  };
  let maxStack = 0;
  // Mirrors mvBLAS() in src/webgpu-movers.js (stack of 32, nearest-first, leaf-tightened siblings).
  const traverse = (b, o, d, tMin, tMax) => {
    const inv = d.map(x => 1 / (Math.abs(x) < 1e-12 ? 1e-12 : x));
    const qInv = b.qScale.map((s, k) => s * inv[k]), qOff = b.qMin.map((m, k) => (m - o[k]) * inv[k]);
    const st = { best: tMax };
    let found = false;
    if (b.single) {
      const nd = node(b.nodeBase);
      if (box(nd, qInv, qOff, st.best) >= 0) found = leaf(nd[3], o, d, tMin, st);
      return found ? st.best : -1;
    }
    const stack = [];
    let pair = b.nodeBase;
    for (let visit = 0; visit < 4096; visit++) {
      const L = node(pair), R = node(pair + 1);
      const tL = box(L, qInv, qOff, st.best), tR = box(R, qInv, qOff, st.best);
      let hL = tL >= 0, hR = tR >= 0;
      if (hL && (L[3] & 0x80000000)) { if (leaf(L[3], o, d, tMin, st)) found = true; hL = false; }
      if (hR && (R[3] & 0x80000000)) { if (leaf(R[3], o, d, tMin, st)) found = true; hR = false; }
      hL = hL && tL <= st.best;
      hR = hR && tR <= st.best;
      if (hL && hR) {
        assert.ok(stack.length < 32, 'mover BLAS traversal stack overflow');
        stack.push(tL <= tR ? R[3] : L[3]);
        maxStack = Math.max(maxStack, stack.length);
        pair = tL <= tR ? L[3] : R[3];
        continue;
      }
      if (hL) { pair = L[3]; continue; }
      if (hR) { pair = R[3]; continue; }
      if (!stack.length) break;
      pair = stack.pop();
    }
    return found ? st.best : -1;
  };
  const brute = (m, o, d, tMin, tMax) => {
    const n = m.indices ? m.indices.length / 3 : m.positions.length / 9;
    const P = (i) => [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
    let best = -1, bt = tMax;
    for (let t = 0; t < n; t++) {
      const a = P(m.indices ? m.indices[t * 3] : t * 3);
      const b = P(m.indices ? m.indices[t * 3 + 1] : t * 3 + 1);
      const c = P(m.indices ? m.indices[t * 3 + 2] : t * 3 + 2);
      const h = triHit(a, sub(b, a), sub(c, a), o, d, tMin, bt);
      if (h > 0) { bt = h; best = h; }
    }
    return best;
  };
  let seed = 12345;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  let rays = 0, hits = 0, mismatches = 0;
  meshes.forEach((m, mi) => {
    const b = blas[mi];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < m.positions.length; i++) { lo[i % 3] = Math.min(lo[i % 3], m.positions[i]); hi[i % 3] = Math.max(hi[i % 3], m.positions[i]); }
    for (let k = 0; k < (mi === 0 ? 400 : 1200); k++) {
      const o = lo.map((l, a) => l - 0.5 * (hi[a] - l) + rnd() * 2 * (hi[a] - l));
      const tgt = lo.map((l, a) => l + rnd() * (hi[a] - l));
      let d = sub(tgt, o);
      const len = Math.hypot(...d);
      d = d.map(x => x / len);
      const tMax = rnd() < 0.3 ? rnd() * len * 1.5 : 1e4;
      const a = traverse(b, o, d, 0.002, tMax), c = brute(m, o, d, 0.002, tMax);
      rays++;
      if (c > 0) hits++;
      if ((a > 0) !== (c > 0) || (a > 0 && Math.abs(a - c) > 1e-5 * Math.max(1, c))) mismatches++;
    }
  });
  assert.equal(mismatches, 0, `mover BLAS disagreed with brute force on ${mismatches}/${rays} rays`);
  assert.ok(hits > rays * 0.3, 'mover BLAS test rays should mostly hit');

  const M = composeMat4([1, 2, 3], [0.2, 0.3, 0.1, Math.sqrt(1 - 0.14)], [0.5, 2, 1.5]);
  const I = invertMat4(M);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let v = 0;
    for (let k = 0; k < 4; k++) v += I[k * 4 + r] * M[c * 4 + k];
    assert.ok(Math.abs(v - (c === r ? 1 : 0)) < 1e-5, 'invertMat4(composeMat4()) must be identity');
  }
  const wgsl = moverUniformWGSL(1) + moverTraceWGSL(2, 3);
  for (const fn of ['fn traceMovers', 'fn moversOccluded', 'fn mvBLAS', 'fn mvNormalToWorld', 'array<u32, 32>']) assert.ok(wgsl.includes(fn), fn);
  console.log(`   OK (${rays} rays, ${hits} hits, 0 mismatches, max stack ${maxStack}/32, ${blas[1].nodeCount.toLocaleString()} nodes, BLAS ${blasMs.toFixed(1)} ms)`);
}

console.log('\nAll package Node tests PASSED!');
