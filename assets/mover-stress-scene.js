// Mover stress test for the WebGPU backend: the curved Atrium (13.7k static triangles) plus 8 moving
// objects with ~266k moving triangles: the Stanford dragon (202k), the Stanford bunny (twice: clay on the
// floor and chrome in flight), a high-res teapot and torus knot, a 1.2 cm thin ring, an open canopy
// plate over the bunny, and the analytic gold mirror sphere. Used by webgpu.html ("Stress test") and
// benchmark/bench_movers.cjs. The scanned meshes (Stanford 3D Scanning Repository) are generated
// locally by benchmark/build-stress-meshes.cjs.
import * as T from 'three';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';
import { composeMat4 } from '../src/webgpu.js';

// Between the two front columns, looking down at the movers in front of the pedestal
export const STRESS_CAMERA = {
  position: [0.55, -5.4, 2.4],
  target: [-0.05, -2.2, 0.8],
};

export const STRESS_MESH_URLS = {
  dragon: './assets/stress/stanford-dragon-202k.b64.txt',
  bunny: './assets/stress/stanford-bunny.b64.txt',
};

// base64(gzip([u32 nVerts, u32 nTris, f32 xyz..., u32 abc...])), see benchmark/build-stress-meshes.cjs
async function loadPackedMesh(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw Error(`Missing ${url}. Run "npm i --no-save bunny@1.0.1 stanford-dragon@1.1.1 && node benchmark/build-stress-meshes.cjs" first.`);
  }
  const bin = atob((await resp.text()).trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const raw = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const dv = new DataView(raw);
  const nv = dv.getUint32(0, true), nt = dv.getUint32(4, true);
  return {
    positions: new Float32Array(raw, 8, nv * 3),
    indices: new Uint32Array(raw, 8 + nv * 12, nt * 3),
  };
}

// Z-up (scans are Y-up), centered on the origin, longest side = size. halfZ rests the mesh on z = 0.
function toMoverMesh(geometry, size, yUp) {
  const g = geometry.clone();
  if (yUp) g.rotateX(Math.PI / 2);
  g.computeBoundingBox();
  const c = g.boundingBox.getCenter(new T.Vector3());
  const ext = g.boundingBox.getSize(new T.Vector3());
  const s = size / Math.max(ext.x, ext.y, ext.z);
  g.translate(-c.x, -c.y, -c.z);
  g.scale(s, s, s);
  g.computeVertexNormals();
  return {
    positions: new Float32Array(g.attributes.position.array),
    normals: new Float32Array(g.attributes.normal.array),
    indices: g.index ? new Uint32Array(g.index.array) : null,
    halfZ: 0.5 * ext.z * s,
  };
}

function indexedGeometry({ positions, indices }) {
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.BufferAttribute(positions, 3));
  g.setIndex(new T.BufferAttribute(indices, 1));
  return g;
}

// Registers the stress-test meshes with a WebGPUFastSDFAO instance (exact BLAS + baked SDF each).
export async function addStressMeshes(gpu, urls = STRESS_MESH_URLS) {
  const [dragonRaw, bunnyRaw] = await Promise.all([loadPackedMesh(urls.dragon), loadPackedMesh(urls.bunny)]);
  const meshes = {
    dragon: toMoverMesh(indexedGeometry(dragonRaw), 1.25, true),
    bunny: toMoverMesh(indexedGeometry(bunnyRaw), 0.55, true),
    teapot: toMoverMesh(new TeapotGeometry(1, 16), 0.6, true),
    knot: toMoverMesh(new T.TorusKnotGeometry(0.5, 0.16, 400, 48), 0.55, false),
    ring: toMoverMesh(new T.TorusGeometry(0.3, 0.012, 16, 160), 0.624, false),
    plate: toMoverMesh(new T.PlaneGeometry(0.8, 0.55), 0.8, false),
  };
  const ids = {}, halfZ = {}, stats = {};
  for (const [name, m] of Object.entries(meshes)) {
    const t0 = performance.now();
    ids[name] = gpu.addMoverMesh(m);
    stats[name] = { triangles: (m.indices ? m.indices.length : m.positions.length / 3) / 3, blasMs: performance.now() - t0 };
    halfZ[name] = m.halfZ;
  }
  return { ids, halfZ, stats };
}

function quatFromEuler(x, y, z) {
  const q = new T.Quaternion().setFromEuler(new T.Euler(x, y, z));
  return [q.x, q.y, q.z, q.w];
}

// The 8 movers at time t (seconds): three resting/bobbing objects in front of the pedestal (dragon,
// bunny under a red canopy, teapot by the bench) and four phase-locked on a 2.1 m orbit.
export function stressMovers({ ids, halfZ }, t) {
  const M = (pos, rx, ry, rz, s = 1) => composeMat4(pos, quatFromEuler(rx, ry, rz), s);
  const a = t * 0.55;
  const orbit = (ph, z, dz) => [2.1 * Math.cos(a + ph), 2.1 * Math.sin(a + ph), z + dz * Math.sin(2 * (a + ph))];
  return [
    { mesh: ids.dragon, matrix: M([-0.35, -2.35, halfZ.dragon], 0, 0, 0.35 + 0.4 * t), color: [0.30, 0.72, 0.52], roughness: 0.45 },
    { mesh: ids.bunny, matrix: M([0.85, -2.85, halfZ.bunny], 0, 0, 2.6 - 0.6 * t), color: [0.86, 0.80, 0.72], roughness: 0.7 },
    { mesh: ids.plate, matrix: M([0.85, -2.6, 1.0 + 0.06 * Math.sin(t * 1.3)], 0.25, 0.1, 0.3), color: [0.90, 0.22, 0.14], roughness: 0.6 },
    { mesh: ids.teapot, matrix: M([-1.05, -2.95, 0.8 + 0.08 * Math.sin(t * 1.7)], 0, 0, 0.8 * t), color: [0.95, 0.55, 0.20], roughness: 0.5 },
    { mesh: ids.knot, matrix: M(orbit(0, 1.55, 0.15), t * 0.7, t * 0.4, 0), color: [0.22, 0.42, 0.92], roughness: 0.35 },
    { pos: orbit(Math.PI, 1.3, 0.2), radius: 0.3, color: [0.98, 0.80, 0.36], roughness: -0.04 },
    { mesh: ids.ring, matrix: M(orbit(Math.PI / 2, 1.9, 0.12), t * 0.9, 0.5, 0), color: [0.85, 0.65, 0.30], roughness: -0.3 },
    { mesh: ids.bunny, matrix: M(orbit(1.5 * Math.PI, 1.25, 0.15), 0, 0, -t, 0.42 / 0.55), color: [0.92, 0.93, 0.95], roughness: -0.05 },
  ];
}
