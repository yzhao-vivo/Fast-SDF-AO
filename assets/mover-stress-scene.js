// Mover stress test for the WebGPU backend: the curved Atrium (13.7k static triangles) plus 8 moving
// objects with ~266k moving triangles: the Stanford dragon (202k), the Stanford bunny (twice: clay on the
// floor and chrome in flight), a high-res teapot and torus knot, a 1.2 cm thin ring, an open canopy
// plate over the bunny, and the analytic gold mirror sphere. Used by webgpu.html ("Stress test") and
// benchmark/bench_movers.cjs. The scanned meshes (Stanford 3D Scanning Repository) are generated
// locally by benchmark/build-stress-meshes.cjs.
//
// The "Sponza + dragon" scene: the Dabrovic Sponza (66k static tris, unsigned 12 cm SDF) with the 202k-tri
// dragon walking the courtyard, a gold mirror sphere, a chrome bunny in flight and a teapot in the arcade.
// Built locally by benchmark/build_sponza.py (Sponza by Marko Dabrovic, non-commercial use with credit).
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

export const SPONZA_URLS = {
  meta: './assets/stress/sponza.json',
  mesh: './assets/stress/sponza-mesh.b64.txt',
  sdf: './assets/stress/sponza-sdf.b64.txt',
};

async function fetchB64Gzip(url, howToBuild) {
  const resp = await fetch(url);
  if (!resp.ok) throw Error(`Missing ${url}. ${howToBuild}`);
  const bin = atob((await resp.text()).trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

// base64(gzip([u32 nVerts, u32 nTris, f32 xyz..., u32 abc...])), see benchmark/build-stress-meshes.cjs
async function loadPackedMesh(url) {
  const raw = await fetchB64Gzip(url, 'Run "npm i --no-save bunny@1.0.1 stanford-dragon@1.1.1 && node benchmark/build-stress-meshes.cjs" first.');
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

// ---- Sponza + dragon ----
const SPONZA_HOWTO = 'Run "npm i --no-save jp.keijiro.dabrovic-sponza@1.0.0 && python3 benchmark/build_sponza.py" first.';

// Static scene data for gpu.loadScene(): mesh, unsigned FP16 SDF and the contact BVH.
export async function loadSponzaScene(buildContactBVH, urls = SPONZA_URLS) {
  const metaResp = await fetch(urls.meta);
  if (!metaResp.ok) throw Error(`Missing ${urls.meta}. ${SPONZA_HOWTO}`);
  const meta = await metaResp.json();
  const [mesh, sdf] = await Promise.all([fetchB64Gzip(urls.mesh, SPONZA_HOWTO), fetchB64Gzip(urls.sdf, SPONZA_HOWTO)]);
  const dv = new DataView(mesh);
  const nv = dv.getUint32(0, true), nt = dv.getUint32(4, true);
  let o = 16;
  const take = (Ctor, n) => { const a = new Ctor(mesh, o, n); o += n * 4; return a; };
  const data = {
    positions: take(Float32Array, nv * 3),
    normals: take(Float32Array, nv * 3),
    colors: take(Float32Array, nv * 3),
    roughness: take(Float32Array, nv),
    indices: take(Uint32Array, nt * 3),
  };
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.BufferAttribute(data.positions, 3));
  g.setIndex(new T.BufferAttribute(data.indices, 1));
  const v = meta.voxel;
  const contactRange = Math.max(0.18, 0.12 + 0.5 * Math.hypot(v, v, v));
  const bvh = buildContactBVH([new T.Mesh(g)], T, { gridSize: 'auto', range: contactRange + 0.0005 });
  return {
    ...data, sdfHalf: new Uint16Array(sdf), sdfDims: meta.sdfDims, sdfMin: meta.sdfMin, sdfSpan: meta.sdfSpan,
    bvh, contactRange, cam: meta.camera, useGI: true, meta,
  };
}

// Dragon (3 m) walking a loop down the courtyard between the column rows, facing where it walks; a gold
// mirror sphere floating above it, a chrome bunny flying at gallery height, a teapot in the shaded arcade.
export function sponzaMovers({ ids, halfZ }, meta, t) {
  const M = (pos, rx, ry, rz, s = 1) => composeMat4(pos, quatFromEuler(rx, ry, rz), s);
  const floor = meta.floorZ;
  const sd = 3.0 / 1.25, sb = 1.2 / 0.55, st = 1.0 / 0.6;
  const th = 0.22 * t;
  const heading = Math.atan2(0.6 * Math.cos(th), -8.5 * Math.sin(th));
  const a = 0.35 * t + 1.0, b = 0.3 * t;
  return [
    // The scanned dragon's head points along local -x, hence the half turn to face its walking direction
    { mesh: ids.dragon, matrix: M([8.5 * Math.cos(th), 0.6 * Math.sin(th), floor + halfZ.dragon * sd], 0, 0, heading + Math.PI, sd), color: [0.30, 0.72, 0.52], roughness: 0.45 },
    { pos: [6.0 * Math.cos(a), 1.0 * Math.sin(a), 3.4], radius: 0.8, color: [0.98, 0.80, 0.36], roughness: -0.04 },
    { mesh: ids.bunny, matrix: M([-6.0 * Math.cos(b), 1.1 * Math.sin(b), 5.2], 0, 0, -0.5 * t, sb), color: [0.92, 0.93, 0.95], roughness: -0.05 },
    { mesh: ids.teapot, matrix: M([4.5, -5.0, floor + halfZ.teapot * st], 0, 0, 0.6 * t, st), color: [0.95, 0.55, 0.20], roughness: 0.5 },
  ];
}
