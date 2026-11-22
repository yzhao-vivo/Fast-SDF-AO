// Arbitrary-mesh moving objects for the WebGPU backend.
//
// Each registered mover mesh gets a local-space BLAS in the same quantized dual-child BVH2 node
// format as the static contact BVH (16-bit boxes relative to the mesh bounds, sibling pairs,
// leaves of <= 7 triangles stored as v0/e1/e2). Up to MAX_MOVERS rigid instances per frame carry a
// world<->local transform; every ray (GI/AO, sun & orb shadows, glossy and mirror reflections) is
// transformed into each candidate instance's local space and traced exactly against its triangles.
// The transform is affine and the local direction is left unnormalized, so local t == world t.
//
// A second mode reproduces the original SDF method for comparison: every mesh is also baked into a
// local signed distance field (one slab of a shared 3D atlas), and in that mode movers are min()'d into
// the scene distance at their current transform, so every ray sphere-traces them like static geometry.
// mv.count.y selects the mode: 0 = exact BVH tracing, 1 = baked SDF (original method).

export const MAX_MOVERS = 8;
export const MOVER_STRIDE_FLOATS = 64; // 2 x mat4 + 8 x vec4 = 256 B per instance
export const MOVER_HEADER_FLOATS = 16; // count/mode/flags + static-scene BLAS frame
export const MOVER_UBO_BYTES = MOVER_HEADER_FLOATS * 4 + MAX_MOVERS * MOVER_STRIDE_FLOATS * 4;
export const MOVER_KIND_BVH = 0;
export const MOVER_KIND_SPHERE = 1; // analytic unit sphere in local space (exact, no BLAS)
export const MOVER_SDF_RES = 64;    // baked local SDF resolution per mesh (64^3 FP16 = 512 KiB)
export const MOVER_SDF_SLABS = 16;  // meshes the shared SDF atlas holds (64 x 64 x 1024)
const MAX_LEAF = 7;
const SAH_BINS = 12;
const SAH_MAX_DEPTH = 20; // deeper levels use median splits so traversal depth stays bounded

// Uniform block + helpers needed by any shader that reads mover instances (raster or trace).
export function moverUniformWGSL(binding) {
  return /* wgsl */`
struct Mover {
  w2l: mat4x4<f32>,
  l2w: mat4x4<f32>,
  sphere: vec4<f32>,   // world-space bounding sphere: xyz=center, w=radius
  colRough: vec4<f32>, // rgb=albedo tint, a=roughness override (used when info.w == 1)
  qMin: vec4<f32>,     // local-space BLAS quantization origin
  qScale: vec4<f32>,   // local-space BLAS quantization step (span / 65535)
  info: vec4<u32>,     // x=kind (0=triangle BLAS, 1=analytic unit sphere), y=root node, z=1 if root is one leaf, w=roughness override
  sdfBox: vec4<f32>,   // baked-SDF mode: xyz=local SDF box min, w=local->world distance scale (smallest axis scale)
  sdfInv: vec4<f32>,   // baked-SDF mode: xyz=1/box span, w=atlas slab (-1 = not baked)
  sdfAlb: vec4<f32>,   // baked-SDF mode: rgb=mesh mean albedo x tint (no per-triangle colors in an SDF)
};
struct MoverU {
  count: vec4<u32>,    // x=active movers, y=mode (0=exact BVH, 1=baked SDF), z=1: exact static-scene reflections
  stat: vec4<u32>,     // static-scene BLAS: x=root node, y=1 if the root is one leaf
  statQMin: vec4<f32>, // static-scene BLAS quantization origin (world space)
  statQScale: vec4<f32>,
  m: array<Mover, ${MAX_MOVERS}>,
};
@group(0) @binding(${binding}) var<uniform> mv: MoverU;

// Local -> world normal via the inverse transpose (= transpose of the world->local matrix).
fn mvNormalToWorld(i: u32, nl: vec3<f32>) -> vec3<f32> {
  let W = mv.m[i].w2l;
  return normalize(vec3<f32>(dot(W[0].xyz, nl), dot(W[1].xyz, nl), dot(W[2].xyz, nl)));
}
`;
}

// Ray tracing against all mover instances: exact BVH hits (mode 0) and baked-SDF distance queries
// (mode 1). Requires moverUniformWGSL() and a filtering sampler named linSamp in the same module.
export function moverTraceWGSL(bindingNodes, bindingTris, bindingAtlas) {
  return /* wgsl */`
@group(0) @binding(${bindingNodes}) var<storage, read> mvNodes: array<vec4<u32>>;
@group(0) @binding(${bindingTris}) var<storage, read> mvTris: array<vec4<f32>>;
@group(0) @binding(${bindingAtlas}) var mvSdfAtlas: texture_3d<f32>;

struct MvHit {
  t: f32,         // world distance along the ray (< 0 on miss)
  n: vec3<f32>,   // world-space geometric normal, flipped to face the ray
  alb: vec3<f32>, // albedo at the hit: instance tint x per-triangle color
  metal: f32,     // 1 for a metal (mirror-like) surface
  mover: f32,     // 1 for a moving object, 0 for the static scene
};

fn mvBox(nd: vec4<u32>, qInv: vec3<f32>, qOff: vec3<f32>, tMax: f32) -> f32 {
  let lo = vec3<f32>(f32(nd.x & 0xFFFFu), f32(nd.x >> 16u), f32(nd.y & 0xFFFFu));
  let hi = vec3<f32>(f32(nd.y >> 16u), f32(nd.z & 0xFFFFu), f32(nd.z >> 16u));
  let a = lo * qInv + qOff;
  let b = hi * qInv + qOff;
  let t0 = max(0.0, max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z)));
  let t1 = min(tMax, min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z)));
  return select(-1.0, t0, t1 >= t0);
}

fn mvLeaf(leafInfo: u32, o: vec3<f32>, d: vec3<f32>, tMin: f32, bestT: ptr<function, f32>, bestTri: ptr<function, u32>) -> bool {
  let first = (leafInfo & 0x7FFFFFFFu) >> 3u;
  let count = leafInfo & 7u;
  var hit = false;
  for (var j = 0u; j < 7u; j++) {
    if (j >= count) { break; }
    let tri = (first + j) * 3u;
    let v0 = mvTris[tri].xyz;
    let e1 = mvTris[tri + 1u].xyz;
    let e2 = mvTris[tri + 2u].xyz;
    let h = cross(d, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-12) { continue; }
    let invDet = 1.0 / det;
    let s = o - v0;
    let uu = dot(s, h) * invDet;
    if (uu < 0.0 || uu > 1.0) { continue; }
    let q = cross(s, e1);
    let vv = dot(d, q) * invDet;
    if (vv < 0.0 || uu + vv > 1.0) { continue; }
    let dist = dot(e2, q) * invDet;
    if (dist > tMin && dist < *bestT) {
      *bestT = dist;
      *bestTri = tri;
      hit = true;
    }
  }
  return hit;
}

// Closest (or any, when anyHit) triangle hit of the BLAS rooted at node "root" (quantized with qMin/qScale) for a
// ray in that BLAS's space. Returns t or -1.
fn mvBLASAt(root: u32, single: bool, qMin: vec3<f32>, qScale: vec3<f32>, o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32, anyHit: bool, bestTri: ptr<function, u32>) -> f32 {
  let inv = vec3<f32>(1.0) / vec3<f32>(
    select(d.x, 1e-12, abs(d.x) < 1e-12),
    select(d.y, 1e-12, abs(d.y) < 1e-12),
    select(d.z, 1e-12, abs(d.z) < 1e-12)
  );
  let qInv = qScale * inv;
  let qOff = (qMin - o) * inv;
  var bestT = tMax;
  var found = false;
  if (single) {
    let nd = mvNodes[root];
    if (mvBox(nd, qInv, qOff, bestT) >= 0.0) { found = mvLeaf(nd.w, o, d, tMin, &bestT, bestTri); }
    return select(-1.0, bestT, found);
  }
  var stack: array<u32, 32>;
  var sp = 0u;
  var pair = root;
  for (var visit = 0u; visit < 4096u; visit++) {
    let ndL = mvNodes[pair];
    let ndR = mvNodes[pair + 1u];
    let tL = mvBox(ndL, qInv, qOff, bestT);
    let tR = mvBox(ndR, qInv, qOff, bestT);
    var hitL = tL >= 0.0;
    var hitR = tR >= 0.0;
    if (hitL && (ndL.w & 0x80000000u) != 0u) {
      if (mvLeaf(ndL.w, o, d, tMin, &bestT, bestTri)) { found = true; }
      hitL = false;
    }
    if (hitR && (ndR.w & 0x80000000u) != 0u) {
      if (mvLeaf(ndR.w, o, d, tMin, &bestT, bestTri)) { found = true; }
      hitR = false;
    }
    if (anyHit && found) { break; }
    // A leaf hit above may have shortened bestT: drop an internal sibling whose box now starts beyond it.
    hitL = hitL && tL <= bestT;
    hitR = hitR && tR <= bestT;
    if (hitL && hitR) {
      if (tL <= tR) {
        if (sp < 32u) { stack[sp] = ndR.w; sp += 1u; }
        pair = ndL.w;
      } else {
        if (sp < 32u) { stack[sp] = ndL.w; sp += 1u; }
        pair = ndR.w;
      }
      continue;
    }
    if (hitL) { pair = ndL.w; continue; }
    if (hitR) { pair = ndR.w; continue; }
    if (sp == 0u) { break; }
    sp -= 1u;
    pair = stack[sp];
  }
  return select(-1.0, bestT, found);
}

// Mover i's BLAS for a ray already in its local space.
fn mvBLAS(i: u32, o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32, anyHit: bool, bestTri: ptr<function, u32>) -> f32 {
  let info = mv.m[i].info;
  return mvBLASAt(info.y, info.z == 1u, mv.m[i].qMin.xyz, mv.m[i].qScale.xyz, o, d, tMin, tMax, anyHit, bestTri);
}

// Entering hit of the local unit sphere. Origins inside the sphere (its own tessellated surface,
// whose faces sit slightly inside the analytic sphere) never report a hit, so no self-occlusion.
fn mvSphereLocal(lo: vec3<f32>, ld: vec3<f32>, tMin: f32, tMax: f32) -> f32 {
  let a = dot(ld, ld);
  let b = dot(lo, ld);
  let c = dot(lo, lo) - 1.0;
  if (c < 0.0) { return -1.0; }
  let disc = b * b - a * c;
  if (disc < 0.0) { return -1.0; }
  let t = (-b - sqrt(disc)) / a;
  return select(-1.0, t, t > tMin && t < tMax);
}

// Conservative world-space bounding-sphere test for the segment [0, tMax] of a unit-length ray.
fn mvMayHit(i: u32, o: vec3<f32>, d: vec3<f32>, tMax: f32) -> bool {
  let sph = mv.m[i].sphere;
  let oc = o - sph.xyz;
  let b = dot(oc, d);
  let c = dot(oc, oc) - sph.w * sph.w;
  if (c > 0.0 && b > 0.0) { return false; }
  let disc = b * b - c;
  if (disc < 0.0) { return false; }
  return -b - sqrt(disc) <= tMax;
}

// Closest exact hit over all mover instances for a unit-length world ray within (tMin, tMax).
// Always a miss in baked-SDF mode, where movers live in the distance field instead.
fn traceMovers(o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32) -> MvHit {
  var best = MvHit(-1.0, vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, 1.0);
  if (mv.count.y != 0u) { return best; }
  var tCur = tMax;
  let cnt = min(mv.count.x, ${MAX_MOVERS}u);
  for (var i = 0u; i < cnt; i++) {
    if (!mvMayHit(i, o, d, tCur)) { continue; }
    let W = mv.m[i].w2l;
    let lo = (W * vec4<f32>(o, 1.0)).xyz;
    let ld = (W * vec4<f32>(d, 0.0)).xyz;
    if (mv.m[i].info.x == 1u) {
      let t = mvSphereLocal(lo, ld, tMin, tCur);
      if (t > 0.0) {
        tCur = t;
        best = MvHit(t, mvNormalToWorld(i, lo + ld * t), mv.m[i].colRough.rgb, mvMetal(i, 1.0), 1.0);
      }
    } else {
      var tri = 0u;
      let t = mvBLAS(i, lo, ld, tMin, tCur, false, &tri);
      if (t > 0.0) {
        tCur = t;
        let a = mvTris[tri];
        let b = mvTris[tri + 1u];
        let c = mvTris[tri + 2u];
        // Per-triangle albedo; a negative red lane marks a metal triangle (see buildMoverBLAS callers)
        best = MvHit(t, mvNormalToWorld(i, cross(b.xyz, c.xyz)), mv.m[i].colRough.rgb * abs(vec3<f32>(a.w, b.w, c.w)), mvMetal(i, a.w), 1.0);
      }
    }
  }
  if (best.t > 0.0 && dot(best.n, d) > 0.0) { best.n = -best.n; }
  return best;
}

// Metal: the instance's roughness override when it has one, else the triangle's metal mark (red < 0).
fn mvMetal(i: u32, triRed: f32) -> f32 {
  if (mv.m[i].info.w != 0u) { return select(0.0, 1.0, mv.m[i].colRough.a < 0.0); }
  return select(0.0, 1.0, triRed < 0.0);
}

// Exact closest hit on the static scene's triangle BLAS (exact reflections). Miss when disabled.
fn traceStatic(o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32) -> MvHit {
  var best = MvHit(-1.0, vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, 0.0);
  if (mv.count.z == 0u) { return best; }
  var tri = 0u;
  let t = mvBLASAt(mv.stat.x, mv.stat.y == 1u, mv.statQMin.xyz, mv.statQScale.xyz, o, d, tMin, tMax, false, &tri);
  if (t > 0.0) {
    let a = mvTris[tri];
    let b = mvTris[tri + 1u];
    let c = mvTris[tri + 2u];
    var n = normalize(cross(b.xyz, c.xyz));
    if (dot(n, d) > 0.0) { n = -n; }
    best = MvHit(t, n, abs(vec3<f32>(a.w, b.w, c.w)), select(0.0, 1.0, a.w < 0.0), 0.0);
  }
  return best;
}

// Baked-SDF mover mode: sphere-trace only the movers' distance fields (exact reflections still need to
// see them while the static scene is hit exactly).
fn traceMoversSdf(o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32) -> MvHit {
  var best = MvHit(-1.0, vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, 1.0);
  if (!mvSdfMode()) { return best; }
  var t = tMin;
  for (var k = 0u; k < 64u; k++) {
    let p = o + d * t;
    let c = mvSdfClosest(p, 1e4);
    if (c.y < 0.0) { break; }
    if (c.x < 0.01) {
      let i = u32(c.y);
      best = MvHit(t, mvSdfNormal(i, p, d), mv.m[i].sdfAlb.rgb, mvMetal(i, 1.0), 1.0);
      break;
    }
    t += max(c.x, 0.008);
    if (t > tMax) { break; }
  }
  return best;
}

// Closest exact hit over the static scene and the movers (in either mover mode).
fn traceScene(o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32) -> MvHit {
  var h = traceStatic(o, d, tMin, tMax);
  let lim = select(tMax, h.t, h.t > 0.0);
  var hm = traceMovers(o, d, tMin, lim);
  if (hm.t < 0.0) { hm = traceMoversSdf(o, d, tMin, lim); }
  if (hm.t > 0.0) { h = hm; }
  return h;
}

// Any-hit shadow/visibility query against all mover instances (never occludes in baked-SDF mode).
fn moversOccluded(o: vec3<f32>, d: vec3<f32>, tMin: f32, tMax: f32) -> bool {
  if (mv.count.y != 0u) { return false; }
  let cnt = min(mv.count.x, ${MAX_MOVERS}u);
  for (var i = 0u; i < cnt; i++) {
    if (!mvMayHit(i, o, d, tMax)) { continue; }
    let W = mv.m[i].w2l;
    let lo = (W * vec4<f32>(o, 1.0)).xyz;
    let ld = (W * vec4<f32>(d, 0.0)).xyz;
    if (mv.m[i].info.x == 1u) {
      if (mvSphereLocal(lo, ld, tMin, tMax) > 0.0) { return true; }
    } else {
      var tri = 0u;
      if (mvBLAS(i, lo, ld, tMin, tMax, true, &tri) > 0.0) { return true; }
    }
  }
  return false;
}

// ---- Baked-SDF mode (the original method): movers answer distance queries, not ray queries ----

fn mvSdfMode() -> bool { return mv.count.y == 1u && mv.count.x > 0u; }

// Conservative world-space distance from p to mover i: the analytic sphere's exact distance, or the
// mover's baked local SDF (trilinear, FP16). Outside the baked box the bound max(dBox, dEdge - dBox)
// holds by the triangle inequality, so sphere tracing never steps through the mesh.
fn mvSdfDistance(i: u32, p: vec3<f32>) -> f32 {
  let lp = (mv.m[i].w2l * vec4<f32>(p, 1.0)).xyz;
  let sc = mv.m[i].sdfBox.w;
  if (mv.m[i].info.x == 1u) { return (length(lp) - 1.0) * sc; }
  let slab = mv.m[i].sdfInv.w;
  if (slab < 0.0) { return 1e4; }
  let inv = mv.m[i].sdfInv.xyz;
  let uvw = (lp - mv.m[i].sdfBox.xyz) * inv;
  let dBox = length(max(max(-uvw, uvw - vec3<f32>(1.0)), vec3<f32>(0.0)) / inv);
  let res = f32(textureDimensions(mvSdfAtlas).x);
  let slabs = f32(textureDimensions(mvSdfAtlas).z) / res;
  let q = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0));
  let qz = clamp(q.z, 0.5 / res, 1.0 - 0.5 / res); // stay inside this mesh's slab
  let dEdge = textureSampleLevel(mvSdfAtlas, linSamp, vec3<f32>(q.x, q.y, (slab + qz) / slabs), 0.0).r;
  return max(dBox, dEdge - dBox) * sc;
}

// Nearest mover closer than dCap: (distance, index) or (dCap, -1). Bounding spheres skip far movers.
fn mvSdfClosest(p: vec3<f32>, dCap: f32) -> vec2<f32> {
  var best = vec2<f32>(dCap, -1.0);
  let cnt = min(mv.count.x, ${MAX_MOVERS}u);
  for (var i = 0u; i < cnt; i++) {
    let sph = mv.m[i].sphere;
    if (length(p - sph.xyz) - sph.w >= best.x) { continue; }
    let dm = mvSdfDistance(i, p);
    if (dm < best.x) { best = vec2<f32>(dm, f32(i)); }
  }
  return best;
}

// Which mover a baked-SDF-mode march hit at p landed on: closer than eps and not behind static geometry.
fn mvSdfHitIndex(p: vec3<f32>, staticD: f32, eps: f32) -> i32 {
  if (!mvSdfMode()) { return -1; }
  return i32(mvSdfClosest(p, min(staticD + 0.004, eps)).y);
}

// Surface normal from the mover's distance gradient (central differences), facing against rayDir.
fn mvSdfNormal(i: u32, p: vec3<f32>, rayDir: vec3<f32>) -> vec3<f32> {
  let h = 0.006;
  let g = vec3<f32>(
    mvSdfDistance(i, p + vec3<f32>(h, 0.0, 0.0)) - mvSdfDistance(i, p - vec3<f32>(h, 0.0, 0.0)),
    mvSdfDistance(i, p + vec3<f32>(0.0, h, 0.0)) - mvSdfDistance(i, p - vec3<f32>(0.0, h, 0.0)),
    mvSdfDistance(i, p + vec3<f32>(0.0, 0.0, h)) - mvSdfDistance(i, p - vec3<f32>(0.0, 0.0, h))
  );
  let l = length(g);
  let n = select(-rayDir, g / l, l > 1e-8);
  return select(n, -n, dot(n, rayDir) > 0.0);
}
`;
}

// ---------------------------------------------------------------------------------------------
// CPU side

// Binned-SAH BVH2 over a triangle mesh in its local space, packed like contact-bvh.js (gridSize 0):
// nodes are vec4<u32> {lo.xy, lo.z|hi.x, hi.yz, info}; info is a child-pair index for internal nodes
// or 0x80000000 | first << 3 | count for leaves. Triangles are 3 x vec4<f32> {v0, e1, e2} with the
// per-triangle albedo in the .w lanes. Indices are local to this mesh (rebased when packed).
export function buildMoverBLAS(positions, indices = null, triColors = null) {
  const triCount = indices ? Math.floor(indices.length / 3) : Math.floor(positions.length / 9);
  if (triCount <= 0) throw Error('Mover mesh has no triangles');
  if (triCount * 8 >= 0x7FFFFFFF) throw Error('Mover mesh exceeds BLAS triangle addressing range');

  const v0 = new Float32Array(triCount * 3), e1 = new Float32Array(triCount * 3), e2 = new Float32Array(triCount * 3);
  const tLo = new Float32Array(triCount * 3), tHi = new Float32Array(triCount * 3), cen = new Float32Array(triCount * 3);
  const gLo = [Infinity, Infinity, Infinity], gHi = [-Infinity, -Infinity, -Infinity];
  for (let t = 0; t < triCount; t++) {
    const ia = indices ? indices[t * 3] : t * 3;
    const ib = indices ? indices[t * 3 + 1] : t * 3 + 1;
    const ic = indices ? indices[t * 3 + 2] : t * 3 + 2;
    for (let k = 0; k < 3; k++) {
      const a = positions[ia * 3 + k], b = positions[ib * 3 + k], c = positions[ic * 3 + k];
      v0[t * 3 + k] = a;
      e1[t * 3 + k] = b - a;
      e2[t * 3 + k] = c - a;
      const lo = Math.min(a, b, c), hi = Math.max(a, b, c);
      tLo[t * 3 + k] = lo;
      tHi[t * 3 + k] = hi;
      cen[t * 3 + k] = 0.5 * (lo + hi);
      if (lo < gLo[k]) gLo[k] = lo;
      if (hi > gHi[k]) gHi[k] = hi;
    }
  }
  if (!gLo.every(Number.isFinite) || !gHi.every(Number.isFinite)) throw Error('Mover mesh has non-finite positions');

  const pad = 1e-4;
  const qMin = gLo.map(v => v - pad);
  const span = gLo.map((v, k) => Math.max(gHi[k] - v + 2 * pad, 1e-4));
  const qInv = span.map(s => 65535 / s);

  const idx = new Int32Array(triCount);
  for (let i = 0; i < triCount; i++) idx[i] = i;
  const nodes = new Uint32Array((2 * triCount + 2) * 4);
  const ord = new Int32Array(triCount);
  let nodeLen = 0, ordLen = 0;

  const packBox = (at, lo, hi, info) => {
    const q = (v, k, up) => Math.max(0, Math.min(65535, up ? Math.ceil((v - qMin[k]) * qInv[k]) : Math.floor((v - qMin[k]) * qInv[k])));
    const lx = q(lo[0], 0, false), ly = q(lo[1], 1, false), lz = q(lo[2], 2, false);
    const hx = Math.max(lx, q(hi[0], 0, true)), hy = Math.max(ly, q(hi[1], 1, true)), hz = Math.max(lz, q(hi[2], 2, true));
    const o = at * 4;
    nodes[o] = (lx | (ly << 16)) >>> 0;
    nodes[o + 1] = (lz | (hx << 16)) >>> 0;
    nodes[o + 2] = (hy | (hz << 16)) >>> 0;
    nodes[o + 3] = info >>> 0;
  };

  const rangeBounds = (s, e) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const clo = [Infinity, Infinity, Infinity], chi = [-Infinity, -Infinity, -Infinity];
    for (let i = s; i < e; i++) {
      const t = idx[i] * 3;
      for (let k = 0; k < 3; k++) {
        if (tLo[t + k] < lo[k]) lo[k] = tLo[t + k];
        if (tHi[t + k] > hi[k]) hi[k] = tHi[t + k];
        if (cen[t + k] < clo[k]) clo[k] = cen[t + k];
        if (cen[t + k] > chi[k]) chi[k] = cen[t + k];
      }
    }
    return { lo, hi, clo, chi };
  };

  const bCnt = new Int32Array(SAH_BINS);
  const bLo = new Float64Array(SAH_BINS * 3), bHi = new Float64Array(SAH_BINS * 3);
  const rArea = new Float64Array(SAH_BINS), rCnt = new Int32Array(SAH_BINS);
  const area = (lx, ly, lz, hx, hy, hz) => {
    const dx = hx - lx, dy = hy - ly, dz = hz - lz;
    return dx * dy + dy * dz + dz * dx;
  };

  // In-place quickselect so idx[k] holds the k-th smallest centroid along axis (median split).
  const select = (s, e, k, axis) => {
    let l = s, r = e - 1;
    while (l < r) {
      const pivot = cen[idx[(l + r) >> 1] * 3 + axis];
      let i = l, j = r;
      while (i <= j) {
        while (cen[idx[i] * 3 + axis] < pivot) i++;
        while (cen[idx[j] * 3 + axis] > pivot) j--;
        if (i <= j) { const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; i++; j--; }
      }
      if (k <= j) r = j; else if (k >= i) l = i; else break;
    }
  };

  const split = (s, e, clo, chi, depth) => {
    if (depth < SAH_MAX_DEPTH) {
      let bestCost = Infinity, bestAx = -1, bestBin = 0;
      for (let ax = 0; ax < 3; ax++) {
        const ext = chi[ax] - clo[ax];
        if (ext <= 1e-9) continue;
        const sc = SAH_BINS / ext;
        bCnt.fill(0); bLo.fill(Infinity); bHi.fill(-Infinity);
        for (let i = s; i < e; i++) {
          const t = idx[i] * 3;
          let bi = ((cen[t + ax] - clo[ax]) * sc) | 0;
          if (bi >= SAH_BINS) bi = SAH_BINS - 1; else if (bi < 0) bi = 0;
          bCnt[bi]++;
          for (let k = 0; k < 3; k++) {
            if (tLo[t + k] < bLo[bi * 3 + k]) bLo[bi * 3 + k] = tLo[t + k];
            if (tHi[t + k] > bHi[bi * 3 + k]) bHi[bi * 3 + k] = tHi[t + k];
          }
        }
        let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity, rx1 = -Infinity, ry1 = -Infinity, rz1 = -Infinity, rc = 0;
        for (let b = SAH_BINS - 1; b > 0; b--) {
          rc += bCnt[b];
          rx0 = Math.min(rx0, bLo[b * 3]); ry0 = Math.min(ry0, bLo[b * 3 + 1]); rz0 = Math.min(rz0, bLo[b * 3 + 2]);
          rx1 = Math.max(rx1, bHi[b * 3]); ry1 = Math.max(ry1, bHi[b * 3 + 1]); rz1 = Math.max(rz1, bHi[b * 3 + 2]);
          rCnt[b] = rc;
          rArea[b] = rc ? area(rx0, ry0, rz0, rx1, ry1, rz1) : 0;
        }
        let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity, lx1 = -Infinity, ly1 = -Infinity, lz1 = -Infinity, lc = 0;
        for (let b = 0; b < SAH_BINS - 1; b++) {
          lc += bCnt[b];
          lx0 = Math.min(lx0, bLo[b * 3]); ly0 = Math.min(ly0, bLo[b * 3 + 1]); lz0 = Math.min(lz0, bLo[b * 3 + 2]);
          lx1 = Math.max(lx1, bHi[b * 3]); ly1 = Math.max(ly1, bHi[b * 3 + 1]); lz1 = Math.max(lz1, bHi[b * 3 + 2]);
          if (!lc || !rCnt[b + 1]) continue;
          const cost = area(lx0, ly0, lz0, lx1, ly1, lz1) * lc + rArea[b + 1] * rCnt[b + 1];
          if (cost < bestCost) { bestCost = cost; bestAx = ax; bestBin = b + 1; }
        }
      }
      if (bestAx >= 0) {
        const sc = SAH_BINS / (chi[bestAx] - clo[bestAx]);
        let i = s, j = e - 1;
        while (i <= j) {
          let bi = ((cen[idx[i] * 3 + bestAx] - clo[bestAx]) * sc) | 0;
          if (bi >= SAH_BINS) bi = SAH_BINS - 1;
          if (bi < bestBin) i++; else { const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; j--; }
        }
        if (i > s && i < e) return i;
      }
    }
    let axis = 0;
    if (chi[1] - clo[1] > chi[axis] - clo[axis]) axis = 1;
    if (chi[2] - clo[2] > chi[axis] - clo[axis]) axis = 2;
    const m = (s + e) >> 1;
    select(s, e, m, axis);
    return m;
  };

  // Tasks: [nodeSlot, start, end, depth]. The root is a sibling pair at nodes 0/1 (or one leaf node).
  const tasks = [];
  const single = triCount <= MAX_LEAF;
  if (single) {
    nodeLen = 1;
    tasks.push([0, 0, triCount, 0]);
  } else {
    const rb = rangeBounds(0, triCount);
    const m = split(0, triCount, rb.clo, rb.chi, 0);
    nodeLen = 2;
    tasks.push([1, m, triCount, 1], [0, 0, m, 1]);
  }
  while (tasks.length) {
    const [slot, s, e, depth] = tasks.pop();
    const b = rangeBounds(s, e);
    const n = e - s;
    if (n <= MAX_LEAF) {
      const first = ordLen;
      for (let i = s; i < e; i++) ord[ordLen++] = idx[i];
      packBox(slot, b.lo, b.hi, (0x80000000 | (first << 3) | n) >>> 0);
      continue;
    }
    const m = split(s, e, b.clo, b.chi, depth);
    const pair = nodeLen;
    nodeLen += 2;
    packBox(slot, b.lo, b.hi, pair);
    tasks.push([pair + 1, m, e, depth + 1], [pair, s, m, depth + 1]);
  }

  const tris = new Float32Array(triCount * 12);
  for (let r = 0; r < triCount; r++) {
    const k = ord[r], o = r * 12;
    tris[o] = v0[k * 3]; tris[o + 1] = v0[k * 3 + 1]; tris[o + 2] = v0[k * 3 + 2];
    tris[o + 4] = e1[k * 3]; tris[o + 5] = e1[k * 3 + 1]; tris[o + 6] = e1[k * 3 + 2];
    tris[o + 8] = e2[k * 3]; tris[o + 9] = e2[k * 3 + 1]; tris[o + 10] = e2[k * 3 + 2];
    tris[o + 3] = triColors ? triColors[k * 3] : 1;
    tris[o + 7] = triColors ? triColors[k * 3 + 1] : 1;
    tris[o + 11] = triColors ? triColors[k * 3 + 2] : 1;
  }

  return {
    nodes: nodes.slice(0, nodeLen * 4),
    tris,
    nodeCount: nodeLen,
    triCount,
    single,
    qMin,
    span,
    qScale: span.map(s => s / 65535),
  };
}

// Rebase per-mesh BLAS node/triangle indices into shared storage buffers.
export function packMoverBLAS(blasList) {
  let nodeTotal = 0, triTotal = 0;
  for (const b of blasList) {
    b.nodeBase = nodeTotal;
    b.triBase = triTotal;
    nodeTotal += b.nodeCount;
    triTotal += b.triCount;
  }
  if (triTotal * 8 >= 0x7FFFFFFF) throw Error('Total mover triangles exceed BLAS addressing range');
  const nodes = new Uint32Array(Math.max(1, nodeTotal) * 4);
  const tris = new Float32Array(Math.max(1, triTotal) * 12);
  for (const b of blasList) {
    for (let k = 0; k < b.nodeCount; k++) {
      const o = (b.nodeBase + k) * 4;
      nodes[o] = b.nodes[k * 4];
      nodes[o + 1] = b.nodes[k * 4 + 1];
      nodes[o + 2] = b.nodes[k * 4 + 2];
      const info = b.nodes[k * 4 + 3];
      if (info & 0x80000000) {
        const first = ((info & 0x7FFFFFFF) >>> 3) + b.triBase;
        nodes[o + 3] = (0x80000000 | (first << 3) | (info & 7)) >>> 0;
      } else {
        nodes[o + 3] = info + b.nodeBase;
      }
    }
    tris.set(b.tris, b.triBase * 12);
  }
  return { nodes, tris };
}

// Area-weighted vertex normals for meshes supplied without normals.
export function computeVertexNormals(positions, indices) {
  const n = new Float32Array(positions.length);
  const triCount = indices ? indices.length / 3 : positions.length / 9;
  for (let t = 0; t < triCount; t++) {
    const a = indices ? indices[t * 3] : t * 3, b = indices ? indices[t * 3 + 1] : t * 3 + 1, c = indices ? indices[t * 3 + 2] : t * 3 + 2;
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const e1x = positions[b * 3] - ax, e1y = positions[b * 3 + 1] - ay, e1z = positions[b * 3 + 2] - az;
    const e2x = positions[c * 3] - ax, e2y = positions[c * 3 + 1] - ay, e2z = positions[c * 3 + 2] - az;
    const fx = e1y * e2z - e1z * e2y, fy = e1z * e2x - e1x * e2z, fz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) { n[v * 3] += fx; n[v * 3 + 1] += fy; n[v * 3 + 2] += fz; }
  }
  for (let v = 0; v < n.length; v += 3) {
    const l = Math.hypot(n[v], n[v + 1], n[v + 2]) || 1;
    n[v] /= l; n[v + 1] /= l; n[v + 2] /= l;
  }
  return n;
}

// Column-major 4x4 from translation, unit quaternion [x, y, z, w] and scale (number or [sx, sy, sz]).
export function composeMat4(pos = [0, 0, 0], quat = [0, 0, 0, 1], scale = 1) {
  const [x, y, z, w] = quat;
  const [sx, sy, sz] = typeof scale === 'number' ? [scale, scale, scale] : scale;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return new Float32Array([
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    pos[0], pos[1], pos[2], 1,
  ]);
}

// General column-major 4x4 inverse (null when singular).
export function invertMat4(m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det || !Number.isFinite(det)) return null;
  det = 1 / det;
  return new Float32Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * det,
    (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det,
    (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det,
    (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det,
    (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det,
    (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det,
    (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det,
    (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det,
    (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ]);
}
