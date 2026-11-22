// In-Browser WebGPU Compute Exact-BVH SDF Baker:
// Evaluates exact closest point-to-triangle distance (Ericson 5.1.5) and signed parity (+Z raycast)
// against a global Dual-Child Quantized SAH BVH2, packing IEEE-754 binary16 directly into a GPU
// buffer and copying straight into a hardware-filtered r16float 3D GPUTexture (zero CPU readback).

const BAKE_SDF_WGSL = /* wgsl */`
struct BakeU {
  sdfMin: vec4<f32>,   // xyz=min, w=clip
  sdfSpan: vec4<f32>,  // xyz=span, w=0
  rootMin: vec4<f32>,  // xyz=bvhRootMin, w=0
  rootSpan: vec4<f32>, // xyz=bvhRootSpan, w=0
  dims: vec4<u32>,     // x=nx, y=ny, z=nz, w=nodeCount
  stride: vec4<u32>,   // x=rowU32Stride (padded to 256B / 4), y=0, z=0, w=0
};

@group(0) @binding(0) var<uniform> u: BakeU;
@group(0) @binding(1) var<storage, read> bvhNodes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bvhTris: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outHalfPairs: array<u32>;

// Exact squared distance from point p to triangle (a, a+e1, a+e2) (Ericson 5.1.5)
fn triDist2(p: vec3<f32>, a: vec3<f32>, ab: vec3<f32>, ac: vec3<f32>) -> f32 {
  let b = a + ab;
  let c = a + ac;
  let ap = p - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return dot(ap, ap); }
  let bp = p - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return dot(bp, bp); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    let v = d1 / (d1 - d3);
    let q = p - (a + ab * v);
    return dot(q, q);
  }
  let cp = p - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return dot(cp, cp); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    let w = d2 / (d2 - d6);
    let q = p - (a + ac * w);
    return dot(q, q);
  }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    let q = p - (b + (c - b) * w);
    return dot(q, q);
  }
  let denom = 1.0 / (va + vb + vc);
  let v = vb * denom;
  let w = vc * denom;
  let q = p - (a + ab * v + ac * w);
  return dot(q, q);
}

fn boxDist2(p: vec3<f32>, nd: vec4<u32>, qMin: vec3<f32>, qScale: vec3<f32>) -> f32 {
  let lo = qMin + vec3<f32>(f32(nd.x & 0xFFFFu), f32(nd.x >> 16u), f32(nd.y & 0xFFFFu)) * qScale;
  let hi = qMin + vec3<f32>(f32(nd.y >> 16u), f32(nd.z & 0xFFFFu), f32(nd.z >> 16u)) * qScale;
  let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
  return dot(d, d);
}

fn boxHitZ(p: vec3<f32>, nd: vec4<u32>, qMin: vec3<f32>, qScale: vec3<f32>) -> bool {
  let lo = qMin + vec3<f32>(f32(nd.x & 0xFFFFu), f32(nd.x >> 16u), f32(nd.y & 0xFFFFu)) * qScale;
  let hi = qMin + vec3<f32>(f32(nd.y >> 16u), f32(nd.z & 0xFFFFu), f32(nd.z >> 16u)) * qScale;
  return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && hi.z >= p.z;
}

fn evalLeafDist(leafMeta: u32, p: vec3<f32>, bestD2: ptr<function, f32>) {
  let first = (leafMeta & 0x7FFFFFFFu) >> 3u;
  let count = leafMeta & 7u;
  for (var j = 0u; j < 7u; j++) {
    if (j >= count) { break; }
    let tri = (first + j) * 3u;
    let d2 = triDist2(p, bvhTris[tri].xyz, bvhTris[tri + 1u].xyz, bvhTris[tri + 2u].xyz);
    if (d2 < *bestD2) { *bestD2 = d2; }
  }
}

fn evalLeafSign(leafMeta: u32, p: vec3<f32>) -> i32 {
  let first = (leafMeta & 0x7FFFFFFFu) >> 3u;
  let count = leafMeta & 7u;
  var winding = 0;
  // Slightly jittered ray xy prevents exact vertex/edge ties on axis-aligned boxes
  let px = p.x + 1.37e-5;
  let py = p.y + 2.91e-5;
  for (var j = 0u; j < 7u; j++) {
    if (j >= count) { break; }
    let tri = (first + j) * 3u;
    let v0 = bvhTris[tri].xyz;
    let e1 = bvhTris[tri + 1u].xyz;
    let e2 = bvhTris[tri + 2u].xyz;
    let det = e1.x * e2.y - e1.y * e2.x;
    if (abs(det) < 1e-12) { continue; }
    let invD = 1.0 / det;
    let sx = px - v0.x;
    let sy = py - v0.y;
    let uu = (sx * e2.y - sy * e2.x) * invD;
    if (uu < 0.0 || uu > 1.0) { continue; }
    let vv = (e1.x * sy - e1.y * sx) * invD;
    if (vv < 0.0 || uu + vv > 1.0) { continue; }
    let tz = v0.z + uu * e1.z + vv * e2.z;
    if (tz > p.z) {
      winding += select(-1, 1, det > 0.0);
    }
  }
  return winding;
}

fn evalSignedDistance(p: vec3<f32>) -> f32 {
  let clip = u.sdfMin.w;
  let cnt = u.dims.w;
  if (cnt == 0u) { return clip; }
  let qMin = u.rootMin.xyz;
  let qScale = u.rootSpan.xyz * (1.0 / 65535.0);

  // 1) Closest point-to-triangle distance via BVH2
  var bestD2 = clip * clip;
  if (cnt == 1u) {
    let nd = bvhNodes[0u];
    if (boxDist2(p, nd, qMin, qScale) < bestD2) { evalLeafDist(nd.w, p, &bestD2); }
  } else {
    var stack: array<u32, 32>;
    var stackD: array<f32, 32>;
    var sp = 0u;
    var pair = 0u;
    for (var visit = 0u; visit < 2048u; visit++) {
      let ndL = bvhNodes[pair];
      let ndR = bvhNodes[pair + 1u];
      let dL = boxDist2(p, ndL, qMin, qScale);
      let dR = boxDist2(p, ndR, qMin, qScale);
      var goL = dL < bestD2;
      var goR = dR < bestD2;
      if (goL && (ndL.w & 0x80000000u) != 0u) {
        evalLeafDist(ndL.w, p, &bestD2);
        goL = false;
      }
      if (goR && (ndR.w & 0x80000000u) != 0u) {
        evalLeafDist(ndR.w, p, &bestD2);
        goR = false;
      }
      if (goL && dL >= bestD2) { goL = false; }
      if (goR && dR >= bestD2) { goR = false; }
      if (goL && goR) {
        if (dL <= dR) {
          if (sp < 32u) { stack[sp] = ndR.w; stackD[sp] = dR; sp += 1u; }
          pair = ndL.w;
        } else {
          if (sp < 32u) { stack[sp] = ndL.w; stackD[sp] = dL; sp += 1u; }
          pair = ndR.w;
        }
        continue;
      }
      if (goL) { pair = ndL.w; continue; }
      if (goR) { pair = ndR.w; continue; }
      var popped = false;
      while (sp > 0u) {
        sp -= 1u;
        if (stackD[sp] < bestD2) {
          pair = stack[sp];
          popped = true;
          break;
        }
      }
      if (!popped) { break; }
    }
  }

  let dist = sqrt(bestD2);
  // Outside the mesh AABB is always positive (exterior)
  if (any(p < qMin) || any(p > qMin + u.rootSpan.xyz)) {
    return dist;
  }

  // 2) Winding-number sign via +Z ray through BVH2
  var winding = 0;
  if (cnt == 1u) {
    let nd = bvhNodes[0u];
    if (boxHitZ(p, nd, qMin, qScale)) { winding += evalLeafSign(nd.w, p); }
  } else {
    var stack: array<u32, 32>;
    var sp = 0u;
    var pair = 0u;
    for (var visit = 0u; visit < 2048u; visit++) {
      let ndL = bvhNodes[pair];
      let ndR = bvhNodes[pair + 1u];
      var goL = boxHitZ(p, ndL, qMin, qScale);
      var goR = boxHitZ(p, ndR, qMin, qScale);
      if (goL && (ndL.w & 0x80000000u) != 0u) { winding += evalLeafSign(ndL.w, p); goL = false; }
      if (goR && (ndR.w & 0x80000000u) != 0u) { winding += evalLeafSign(ndR.w, p); goR = false; }
      if (goL && goR) {
        if (sp < 32u) { stack[sp] = ndR.w; sp += 1u; }
        pair = ndL.w;
        continue;
      }
      if (goL) { pair = ndL.w; continue; }
      if (goR) { pair = ndR.w; continue; }
      if (sp == 0u) { break; }
      sp -= 1u;
      pair = stack[sp];
    }
  }
  return select(dist, -dist, winding != 0);
}

// Each thread evaluates 2 adjacent X voxels and packs them into 1 u32 via pack2x16float
// so the output buffer copies directly via copyBufferToTexture into an r16float 3D texture.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = u.dims.x;
  let ny = u.dims.y;
  let nz = u.dims.z;
  let xPair = gid.x;
  let y = gid.y;
  let z = gid.z;
  if (xPair * 2u >= nx || y >= ny || z >= nz) { return; }

  let step = u.sdfSpan.xyz / vec3<f32>(f32(nx), f32(ny), f32(nz));
  let py = u.sdfMin.y + (f32(y) + 0.5) * step.y;
  let pz = u.sdfMin.z + (f32(z) + 0.5) * step.z;
  let px0 = u.sdfMin.x + (f32(xPair * 2u) + 0.5) * step.x;
  let px1 = u.sdfMin.x + (f32(xPair * 2u + 1u) + 0.5) * step.x;

  let d0 = evalSignedDistance(vec3<f32>(px0, py, pz));
  let d1 = select(u.sdfMin.w, evalSignedDistance(vec3<f32>(px1, py, pz)), xPair * 2u + 1u < nx);

  let outIdx = (z * ny + y) * u.stride.x + xPair;
  outHalfPairs[outIdx] = pack2x16float(vec2<f32>(d0, d1));
}
`;

const pipelineCache = new WeakMap();

// outTexture/outOrigin bake into a region of an existing r16float 3D texture (e.g. one slab of the
// moving-object SDF atlas) instead of allocating a new texture. All GPU work is submitted before the
// first await, so callers may skip awaiting: later submissions on the queue see the finished bake.
export async function bakeSDFOnGPU(device, globalBvh, { sdfMin, sdfSpan, sdfDims, clip = 1.5, outTexture = null, outOrigin = [0, 0, 0] }) {
  const t0 = performance.now();
  let pipe = pipelineCache.get(device);
  if (!pipe) {
    const mod = device.createShaderModule({ code: BAKE_SDF_WGSL });
    pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    pipelineCache.set(device, pipe);
  }

  const [nx, ny, nz] = sdfDims;
  // copyBufferToTexture requires bytesPerRow to be a multiple of 256
  const bytesPerRow = Math.ceil((nx * 2) / 256) * 256;
  const rowU32Stride = bytesPerRow / 4;
  const bufSize = bytesPerRow * ny * nz;

  const nodeData = globalBvh.uniforms.contactNodes.value.image.data;
  const triData  = globalBvh.uniforms.contactTriangles.value.image.data;
  const mkBuf = (arr) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(arr.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, arr);
    return b;
  };
  const nodesBuf = mkBuf(nodeData);
  const trisBuf  = mkBuf(triData);
  const outBuf   = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const rMin = globalBvh.uniforms.contactRootMin.value;
  const rSpan = globalBvh.uniforms.contactRootSpan.value;
  const uData = new ArrayBuffer(96);
  const uf = new Float32Array(uData), uu = new Uint32Array(uData);
  uf.set([sdfMin[0], sdfMin[1], sdfMin[2], clip], 0);
  uf.set([sdfSpan[0], sdfSpan[1], sdfSpan[2], 0], 4);
  uf.set([rMin.x, rMin.y, rMin.z, 0], 8);
  uf.set([rSpan.x, rSpan.y, rSpan.z, 0], 12);
  uu.set([nx, ny, nz, globalBvh.nodes], 16);
  uu.set([rowU32Stride, 0, 0, 0], 20);
  const uBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uBuf, 0, uData);

  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uBuf } },
      { binding: 1, resource: { buffer: nodesBuf } },
      { binding: 2, resource: { buffer: trisBuf } },
      { binding: 3, resource: { buffer: outBuf } },
    ],
  });

  const sdfTex = outTexture || device.createTexture({
    size: [nx, ny, nz],
    dimension: '3d',
    format: 'r16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(Math.ceil(nx / 2) / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
  pass.end();
  enc.copyBufferToTexture({ buffer: outBuf, bytesPerRow, rowsPerImage: ny }, { texture: sdfTex, origin: outOrigin }, [nx, ny, nz]);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  nodesBuf.destroy();
  trisBuf.destroy();
  outBuf.destroy();
  uBuf.destroy();

  return { sdfTex, bakeMs: performance.now() - t0 };
}
