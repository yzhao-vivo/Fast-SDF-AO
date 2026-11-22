#!/usr/bin/env node
// WebGPU moving-object benchmark: quality and cost of Exact BVH vs Baked SDF (original method) movers on
// the stress scene (curved Atrium, 13.7k static tris + 8 movers, ~270k moving tris incl. the Stanford dragon).
//
// Quality: freezes the movers, lets each mode converge, reads the renderer's final half-res AO / GI /
// sun-shadow buffers and its G-buffer back from the GPU, and scores them against tools/trace_gi
// (512-ray reference) traced on the exact same world-space triangles (static scene + posed movers).
// A second reference without movers marks the pixels the movers change ("mover-affected").
// Cost: runs the in-page benchmark of webgpu.html ("Run benchmark") with the movers animating.
//
// Setup:  npm i --no-save bunny@1.0.1 stanford-dragon@1.1.1 && node benchmark/build-stress-meshes.cjs
// Run:    node benchmark/bench_movers.cjs [--frames 48] [--perf-warmup 30] [--perf-frames 120]
//           [--swiftshader] [--headed] [--channel chrome] [--three-dir node_modules/three]
// Output: benchmark/mover-bench/ (raw buffers, references, metrics.json, report.md, images/*.png)
// Needs Playwright (require('playwright')), a C++17 compiler, and python3 with numpy + Pillow.
'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const { execFileSync } = require('child_process');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(__dirname, 'mover-bench');
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const flag = (name) => args.includes(`--${name}`);
const FRAMES = Number(opt('frames', 48));
const PERF_WARMUP = Number(opt('perf-warmup', 30));
const PERF_FRAMES = Number(opt('perf-frames', 120));
const POSE_T = Number(opt('pose', 2.2));
const SWIFTSHADER = flag('swiftshader');
const THREE_DIR = opt('three-dir', null);

// Headless SwiftShader cannot present to a canvas: render into an offscreen texture instead.
const OFFSCREEN_CANVAS_SHIM = `
GPUCanvasContext.prototype.configure = function (cfg) { this.__cfg = cfg; this.__tex = null; };
GPUCanvasContext.prototype.getCurrentTexture = function () {
  if (!this.__tex) this.__tex = this.__cfg.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: this.__cfg.format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
  return this.__tex;
};
// Copy the offscreen frame into a 2D canvas (#__out) so it can be screenshotted.
window.__grab = async (canvas) => {
  const ctx = canvas.getContext('webgpu'), tex = ctx.__tex, dev = ctx.__cfg.device, w = tex.width, h = tex.height;
  const bpr = Math.ceil(w * 4 / 256) * 256;
  const buf = dev.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange()), bgra = ctx.__cfg.format.startsWith('bgra');
  let out = document.getElementById('__out');
  if (!out) { out = document.createElement('canvas'); out.id = '__out'; out.style.cssText = 'position:fixed;left:0;top:0;z-index:99999'; document.body.appendChild(out); }
  out.width = w; out.height = h;
  const c2 = out.getContext('2d'), img = c2.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const s = y * bpr + x * 4, d = (y * w + x) * 4;
    img.data[d] = src[s + (bgra ? 2 : 0)]; img.data[d + 1] = src[s + 1]; img.data[d + 2] = src[s + (bgra ? 0 : 2)]; img.data[d + 3] = 255;
  }
  c2.putImageData(img, 0, 0);
  buf.unmap(); buf.destroy();
};`;

function serve() {
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.txt': 'text/plain', '.gz': 'application/gzip' };
  const server = http.createServer((req, res) => {
    const file = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(r => server.listen(0, '127.0.0.1', () => r(server)));
}

// In-page helpers: GPU texture readback (any float-sampleable 2D texture, every `stride`-th pixel)
// and world-space triangle export of the posed movers.
const PAGE_HELPERS = `
window.__bench = {
  async readTex(tex, stride) {
    const g = window.wgpuDemo.gpu, d = g.device;
    if (!this.pipe) {
      const mod = d.createShaderModule({ code: \`
        @group(0) @binding(0) var src: texture_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
        @group(0) @binding(2) var<uniform> P: vec4<u32>;
        @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x >= P.x || id.y >= P.y) { return; }
          dst[id.y * P.x + id.x] = textureLoad(src, vec2<i32>(id.xy * P.z), 0);
        }\` });
      this.bgl = d.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ] });
      this.pipe = d.createComputePipeline({ layout: d.createPipelineLayout({ bindGroupLayouts: [this.bgl] }), compute: { module: mod, entryPoint: 'main' } });
    }
    const w = Math.ceil(tex.width / stride), h = Math.ceil(tex.height / stride), bytes = w * h * 16;
    const dst = d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rd = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const ub = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(ub, 0, new Uint32Array([w, h, stride, 0]));
    const bg = d.createBindGroup({ layout: this.bgl, entries: [
      { binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: dst } }, { binding: 2, resource: { buffer: ub } }] });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); pass.end();
    enc.copyBufferToBuffer(dst, 0, rd, 0, bytes);
    d.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const out = this.b64(new Uint8Array(rd.getMappedRange()));
    rd.unmap(); [dst, rd, ub].forEach(b => b.destroy());
    return { w, h, data: out };
  },
  b64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); },
  // The buffers the last composite pass used
  async capture(stride) {
    const g = window.wgpuDemo.gpu, idx = 1 - g.pingIdx, bg = g._lastCompBG;
    const aogi = bg === g.compBGD2[idx] ? g.denoiseB : (bg === g.compBGD1[idx] ? g.denoiseA : g.histAOGI[idx]);
    const res = {};
    for (const [k, tex] of Object.entries({ aogi, meta: g.histMeta[idx], pos: g.gPosDepth[idx], nor: g.gNormal[idx], alb: g.gAlbedo })) res[k] = await this.readTex(tex, stride);
    return res;
  },
  // 15 floats per triangle (v0, v1, v2, unit normal, albedo) in world space, for tools/trace_gi
  async moverTriangles() {
    const { composeMat4 } = await import('./src/webgpu.js');
    const g = window.wgpuDemo.gpu, out = [];
    for (const o of g.dynObjects) {
      const mesh = g.moverMeshes.get(o.mesh ?? g.SPHERE_MESH);
      const M = o.matrix ? Float32Array.from(o.matrix) : composeMat4(o.pos || [0, 0, 0], o.rotation || [0, 0, 0, 1], o.scale ?? o.radius ?? 1);
      const tint = o.color || (mesh.blas ? [1, 1, 1] : [0.8, 0.8, 0.8]);
      const X = (x, y, z) => [M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]];
      const push = (a, b, c, alb) => {
        const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const l = Math.hypot(...n) || 1;
        out.push(...a, ...b, ...c, n[0] / l, n[1] / l, n[2] / l, ...alb);
      };
      if (mesh.blas) {
        const t = mesh.blas.tris;
        for (let i = 0; i < mesh.blas.triCount; i++) {
          const o12 = i * 12, v = [t[o12], t[o12 + 1], t[o12 + 2]];
          push(X(...v), X(v[0] + t[o12 + 4], v[1] + t[o12 + 5], v[2] + t[o12 + 6]), X(v[0] + t[o12 + 8], v[1] + t[o12 + 9], v[2] + t[o12 + 10]),
            [tint[0] * t[o12 + 3], tint[1] * t[o12 + 7], tint[2] * t[o12 + 11]]);
        }
      } else { // analytic sphere: finely tessellated for the reference
        const U = 160, V = 80, P = (i, j) => { const th = 2 * Math.PI * i / U, ph = Math.PI * j / V; return X(Math.sin(ph) * Math.cos(th), Math.sin(ph) * Math.sin(th), Math.cos(ph)); };
        for (let j = 0; j < V; j++) for (let i = 0; i < U; i++) { push(P(i, j), P(i, j + 1), P(i + 1, j + 1), tint); push(P(i, j), P(i + 1, j + 1), P(i + 1, j), tint); }
      }
    }
    return this.b64(new Uint8Array(new Float32Array(out).buffer));
  },
};`;

async function main() {
  fs.mkdirSync(path.join(OUT, 'bin'), { recursive: true });
  if (!fs.existsSync(path.join(ROOT, 'assets/stress/stanford-dragon-202k.b64.txt'))) {
    throw Error('Missing stress meshes: npm i --no-save bunny@1.0.1 stanford-dragon@1.1.1 && node benchmark/build-stress-meshes.cjs');
  }
  const tracer = path.join(OUT, 'bin', 'trace_gi');
  console.log('Compiling tools/trace_gi.cpp…');
  execFileSync('c++', ['-O3', '-std=c++17', '-pthread', path.join(ROOT, 'tools/trace_gi.cpp'), '-o', tracer], { stdio: 'inherit' });

  const server = await serve();
  const url = `http://127.0.0.1:${server.address().port}/webgpu.html`;
  const launchArgs = ['--enable-unsafe-webgpu', '--disable-gpu-watchdog'];
  if (SWIFTSHADER) launchArgs.push('--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader');
  const browser = await chromium.launch({ channel: opt('channel', 'chromium'), headless: !flag('headed'), args: launchArgs });
  const page = await browser.newPage({ viewport: { width: 1320, height: 900 } });
  const logs = [];
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
  if (SWIFTSHADER) await page.addInitScript(OFFSCREEN_CANVAS_SHIM);
  if (THREE_DIR) {
    await page.route('https://cdn.jsdelivr.net/npm/three@0.170.0/**', route => {
      const f = path.join(path.resolve(THREE_DIR), route.request().url().replace('https://cdn.jsdelivr.net/npm/three@0.170.0/', ''));
      fs.existsSync(f) ? route.fulfill({ path: f, contentType: 'text/javascript' }) : route.abort();
    });
  }
  page.setDefaultTimeout(0);
  await page.goto(url);
  await page.waitForFunction(() => window.wgpuDemo && window.wgpuDemo.gpu.sceneRes);
  await page.addScriptTag({ content: PAGE_HELPERS });

  console.log('Loading the stress scene…');
  const setup = await page.evaluate(async (t) => {
    const D = window.wgpuDemo, g = D.gpu;
    D.stopLoop();
    await D.selectScene('stress');
    D.setMovers(true, t);
    for (let i = 0; i < 2; i++) { D.renderOnce(); await g.device.queue.onSubmittedWorkDone(); } // finish the SDF bakes
    const meshes = [...g.moverMeshes.values()].map(m => ({ id: m.id, triangles: m.triangles, sdfBakeMs: m.sdfBakeMs ?? null,
      blasBytes: m.blas ? m.blas.nodes.byteLength + m.blas.tris.byteLength : 0 }));
    const info = g.adapterInfo;
    return { device: info.description || info.architecture || info.vendor || 'WebGPU', hasTimestamp: g.hasTimestamp,
      sunDir: g.sunDir, stats: D.getStressSet().stats, meshes, width: g.width, height: g.height, aoW: g.aoW, aoH: g.aoH,
      quality: g.quality, atlasBytes: g.moverSdfAtlas.width * g.moverSdfAtlas.height * g.moverSdfAtlas.depthOrArrayLayers * 2 };
  }, POSE_T);
  console.log(`  ${setup.device}, timestamps: ${setup.hasTimestamp}`);

  // ---- Quality: converge each mode on the frozen pose, then read back ----
  const STRIDE = 2;
  for (const mode of ['bvh', 'sdf']) {
    console.log(`Converging ${mode} (${FRAMES} frames)…`);
    const t0 = Date.now();
    const cap = await page.evaluate(async ({ mode, frames, stride }) => {
      const D = window.wgpuDemo, g = D.gpu;
      D.setMoverTracing(mode);
      g.clearHash();
      g.viewMode = 0;
      for (let i = 0; i < frames; i++) { D.renderOnce(); await g.device.queue.onSubmittedWorkDone(); }
      return window.__bench.capture(stride);
    }, { mode, frames: FRAMES, stride: STRIDE });
    for (const [k, v] of Object.entries(cap)) fs.writeFileSync(path.join(OUT, `gpu_${mode}_${k}.f32`), Buffer.from(v.data, 'base64'));
    setup.capW = cap.aogi.w; setup.capH = cap.aogi.h;
    console.log(`  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    for (const [vm, name] of [[0, 'shaded'], [1, 'ao'], [2, 'gi']]) {
      const grabbed = await page.evaluate(async (vm) => {
        const D = window.wgpuDemo, g = D.gpu;
        g.viewMode = vm; D.renderOnce(); await g.device.queue.onSubmittedWorkDone();
        g.viewMode = 0;
        if (!window.__grab) return false;
        await window.__grab(document.querySelector('#c'));
        return true;
      }, vm);
      fs.mkdirSync(path.join(OUT, 'images'), { recursive: true });
      await page.locator(grabbed ? '#__out' : '#c').screenshot({ path: path.join(OUT, 'images', `screen_${mode}_${name}.png`) });
    }
  }

  // ---- Reference: exact world-space triangles (static Atrium + posed movers) ----
  console.log('Exporting triangles…');
  const moverTris = Buffer.from(await page.evaluate(() => window.__bench.moverTriangles()), 'base64');
  const { buildAtriumGeometryData } = await import(require('url').pathToFileURL(path.join(ROOT, 'assets/atrium-scene.js')).href);
  const A = buildAtriumGeometryData();
  const st = new Float32Array(A.triangleCount * 15);
  for (let t = 0; t < A.triangleCount; t++) {
    const vs = [0, 1, 2].map(k => A.indices[t * 3 + k]);
    const P = vs.map(v => [A.positions[v * 3], A.positions[v * 3 + 1], A.positions[v * 3 + 2]]);
    const e1 = P[1].map((x, k) => x - P[0][k]), e2 = P[2].map((x, k) => x - P[0][k]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(...n) || 1;
    const alb = [0, 1, 2].map(k => vs.reduce((s, v) => s + A.colors[v * 3 + k], 0) / 3);
    st.set([...P[0], ...P[1], ...P[2], n[0] / l, n[1] / l, n[2] / l, ...alb], t * 15);
  }
  fs.writeFileSync(path.join(OUT, 'tris_static.f32'), Buffer.from(st.buffer));
  fs.writeFileSync(path.join(OUT, 'tris_full.f32'), Buffer.concat([Buffer.from(st.buffer), moverTris]));

  // Samples = valid G-buffer pixels (identical rasterization in both modes)
  const readF32 = (f) => { const b = fs.readFileSync(path.join(OUT, f)); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
  const pos = readF32('gpu_bvh_pos.f32'), nor = readF32('gpu_bvh_nor.f32'), alb = readF32('gpu_bvh_alb.f32');
  const valid = [], samples = [];
  for (let i = 0; i < pos.length / 4; i++) {
    if (pos[i * 4 + 3] > 0 && nor[i * 4 + 3] > 0.5) {
      valid.push(i);
      samples.push(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2], nor[i * 4], nor[i * 4 + 1], nor[i * 4 + 2], alb[i * 4], alb[i * 4 + 1], alb[i * 4 + 2]);
    }
  }
  fs.writeFileSync(path.join(OUT, 'samples.f32'), Buffer.from(new Float32Array(samples).buffer));
  fs.writeFileSync(path.join(OUT, 'valid.u32'), Buffer.from(new Uint32Array(valid).buffer));
  const [sx, sy, sz] = setup.sunDir;
  for (const which of ['full', 'static']) {
    console.log(`Tracing the 512-ray reference (${which}, ${valid.length.toLocaleString()} pixels)…`);
    const t0 = Date.now();
    execFileSync(tracer, [path.join(OUT, `tris_${which}.f32`), path.join(OUT, 'samples.f32'), path.join(OUT, `ref_${which}.f32`),
      String(sx), String(sy), String(sz), path.join(OUT, `sun_${which}.f32`)], { stdio: 'inherit' });
    setup[`refSeconds_${which}`] = (Date.now() - t0) / 1000;
  }

  // ---- Cost: in-page benchmark with the movers animating ----
  console.log(`Timing (${PERF_WARMUP} warm-up + ${PERF_FRAMES} frames per condition)…`);
  const perf = await page.evaluate(({ warmup, frames }) => window.wgpuDemo.runBenchmark({ warmup, frames }), { warmup: PERF_WARMUP, frames: PERF_FRAMES });

  fs.writeFileSync(path.join(OUT, 'run.json'), JSON.stringify({ setup, perf, frames: FRAMES, pose: POSE_T, stride: STRIDE,
    triangles: { static: A.triangleCount, movers: moverTris.length / 60 }, swiftshader: SWIFTSHADER, logs }, null, 2));
  await browser.close();
  server.close();
  console.log('Scoring…');
  execFileSync('python3', [path.join(__dirname, 'eval_movers.py'), OUT], { stdio: 'inherit' });
}

main().catch(e => { console.error(e); process.exit(1); });
