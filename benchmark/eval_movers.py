#!/usr/bin/env python3
"""Scores a benchmark/bench_movers.cjs run: Exact BVH vs Baked SDF (original) movers against the
tools/trace_gi 512-ray reference. Writes metrics.json, report.md and images/ into the run directory.

"Mover-affected" pixels are those where the reference with movers differs from the reference without
them (AO > 0.02, GI > 0.02 in any channel, sun visibility > 0.05): the only pixels where the two mover
methods can differ, since the static scene is traced identically in both.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name('mover-bench'))
run = json.loads((out / 'run.json').read_text())
setup = run['setup']
W, H = setup['capW'], setup['capH']
valid = np.fromfile(out / 'valid.u32', dtype=np.uint32)


def ref(which):
    r = np.fromfile(out / f'ref_{which}.f32', np.float32).reshape(-1, 10)
    return {'ao': r[:, 0], 'gi': r[:, 1:4], 'sun': np.fromfile(out / f'sun_{which}.f32', np.float32)}


def gpu(mode):
    aogi = np.fromfile(out / f'gpu_{mode}_aogi.f32', np.float32).reshape(-1, 4)[valid]
    meta = np.fromfile(out / f'gpu_{mode}_meta.f32', np.float32).reshape(-1, 4)[valid]
    return {'ao': np.clip(aogi[:, 3], 0, 1), 'gi': np.maximum(aogi[:, :3], 0), 'sun': np.clip(meta[:, 0], 0, 1)}


R, S = ref('full'), ref('static')
G = {m: gpu(m) for m in ('bvh', 'sdf')}


def absdiff(a, b):
    d = np.abs(a - b)
    return d.mean(1) if d.ndim == 2 else d


masks = {
    'ao': absdiff(R['ao'], S['ao']) > 0.02,
    'gi': np.abs(R['gi'] - S['gi']).max(1) > 0.02,
    'sun': absdiff(R['sun'], S['sun']) > 0.05,
}

metrics = {'pixels': int(valid.size), 'affected': {k: int(v.sum()) for k, v in masks.items()}, 'modes': {}}
for mode, g in G.items():
    mm = {}
    for q in ('ao', 'gi', 'sun'):
        err = absdiff(g[q], R[q])
        sig = (g[q] - R[q]).mean(1) if g[q].ndim == 2 else g[q] - R[q]
        m = masks[q]
        mm[q] = {
            'mae_all': float(err.mean()),
            'mae_affected': float(err[m].mean()) if m.any() else None,
            'bias_affected': float(sig[m].mean()) if m.any() else None,
            'p95_affected': float(np.percentile(err[m], 95)) if m.any() else None,
        }
    metrics['modes'][mode] = mm
# How large the movers' effect is on those pixels (what ignoring the movers entirely would cost)
metrics['ignore_movers'] = {q: float(absdiff(S[q], R[q])[masks[q]].mean()) for q in ('ao', 'gi', 'sun')}


# ---------------- images ----------------
def to_img(vals, kind):
    img = np.full((W * H, 3), [24, 30, 40], np.float32)
    if kind == 'gray':
        v = np.clip(vals, 0, 1)[:, None] * 255
        img[valid] = np.repeat(v, 3, 1)
    elif kind == 'gi':
        x = np.maximum(vals, 0) * 1.15
        y = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
        img[valid] = np.where(y < 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055) * 255
    elif kind == 'err':  # 0 -> black, >= 0.25 -> bright (magma-like ramp)
        t = np.clip(vals / 0.25, 0, 1)[:, None]
        img[valid] = np.concatenate([255 * np.sqrt(t), 210 * t ** 1.6, 90 * t ** 3 + 30 * t], 1)
    elif kind == 'mask':
        img[valid] = np.where(vals[:, None], [235, 120, 60], [70, 76, 88])
    return Image.fromarray(img.reshape(H, W, 3).astype(np.uint8))


(out / 'images').mkdir(exist_ok=True)
rows = [
    ('AO', 'gray', R['ao'], G['bvh']['ao'], G['sdf']['ao']),
    ('Sun visibility', 'gray', R['sun'], G['bvh']['sun'], G['sdf']['sun']),
    ('GI bounce', 'gi', R['gi'], G['bvh']['gi'], G['sdf']['gi']),
]
cols = ['Reference (512 rays)', 'Exact BVH', 'Baked SDF (original)', '|error| Exact BVH', '|error| Baked SDF']
pad, head = 6, 18
montage = Image.new('RGB', (len(cols) * (W + pad) + pad + 110, len(rows) * (H + pad) + pad + head), (16, 20, 28))
draw = ImageDraw.Draw(montage)
for c, name in enumerate(cols):
    draw.text((110 + pad + c * (W + pad), 4), name, fill=(225, 230, 238))
for r, (label, kind, ref_v, bvh_v, sdf_v) in enumerate(rows):
    y = head + pad + r * (H + pad)
    draw.text((8, y + H // 2 - 6), label, fill=(225, 230, 238))
    tiles = [to_img(ref_v, kind), to_img(bvh_v, kind), to_img(sdf_v, kind),
             to_img(absdiff(bvh_v, ref_v), 'err'), to_img(absdiff(sdf_v, ref_v), 'err')]
    for c, t in enumerate(tiles):
        montage.paste(t, (110 + pad + c * (W + pad), y))
montage.save(out / 'images' / 'quality_montage.png')
mask_img = Image.new('RGB', (3 * (W + pad) + pad, H + pad * 2 + head), (16, 20, 28))
d2 = ImageDraw.Draw(mask_img)
for c, q in enumerate(('ao', 'gi', 'sun')):
    d2.text((pad + c * (W + pad), 4), f'mover-affected: {q} ({masks[q].sum():,} px)', fill=(225, 230, 238))
    mask_img.paste(to_img(masks[q], 'mask'), (pad + c * (W + pad), head + pad))
mask_img.save(out / 'images' / 'affected_masks.png')

# ---------------- report ----------------
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
f = lambda v, d=4: '—' if v is None else f'{v:.{d}f}'
mB, mS = metrics['modes']['bvh'], metrics['modes']['sdf']
lines = [
    '# Moving-object benchmark: Exact BVH vs Baked SDF (original)', '',
    f"Scene: {run.get('sceneLabel', 'curved Atrium + 8 movers')}; {int(run['triangles']['static']):,} static tris, "
    f"{int(run['triangles']['movers']):,} moving tris. {setup['width']}×{setup['height']}, `{setup['quality']}` preset, "
    f"{run['frames']} converged frames per mode, movers frozen at t = {run['pose']}. Device: {setup['device']}"
    + (' (SwiftShader, CPU-emulated WebGPU)' if run.get('swiftshader') else '') + '.', '',
    f"Reference: `tools/trace_gi`, 512 cosine rays + 64 solar-disk shadow rays per pixel on the exact world-space triangles, "
    f"{metrics['pixels']:,} pixels (every 2nd half-res pixel). Mover-affected pixels: AO {metrics['affected']['ao']:,}, "
    f"GI {metrics['affected']['gi']:,}, sun {metrics['affected']['sun']:,}.", '',
    '## Quality (mean absolute error vs reference, lower is better)', '',
    '| Metric | Exact BVH | Baked SDF (original) | Ignoring movers |', '|---|---:|---:|---:|',
]
for q, name in (('ao', 'AO'), ('sun', 'Sun visibility'), ('gi', 'GI bounce (linear RGB)')):
    lines.append(f"| {name}, mover-affected pixels | **{f(mB[q]['mae_affected'])}** | {f(mS[q]['mae_affected'])} | {f(metrics['ignore_movers'][q])} |")
    lines.append(f"| {name}, mover-affected p95 | {f(mB[q]['p95_affected'])} | {f(mS[q]['p95_affected'])} | |")
    lines.append(f"| {name}, mover-affected bias (GPU − ref) | {f(mB[q]['bias_affected'])} | {f(mS[q]['bias_affected'])} | |")
    lines.append(f"| {name}, all pixels | {f(mB[q]['mae_all'])} | {f(mS[q]['mae_all'])} | |")
lines += ['', '"Ignoring movers" is the reference without movers scored against the reference with them: the size of the effect a method has to capture.', '']
perf = run['perf']
lines += ['## Cost (median per frame, movers animating)', '', perf['markdown'], '']
lines += ['## Build cost', '', '| Mesh | Triangles | Exact BVH: CPU build (ms) | BVH memory (MiB) | Baked SDF: GPU bake (ms) |', '|---|---:|---:|---:|---:|']
names = ['dragon', 'bunny', 'teapot', 'knot', 'ring', 'plate']
meshes = [m for m in setup['meshes'] if m['blasBytes'] > 0]
for name, m in zip(names, meshes[-len(names):]):
    st = setup['stats'][name]
    lines.append(f"| {name} | {st['triangles']:,} | {st['blasMs']:.0f} | {m['blasBytes'] / 1048576:.2f} | {f(m['sdfBakeMs'], 0)} |")
lines += ['', f"Baked SDF atlas: {setup['atlasBytes'] / 1048576:.1f} MiB (16 slabs of 64³ FP16). "
          'CPU build times include packing all mover BVHs into the shared GPU buffers.', '',
          '![quality](images/quality_montage.png)', '', '![affected](images/affected_masks.png)', '']
(out / 'report.md').write_text('\n'.join(lines))
print('\n'.join(lines[:24]))
