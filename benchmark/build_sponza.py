#!/usr/bin/env python3
"""Converts the Dabrovic Sponza into the WebGPU demo's "Sponza + dragon" scene (webgpu.html).

    npm i --no-save jp.keijiro.dabrovic-sponza@1.0.0
    python3 benchmark/build_sponza.py [--pkg node_modules/jp.keijiro.dabrovic-sponza] [--voxel 0.12]

Writes into assets/stress/ (gitignored; the model is not redistributed here):
  sponza-mesh.b64.txt  base64(gzip([u32 nVerts, nTris, 1, 1], f32 xyz, f32 normal, f32 rgb albedo, f32 roughness, u32 abc))
  sponza-sdf.b64.txt   base64(gzip(FP16 distance field)), unsigned: the model is not closed (3.5% open edges)
  sponza.json          SDF bounds/dims, camera, floor height, triangle count
Z-up metres (the OBJ is Y-up metres). Albedo per material = linear average of its diffuse texture; smooth
normals with a 40-degree crease. Sponza: Marko Dabrovic (cgtechniques.com, 2002); UVs and fixes by Kenzie
Lamar (Vicarious Visions); free for non-commercial use with attribution.
"""
import argparse
import base64
import gzip
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--pkg', default=str(ROOT / 'node_modules' / 'jp.keijiro.dabrovic-sponza'))
ap.add_argument('--voxel', type=float, default=0.12)
ap.add_argument('--out', default=str(ROOT / 'assets' / 'stress'))
a = ap.parse_args()
src = Path(a.pkg) / 'DabrovicSponza'
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# ---- materials: linear average texture colour ----
tex_dir = src / 'Textures'
tex_files = {p.name.lower(): p for p in tex_dir.iterdir()}
albedo, mat = {}, None
for line in (src / 'sponza.mtl').read_text(errors='ignore').splitlines():
    t = line.split()
    if not t:
        continue
    if t[0] == 'newmtl':
        mat = t[1]
        albedo[mat] = [0.7, 0.7, 0.7]
    elif t[0] == 'Kd' and mat:
        albedo[mat] = [float(x) for x in t[1:4]]
    elif t[0] == 'map_Kd' and mat:
        f = tex_files.get(t[-1].lower())
        if f:
            px = np.asarray(Image.open(f).convert('RGB').resize((64, 64)), np.float32) / 255.0
            lin = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4)
            albedo[mat] = [float(x) for x in lin.reshape(-1, 3).mean(0)]
roughness = {m: (0.35 if m == 'sp_00_pod' else 0.8) for m in albedo}  # polished floor, matte stone elsewhere

# ---- geometry ----
V, tris, tmat = [], [], []
cur = None
for line in (src / 'sponza.obj').read_text(errors='ignore').splitlines():
    if line.startswith('v '):
        x, y, z = (float(s) for s in line.split()[1:4])
        V.append((x, -z, y))  # Y-up -> Z-up (rotation, keeps winding)
    elif line.startswith('usemtl'):
        cur = line.split()[1]
    elif line.startswith('f '):
        idx = [int(s.split('/')[0]) for s in line.split()[1:]]
        idx = [i - 1 if i > 0 else len(V) + i for i in idx]
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
            tmat.append(cur)
P = np.array(V, np.float64)
T = np.array(tris, np.int64)
fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
area2 = np.linalg.norm(fn, axis=1)
keep = area2 > 1e-10
T, fn, area2 = T[keep], fn[keep], area2[keep]
tmat = [m for m, k in zip(tmat, keep) if k]
fu = fn / area2[:, None]

# Smooth normals across faces sharing a welded position, within a 40-degree crease
weld = {}
wid = np.array([weld.setdefault((round(x, 4), round(y, 4), round(z, 4)), len(weld)) for x, y, z in P])
faces_of = defaultdict(list)
for f, tri in enumerate(T):
    for v in tri:
        faces_of[wid[v]].append(f)
cos_crease = math.cos(math.radians(40))
verts, vkey = [], {}
out_idx = np.zeros_like(T)
for f, tri in enumerate(T):
    for c, v in enumerate(tri):
        n = np.zeros(3)
        for g in faces_of[wid[v]]:
            if fu[g] @ fu[f] >= cos_crease:
                n += fn[g]  # area weighted
        n /= np.linalg.norm(n) or 1.0
        key = (wid[v], tmat[f], tuple(np.round(n, 3)))
        if key not in vkey:
            vkey[key] = len(verts)
            verts.append((*P[v], *n, *albedo.get(tmat[f], [0.7] * 3), roughness.get(tmat[f], 0.8)))
        out_idx[f, c] = vkey[key]
VA = np.array(verts, np.float32)
nv, nt = len(VA), len(T)
lo, hi = P.min(0), P.max(0)
print(f'Sponza: {nt:,} triangles, {nv:,} vertices, bounds {np.round(lo, 2)} .. {np.round(hi, 2)}')

# Floor height: the most common height among large upward-facing triangles
up = fu[:, 2] > 0.95
zs = np.round(P[T[up, 0], 2], 2)
floor_z = float(np.bincount(((zs - zs.min()) * 100).astype(int), weights=area2[up]).argmax() / 100 + zs.min())

blob = (np.array([nv, nt, 1, 1], np.uint32).tobytes() + VA[:, 0:3].tobytes() + VA[:, 3:6].tobytes()
        + VA[:, 6:9].tobytes() + VA[:, 9].copy().tobytes() + out_idx.astype(np.uint32).tobytes())
(out / 'sponza-mesh.b64.txt').write_text(base64.b64encode(gzip.compress(blob, 9)).decode())

# ---- unsigned SDF with the repo's exact CPU baker ----
vox, pad, block = a.voxel, 0.5, 8
dims = [int(math.ceil((hi[k] - lo[k] + 2 * pad) / vox / block) * block) for k in range(3)]
span = [d * vox for d in dims]
sdf_min = [float((lo[k] + hi[k]) / 2 - span[k] / 2) for k in range(3)]
obj_tmp, f16_tmp = out / 'sponza-zup.tmp.obj', out / 'sponza.tmp.f16'
with open(obj_tmp, 'w') as f:
    f.writelines(f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in P)
    f.writelines(f'f {i + 1} {j + 1} {k + 1}\n' for i, j, k in T)
cmd = ['node', str(ROOT / 'bin' / 'bake-sdf.mjs'), '--obj', str(obj_tmp), '--lo', *map(str, sdf_min), '--span', *map(str, span),
       '--res-xyz', *map(str, dims), '--clip', '1.5', '--unsigned', '--f16', '--validate', '400', '--out', str(f16_tmp)]
print('Baking', 'x'.join(map(str, dims)), f'unsigned SDF ({vox * 100:.0f} cm voxels)…', flush=True)
stats = json.loads(subprocess.check_output(cmd, text=True).strip().splitlines()[-1])
raw = f16_tmp.read_bytes()
assert len(raw) == dims[0] * dims[1] * dims[2] * 2, 'unexpected SDF size'
(out / 'sponza-sdf.b64.txt').write_text(base64.b64encode(gzip.compress(raw, 9)).decode())
obj_tmp.unlink()
f16_tmp.unlink()

meta = {
    'triangles': nt, 'vertices': nv, 'boundsMin': lo.tolist(), 'boundsMax': hi.tolist(), 'floorZ': floor_z,
    'sdfMin': sdf_min, 'sdfSpan': span, 'sdfDims': dims, 'voxel': vox, 'unsigned': True,
    'bake': {'ms': stats['ms']['total'], 'validate': stats.get('validate')},
    'camera': {'position': [-12.6, -1.4, 2.2], 'target': [1.5, 0.5, 2.4]},
    'credit': 'Sponza by Marko Dabrovic (cgtechniques.com, 2002); UVs and crack fixes by Kenzie Lamar (Vicarious Visions)',
}
(out / 'sponza.json').write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ('triangles', 'floorZ', 'sdfDims', 'bake')}))
for f in ('sponza-mesh.b64.txt', 'sponza-sdf.b64.txt'):
    print(f'{f}: {(out / f).stat().st_size / 1048576:.2f} MiB')
