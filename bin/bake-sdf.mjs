#!/usr/bin/env node
// Zero-dependency CLI wrapper for tools/bake_sdf.cpp.
// Compiles the native C++17 baker on first run into os.tmpdir()/fast-sdf-ao-bake
// (rebuilding automatically when tools/bake_sdf.cpp changes) and forwards all arguments.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const srcPath = path.join(repoRoot, 'tools', 'bake_sdf.cpp');
const binName = process.platform === 'win32' ? 'fast-sdf-ao-bake.exe' : 'fast-sdf-ao-bake';
const binPath = path.join(os.tmpdir(), binName);

const args = process.argv.slice(2);

if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
  console.log(`fast-sdf-ao bake-sdf — Exact multi-threaded signed distance field baker

Usage:
  npx bake-sdf --obj scene.obj --lo x y z --span sx sy sz --out scene.f16 [options]
  npx bake-sdf --points scan.ply --lo x y z --span sx sy sz --out scan.f32 [options]
  npx bake-sdf --boxes boxes.json --lo x y z --span sx sy sz --out boxes.f16 [options]

Options:
  --res N                  Cubic resolution N^3 (default: 128)
  --res-xyz NX NY NZ       Anisotropic / isotropic grid dimensions (multiples of --block)
  --f16                    Write IEEE-754 binary16 (half-float, 2 bytes/voxel)
  --clip D                 Clamp distances to [-D, +D] (default: 1.5)
  --unsigned               Distances only, no sign (for meshes that are not closed)
  --sign z|vote            Sign by the +z winding number (default) or by the majority of the +x/+y/+z
                           winding numbers, robust to holes in nearly closed meshes (e.g. architecture)
  --reach R                Skip resolving voxels beyond R + 1 voxel diagonal
  --brick-out FILE         Write 2-level sparse narrow-band 8^3 brick map
  --threads N              Worker thread count (default: all cores)
  --block B                Lipschitz culling block size (default: 8, 0 disables)
  --validate K             Brute-force verify K random voxels after baking (exits 1 on mismatch)

Example:
  npx bake-sdf --obj scene.obj --lo -1 -1 -1 --span 26 18 8 --res-xyz 256 176 80 --clip 1.5 --f16 --validate 2000 --out scene.f16
`);
  process.exit(0);
}

if (!fs.existsSync(srcPath)) {
  console.error(`Error: baker source not found at ${srcPath}`);
  process.exit(1);
}

const needBuild = !fs.existsSync(binPath) || fs.statSync(binPath).mtimeMs < fs.statSync(srcPath).mtimeMs;
if (needBuild) {
  const compiler = process.env.CXX || 'c++';
  const flags = ['-O3', '-std=c++17', '-pthread'];
  if (process.platform === 'darwin') {
    try {
      const sdk = execFileSync('xcrun', ['--show-sdk-path'], { encoding: 'utf8' }).trim();
      const cxxInc = path.join(sdk, 'usr', 'include', 'c++', 'v1');
      if (fs.existsSync(cxxInc)) flags.push('-isystem', cxxInc);
    } catch {
      // Fall back to default compiler include paths
    }
  }
  const build = spawnSync(compiler, [...flags, srcPath, '-o', binPath], { stdio: 'inherit' });
  if (build.status !== 0) {
    console.error(`Failed to compile ${srcPath} with ${compiler}`);
    process.exit(build.status ?? 1);
  }
}

const res = spawnSync(binPath, args, { stdio: 'inherit' });
process.exit(res.status ?? 1);
