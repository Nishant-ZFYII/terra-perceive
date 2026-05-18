#!/usr/bin/env node
/**
 * Recreate the web/public/ symlinks that point at sibling directories
 * outside the web/ subtree (docs/assets, results_m{4,5,6}).
 *
 * These dirs live above web/ in the repo layout (mirroring the project
 * root). The Astro build needs them inside web/public/ so they end up in
 * dist/. Symlinks committed with absolute paths break on CI runners with
 * different checkout roots — this script rewrites them as RELATIVE
 * symlinks every time, so it doesn't matter what git stores or where the
 * repo is checked out.
 *
 * Wired into `predev` and `prebuild` in package.json, so it runs before
 * any astro command.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(WEB_ROOT, 'public');

// Each entry: link name inside web/public/, and the symlink target as a
// path RELATIVE to web/public/. From web/public/, going up two levels lands
// at the repo root, where docs/ and results_m*/ live.
const LINKS = [
  { name: 'assets',     target: '../../docs/assets' },
  { name: 'results_m4', target: '../../results_m4' },
  { name: 'results_m5', target: '../../results_m5' },
  { name: 'results_m6', target: '../../results_m6' },
];

fs.mkdirSync(PUBLIC_DIR, { recursive: true });

for (const { name, target } of LINKS) {
  const linkPath = path.join(PUBLIC_DIR, name);
  const absoluteTarget = path.resolve(PUBLIC_DIR, target);

  // Skip silently if the source dir doesn't exist — keeps the script
  // tolerant of repos that don't have all the sibling dirs (e.g. a future
  // contributor without results_m6/).
  if (!fs.existsSync(absoluteTarget)) {
    console.warn(`[setup-public-symlinks] skipping ${name}: target ${absoluteTarget} does not exist`);
    continue;
  }

  // Remove anything currently at the link path (broken symlink, old
  // symlink with wrong target, or stale file). lstat doesn't follow
  // symlinks so it catches broken ones too.
  try {
    const stats = fs.lstatSync(linkPath);
    if (stats.isDirectory() && !stats.isSymbolicLink()) {
      console.warn(`[setup-public-symlinks] ${linkPath} is a real directory, leaving alone`);
      continue;
    }
    fs.unlinkSync(linkPath);
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }

  fs.symlinkSync(target, linkPath, 'dir');
  console.log(`[setup-public-symlinks] linked public/${name} → ${target}`);
}
