#!/usr/bin/env node
/**
 * Post-build cleanup of dist/assets/, dist/results_m{4,5,6}/.
 *
 * Astro's `public/` symlinks copy the entire docs/assets/ tree (and the
 * results_m{4,5,6} symlinks) into dist/. That includes large files the blog
 * never references — backup `.prev_*` copies, alternate-encoding `.transparent_*`
 * copies, mp4/gif pairs where only one is used, etc. This script walks dist/
 * and deletes media files (mp4/gif/png/jpg/svg/webm) that no markdown file
 * under docs/ references, plus any file matching the backup-name patterns.
 *
 * Run by `postbuild` in package.json — fires once after `astro build`
 * completes, before the dist/ is uploaded to Pages.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(WEB_ROOT, '..');
const DIST_DIR = path.join(WEB_ROOT, 'dist');
const DOCS_DIR = path.join(REPO_ROOT, 'docs');
const SHADOW_DIR = path.join(WEB_ROOT, 'src', '.processed-docs');
const SRC_DIR = path.join(WEB_ROOT, 'src');

const BACKUP_NAME_PATTERNS = [
  /\.prev[_.][\w.-]+$/,    // foo.gif.prev_v1, foo.svg.prev_overlap
  /\.transparent_[\w.-]+$/, // foo.svg.transparent_v2
  /\.tmp\.[\w.-]+$/,        // foo.tmp.svg
  /_tmp\.[\w]+$/,           // foo_tmp.svg
];

const MEDIA_EXTENSIONS = new Set(['.mp4', '.webm', '.gif', '.png', '.jpg', '.jpeg', '.svg']);

function collectReferencedAssets() {
  // Scan markdown sources (original + shadow) AND Astro components / TS
  // sources. The hero clips on the landing page are referenced from
  // src/pages/index.astro, not from any .md file.
  const mdSources = [DOCS_DIR, SHADOW_DIR].filter((d) => fs.existsSync(d));
  const codeSources = [SRC_DIR];

  const referenced = new Set();
  // Same regex matches markdown img/link `(path)`, src/href attrs, and the
  // `withBase('/assets/...')` helper calls used in Astro components.
  const re = /(?:\(|src=["']|href=["']|['"`])([^)"'`]+\.(?:mp4|webm|gif|png|jpg|jpeg|svg))/gi;

  function scanFile(filepath) {
    const content = fs.readFileSync(filepath, 'utf-8');
    let m;
    re.lastIndex = 0;
    while ((m = re.exec(content))) {
      const cleaned = m[1].replace(/^(\.\.\/|\.\/|\/)+/, '');
      referenced.add(cleaned);
    }
  }

  for (const dir of mdSources) {
    for (const filename of fs.readdirSync(dir)) {
      if (filename.endsWith('.md')) scanFile(path.join(dir, filename));
    }
  }
  for (const dir of codeSources) {
    for (const filepath of walk(dir)) {
      const ext = path.extname(filepath);
      if (['.astro', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.html'].includes(ext)) {
        scanFile(filepath);
      }
    }
  }
  return referenced;
}

function* walk(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      yield* walk(p);
    } else if (ent.isFile()) {
      yield p;
    }
  }
}

function isBackupName(filename) {
  return BACKUP_NAME_PATTERNS.some((re) => re.test(filename));
}

function main() {
  if (!fs.existsSync(DIST_DIR)) {
    console.error('[prune-dist] dist/ not found — run astro build first.');
    process.exit(1);
  }

  const referenced = collectReferencedAssets();
  console.log(`[prune-dist] ${referenced.size} unique asset paths referenced in markdown`);

  let kept = 0;
  let deletedBackup = 0;
  let deletedUnref = 0;
  let bytesFreed = 0;

  for (const filepath of walk(DIST_DIR)) {
    // Skip the Astro build output dir _astro/ (CSS/JS bundles only — no media).
    if (filepath.includes(`${path.sep}_astro${path.sep}`)) continue;

    const filename = path.basename(filepath);
    const ext = path.extname(filepath).toLowerCase();
    const size = fs.statSync(filepath).size;

    // Backup-pattern check FIRST — runs regardless of extension, since the
    // ext on "foo.mp4.prev_v1" is ".prev_v1", not ".mp4".
    if (isBackupName(filename)) {
      fs.unlinkSync(filepath);
      deletedBackup++;
      bytesFreed += size;
      continue;
    }

    // For media files, also check whether anything references them.
    if (MEDIA_EXTENSIONS.has(ext)) {
      const rel = path.relative(DIST_DIR, filepath);
      if (!referenced.has(rel)) {
        fs.unlinkSync(filepath);
        deletedUnref++;
        bytesFreed += size;
        continue;
      }
      kept++;
    }
    // Non-media, non-backup file — leave alone (HTML, CSS, sitemap, etc.)
  }

  const mbFreed = (bytesFreed / 1024 / 1024).toFixed(1);
  console.log(
    `[prune-dist] kept ${kept} files · removed ${deletedBackup} backup, ` +
      `${deletedUnref} unreferenced · freed ${mbFreed} MB`
  );
}

main();
