#!/usr/bin/env node
/**
 * Build a shadow copy of docs/*.md with MathJax-style math syntax repaired.
 *
 * The Astro content collection reads from `web/src/.processed-docs/` (this
 * script's output), not directly from `docs/`. The original docs/ stays
 * pristine — no content edits, per the project's content rules.
 *
 * Repaired patterns:
 *   1. `$$X$$` used inline (mid-paragraph, single-line, no matrix markers)
 *      → `$X$` so remark-math treats it as inline math.
 *   2. Mismatched delimiters like `$X$$, $$Y$` → `$X$, $Y$` by collapsing the
 *      middle `$$ ... $$` pair.
 *
 * Real multi-line display blocks (with `\begin{...}` or `\\` line breaks) are
 * left alone — those parse correctly as display math.
 *
 * Run via the `predev` / `prebuild` npm script.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');     // web/
const REPO_ROOT = path.resolve(PROJECT_ROOT, '..');     // terra-perceive/
const DOCS_DIR = path.join(REPO_ROOT, 'docs');
const SHADOW_DIR = path.join(PROJECT_ROOT, 'src', '.processed-docs');

function fixMath(source) {
  // Pass 1 — single-line `$$inline$$` → `$inline$`.
  let out = source.replace(/\$\$([^$\n]*?)\$\$/g, (match, inner) => {
    if (inner.includes('\\begin{') || inner.includes('\\\\')) return match;
    if (inner.length > 100) return match;
    return `$${inner}$`;
  });

  // Pass 2 — multi-line `$$...$$` blocks need `$$` on its own line for
  // micromark-extension-math to recognise them. Insert a newline after the
  // opening `$$` and before the closing `$$` when the content spans multiple
  // lines. The `[\s\S]*?` matches across newlines; we only rewrite blocks
  // that actually span lines AND already contain content adjacent to the
  // delimiters (the broken pattern).
  out = out.replace(/\$\$([^\n]+(?:\n(?!\$\$)[^\n]*)+?)\$\$/g, (match, inner) => {
    // Sanity: must actually span multiple lines.
    if (!inner.includes('\n')) return match;
    return `$$\n${inner.trim()}\n$$`;
  });

  return out;
}

function main() {
  if (!fs.existsSync(DOCS_DIR)) {
    console.error(`[build-shadow-docs] docs/ not found at ${DOCS_DIR}`);
    process.exit(1);
  }

  fs.mkdirSync(SHADOW_DIR, { recursive: true });

  const entries = fs.readdirSync(DOCS_DIR).filter((f) => f.endsWith('.md'));
  let fixedCount = 0;
  for (const filename of entries) {
    const srcPath = path.join(DOCS_DIR, filename);
    const dstPath = path.join(SHADOW_DIR, filename);
    const src = fs.readFileSync(srcPath, 'utf-8');
    const fixed = fixMath(src);
    fs.writeFileSync(dstPath, fixed, 'utf-8');
    if (src !== fixed) fixedCount++;
  }

  console.log(
    `[build-shadow-docs] wrote ${entries.length} files to ${path.relative(PROJECT_ROOT, SHADOW_DIR)} ` +
      `(${fixedCount} had math syntax fixes applied)`
  );
}

main();
