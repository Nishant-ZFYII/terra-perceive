/**
 * Build a URL that respects the configured `base` in astro.config.mjs.
 *
 * Astro sets `import.meta.env.BASE_URL` based on the active base config —
 * `/terra-perceive/` in production, `/terra-perceive/` in dev when the same
 * base is set, `/` when no base.
 *
 * Pass either:
 *   - a root-absolute path like `/assets/m10/foo.mp4` (leading slash stripped)
 *   - a relative path like `assets/m10/foo.mp4`
 *
 * Returns a base-prefixed URL safe to use as `src` / `href`.
 */
export function withBase(path: string): string {
  // Astro docs claim BASE_URL always has a trailing slash, but observation
  // shows it can be `/terra-perceive` (no slash) when base is set without
  // one. Normalize defensively.
  const rawBase = import.meta.env.BASE_URL || '/';
  const base = rawBase.endsWith('/') ? rawBase : rawBase + '/';
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('//') || path.startsWith('data:')) {
    return path;
  }
  if (path.startsWith(base)) return path;
  const clean = path.replace(/^\/+/, '');
  return base + clean;
}
