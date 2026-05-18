import { visit } from 'unist-util-visit';

/**
 * Rewrite all relative image src in markdown to absolute public-URL paths,
 * optionally prefixed with the site's `base` URL.
 *
 * The docs/ markdown was authored when the site lived at the project root,
 * so image references use paths like:
 *   - `assets/m13/foo.gif`        → /<base>/assets/m13/foo.gif (via symlink)
 *   - `../results_m4/foo.png`     → /<base>/results_m4/foo.png (via symlink)
 *
 * Without this rewrite, Astro's content-asset pipeline treats them as imports
 * and fails because they live outside the content collection root.
 *
 * Sibling directories are symlinked into web/public/ at scaffold time.
 *
 * @param {{ base?: string }} options
 *   base — leading-slashed, trailing-slashed base URL (e.g. '/terra-perceive/').
 *          Defaults to '/' if not provided.
 */
export function remarkPublicImages(options = {}) {
  const base = (options.base || '/').replace(/\/?$/, '/'); // ensure trailing slash
  return (tree) => {
    visit(tree, 'image', (node) => {
      if (!node.url) return;
      if (
        node.url.startsWith('http://') ||
        node.url.startsWith('https://') ||
        node.url.startsWith(base) ||
        node.url.startsWith('data:')
      ) {
        return;
      }
      const cleaned = node.url.replace(/^(\.\.\/|\.\/|\/)+/, '');
      node.url = base + cleaned;
    });
  };
}
