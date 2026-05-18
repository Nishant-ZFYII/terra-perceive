import { visit } from 'unist-util-visit';

/**
 * Promote markdown images to <figure><img><figcaption> blocks.
 *
 * The project's blog style uses the alt-text-as-caption pattern (alt text is
 * a full description, not a label). Minima rendered alt text invisibly for
 * sighted readers; this plugin makes it visible without changing the source
 * markdown.
 *
 * Behavior:
 *   - A paragraph containing only one <img> becomes <figure><img><figcaption>.
 *   - The first sentence/phrase of the figcaption is wrapped in <strong>
 *     (colah pattern) so it works as a scannable topic sentence.
 *   - The img's alt attribute is cleared once a visible figcaption exists —
 *     screen readers read figcaption as the figure's accessible name, and
 *     keeping alt would (a) cause "double-read" for assistive tech and
 *     (b) duplicate the caption visually when an image fails to load.
 *   - Inline images (alongside other text in the same paragraph) are left
 *     alone — they're not figures.
 *   - All <img> elements get loading="lazy" and decoding="async".
 */
export function rehypeFigureCaptions() {
  return (tree) => {
    visit(tree, 'element', (node, _index, parent) => {
      if (node.tagName !== 'img') return;

      node.properties = node.properties || {};
      if (node.properties.loading === undefined) {
        node.properties.loading = 'lazy';
      }
      if (node.properties.decoding === undefined) {
        node.properties.decoding = 'async';
      }

      // Promote to figure only when the image is the sole child of a paragraph.
      if (!parent || parent.tagName !== 'p') return;
      const significant = parent.children.filter(
        (c) => !(c.type === 'text' && c.value.trim() === '')
      );
      if (significant.length !== 1 || significant[0] !== node) return;

      const alt = (node.properties.alt || '').toString().trim();
      if (!alt) {
        parent.tagName = 'figure';
        return;
      }

      // Split the alt text into a "topic sentence" head + remainder so we can
      // bold the head. Heuristic: first period (followed by space or end) up
      // to a max of 120 chars; otherwise the whole alt becomes the head if
      // it's short. Long single-sentence alts fall back to bolding the first
      // ~12 words.
      const { head, tail } = splitForBoldHead(alt);

      const figcaptionChildren = [
        { type: 'element', tagName: 'strong', properties: {}, children: [{ type: 'text', value: head }] },
      ];
      if (tail) {
        figcaptionChildren.push({ type: 'text', value: ' ' + tail });
      }

      // Tag the img with src-extension data so CSS can target SVGs vs raster.
      const src = (node.properties.src || '').toString();
      const ext = src.split('?')[0].split('.').pop()?.toLowerCase() ?? '';
      if (ext) {
        node.properties['data-ext'] = ext;
      }

      // Clear alt now that a visible figcaption exists — avoids the double
      // caption when the image is broken and avoids screen-reader double-read.
      delete node.properties.alt;
      node.properties.alt = ''; // keep attribute present for HTML validity

      parent.tagName = 'figure';
      parent.children = [
        node,
        {
          type: 'element',
          tagName: 'figcaption',
          properties: {},
          children: figcaptionChildren,
        },
      ];
    });
  };
}

function splitForBoldHead(alt) {
  // Try first period followed by whitespace.
  const periodIdx = alt.search(/\.\s+/);
  if (periodIdx > 0 && periodIdx <= 140) {
    return { head: alt.slice(0, periodIdx + 1), tail: alt.slice(periodIdx + 2) };
  }
  // No period in range — if alt is short, the whole thing is the head.
  if (alt.length <= 120) {
    return { head: alt, tail: '' };
  }
  // Long single-clause alt: bold the first ~12 words.
  const words = alt.split(/\s+/);
  if (words.length <= 12) return { head: alt, tail: '' };
  return {
    head: words.slice(0, 12).join(' ') + '…',
    tail: words.slice(12).join(' '),
  };
}
