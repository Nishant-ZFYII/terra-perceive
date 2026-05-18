import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';
import expressiveCode from 'astro-expressive-code';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkPublicImages } from './src/plugins/remark-public-images.mjs';
import { remarkFixInlineMath } from './src/plugins/remark-fix-inline-math.mjs';
import { rehypeFigureCaptions } from './src/plugins/rehype-figure-captions.mjs';

// Production deploys to https://nishant-zfyii.github.io/terra-perceive/.
// `base` is set unconditionally so dev mode matches prod URL structure —
// localhost serves at http://localhost:4321/terra-perceive/. All internal
// links and assets MUST go through the `withBase()` helper or the
// `remark-public-images` plugin (both honor `import.meta.env.BASE_URL`)
// so they stay correct in both environments.
export default defineConfig({
  site: 'https://nishant-zfyii.github.io',
  base: '/terra-perceive',
  trailingSlash: 'ignore',
  integrations: [
    expressiveCode({
      themes: ['github-dark', 'github-light'],
      themeCssSelector: (theme) => `[data-theme='${theme === 'github-dark' ? 'dark' : 'light'}']`,
      styleOverrides: {
        borderRadius: '0.5rem',
        codeFontSize: '0.875rem',
        codeLineHeight: '1.6',
      },
    }),
    mdx(),
    tailwind({ applyBaseStyles: false }),
    sitemap(),
  ],
  markdown: {
    // remarkFixInlineMath is a no-op (math repair moved to the shadow-docs
    // build script). remarkPublicImages receives the same `base` so img URLs
    // are correctly prefixed for production. remarkMath converts math syntax
    // to AST nodes that rehypeKatex then renders.
    remarkPlugins: [
      remarkFixInlineMath,
      [remarkPublicImages, { base: '/terra-perceive' }],
      remarkMath,
    ],
    rehypePlugins: [[rehypeKatex, { strict: false }], rehypeFigureCaptions],
    shikiConfig: { theme: 'github-dark' },
  },
  vite: {
    server: {
      fs: {
        allow: ['..'],
      },
    },
  },
});
