import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Posts are loaded from `src/.processed-docs/`, a shadow copy of `../docs/`
// rebuilt by `scripts/build-shadow-docs.mjs` (run via predev/prebuild npm
// scripts). The shadow exists to repair `$$inline$$` math syntax that the
// original docs use but micromark rejects — see the script header for detail.
// The shadow dir is gitignored.
const posts = defineCollection({
  loader: glob({
    pattern: '*.md',
    base: './src/.processed-docs',
  }),
  schema: z
    .object({
      title: z.string().optional(),
      date: z.coerce.date().optional(),
      layout: z.string().optional(),
      description: z.string().optional(),
    })
    .passthrough(),
});

export const collections = { posts };
