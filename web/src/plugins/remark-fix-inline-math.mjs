// Math pre-processing now happens at the source-string level inside
// src/content.config.ts (buildShadowDocs). This plugin is intentionally a
// no-op, kept as the registered slot in astro.config so removing it doesn't
// require touching the config.
export function remarkFixInlineMath() {
  return () => {};
}
