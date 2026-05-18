/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter var', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['Newsreader', 'ui-serif', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        // Light mode foundation
        paper: '#fbfaf7',          // warm off-white, easier on eyes than pure white
        ink: '#1a1a1a',
        muted: '#6b7280',
        // Dark mode foundation
        carbon: '#0a0a0b',         // near-black with a hint of warmth
        chalk: '#e8e8ea',
        smoke: '#9ca3af',
        // Accent — desaturated teal that reads "engineered" in both modes
        accent: {
          DEFAULT: '#0d9488',      // teal-600
          dark: '#2dd4bf',         // teal-400 for dark mode
        },
        // Subtle surface colors
        surface: {
          light: '#f4f2ee',
          dark: '#16161a',
        },
        border: {
          light: '#e7e5e0',
          dark: '#27272a',
        },
      },
      maxWidth: {
        prose: '42rem',     // 672px — comfortable for body text
        wide: '64rem',      // 1024px — for hero images, code, page chrome
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            maxWidth: 'none',
            color: theme('colors.ink'),
            fontSize: '1.125rem',
            lineHeight: '1.75',
          },
        },
      }),
    },
  },
};
