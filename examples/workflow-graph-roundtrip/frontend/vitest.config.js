import { defineConfig } from 'vitest/config';

// Standalone Vitest config (kept separate from vite.config.js so the unit tests
// run without the Svelte plugin or a dev server — they cover the pure field /
// operation helpers only, and must run fully offline).
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
  },
});
