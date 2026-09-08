import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: { proxy: { '/api': 'http://127.0.0.1:8585' } },
  // Keep placeholder.txt so the Go embed package also compiles before the
  // first frontend build in a fresh checkout.
  build: { outDir: 'dist', emptyOutDir: false },
});
