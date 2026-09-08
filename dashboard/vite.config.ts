import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

/**
 * The dashboard talks to the FastAPI scoring service.
 *
 * In development, `/api` is proxied to the local service so the browser sees a
 * single origin and CORS never enters the picture. In production the built
 * assets are served by that same service, so the identical relative URLs work
 * unchanged — there is no environment-specific API base URL to get wrong.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ready': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Recharts is by far the largest dependency and changes rarely, so it is
    // split into its own chunk that browsers keep across deploys. React is not
    // split: it sits inside Recharts' own dependency graph, so forcing it into a
    // separate chunk just produces an empty file.
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
        },
      },
    },
  },
});
