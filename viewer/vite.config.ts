import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  build: {
    target: 'esnext',
  },
  worker: {
    format: 'es', // required for geotiff's decoder pool workers
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
  },
});
