import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/recharts/') || id.includes('/node_modules/victory-vendor/')) return 'recharts-vendor';
          if (id.includes('/node_modules/@visx/') || id.includes('/node_modules/d3-array/')) return 'ohlc-vendor';
          if (id.includes('/node_modules/motion/') || id.includes('/node_modules/motion-dom/') || id.includes('/node_modules/motion-utils/')) return 'motion-vendor';
          if (id.includes('/node_modules/animejs/')) return 'anime-vendor';
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
