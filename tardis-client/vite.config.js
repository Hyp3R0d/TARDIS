import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Electron loads the production renderer from a file:// URL. Relative
  // asset paths keep the bundled UI self-contained instead of resolving to
  // the filesystem root (C:\\assets).
  base: './',
  emptyOutDir: true,
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8787',
    },
  },
})
