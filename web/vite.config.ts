import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/app/',
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/flow': 'http://127.0.0.1:8001',
      '/team': 'http://127.0.0.1:8001',
      '/agent': 'http://127.0.0.1:8001',
      '/ask': 'http://127.0.0.1:8001',
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
