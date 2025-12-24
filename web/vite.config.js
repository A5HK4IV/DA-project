import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcdd from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),
  tailwindcdd(),
  ],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:5000', rewrite: (path) => path.replace(/^\/api/, '') },
    }
  }
})
