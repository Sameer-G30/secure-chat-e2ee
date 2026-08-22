import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['libsodium-wrappers']
  },
  resolve: {
    alias: {
      // Use the correct export path for libsodium
      'libsodium-wrappers': 'libsodium-wrappers/dist/modules-sumo/libsodium-wrappers.js'
    }
  }
})