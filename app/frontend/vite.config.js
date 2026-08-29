import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// The React plugin enables the automatic JSX runtime (so components do not need
// to `import React`) and Fast Refresh. Without it, Vite's default classic JSX
// transform emits `React.createElement` and the app fails with
// "React is not defined".
export default defineConfig({
  plugins: [react()],
})
