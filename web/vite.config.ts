import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'; // Use the Tailwind CSS v4 Vite plugin
import tsconfigPaths from 'vite-tsconfig-paths';

// [https://vitejs.dev/config/](https://vitejs.dev/config/)
export default defineConfig({
  plugins: [
    tsconfigPaths(), // Enables tsconfig path aliases like @/*
    tailwindcss(),   // Integrates Tailwind CSS v4 via its Vite plugin
    react()          // Standard React plugin
  ]
});