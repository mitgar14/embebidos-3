// web/vite.config.ts
import { defineConfig } from 'vite';
import tailwindcss  from '@tailwindcss/vite';
import solidPlugin  from 'vite-plugin-solid';

export default defineConfig({
  plugins: [
    tailwindcss(),  // PRIMERO: procesa CSS antes de que Solid compile JSX
    solidPlugin(),
  ],
  server:  { port: 3000 },
  preview: { port: 4173 },
  build:   { target: 'esnext' },
});
