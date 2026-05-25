// web/src/routes/guia.tsx
// Guía de conexión (Fase 5, Plan 02): esqueleto de ruta navegable.
// El contenido 3D (escena Three.js, overlays, pasos de cableado) se añade en el Plan 05-02.

import { A } from '@solidjs/router';
import { ThemeToggle } from '../components/ThemeToggle';

const BTN =
  'rounded-md border border-border px-3 py-1.5 text-sm text-text-primary ' +
  'hover:border-accent hover:bg-bg-surface transition-colors';

function IconBack() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

export default function Guia() {
  return (
    <div class="min-h-screen bg-bg-app flex flex-col">
      <header class="flex items-center justify-between border-b border-border px-6 h-14">
        <div class="flex items-center gap-3">
          <A href="#/" class={BTN} aria-label="Volver al hub">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Guía de conexión</span>
        </div>
        <ThemeToggle />
      </header>

      <main class="flex-1">
        <div
          id="guia-container"
          style="width:100%;height:calc(100vh - 3.5rem)"
        />
      </main>
    </div>
  );
}
