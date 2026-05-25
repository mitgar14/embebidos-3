// web/src/routes/control.tsx
// Control de servos (Fase 5, Plan 03): esqueleto de ruta navegable.
// La lógica MQTT y el panel de estado dual se añaden en el Plan 05-03.

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

export default function Control() {
  return (
    <div class="min-h-screen bg-bg-app flex flex-col">
      <header class="flex items-center justify-between border-b border-border px-6 h-14">
        <div class="flex items-center gap-3">
          <A href="#/" class={BTN} aria-label="Volver al hub">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Control de servos</span>
        </div>
        <ThemeToggle />
      </header>

      <main class="flex-1 flex items-center justify-center">
        <p class="text-sm text-text-secondary">Cargando…</p>
      </main>
    </div>
  );
}
