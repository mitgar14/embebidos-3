// web/src/lib/theme.ts
import { createSignal, createEffect } from 'solid-js';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';

/** Lee la preferencia guardada o la infiere del sistema (fallback: dark). */
function resolveInitialTheme(): Theme {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

// El tema inicial ya fue aplicado al DOM por el script anti-FOUC de index.html.
// Solo necesitamos sincronizar el signal con ese estado inicial.
const [theme, setTheme] = createSignal<Theme>(resolveInitialTheme());

/** Efecto reactivo: cuando el signal cambia, actualiza el DOM y localStorage. */
createEffect(() => {
  const t = theme();
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem(STORAGE_KEY, t);
});

/** Toggle entre claro y oscuro. */
export function toggleTheme(): void {
  setTheme(t => (t === 'dark' ? 'light' : 'dark'));
}

/** Signal de solo lectura para consumir en componentes. */
export { theme };
