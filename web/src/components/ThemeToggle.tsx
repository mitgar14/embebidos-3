// web/src/components/ThemeToggle.tsx
import { theme, toggleTheme } from '../lib/theme';

/** Icono de sol — 20px, para modo oscuro (indica que se puede cambiar a claro) */
function SunIcon() {
  return (
    <svg
      aria-hidden="true"
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      {/* Circulo central */}
      <circle cx="10" cy="10" r="3.5" />
      {/* Rayos cardinales y diagonales */}
      <line x1="10" y1="1.5"  x2="10" y2="3.5"  />
      <line x1="10" y1="16.5" x2="10" y2="18.5" />
      <line x1="1.5"  y1="10" x2="3.5"  y2="10" />
      <line x1="16.5" y1="10" x2="18.5" y2="10" />
      <line x1="3.55"  y1="3.55"  x2="4.97"  y2="4.97"  />
      <line x1="15.03" y1="15.03" x2="16.45" y2="16.45" />
      <line x1="16.45" y1="3.55"  x2="15.03" y2="4.97"  />
      <line x1="4.97"  y1="15.03" x2="3.55"  y2="16.45" />
    </svg>
  );
}

/** Icono de luna creciente — 20px, para modo claro (indica que se puede cambiar a oscuro) */
function MoonIcon() {
  return (
    <svg
      aria-hidden="true"
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M17 13.5A7.5 7.5 0 0 1 6.5 3a7.5 7.5 0 1 0 10.5 10.5z" />
    </svg>
  );
}

/** Botón que alterna entre tema claro y oscuro. Sin emojis, con SVG inline. */
export function ThemeToggle() {
  return (
    <button
      onClick={toggleTheme}
      aria-label={`Cambiar a tema ${theme() === 'dark' ? 'claro' : 'oscuro'}`}
      class="rounded-md p-2 text-text-secondary hover:text-text-primary hover:bg-bg-surface"
    >
      {theme() === 'dark' ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}
