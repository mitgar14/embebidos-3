// web/src/components/Placeholder.tsx
// Walking skeleton — demuestra el sistema de diseño dual-theme end-to-end.
// Reemplazado en Fase 2 (hub real) y Fases 3-4 (superficies completas).

import { useNavigate } from '@solidjs/router';
import { theme } from '../lib/theme';
import { ThemeToggle } from './ThemeToggle';

// Icono de velocímetro para Dashboard
function DashboardIcon() {
  return (
    <svg aria-hidden="true" width="20" height="20" viewBox="0 0 20 20" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10 2a8 8 0 1 0 4.9 14.2" />
      <path d="M10 10L14 6" />
      <circle cx="10" cy="10" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

// Icono de CPU/chip para Engine del Modelo
function EngineIcon() {
  return (
    <svg aria-hidden="true" width="20" height="20" viewBox="0 0 20 20" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <rect x="5" y="5" width="10" height="10" rx="1" />
      <line x1="8" y1="5" x2="8" y2="2" />
      <line x1="12" y1="5" x2="12" y2="2" />
      <line x1="8" y1="18" x2="8" y2="15" />
      <line x1="12" y1="18" x2="12" y2="15" />
      <line x1="5" y1="8" x2="2" y2="8" />
      <line x1="5" y1="12" x2="2" y2="12" />
      <line x1="15" y1="8" x2="18" y2="8" />
      <line x1="15" y1="12" x2="18" y2="12" />
    </svg>
  );
}

// Icono de etiqueta/tag para Labelling
function LabellingIcon() {
  return (
    <svg aria-hidden="true" width="20" height="20" viewBox="0 0 20 20" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="2" width="16" height="16" rx="2" />
      <rect x="5" y="6" width="6" height="4" rx="0.5" />
      <line x1="5" y1="14" x2="15" y2="14" />
      <line x1="5" y1="11.5" x2="13" y2="11.5" />
    </svg>
  );
}

// Icono de chevron derecha
function ChevronRightIcon() {
  return (
    <svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M6 4l4 4-4 4" />
    </svg>
  );
}

interface NavRowProps {
  label: string;
  description: string;
  icon: () => Element;
  path: string;
}

function NavRow(props: NavRowProps) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(props.path)}
      class="w-full flex items-center gap-4 h-16 px-4 border-b border-border text-left
             hover:bg-bg-surface transition-colors"
    >
      <span class="flex-shrink-0 text-text-secondary">
        <props.icon />
      </span>
      <span class="flex-1 min-w-0">
        <span class="block text-base font-semibold text-text-primary leading-tight">
          {props.label}
        </span>
        <span class="block text-sm text-text-secondary truncate">
          {props.description}
        </span>
      </span>
      <span class="flex-shrink-0 text-text-secondary">
        <ChevronRightIcon />
      </span>
    </button>
  );
}

/** Componente de prueba del walking skeleton — muestra tokens y toggle dual-theme. */
export function Placeholder() {
  return (
    <div class="min-h-screen bg-bg-app">
      {/* Barra superior con toggle */}
      <div class="flex justify-end px-6 pt-6">
        <ThemeToggle />
      </div>

      {/* Contenido centrado */}
      <main class="max-w-lg mx-auto py-12 px-6">
        {/* Encabezado */}
        <h1 class="text-2xl font-semibold text-text-primary mb-1">
          embebidos-3 — Consola Web
        </h1>
        <p class="text-sm text-text-secondary mb-8">
          Walking Skeleton — Fase 1
        </p>

        {/* Navegación: tres filas sin cards, separadas por border-b */}
        <nav class="border-t border-border">
          <NavRow
            label="Dashboard"
            description="Video en vivo + overlay de detección a 14 fps"
            icon={DashboardIcon}
            path="/dashboard"
          />
          <NavRow
            label="Engine del Modelo"
            description="Estado del modelo, build y logs en vivo"
            icon={EngineIcon}
            path="/engine"
          />
          <NavRow
            label="Labelling"
            description="Anotación de imágenes con drag/resize/export"
            icon={LabellingIcon}
            path="/labelling"
          />
        </nav>

        {/* Estado del tema — diagnóstico para el walking skeleton */}
        <p class="mt-8 font-mono text-sm text-text-secondary tabular">
          Tema: {theme() === 'dark' ? 'oscuro' : 'claro'}
        </p>
      </main>
    </div>
  );
}
