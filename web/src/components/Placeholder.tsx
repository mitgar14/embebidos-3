// web/src/components/Placeholder.tsx
// Hub de inicio (Fase 2): tres destinos navegables + estado del Nano inline.
// (Las superficies destino se completan en Fases 3-4.)

import { createSignal, onCleanup } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { ThemeToggle } from './ThemeToggle';
import { NanoStatus } from './NanoStatus';
import { getWsUrl } from '../lib/ws';

/** Host del Nano configurado (se muestra discreto en el footer). */
function nanoTarget(): string {
  try { return new URL(getWsUrl()).host; } catch { return ''; }
}

// ─── Grupo de tooltips: hover-intent + skip-delay ────────────────────────────
// El primer tooltip espera OPEN_DELAY (intención real, evita roces accidentales).
// Mientras el grupo sigue activo, pasar de un tile a otro abre instantáneo.
const OPEN_DELAY   = 200; // ms antes de mostrar el primer tooltip
const GROUP_WINDOW = 300; // ms de gracia tras salir para mantener el grupo vivo

const [groupActive, setGroupActive] = createSignal(false);
let groupResetTimer: ReturnType<typeof setTimeout> | null = null;

function keepGroupActive() {
  if (groupResetTimer) { clearTimeout(groupResetTimer); groupResetTimer = null; }
  setGroupActive(true);
}
function scheduleGroupReset() {
  if (groupResetTimer) clearTimeout(groupResetTimer);
  groupResetTimer = setTimeout(() => setGroupActive(false), GROUP_WINDOW);
}

// Icono de conector/placa para Guía de conexión
function GuiaIcon() {
  return (
    <svg aria-hidden="true" width="26" height="26" viewBox="0 0 20 20" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      {/* Cuerpo de la placa */}
      <rect x="4" y="6" width="12" height="8" rx="1" />
      {/* Pines superiores */}
      <line x1="7" y1="6" x2="7" y2="3" />
      <line x1="10" y1="6" x2="10" y2="3" />
      <line x1="13" y1="6" x2="13" y2="3" />
      {/* Pines inferiores */}
      <line x1="7" y1="14" x2="7" y2="17" />
      <line x1="13" y1="14" x2="13" y2="17" />
    </svg>
  );
}

// Icono de velocímetro para Dashboard
function DashboardIcon() {
  return (
    <svg aria-hidden="true" width="26" height="26" viewBox="0 0 20 20" fill="none"
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
    <svg aria-hidden="true" width="26" height="26" viewBox="0 0 20 20" fill="none"
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
    <svg aria-hidden="true" width="26" height="26" viewBox="0 0 20 20" fill="none"
      stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="2" width="16" height="16" rx="2" />
      <rect x="5" y="6" width="6" height="4" rx="0.5" />
      <line x1="5" y1="14" x2="15" y2="14" />
      <line x1="5" y1="11.5" x2="13" y2="11.5" />
    </svg>
  );
}

interface DestTileProps {
  label: string;
  description: string;
  icon: () => Element;
  path: string;
}

/**
 * Cubo de destino: icono arriba, título debajo. La descripción aparece como
 * tooltip de UI propio (elemento <span role="tooltip">, no el title nativo)
 * en hover (con delay e intención) y en focus por teclado (inmediato).
 */
function DestTile(props: DestTileProps) {
  const navigate = useNavigate();
  const [open, setOpen]       = createSignal(false);
  const [instant, setInstant] = createSignal(false);
  let openTimer: ReturnType<typeof setTimeout> | null = null;

  function showByHover() {
    if (openTimer) { clearTimeout(openTimer); openTimer = null; }
    const skip = groupActive();           // grupo activo -> sin delay ni animación
    setInstant(skip);
    if (skip) {
      setOpen(true);
      keepGroupActive();
    } else {
      openTimer = setTimeout(() => {
        setOpen(true);
        keepGroupActive();
      }, OPEN_DELAY);
    }
  }

  function showByFocus() {
    if (openTimer) { clearTimeout(openTimer); openTimer = null; }
    setInstant(false);                     // teclado: siempre con animación
    setOpen(true);
  }

  function hide() {
    if (openTimer) { clearTimeout(openTimer); openTimer = null; }
    setOpen(false);
    scheduleGroupReset();
  }

  onCleanup(() => { if (openTimer) clearTimeout(openTimer); });

  return (
    <div class="relative" onMouseEnter={showByHover} onMouseLeave={hide}>
      <button
        onClick={() => navigate(props.path)}
        aria-label={`${props.label}. ${props.description}`}
        onFocus={showByFocus}
        onBlur={hide}
        class="flex flex-col items-center justify-center gap-3 w-44 h-36 rounded-md
               border border-border text-text-secondary
               hover:border-accent hover:text-text-primary hover:bg-bg-surface
               focus-visible:border-accent transition-colors"
      >
        <props.icon />
        <span class="text-sm font-semibold text-text-primary">{props.label}</span>
      </button>

      {/* Tooltip de UI (no nativo). Animación opacity + transform en .tooltip-popup,
          aislada de la transición global para no sentirse a bajos fps. */}
      <span
        role="tooltip"
        data-instant={instant() ? '' : undefined}
        classList={{ 'tooltip-open': open() }}
        class="tooltip-popup pointer-events-none absolute left-1/2 top-full z-20 mt-2
               whitespace-nowrap rounded-md border border-border bg-bg-panel px-2.5 py-1
               text-xs text-text-secondary"
      >
        {props.description}
      </span>
    </div>
  );
}

/** Componente de prueba del walking skeleton: tokens, toggle dual-theme y estado WS. */
export function Placeholder() {
  return (
    <div class="min-h-screen bg-bg-app flex flex-col">
      {/* Barra superior (chrome): marca a la izquierda, estado del Nano + tema a la derecha */}
      <header class="flex items-center justify-between border-b border-border px-6 h-14">
        <span class="flex items-center gap-2 font-semibold text-text-primary">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-accent" aria-hidden="true">
            <path d="M3 6h18" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
            <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            <line x1="10" x2="10" y1="11" y2="17" />
            <line x1="14" x2="14" y1="11" y2="17" />
          </svg>
          Tiny Trash
        </span>
        <div class="flex items-center gap-4">
          <NanoStatus />
          <ThemeToggle />
        </div>
      </header>

      {/* Hero centrado: encabezado, contexto y los tres destinos en fila */}
      <main class="flex-1 flex flex-col items-center justify-center px-6 text-center">
        <h1 class="text-2xl font-semibold tracking-tight text-text-primary">
          Elige una herramienta
        </h1>
        <p class="mt-2 max-w-md text-sm text-text-secondary">
          Clasificador de residuos en tiempo real sobre Jetson Nano.
        </p>

        <nav class="mt-10 flex flex-wrap items-center justify-center gap-5">
          <DestTile
            label="Guía de conexión"
            description="Cableado 3D interactivo ESP32, PCA9685 y servos"
            icon={GuiaIcon}
            path="/guia"
          />
          <DestTile
            label="Dashboard"
            description="Video en vivo y overlay de detección a 14 fps"
            icon={DashboardIcon}
            path="/dashboard"
          />
          <DestTile
            label="Engine del Modelo"
            description="Estado del modelo, build y logs en vivo"
            icon={EngineIcon}
            path="/engine"
          />
          <DestTile
            label="Labelling"
            description="Anotación de imágenes con drag, resize y export"
            icon={LabellingIcon}
            path="/labelling"
          />
        </nav>
      </main>

      {/* Footer: a qué Nano apunta la consola, sin protagonismo */}
      <footer class="px-6 py-4 text-center">
        <span class="font-mono text-xs text-text-secondary opacity-70">
          Nano · {nanoTarget()}
        </span>
      </footer>
    </div>
  );
}
