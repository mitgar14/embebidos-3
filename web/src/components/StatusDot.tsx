// web/src/components/StatusDot.tsx
// Punto de estado de 8px — sin texto, sin acento de borde.
// Los colores son semánticos de estado, no tokens del sistema de diseño.

export type DotStatus = 'active' | 'connecting' | 'reconnecting' | 'closed';

interface StatusDotProps {
  status: DotStatus;
}

/** Mapa de color por estado. */
const STATUS_COLORS: Record<DotStatus, string> = {
  active:       '#10b981',  // verde esmeralda
  connecting:   '#f59e0b',  // ámbar
  reconnecting: '#f59e0b',  // ámbar (parpadea)
  closed:       '#5a6169',  // Slate dark-8 — muted
};

/** Componente dot de 8px de diámetro que indica el estado de conexión. */
export function StatusDot(props: StatusDotProps) {
  const isAnimated = () => props.status === 'connecting' || props.status === 'reconnecting';

  return (
    <div
      class="w-2 h-2 rounded-full flex-shrink-0"
      style={{
        'background-color': STATUS_COLORS[props.status],
        animation: isAnimated() ? 'statusPulse 1.2s ease-in-out infinite' : undefined,
      }}
      role="status"
      aria-label={props.status}
    />
  );
}

// Keyframe de parpadeo inyectado como <style> global una sola vez
const style = document.createElement('style');
style.textContent = `
  @keyframes statusPulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
  }
  @media (prefers-reduced-motion: reduce) {
    @keyframes statusPulse { 0%, 100% { opacity: 1; } }
  }
`;
document.head.appendChild(style);
