// web/src/lib/platform.ts
// Detecta el SO del host basándose en navigator.platform (deprecated pero universal
// y disponible en HTTP, a diferencia de navigator.userAgentData que solo funciona en HTTPS).

export type OS = 'mac' | 'windows' | 'linux' | 'unknown';

/** Detecta el sistema operativo del usuario. */
export function detectOS(): OS {
  // Preferir userAgentData si está disponible (Chrome moderno, HTTPS)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const uadPlatform: string | undefined = (navigator as any).userAgentData?.platform;
  if (uadPlatform) {
    const p = uadPlatform.toLowerCase();
    if (p.includes('mac')) return 'mac';
    if (p.includes('win')) return 'windows';
    if (p.includes('linux')) return 'linux';
  }

  // Fallback: navigator.platform (deprecated pero funciona en HTTP local)
  const platform = navigator.platform.toLowerCase();
  if (platform.startsWith('mac')) return 'mac';
  if (platform.startsWith('win')) return 'windows';
  if (platform.startsWith('linux') || platform.includes('linux')) return 'linux';

  // Fallback: userAgent
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes('mac os x') || ua.includes('macintosh')) return 'mac';
  if (ua.includes('windows')) return 'windows';
  if (ua.includes('linux')) return 'linux';

  return 'unknown';
}

/** Retorna la tecla modificadora principal para el SO detectado. */
export function modifierKey(): string {
  return detectOS() === 'mac' ? '⌘' : 'Ctrl';
}
