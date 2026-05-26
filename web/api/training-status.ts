// web/api/training-status.ts
//
// Proxy serverless (Vercel, Edge runtime) que lee los artefactos de
// entrenamiento desde un repo PRIVADO de HF Hub y los expone al frontend SIN
// filtrar el token. El dashboard SolidJS hace polling a /api/training-status
// cada ~30 s.
//
// Seguridad: requiere la env var HF_TOKEN en el proyecto Vercel. Debe ser un
// token fine-grained de SOLO LECTURA, con alcance al único repo de training.
// NUNCA se expone al cliente: el fetch a HF ocurre del lado servidor.
//
// Fuente de datos: el CommitScheduler del notebook publica en el repo del
// modelo (runs/heartbeat.jsonl, runs/detect/train/results.csv,
// manifests/manifest.json). manifest.json solo aparece cuando el run terminó.

export const config = { runtime: 'edge' };

// process.env en Edge runtime de Vercel (sin depender de @types/node).
declare const process: { env: Record<string, string | undefined> };

const REPO = 'mitgar14/embebidos-3-models-v1d';
const REPO_TYPE: 'model' | 'dataset' = 'model';
const BASE = `https://huggingface.co/${REPO_TYPE === 'dataset' ? 'datasets/' : ''}${REPO}/resolve/main`;

async function fetchText(path: string, token: string): Promise<string | null> {
  const r = await fetch(`${BASE}/${path}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: 'no-store',
  });
  if (!r.ok) return null; // 404 mientras el archivo aún no existe en el repo
  return r.text();
}

export default async function handler(_req: Request): Promise<Response> {
  const token = process.env.HF_TOKEN;
  if (!token) {
    return new Response(
      JSON.stringify({ ok: false, error: 'HF_TOKEN no configurado en el servidor' }),
      { status: 500, headers: { 'content-type': 'application/json' } },
    );
  }

  const [heartbeatRaw, manifestRaw, evalRaw] = await Promise.all([
    fetchText('runs/heartbeat.jsonl', token),
    fetchText('manifests/manifest.json', token),
    fetchText('manifests/eval_summary.json', token),
  ]);

  // heartbeat.jsonl: una línea JSON por tick (epoch, loss, grad_norm, lr,
  // gpu_mem_mb, elapsed_s, eta_s). Tomamos la última línea válida.
  let latest: unknown = null;
  if (heartbeatRaw) {
    const lines = heartbeatRaw.trim().split('\n').filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        latest = JSON.parse(lines[i]);
        break;
      } catch {
        // última línea puede estar a medio escribir; probar la anterior
      }
    }
  }

  const done = manifestRaw !== null;
  let manifest: unknown = null;
  if (manifestRaw) {
    try {
      manifest = JSON.parse(manifestRaw);
    } catch {
      /* manifest a medio commitear; se resuelve en el siguiente poll */
    }
  }

  // eval_summary.json: métricas finales (mAP50, mAP50_95, precision/recall medios
  // y por clase) sobre el split de validación. Lo escribe la sección de
  // evaluación del notebook al terminar, normalmente junto con manifest.json.
  // Sus claves son conocidas y estables, así que el frontend las consume directo.
  let evalSummary: unknown = null;
  if (evalRaw) {
    try {
      evalSummary = JSON.parse(evalRaw);
    } catch {
      /* a medio commitear; se resuelve en el siguiente poll */
    }
  }

  return new Response(
    JSON.stringify({
      ok: true,
      running: latest !== null && !done,
      done,
      latest,
      manifest,
      eval: evalSummary,
      ts: Date.now(),
    }),
    {
      headers: {
        'content-type': 'application/json',
        // El CDN de Vercel sirve la respuesta cacheada ~20 s y revalida en
        // background; si HF falla un instante, no rompe la UI.
        'cache-control': 's-maxage=20, stale-while-revalidate=40',
      },
    },
  );
}
