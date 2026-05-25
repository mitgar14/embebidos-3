// web/src/lib/labelStore.ts
// Persistencia de la sesión de Labelling en IndexedDB para sobrevivir recargas
// o cierres accidentales. Ventana de recuperación de 30 minutos desde la última
// actividad: al reabrir dentro de ese plazo el estado se restaura tal cual; si
// expiró, se descarta. IndexedDB (no localStorage) porque hay que guardar los
// File de las imágenes como Blob, lo que excede el límite de localStorage.
//
// Dos object stores:
//   blobs: id -> { name, blob, w, h }   (pesado; se escribe solo al cargar imágenes)
//   meta:  'current' -> { savedAt, idx, order, boxes }   (liviano; se reescribe en cada edición)

const DB_NAME = 'tiny-trash-labelling';
const DB_VERSION = 1;
const BLOBS = 'blobs';
const META = 'meta';
const META_KEY = 'current';
const TTL_MS = 30 * 60 * 1000; // 30 minutos

export interface StoredBox { x: number; y: number; w: number; h: number; cls: string; }
interface BlobRec { name: string; blob: Blob; w: number; h: number; }
interface MetaRec { savedAt: number; idx: number; order: string[]; boxes: Record<string, StoredBox[]>; }

export interface RestoredItem { id: string; name: string; blob: Blob; w: number; h: number; boxes: StoredBox[]; }
export interface RestoredSession { idx: number; items: RestoredItem[]; }

let dbPromise: Promise<IDBDatabase> | null = null;
function db(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const d = req.result;
        if (!d.objectStoreNames.contains(BLOBS)) d.createObjectStore(BLOBS);
        if (!d.objectStoreNames.contains(META)) d.createObjectStore(META);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbPromise;
}

// Ejecuta una operación dentro de una transacción y resuelve con su resultado.
function run<T>(store: string, mode: IDBTransactionMode, op: (s: IDBObjectStore) => IDBRequest): Promise<T> {
  return db().then(
    (d) =>
      new Promise<T>((resolve, reject) => {
        const tx = d.transaction(store, mode);
        const req = op(tx.objectStore(store));
        req.onsuccess = () => resolve(req.result as T);
        req.onerror = () => reject(req.error);
      }),
  );
}

export async function saveBlob(id: string, rec: BlobRec): Promise<void> {
  await run(BLOBS, 'readwrite', (s) => s.put(rec, id));
}

export async function saveMeta(meta: MetaRec): Promise<void> {
  await run(META, 'readwrite', (s) => s.put(meta, META_KEY));
}

export async function clearSession(): Promise<void> {
  await run(META, 'readwrite', (s) => s.clear());
  await run(BLOBS, 'readwrite', (s) => s.clear());
}

// Lee la sesión guardada. Si no hay, o si expiró (>30 min), devuelve null
// (y limpia lo expirado). Une la metadata liviana con los blobs por id/orden.
export async function loadSession(): Promise<RestoredSession | null> {
  const meta = await run<MetaRec | undefined>(META, 'readonly', (s) => s.get(META_KEY));
  if (!meta) return null;
  if (Date.now() - meta.savedAt > TTL_MS) {
    await clearSession();
    return null;
  }
  const items: RestoredItem[] = [];
  for (const id of meta.order) {
    const rec = await run<BlobRec | undefined>(BLOBS, 'readonly', (s) => s.get(id));
    if (rec) items.push({ id, name: rec.name, blob: rec.blob, w: rec.w, h: rec.h, boxes: meta.boxes[id] ?? [] });
  }
  return { idx: meta.idx ?? 0, items };
}
