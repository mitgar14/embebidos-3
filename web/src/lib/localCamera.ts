// web/src/lib/localCamera.ts
// Cliente WebSocket del modo LOCAL del Dashboard: consume /ws/local del Nano.
//
// Contrato /ws/local (congelado en 06-02, replicado aqui para no explorar el Nano):
//   Nano -> cliente, por cada frame DOS mensajes contiguos en orden estricto:
//     1) BINARIO (ArrayBuffer): bytes JPEG del frame 640x480 NATIVO (4:3, sin squish).
//     2) TEXTO JSON: {ok:true, bboxes:[...], t_infer_ms:<number>, seq:<number>}.
//        bboxes en PIXELES del frame 640x480 (no normalizadas, no 416).
//   Cliente -> Nano (opcional): texto {type:"conf", value:0..1}.
//   Errores Nano -> cliente (texto JSON, luego el Nano cierra el WS):
//     {ok:false, error:"camera_open_failed"}   la camara no abrio
//     {ok:false, error:"local_busy"}           ya hay un modo local activo
//
// El binario llega ANTES que su JSON: guardamos el ultimo ArrayBuffer pendiente y
// lo emparejamos con el siguiente JSON ok:true (orden estricto del Nano). Si llega
// un JSON sin frame pendiente lo ignoramos (desfase); si llegan dos binarios
// seguidos nos quedamos con el ultimo.
//
// Es un WebSocket CRUDO, no el ReconnectingWebSocket global del store remoto: su
// ciclo de vida esta atado al modo local del Dashboard y se cierra al cambiar de
// fuente o salir, para que el Nano libere /dev/video0. NO reutiliza ni toca el ws
// global remoto.
//
// Decodificacion: el JPEG se decodifica fuera del hilo principal (la API de bitmap
// del navegador). NO usamos canvas fuera de pantalla ni Web Worker (diferido a
// V2-02). El consumidor hace drawImage(bitmap, ...) + bitmap.close() y dimensiona
// su canvas a bitmap.width/bitmap.height (el frame es 640x480, NO cuadrado).

import { getWsUrl } from './ws';
import { nanoHttpBase } from '../stores/nanoStore';
import type { BBox } from './detection';

/** URL del WS local: toma getWsUrl() y reemplaza el sufijo final "/ws" por "/ws/local".
 *  Ej.: "ws://100.64.0.2:8000/ws" -> "ws://100.64.0.2:8000/ws/local".
 *  Si la URL configurada no terminara en "/ws" (caso no esperado), se anexa
 *  "/ws/local" de forma defensiva para no romper. */
export function localWsUrl(): string {
  const base = getWsUrl();
  if (base.endsWith('/ws')) return base.slice(0, -3) + '/ws/local';
  return base.replace(/\/+$/, '') + '/ws/local';
}

/** URL del fallback MJPEG del Nano (consumible con <img src>). */
export function mjpegUrl(): string {
  return nanoHttpBase() + '/camera/mjpeg';
}

/** Mensaje JSON por frame del Nano. ok:false viene con error (camera_open_failed / local_busy). */
export interface LocalFrameMsg {
  ok: boolean;
  bboxes?: BBox[];
  t_infer_ms?: number;
  seq?: number;
  error?: string;
}

export interface LocalCameraOpts {
  onFrame: (bitmap: ImageBitmap, msg: LocalFrameMsg) => void;
  onError: (error: string) => void;
  onClose: () => void;
  onOpen?: () => void;
}

export class LocalCameraClient {
  private _opts: LocalCameraOpts;
  private _ws: WebSocket | null = null;
  private _pendingFrame: ArrayBuffer | null = null;   // ultimo binario sin su JSON
  private _closed = false;                              // close() intencional: silencia callbacks

  constructor(opts: LocalCameraOpts) {
    this._opts = opts;
  }

  /** Abre el WS local (binaryType arraybuffer) y cablea los handlers. */
  connect(): void {
    const ws = new WebSocket(localWsUrl());
    ws.binaryType = 'arraybuffer';
    this._ws = ws;

    ws.onopen = () => {
      if (this._closed) return;
      this._opts.onOpen?.();
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this._closed) return;
      const data = (ev as MessageEvent).data as unknown;

      // Binario (ArrayBuffer): es un frame JPEG. Guardamos solo el ultimo pendiente.
      if (typeof data !== 'string') {
        this._pendingFrame = data as ArrayBuffer;
        return;
      }

      // Texto: JSON de bboxes o de error. JSON.parse en try/catch (entrada no confiable).
      let msg: LocalFrameMsg;
      try {
        msg = JSON.parse(data);
      } catch {
        return;   // JSON malformado: ignorar sin romper el bucle
      }

      // Error estructurado del Nano: no hay frame que emparejar; el Nano cerrara el WS.
      if (msg.ok === false) {
        if (msg.error) this._opts.onError(msg.error);
        this._pendingFrame = null;
        return;
      }

      if (msg.ok !== true) return;   // mensaje sin ok:true (ack u otro): ignorar

      // Emparejar el JSON ok:true con su frame binario pendiente.
      const frame = this._pendingFrame;
      this._pendingFrame = null;
      if (!frame) return;            // JSON sin frame pendiente (desfase): ignorar

      // Decodifica fuera del hilo principal; el consumidor cierra el bitmap tras dibujarlo.
      createImageBitmap(new Blob([frame], { type: 'image/jpeg' }))
        .then((bitmap) => {
          if (this._closed) { bitmap.close(); return; }
          this._opts.onFrame(bitmap, msg);
        })
        .catch(() => { /* JPEG corrupto: saltar este frame */ });
    };

    ws.onerror = () => {
      if (this._closed) return;
      this._opts.onError('ws_error');
    };

    ws.onclose = () => {
      if (this._closed) return;
      this._opts.onClose();
    };
  }

  /** Envia la certeza minima al Nano (solo si el WS esta OPEN). value en 0..1. */
  setConf(value01: number): void {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'conf', value: value01 }));
    }
  }

  /** Cierre intencional e idempotente: tras esto ningun callback vuelve a dispararse. */
  close(): void {
    this._closed = true;
    this._pendingFrame = null;
    const ws = this._ws;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      try { ws.close(); } catch { /* ya cerrado */ }
      this._ws = null;
    }
  }
}
