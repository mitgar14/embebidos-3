// web/src/lib/detection.ts
// Protocolo de detección del Nano y dibujo del overlay en canvas.
//
// Contrato WS (verificado contra el dashboard previo, fuente autoritativa):
//   browser -> Nano : frame JPEG binario (ArrayBuffer, sin envoltorio; el
//                     servidor cuenta los frames recibidos por conexión) y
//                     control JSON {type:"conf", value:0..1}.
//   Nano -> browser : JSON por frame con las bboxes en píxeles del frame
//                     enviado. {ok, seq, bboxes, t_infer_ms}. conf_ack y pong
//                     se ignoran (pong ya lo filtra ReconnectingWebSocket).

export type ClsName = 'glass' | 'paper' | 'plastic';

export interface BBox {
  x1: number; y1: number; x2: number; y2: number;
  cls_name: ClsName;
  conf: number;            // 0..1
}

export interface DetectionMsg {
  ok: boolean;
  seq: number;
  bboxes?: BBox[];
  t_infer_ms?: number;
}

// Paleta Wong (daltónico-segura), decisión canónica del proyecto.
export const WONG: Record<ClsName, string> = {
  glass:   '#56B4E9',   // sky blue
  paper:   '#E69F00',   // ámbar
  plastic: '#009E73',   // bluish-green
};

export const CLASS_LABEL_ES: Record<ClsName, string> = {
  glass:   'vidrio',
  paper:   'papel',
  plastic: 'plástico',
};

const BG_BLACK    = '#0a0a0b';   // fondo cuando se oculta la cámara
const LABEL_TEXT  = '#0b0b0c';   // texto sobre el pill de color (alto contraste)
const LABEL_FONT  = '600 13px "Geist Variable", system-ui, sans-serif';

/** Dibuja las bounding boxes en el overlay. Si showVideo es false, pinta el
 *  fondo negro para ver solo las cajas (la opacidad del <video> la maneja la UI). */
export function drawDetections(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  dets: BBox[],
  showVideo: boolean,
): void {
  ctx.clearRect(0, 0, W, H);
  if (!showVideo) {
    ctx.fillStyle = BG_BLACK;
    ctx.fillRect(0, 0, W, H);
  }

  ctx.lineWidth = 2;
  ctx.font = LABEL_FONT;
  ctx.textBaseline = 'alphabetic';

  for (const d of dets) {
    const color = WONG[d.cls_name] ?? '#ffffff';
    const x = d.x1, y = d.y1, w = d.x2 - d.x1, h = d.y2 - d.y1;

    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const labelEs = CLASS_LABEL_ES[d.cls_name] ?? d.cls_name;
    const label = `${labelEs} ${(d.conf * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 10;
    const th = 18;
    const ty = Math.max(th, y);

    ctx.fillStyle = color;
    roundRect(ctx, x, ty - th, tw, th, 3);
    ctx.fill();

    ctx.fillStyle = LABEL_TEXT;
    ctx.fillText(label, x + 5, ty - 5);
  }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/** Exporta un PNG con el frame actual (o fondo negro) y las bboxes superpuestas. */
export function exportSnapshot(video: HTMLVideoElement, overlay: HTMLCanvasElement, showVideo: boolean): void {
  const w = overlay.width, h = overlay.height;
  if (!w || !h) return;

  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  if (!octx) return;

  if (showVideo) {
    octx.drawImage(video, 0, 0, w, h);
  } else {
    octx.fillStyle = BG_BLACK;
    octx.fillRect(0, 0, w, h);
  }
  octx.drawImage(overlay, 0, 0, w, h);

  out.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `snapshot-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}
