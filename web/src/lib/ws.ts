// web/src/lib/ws.ts
/** ReconnectingWebSocket — vanilla, sin dependencias externas (~180 LOC).
 *  Frames JPEG binarios (ArrayBuffer) + control JSON.
 *  Protocolo Nano: {type:"ping"}→{type:"pong"}, {type:"conf"}→{type:"conf_ack"}.
 *  Fuentes: pladaria/reconnecting-websocket (MIT), zimv/WebSocketHeartBeat (MIT).
 */

export type WSStatus = 'connecting' | 'active' | 'reconnecting' | 'closed';

export interface RWSOptions {
  minDelay?:          number;
  maxDelay?:          number;
  growFactor?:        number;
  connectionTimeout?: number;
  minUptime?:         number;
  pingInterval?:      number;
  pongTimeout?:       number;
  binaryType?:        BinaryType;
  onStatusChange?:    (status: WSStatus) => void;
}

const DEFAULTS: Required<Omit<RWSOptions, 'onStatusChange'>> = {
  minDelay:          1000,
  maxDelay:          30000,
  growFactor:        1.5,
  connectionTimeout: 5000,
  minUptime:         3000,
  pingInterval:      20000,
  pongTimeout:       8000,
  binaryType:        'arraybuffer',
};

export class ReconnectingWebSocket extends EventTarget {
  static readonly CONNECTING = 0;
  static readonly OPEN       = 1;
  static readonly CLOSING    = 2;
  static readonly CLOSED     = 3;

  private _url:            string;
  private _opts:           Required<Omit<RWSOptions, 'onStatusChange'>>;
  private _onStatusChange: ((s: WSStatus) => void) | undefined;
  private _ws:             WebSocket | null    = null;
  private _retryCount:     number              = 0;
  private _messageQueue:   (string | ArrayBufferLike | Blob)[] = [];
  private _shouldReconnect = true;
  private _connectLock     = false;
  private _closeCalled     = false;
  private _connectTimer:   ReturnType<typeof setTimeout> | null = null;
  private _uptimeTimer:    ReturnType<typeof setTimeout> | null = null;
  private _pingTimer:      ReturnType<typeof setTimeout> | null = null;
  private _pongTimer:      ReturnType<typeof setTimeout> | null = null;
  private _status:         WSStatus = 'connecting';

  constructor(url: string, options: RWSOptions = {}) {
    super();
    this._url  = url;
    this._opts = { ...DEFAULTS, ...options };
    this._onStatusChange = options.onStatusChange;
    this._connect();

    // Heartbeat inmediato al volver a primer plano (laptop que despierta de sleep)
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this._ws?.readyState !== WebSocket.OPEN) {
        this._retryCount = 0;
        this._connect();
      }
    });
  }

  get readyState(): number {
    return this._ws?.readyState ?? ReconnectingWebSocket.CONNECTING;
  }

  get status(): WSStatus { return this._status; }

  send(data: string | ArrayBufferLike | Blob): void {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(data);
    } else {
      this._messageQueue.push(data);
    }
  }

  close(code = 1000, reason = ''): void {
    this._closeCalled     = true;
    this._shouldReconnect = false;
    this._clearTimers();
    this._ws?.close(code, reason);
    this._setStatus('closed');
  }

  reconnect(): void {
    this._shouldReconnect = true;
    this._closeCalled     = false;
    this._retryCount      = 0;
    this._ws ? this._ws.close(1000) : this._connect();
  }

  private _setStatus(s: WSStatus): void {
    if (this._status === s) return;
    this._status = s;
    this._onStatusChange?.(s);
  }

  private _getDelay(): number {
    const exp = this._opts.minDelay * Math.pow(this._opts.growFactor, this._retryCount);
    return Math.random() * Math.min(exp, this._opts.maxDelay);
  }

  private _connect(): void {
    if (this._connectLock || !this._shouldReconnect) return;
    this._connectLock = true;
    const delay = this._retryCount === 0 ? 0 : this._getDelay();

    setTimeout(() => {
      if (this._closeCalled) { this._connectLock = false; return; }

      this._setStatus(this._retryCount === 0 ? 'connecting' : 'reconnecting');
      const ws = new WebSocket(this._url);
      ws.binaryType = this._opts.binaryType;
      this._ws = ws;
      this._connectLock = false;

      // Timeout de conexión
      this._connectTimer = setTimeout(() => ws.close(), this._opts.connectionTimeout);

      ws.addEventListener('open', (e) => {
        clearTimeout(this._connectTimer!);
        this._uptimeTimer = setTimeout(() => {
          this._retryCount = 0;
        }, this._opts.minUptime);

        // Vaciar cola de mensajes pendientes
        const q = this._messageQueue.splice(0);
        q.forEach(m => ws.send(m));

        this._schedulePing();
        this._setStatus('active');
        this.dispatchEvent(Object.assign(new Event('open'), { originalEvent: e }));
      });

      ws.addEventListener('message', (e: MessageEvent) => {
        this._resetHeartbeat();
        // Filtrar pongs silenciosamente
        if (typeof e.data === 'string') {
          try {
            const parsed = JSON.parse(e.data);
            if (parsed?.type === 'pong') return;
          } catch { /* no es JSON — dejar pasar */ }
        }
        this.dispatchEvent(new MessageEvent('message', { data: e.data }));
      });

      ws.addEventListener('close', (e: CloseEvent) => {
        this._clearTimers();
        this.dispatchEvent(
          Object.assign(new Event('close'), { code: e.code, reason: e.reason })
        );
        if (this._shouldReconnect) {
          this._retryCount++;
          this._connect();
        }
      });

      ws.addEventListener('error', () => this.dispatchEvent(new Event('error')));
    }, delay);
  }

  private _schedulePing(): void {
    clearTimeout(this._pingTimer!);
    this._pingTimer = setTimeout(() => {
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'ping' }));
        this._pongTimer = setTimeout(() => this._ws?.close(), this._opts.pongTimeout);
      }
    }, this._opts.pingInterval);
  }

  private _resetHeartbeat(): void {
    clearTimeout(this._pongTimer!);
    this._schedulePing();
  }

  private _clearTimers(): void {
    clearTimeout(this._connectTimer!);
    clearTimeout(this._uptimeTimer!);
    clearTimeout(this._pingTimer!);
    clearTimeout(this._pongTimer!);
  }
}

// ─── URL del servidor — CONN-03 ────────────────────────────────────────────
const WS_URL_KEY  = 'nano_ws_url';
const DEFAULT_URL = 'ws://100.64.0.2:8000/ws';

export function getWsUrl(): string {
  return localStorage.getItem(WS_URL_KEY) ?? DEFAULT_URL;
}

/**
 * Persiste la URL del servidor WS en localStorage.
 * Valida el esquema antes de guardar (T-02-01).
 * @throws {Error} Si la URL no empieza por 'ws://' o 'wss://'
 */
export function setWsUrl(url: string): void {
  if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
    throw new Error('URL debe comenzar con ws:// o wss://');
  }
  localStorage.setItem(WS_URL_KEY, url);
}
