// web/src/lib/mqtt.ts
// Cliente MQTT sobre WebSocket (WSS) para la página de control de servos.
// Sigue el molde de ReconnectingWebSocket (ws.ts): clase con estado observable,
// instancia a nivel de módulo en el store (servoStore.ts).

import mqttLib, { type MqttClient as MqttClientType, type IClientOptions } from 'mqtt';

export type MqttStatus = 'connecting' | 'active' | 'reconnecting' | 'closed';

export class MqttClient {
  private _client: MqttClientType;
  private _status: MqttStatus = 'connecting';
  private _onStatusChange?: (s: MqttStatus) => void;

  constructor(
    url: string = 'wss://broker.emqx.io:8084/mqtt',
    onStatusChange?: (s: MqttStatus) => void,
  ) {
    this._onStatusChange = onStatusChange;

    const opts: IClientOptions = {
      clientId: 'tiny-trash-' + Math.random().toString(16).slice(2, 8),
      clean: true,
      reconnectPeriod: 2000,
      connectTimeout: 30_000,
      keepalive: 60,
    };

    this._client = mqttLib.connect(url, opts);

    // Mapeo de eventos mqtt.js al vocabulario MqttStatus.
    // 'connect' se emite en la conexión inicial Y en cada reconexión automática
    // (con reconnectPeriod activo), por lo que suscribirse dentro de este handler
    // garantiza que los topics se re-registran tras cada reconexión (clean: true).
    this._client.on('connect', () => this._setStatus('active'));
    this._client.on('reconnect', () => this._setStatus('reconnecting'));
    this._client.on('close', () => this._setStatus('closed'));
    this._client.on('offline', () => this._setStatus('reconnecting'));
  }

  get status(): MqttStatus {
    return this._status;
  }

  subscribe(topic: string): void {
    this._client.subscribe(topic);
  }

  publish(topic: string, payload: string): void {
    this._client.publish(topic, payload);
  }

  /** Reexporta el .on del cliente mqtt para escuchar mensajes entrantes. */
  on(event: 'message', handler: (topic: string, payload: Buffer) => void): void {
    this._client.on(event, handler);
  }

  /** Detiene el cliente limpiamente (no dispara el Last Will). */
  destroy(): void {
    this._client.end(true);
  }

  private _setStatus(s: MqttStatus): void {
    if (this._status === s) return;
    this._status = s;
    this._onStatusChange?.(s);
  }
}
