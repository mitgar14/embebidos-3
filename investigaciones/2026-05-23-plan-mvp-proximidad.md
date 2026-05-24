# Plan MVP — Detector de proximidad / orden de caída

**Modo:** MVP (slice vertical). **Deadline:** demo 2026-05-26 (3 días).
**Deriva de:** [`2026-05-23-deteccion-proximidad-orden-caida.md`](2026-05-23-deteccion-proximidad-orden-caida.md) (investigación, Ronda 1) + grounding SSH del Nano.
**Decisión base:** homografía/IPM + tracker NumPy puro (iou-tracker) + scheduler de servos `smbus`→PCA9685. NO depth NN.

---

## User story

> **Como** sistema de clasificación de residuos sobre la banda,
> **quiero** saber en cada frame qué objeto detectado está más próximo a caer por el hueco (y el *time-to-fall* estimado de los demás como cola ordenada),
> **para** pre-posicionar el servo de la clase correcta antes de que el objeto llegue al hueco y reducir la latencia de actuación.

## Chequeo SPIDR → la historia es compuesta ⇒ se parte

La historia agrupa 5 capacidades (calibración geométrica · mapeo a la banda · tracking+velocidad · cola+timing · actuación de servos · viz) ⇒ **demasiado para un solo slice**. Eje elegido: **Interfaces + Rules** (de afuera hacia adentro, regla simple → completa). Split en **walking skeleton + 3 incrementos**:

| Slice | Entrega | Regla |
|---|---|---|
| **S0 — Walking skeleton** | Homografía calibrada + "next-to-fall" resaltado en el dashboard | nearest por distancia al hueco, **sin** tracking ni servos |
| **S1 — Cola con timing** | Tracker (identidad+velocidad) → `time_to_fall` → cola ordenada en el WS+viz | regla de cola por ETA |
| **S2 — Actuación** | Módulo servos `smbus`→PCA9685; pre-posición del servo de la clase del next-to-fall | actuación con `lookahead` |
| **S3 — Robustez+demo** | Tuning, fallbacks, dry-run de la demo | edge cases mínimos |

> Cada slice es demostrable por sí solo. Si el hardware de servos no está listo, S0+S1 ya muestran la lógica de decisión y la viz (S2 puede correr en modo "log/print" sin servos físicos).

---

## Arquitectura de integración (sobre lo que YA existe)

```
[browser app.js getUserMedia 640×480] --JPEG/WS--> [Nano TRTWorker]
                                                       _postprocess → bboxes {x1,y1,x2,y2,conf,cls,cls_name}
                                                       │
                                                       ▼  (NUEVO)
                                                  proximity.update(bboxes, ts)
                                                   ├─ homografía: bottom-center → (u,v) banda
                                                   ├─ iou-tracker: track_id + EMA velocidad
                                                   ├─ time_to_fall = (v_hueco - v_obj)/vel
                                                   └─ orden + next_to_fall
                                                       │
                          ┌────────────────────────────┼───────────────────────────┐
                          ▼                             ▼                           ▼
              WS JSON enriquecido           servo_controller (NUEVO)        dashboard app.js
              (+track_id,belt_xy,            smbus→PCA9685: pre-posición     drawDetections:
               dist_to_hole,ttf,             del canal de la clase del       línea del hueco,
               order,next_to_fall)           next-to-fall si ttf≤lookahead   highlight + cola/ETA
```

**Punto de enganche:** `TRTWorker.run()` en `scripts/server/nano_server.py`, justo después de `dets = self._postprocess(...)` (línea ~343). El estado del tracker vive como atributo de instancia del worker (hilo único secuencial). `dt` para velocidad = `client_ts_ms`/`t_recv_ms` por frame.

---

## Archivos (nuevos y editados)

| Archivo | Acción | Contenido |
|---|---|---|
| `scripts/server/proximity.py` | **NUEVO** | Homografía (load/apply), mapeo a banda, `IouTracker` (numpy puro), EMA velocidad, `time_to_fall`, orden de cola. Solo `numpy`/`cv2`/`scipy`. |
| `scripts/server/servo_controller.py` | **NUEVO** | Driver PCA9685 vía `smbus` (estilo waveshare) + mapa clase→canal→ángulo + scheduler de pre-posición. |
| `scripts/tools/calibrate_homography.py` | **NUEVO** | Calibración interactiva 4 puntos (`cv2.setMouseCallback`) sobre un snapshot → guarda `calibration/homography.json`. Corre en la laptop. |
| `scripts/tools/servo_smoke.py` | **NUEVO** | Smoke test en el Nano: `i2cdetect` 0x40, home + sweep de los 3 canales. Valida hardware antes de integrar. |
| `calibration/homography.json` | **NUEVO (dato)** | `H` 3×3 + geometría de banda (cm) + coord del hueco + resolución de captura + ángulos servo por clase. |
| `scripts/server/nano_server_constants.py` | EDIT | Constantes: paths de calibración, `SERVO_BUS=1`, `PCA9685_ADDR=0x40`, `SERVO_CHANNELS={glass:0,paper:1,plastic:2}`, `SERVO_HOME/DEFLECT`, `LOOKAHEAD_MS`, `IOU_TRACK_THR`, `EMA_ALPHA`, `DEFAULT_BELT_SPEED`. |
| `scripts/server/nano_server.py` | EDIT | Instanciar `Proximity` + `ServoController` en el worker; llamar `update()` tras `_postprocess`; enriquecer el JSON del WS; disparar servo (flag activable). |
| `scripts/dashboard/app.js` | EDIT | En `drawDetections`: dibujar línea del hueco, resaltar `next_to_fall`, panel con cola + ETA por objeto. |

---

## Tareas atómicas

### S0 — Walking skeleton (Día 1)
- **T0.1** Snapshot de calibración: capturar 1 frame del montaje real (dashboard "snapshot") a la resolución de captura. *(criterio: imagen guardada a la misma res que la inferencia)*
- **T0.2** `calibrate_homography.py`: click de 4 esquinas de un rectángulo de tamaño conocido sobre la banda → `cv2.getPerspectiveTransform` → guardar `H` + geometría + `v_hueco` + resolución en `homography.json`. *(criterio: re-proyectar un 5º punto conocido da error < ~2 cm)*
- **T0.3** `proximity.py` v0: cargar `H`; `belt_xy(bbox)=perspectiveTransform((cx,y2))`; `dist_to_hole`; devolver lista ordenada + `next_to_fall`. Sin tracking. *(criterio: con 2-3 objetos estáticos, marca correctamente el más cercano al hueco)*
- **T0.4** `nano_server.py`: llamar `proximity.update()` tras `_postprocess`; añadir `belt_xy/dist_to_hole/order/next_to_fall` al JSON del WS. *(criterio: el WS emite los campos nuevos sin romper el render existente)*
- **T0.5** `app.js`: dibujar línea del hueco + resaltar el bbox `next_to_fall`. *(criterio: en vivo, el objeto más cercano al hueco se resalta)*

### S1 — Cola con timing (Día 1 PM – Día 2 AM)
- **T1.1** `IouTracker` (numpy puro, ~80 LOC, base bochinski/iou-tracker MIT): asociación greedy por IoU, `track_id`, manejo de altas/bajas por `max_age`. *(criterio: IDs estables mientras el objeto cruza el frame)*
- **T1.2** Velocidad por EMA del `belt_xy` por track (`EMA_ALPHA`); fallback `DEFAULT_BELT_SPEED` si <3 muestras. *(criterio: velocidad converge y es positiva hacia el hueco)*
- **T1.3** `time_to_fall=(v_hueco - v_obj)/vel`; orden de cola por ttf; añadir `track_id/ttf` al WS. *(criterio: cola coherente; ttf decrece frame a frame)*
- **T1.4** `app.js`: panel lateral con la cola (clase + ETA ms) y orden. *(criterio: cola visible y actualizada en vivo)*

### S2 — Actuación (Día 2)
- **T2.1** `servo_smoke.py` en el Nano: `i2cdetect -y -r 1` (esperar 0x40), `set_pwm_freq(50)`, home + sweep de canales 0/1/2. *(criterio: los 3 SG90 se mueven; sin ello, S2 corre en modo log)*
- **T2.2** `servo_controller.py`: driver PCA9685 `smbus` (prescale 121 @50Hz; `set_servo_angle`); mapa clase→canal; `home_all()`. *(criterio: posiciona un canal a un ángulo dado)*
- **T2.3** Scheduler: cuando `next_to_fall.ttf ≤ LOOKAHEAD_MS`, pre-posicionar el servo de su clase (deflect), los otros en home; volver a home tras el paso. Flag `SERVO_ENABLED`. *(criterio: para 1 objeto, el servo correcto deflecta antes de la caída)*
- **T2.4** Integrar en el worker (llamada no bloqueante; el I²C es rápido pero proteger con try/except como el resto). *(criterio: no degrada el FPS por debajo de 10)*

### S3 — Robustez + demo (Día 3)
- **T3.1** Tuning: `conf`, `LOOKAHEAD_MS`, `EMA_ALPHA`, ángulos de deflect; recalibrar `H` si se movió la cámara.
- **T3.2** Fallbacks: sin `homography.json` → degradar a orden por `y2` en imagen (proxy) + warning; engine en standby → no actuar.
- **T3.3** Dry-run completo de la demo end-to-end con objetos reales de las 3 clases; checklist.

---

## Constantes / parámetros nuevos (MVP, ajustables)

```python
# nano_server_constants.py (añadir)
CALIBRATION_PATH   = ROOT / "calibration" / "homography.json"
SERVO_ENABLED      = False          # True cuando el hardware esté validado (T2.1)
SERVO_BUS          = 1              # I2C-1 (pines 3/5 del header 40-pin)
PCA9685_ADDR       = 0x40
SERVO_CHANNELS     = {"glass": 0, "paper": 1, "plastic": 2}
SERVO_HOME_DEG     = 90
SERVO_DEFLECT_DEG  = 30             # ajustar al mecanismo real
LOOKAHEAD_MS       = 400            # latencia SG90 supuesta (300-500)
IOU_TRACK_THR      = 0.3
TRACK_MAX_AGE      = 5              # frames sin match antes de cerrar track
EMA_ALPHA          = 0.4
DEFAULT_BELT_SPEED = None           # cm/frame; None → se exige ≥3 muestras
```

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Cámara en la laptop (browser), no en el Nano | La homografía se calibra sobre un snapshot **a la resolución de captura**; atada al montaje fijo. Si se mueve la cámara, recalibrar (T3.1). |
| Velocidad ruidosa por frame-drops (`queue maxsize=2`) | EMA + `TRACK_MAX_AGE` toleran huecos; fallback de velocidad por defecto. |
| Servos greenfield / wiring | **T2.1 smoke test primero**; `SERVO_ENABLED=False` por defecto → S0/S1 demostrables sin hardware; SG90 jitter → considerar liberar PWM tras el movimiento. |
| Tocar el Nano durante un build | Regla conocida: **NUNCA scp/edit de scripts del Nano con un build corriendo**. Verificar `/jobs/active=null` antes de desplegar; desarrollar en local y desplegar atómico. |
| Romper el pipeline de inferencia | Toda la lógica nueva va envuelta en try/except (patrón del server); si `proximity` falla, el server sigue devolviendo `bboxes` crudos. |
| Plazo 3 días | Orden S0→S3; S0+S1 son el corazón demostrable; S2 puede ir en modo log si el hardware no llega. |

## Definición de "Hecho" (demo-ready)
- Homografía calibrada y persistida; mapeo bbox→banda verificado.
- En vivo: `next_to_fall` resaltado + cola con ETA en el dashboard.
- Para 1 objeto en movimiento, el servo de su clase se pre-posiciona antes del hueco (o lo registra en modo log si `SERVO_ENABLED=False`).
- End-to-end ≥10 FPS; sin regresiones en el pipeline de detección existente.
- Todo en Python 3.6.9 con las libs ya presentes (numpy/cv2/scipy/smbus); sin deps nuevas pesadas.
```
