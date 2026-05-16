# Brainstorming pipeline modelo — preguntas y respuestas

**Fecha:** 2026-05-15
**Relacionado:** `2026-05-15-dashboard-pipeline-request.md`
**Estado:** en curso.

---

## Pregunta 1 — Punto de partida del pipeline

> ¿Cuál es el punto de partida del pipeline que orquesta el dashboard? Es decir, ¿qué artefacto se asume que existe "al principio" cuando el usuario lanza la página?

Opciones presentadas:

- **A. `best.onnx` ya existe (en HF Hub o subido manualmente).** El dashboard solo verifica si el `.onnx` está presente y, si está, lo descarga al Nano y compila el `.engine`. La conversión `.pt → .onnx` queda fuera de scope (la sigue haciendo el notebook de training en Vast.ai). Recomendado: el `.onnx` ya se produce siempre en el notebook, está en HF, no necesitamos PyTorch en el Nano ni en el host.
- **B. `best.pt` en el host (Windows), dashboard hace `.pt → .onnx → .engine`.** El usuario tiene un `.pt` local en el portátil; el launcher local (Python 3.10 + ultralytics + uv) hace la exportación a `.onnx` ahí mismo y luego envía `.onnx` al Nano para compilar engine. Requiere añadir endpoints al `launch_demo.py` y deps ultralytics en `pyproject.toml` del host.
- **C. `best.pt` llega al Nano y todo el pipeline corre allí.** Bajo modelo `.pt` al Nano, instalo ultralytics + PyTorch en JP 4.6.1, hago la conversión y compilo engine. Es lo más invasivo: ultralytics moderno requiere Python 3.8+ y PyTorch para Jetson antiguo es complicado. NO recomendado.

**Respuesta:** **A — `best.onnx` ya existe (en HF Hub o subido manualmente).**

---

## Contexto adicional aportado por el usuario (manifest de HF Hub)

Sección `artifacts` del manifest oficial de HF:

```json
"artifacts": {
  "best_pt": {
    "path": "runs/detect/train/weights/best.pt",
    "sha256": "fabfec53481708a64b21640761312b68aa86f9e9bc6b5ccc0a9e85c8d1cbf4e0",
    "size_mb": 6.22
  },
  "best_onnx": {
    "path": "exports/best.onnx",
    "sha256": "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad",
    "size_mb": 12.17,
    "opset": 11,
    "ir_version": 6
  }
}
```

Cross-ref pendiente: comparar este SHA con el `.onnx` real que tiene la Nano (el usuario dice que el único ONNX corrido fue uno hecho "extraordinariamente fuera del notebook"; hay que confirmar si coincide o no con el de HF Hub).

---

## Pregunta 2 — Fuente del `.onnx`

> Si la verificación inicial detecta que el `.onnx` no está en el Nano, ¿de dónde lo trae el dashboard?

**Respuesta:** **A — Descarga desde HF Hub (`mitgar14/embebidos-3-models`).**

Implicación: el Nano necesita `huggingface_hub` (versión compatible Py3.6) y un `HF_TOKEN` si el repo es privado. Se debe descargar `exports/best.onnx` del último commit del HEAD del repo.

---

## Pregunta 3 — Disparador del pipeline

> ¿La compilación del `.engine` arranca automáticamente cuando el dashboard detecta que falta, o el usuario tiene que pulsar un botón?

**Respuesta:** **arranque automático cuando el dashboard detecta que falta**, con dos requisitos críticos:

1. **Ejecución en cola/background**: el proceso debe sobrevivir al cierre de la pestaña del navegador. La compilación corre en el Nano como un job persistente.
2. **Consultable**: el usuario debe poder volver a abrir el dashboard y ver el estado/avance del job activo. El avance debe ser **explícito** (no opaco).

Implicación: el server necesita un endpoint tipo `GET /jobs/<id>` o `GET /jobs/active` que devuelva estado, porcentaje, ETA y logs. El job ID se persiste server-side (no en localStorage del cliente).

---

## Pregunta 4 — Visibilidad del progreso

> ¿Cómo se muestra al usuario el avance de la compilación TRT?

**Respuesta:** **barra de progreso + logs en vivo**.

Implicación técnica: SSE o WebSocket dedicado para stream de logs (`stdout` de `trtexec`) + parser que extrae porcentaje/fase (TRT imprime fases tipo `[I] Building optimized engine`, `[I] Total Memory Usage`, etc). La barra puede ser aproximada (parse-based) o por tiempo transcurrido contra ETA fijo si parsing es frágil.

---

## Pregunta 5 — Alcance del "siempre activo"

> ¿"Siempre activo" significa solo que el servidor FastAPI sobreviva reboots, o también el dashboard del host?

**Respuesta:** **solo servidor del Nano**.

Implicación: systemd unit en `/etc/systemd/system/embebidos3-server.service` con `WantedBy=multi-user.target` + `Restart=on-failure` + `RestartSec=10`. `systemctl enable embebidos3-server` para auto-start al boot. El dashboard sigue siendo lanzado a demanda por el usuario en su portátil.

---

## Pregunta 6 — Manejo de versiones

> Si en HF Hub hay varios `.onnx` de entrenamientos sucesivos, ¿cuál descarga?

**Respuesta:** **último commit** (HEAD del repo).

Implicación: no se necesita selector de revision/tag. El dashboard llama a `huggingface_hub.hf_hub_download(repo_id="mitgar14/embebidos-3-models", filename="exports/best.onnx", revision="main")` siempre. Posible refinamiento futuro: mostrar el SHA del commit actual como metadata para trazabilidad.

---

## Pregunta 7 — UI en el dashboard

> ¿Dónde aparece el flujo en la página?

**Respuesta:** **una pestaña nueva**.

Implicación: el dashboard pasa de single-view a multi-view. Necesitamos una capa de routing simple (vanilla JS, hash-based o tabs visuales). Las pestañas previsibles: `live` (la actual con video + detección) y `modelo` (nueva, con estado del pipeline + acciones). La navegación entre tabs no debe interrumpir el job en curso.

---

## Pregunta 8 — Comportamiento si falla

> Si la descarga de HF falla o `trtexec` revienta, ¿bloquea o degrada?

**Respuesta:** **modo "sin inferencia"**.

Implicación: el dashboard sigue funcional. La pestaña `live` muestra el preview de la cámara local sin overlays de bbox, con un banner que indica "modelo no disponible — ver pestaña modelo". La pestaña `modelo` muestra el error y un botón para reintentar.

---

## Hallazgos exploratorios (auditoría 2026-05-15)

### HF Hub `mitgar14/embebidos-3-models` (private repo)

**Estructura real**:
- `exports/best.onnx` (12.169.740 B) — único ONNX productivo, en raíz no en `track_b/`.
- `runs/detect/train/weights/best.pt` (6.224.035 B) y `last.pt` (6.224.035 B, **tamaño idéntico a best.pt**, sospechoso de ser el mismo archivo o snapshots equivalentes).
- `runs/detect/train/`: 9 imágenes de batch (labels.jpg, train_batch0-2.jpg, val_batch0-1 labels+pred), `args.yaml`, `results.csv` (10.871 B), `heartbeat.jsonl` (42.311 B).
- `manifests/`: `manifest.json` (3.095 B), `eval_summary.json` (692 B), `gate3_onnx.json` (200 B). **`gate4_polygraphy.json` NO existe como archivo** — está embebido dentro de `manifest.json` como `gates.gate4_polygraphy: {skipped:true, reason:"recovery local sin Docker"}`.
- `track_a/` y `track_b/`: **solo 4 `.gitkeep` cada una, 0 bytes totales**. Scaffolding muerto del setup inicial. Ningún artefacto real ahí.
- `README.md` (1.287 B) y `.gitattributes` (2.097 B).

**Total storage**: 27,87 MB. **Commits**: 5, todos del 2026-05-14, último a las 18:38:31 UTC (`recovery: upload manifests/manifest.json`, hash `65c1634`). Super-squash del 18:20:36 wipeó toda la historia previa — no hay commits anteriores con artefactos distintos accesibles.

**Manifest.json — datos clave**: `incomplete_run: true`, "instancia Vast.ai destruida antes de cells 22-28 (post-hoc recovery)". El ONNX se regeneró el 2026-05-14 18:38:26 UTC en **Win11 + Intel Iris Xe CPU con torch 2.12.0+cpu** — **confirmado**: el ONNX productivo no salió del notebook canónico, fue una corrida local de recuperación. mAP50 test = 0,8891 / mAP50-95 = 0,6973. Classes `[glass, paper, plastic]`, imgsz 416, arch yolov8n.

**Deuda colateral del cleanup**: si borramos `track_a/` y `track_b/`, el README sigue describiendo arquitectura dual-track con tablas y bloque de estructura desactualizado. Hay que decidir si se actualiza también.

### Jetson Nano (`/home/jetson/embebidos-3/`)

**Estructura plana, 26 MB total**:
- `engines/best_fp16.engine` (13 MB, May 14 14:47) — SHA `a30f8f5f...` (hardware-specific, no comparable contra HF).
- `onnx/best.onnx` (12 MB, May 14 14:39) — SHA `223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad`. **MATCH EXACTO con el manifest de HF**. El ONNX local es bit-a-bit el mismo que el de HF Hub.
- `scripts/`: 4 scripts (`nano_install_inference.sh`, `nano_server.py`, `nano_start_server.sh`, `nano_stop_server.sh`) — coinciden 1:1 con `scripts/` del repo local. **Ningún script ad-hoc adicional**.
- `logs/`: `server.log` (135K, activo creciendo) y `trtexec_build.log` (30K, build histórico). NO existen `runs/`, `train*/`, ni JSON huérfanos.
- `test_images/` (268K).
- NO existen carpetas `models/`, `exports/`, `weights/`.

**Hallazgo importante sobre el build**: el `trtexec_build.log` revela que el engine fue compilado con `--workspace=512` (no 1024 como dice la doc del proyecto). El engine resultante de 13 MB funciona bien con 512 MB de workspace — significa que **hay margen para usar 1024 o quedarse en 512** según ergonomía vs riesgo OOM.

**Server activo**: PID 12211, vivo desde May 14 11:54, **83.454 inferencias acumuladas**, last_t_infer 39,65 ms, `gpu_temp_c: 30.0`, `ram_available_mb: 1487`, RSS del proceso 1,44 GB (36,4 % RAM). Lanzado por `nohup` (no systemd). Health endpoint responde correctamente.

**Recursos**: 40 GB libres en `/`, 1,5 GB RAM disponible, swap usado solo 39 MB de 1,9 GB. Temperaturas frías (CPU 29,5 °C, GPU 28,5 °C). No hay throttling térmico.

**Bug capturado en logs**: excepción recurrente en `nano_server.py:360` (`ConnectionClosedOK 1001 going away`) cuando el cliente cierra el WS durante el envío del JSON de error. Defensiva pero ya capturada por uvicorn — no bloquea operación.

**systemd actual**: ninguna unit relacionada (`ls /etc/systemd/system/embebidos*` = vacío). Confirma que la conversión a service es trabajo nuevo, no migración.

---

## Decisiones de cleanup HF Hub (pendientes de confirmación del usuario)

A partir de los hallazgos, propongo tres opciones de cleanup:

- **Cleanup mínimo**: solo borrar los 8 `.gitkeep` de `track_a/**` y `track_b/**` (el comando del subagente). README queda inconsistente pero el resto del repo limpio.
- **Cleanup completo**: borrar `.gitkeep` + reescribir README para reflejar arquitectura Track-B-exclusiva (alineado con el README del repo local, que ya está actualizado). Más prolijo.
- **Cleanup + organizar artefactos**: además de lo anterior, evaluar mover `runs/detect/train/weights/` a `models/` (más estándar HF) y eliminar `last.pt` si es duplicado de `best.pt`. Cambio estructural — afecta paths del manifest.

Recomendación: opción 2. Cero impacto en consumidores (el dashboard nuevo apuntará a las rutas reales, no al README), README queda alineado con el del repo, sin riesgo de cambiar paths.

---

## Constraints adicionales descubiertos

- El `huggingface_hub` SDK necesita compatibilidad con Python 3.6 (Nano). Última versión compat py36 ronda la 0.5.x — hay que verificar API disponible (`hf_hub_download` sí existe desde 0.0.x). Pendiente investigar al implementar.
- El engine actual usa workspace 512 MB, no 1024 — el comando canónico en la doc del proyecto está sobreespecificado. Se puede dejar 512 (validado) o subir a 1024 (más holgura para shapes mayores futuras).
- **Decisión arquitectónica abierta** (nueva, derivada de hallazgos):
  - El job de compilación TRT (15-45 min) **NO puede correr dentro del proceso `nano_server`**, porque mataría toda la concurrencia del server por minutos (`trtexec` consume GPU al 100 %, RAM 2-3 GB pico, y el server ya carga el engine activo en GPU). Esto implica que el job debe ser un **subproceso desacoplado** lanzado por el server (`subprocess.Popen` con PID file en `/tmp/embebidos3-trtexec.pid` o `/run/`), con el server actuando solo como orquestador.

---

## Pregunta 9 — Server vs build (contienda GPU/RAM)

> Si llega un nuevo `.onnx` desde HF, ¿qué hacemos con el server que está corriendo en ese momento? (`trtexec` 15-45 min, GPU 100 %, RAM 2-3 GB pico).

**Respuesta:** **parar server durante la compilación, reanudar al final**.

Implicaciones:
- Durante el job el dashboard `live` muestra el modo "sin inferencia" decidido en Q8 (banner que apunta a la pestaña `modelo`).
- El server necesita un mecanismo de **liberación del TRT context** antes de lanzar `trtexec` (destruir engine + context para liberar los ~250 MB que TRT reserva en GPU), si no, `trtexec` puede OOM en Maxwell.
- Después de compilar, el server necesita **recargar el engine** sin reiniciar el proceso entero (cleaner que un `exec` self-restart). El `TRTWorker` actual NO es reentrante en carga de engine — hay que refactorizarlo para soportar load/unload dinámico.
- Mientras el TRT context está liberado y `trtexec` corre, el server sigue vivo respondiendo `/health`, `/jobs/<id>` y rechazando frames del WS con `{"ok": false, "error": "building"}`. Un único proceso vivo, downtime de inferencia controlado.

Alternativa rechazada: dos daemons separados (server + status-only). Más complejidad sin beneficio claro.

---

## Pregunta 10 — Alcance del cleanup HF Hub

> ¿Qué alcance preferís para el cleanup que mencionaste sobre `track_a/` y `track_b/`?

**Respuesta:** **borrar los 8 `.gitkeep` + reescribir el README del HF Hub a Track-B exclusivo** (alineado con el README local del repo).

Implicaciones:
- Tarea de housekeeping post-aprobación del design. Cero impacto en la implementación del dashboard (las rutas reales `exports/best.onnx` y `runs/detect/train/weights/best.pt` no cambian).
- Se mantienen `last.pt` (pendiente verificar si es duplicado bit-a-bit de `best.pt` — si lo es, candidato a borrado en un cleanup futuro, fuera de scope ahora).
- Comando único: `hf repos delete-files mitgar14/embebidos-3-models track_a/checkpoints/.gitkeep track_a/exports/.gitkeep track_a/logs/.gitkeep track_a/runs/.gitkeep track_b/checkpoints/.gitkeep track_b/exports/.gitkeep track_b/logs/.gitkeep track_b/runs/.gitkeep`
- README: reescritura completa o sync directo desde el local (`huggingface-cli upload mitgar14/embebidos-3-models README.md`).

---

## Asunciones tomadas sin preguntar (verifico inline)

Para no fragmentar el flujo, las siguientes decisiones menores las asumo y se documentan en el spec final:

- **Bug `nano_server.py:360` (`ConnectionClosedOK` recurrente)**: se corrige en el mismo PR del pipeline porque el archivo se va a refactorizar extensivamente (job orquestador, endpoints nuevos, load/unload de engine). Fix trivial (~3 líneas: wrap del `send_text` en try/except).
- **Workspace `trtexec`**: usar **1024 MB** (no 512). Hay margen confirmado por la auditoría (40 GB libres, 1,5 GB RAM disponible con el server vivo). Más holgura es preferible para shapes mayores futuras y reduce riesgo de `BadAlloc` durante el build. Si en práctica revienta, fallback a 512.
- **Primer arranque del Nano sin engine**: el server arranca aunque no haya `.engine` (estado "no model"). Cuando el dashboard se conecta y detecta esa condición, ofrece descargar + compilar como flujo guiado. Esto se alinea con Q3 (arranque auto si falta) y Q8 (modo sin inferencia controlado).
- **Persistencia de metadata del engine actual**: el server escribe `engines/best_fp16.engine.meta.json` con `{onnx_sha256, hf_revision, hf_commit_date, build_started_at, build_completed_at, build_duration_s, trtexec_args}`. Permite cross-ref y trazabilidad para defensa académica.
- **Validación post-build (mini correctness)**: antes de declarar el engine recién compilado como "oficial", el server corre 1-3 imágenes de `/home/jetson/embebidos-3/test_images/` in-process contra el nuevo engine y verifica que produzca al menos N detecciones plausibles. Si pasa, swap; si no, conserva el engine anterior y reporta fallo en el job. Más rápido que volver a lanzar `trtexec` para validar.

---

## Pregunta 11 — Detección de updates en HF Hub

> ¿Cómo se entera el usuario de que apareció un nuevo entrenamiento en HF Hub?

**Respuesta:** **solo manual: botón "verificar actualizaciones" en la pestaña `modelo`**.

Implicaciones:
- Cero polling en background, cero tráfico HF sin pedido explícito.
- Pestaña `modelo` muestra siempre la metadata del engine actual (commit HF de origen + fecha de build).
- Acción "verificar actualizaciones" consulta `GET https://huggingface.co/api/models/mitgar14/embebidos-3-models/tree/main` y compara contra la revision guardada en la metadata. Resultado: "al día" o "nuevo commit disponible: abc1234, hace 2 h".
- Si hay novedad y el usuario hace click en "actualizar engine", se dispara el flujo `descargar .onnx → parar inferencia → trtexec → validar → reanudar`.
- El usuario controla el momento del downtime de 15-45 min. Alineado con perfil académico (visibilidad y control humano).

---

## Pregunta 12 — Gestión del HF_TOKEN en el Nano (repo privado)

> ¿Cómo guardamos el token para descargar el `.onnx` desde el repo privado?

**Respuesta:** **env var `HF_TOKEN` en el systemd unit (`EnvironmentFile=/etc/embebidos3/secrets.env`)**.

Implicaciones:
- Crear `/etc/embebidos3/secrets.env` con permisos `0600 root:jetson` (legible por el user del service, no por otros).
- Contenido: `HF_TOKEN=hf_xxxxxxxx` (token con permisos de read sobre `mitgar14/embebidos-3-models`).
- El systemd unit del server (y del job de build, si se separa) declara `EnvironmentFile=/etc/embebidos3/secrets.env`.
- Setup manual primera vez: el script de instalación deja el archivo con plantilla vacía y un mensaje de error claro si el token no está; el usuario lo pone con un editor. Cero credenciales hardcodeadas en el repo.
- Si después se quiere rotar, basta editar el archivo y `systemctl restart embebidos3-server`.

---

## Pregunta 13 — Robustez ante fallas de compilación y rollback

> ¿El sistema es capaz de devolverse a la versión anterior en caso de fallas?

**Respuesta:** sí, por diseño. Estrategia de robustez en capas:

### Capa 1 — No sobrescribir el engine vivo durante el build

- Engine activo: `engines/best_fp16.engine` + `engines/best_fp16.engine.meta.json`.
- Build nuevo escribe a un **path staging oculto**: `engines/.staging/best_fp16.engine.new`. El server jamás lo carga hasta validar.
- Si el build falla a mitad de camino (OOM, exit code != 0, kill por timeout), el engine vivo está intacto.

### Capa 2 — Validación post-build (mini-correctness in-process)

- Antes del swap, el server carga el `.new` en un TRT context temporal y corre 1-3 imágenes de `/home/jetson/embebidos-3/test_images/` (que ya existen, 268 KB).
- Criterios de aprobación: produce al menos N detecciones con confidence > 0,3 sobre las imágenes esperadas, sin NaN en outputs.
- Si falla la validación: descartar staging, mantener engine vivo, marcar job FAILED con detalle.

### Capa 3 — Swap atómico con backup

- Si valida: backup del engine actual → `engines/.previous/best_fp16.engine.old` + `.meta.json.old`.
- Luego `mv staging/best_fp16.engine.new engines/best_fp16.engine` (atómico en mismo filesystem).
- Server destruye el TRT context viejo y carga el nuevo. Si falla la carga del nuevo: rollback automático (restaurar desde `.previous/`).

### Capa 4 — Rollback manual desde el dashboard

- Pestaña `modelo` muestra "engine anterior disponible (compilado YYYY-MM-DD desde commit `abc1234`)" si existe `.previous/`.
- Botón "revertir a engine anterior" → swap inverso, recarga engine, deja el actual en staging por si el usuario quiere recuperarlo.

### Capa 5 — Auto-recovery en arranque del Nano

El server al arrancar (systemd `restart=on-failure` + boot enable):
1. Si `engines/best_fp16.engine` carga OK → modo normal.
2. Si falla la carga pero existe `engines/.previous/best_fp16.engine.old` válido → cargar el viejo + warning visible en `/health`.
3. Si ninguno carga → modo "no model", espera input del usuario.

### Capa 6 — Manejo de fallos de descarga HF

- Network/auth/checksum mismatch: el job nunca llega a parar el server. Job FAILED en fase "download", engine vivo intacto.
- Verificación SHA256 del `.onnx` descargado contra el manifest de HF (gate extra) antes de pasar a la fase de compilación.

### Capa 7 — Logs por job

- Cada job persiste `logs/jobs/<job_id>.log` con stdout/stderr de `trtexec`, timestamps de fases y resultado. El dashboard muestra logs históricos para diagnóstico.

**Estado:** estrategia de 7 capas aprobada por el usuario el 2026-05-15.

---

## Decisiones finales sobre las secciones del design (2026-05-15)

El usuario cerró la fase de Q&A con las siguientes decisiones rápidas:

### Sección 3 (endpoints + data flow)
- **Cancelación de jobs en curso: SÍ permitida.** Endpoint dedicado, hereda implicancia visual a la Sección 4 (botón "cancelar" en la pestaña `modelo` mientras `phase == building`).

### Sección 4 (UI pestaña `modelo`)
- Hereda lo anterior: botón "cancelar build" visible mientras hay job activo.

### Sección 6 (robustez + retención)
- **Engines viejos**: solo se conserva el último (el inmediatamente anterior) en `.previous/`.
- **Backup obligatorio antes de eliminar cualquier material**: antes de pisar/eliminar contenido de `.previous/` o cualquier engine antiguo, el sistema lo sube a HF Hub primero (preservación académica). Ruta sugerida: `engines-archive/<ISO_timestamp>__<engine_sha8>/` con el `.engine` + su `.meta.json`. Si la subida falla (red/auth/quota), NO se elimina nada local, el cleanup queda pendiente hasta que se desbloquee.
- **Logs**: `logs/jobs/<id>.{log,json}` con TTL de 3 días. Cron o `systemd-tmpfiles` que limpie mayores a 72 h.
- **Timeout de `trtexec`**: 40 minutos. Pasado ese umbral, matar proceso (SIGTERM → SIGKILL si no responde en 30 s) y marcar job FAILED con razón `timeout_40min`.
- **Recuperación si el server muere durante un build**: el builder es independiente (otra unit systemd), sigue corriendo. Cuando el server reinicia, lee `/var/run/embebidos3/job.json` y si encuentra `phase != done|failed|cancelled` + PID del builder vivo → **retoma el reporte de estado** al cliente sin interrumpir el job. Si PID muerto + phase no terminal → marca ABANDONED y limpia staging.

### Sección 7 (setup + migración)
- **Contador de inferencias acumuladas**: se reinicia a 0 al migrar al nuevo systemd. Las 83.454 actuales se pierden (decisión consciente, contador era best-effort en memoria).

---

## Vacíos técnicos detectados (a investigar antes de escribir el spec final)

El usuario pidió resolverlos vía `/investiga` profundidad alta con YouTube + AAI. Lista:

1. **Load/reload reentrante de TRT engine en pycuda 2019.1.2 + TensorRT 8.2.1.8** (Jetson Nano Maxwell sm_53, Python 3.6): patrones canónicos para destruir `ICudaEngine` + `IExecutionContext` y cargar otro sin leak de GPU memory; gotchas con `pycuda.driver.Context` push/pop, `mem_alloc` cleanup, refcounting TRT.
2. **`trtexec` parsing de progreso y manejo de timeout/cancelación**: ¿qué señales/líneas emite TRT 8.2 que permitan estimar progreso? Patrón canónico para matar `trtexec` con SIGTERM elegantemente sin corromper engine parcialmente escrito. Comportamiento ante OOM-killer.
3. **Atomic file swap sobre archivo abierto/mmap por otro proceso (TRT engine cargado)** en ext4 Linux: ¿es seguro `mv` un `.engine` nuevo encima de uno deserializado en memoria del server? Comparativa `rename` atómico vs `unlink+write+rename`.
4. **`trtexec` memory footprint real en Jetson Nano 4 GB con YOLOv8n@416 FP16**: RAM/VRAM pico con `--workspace=1024` vs `512`, mejores prácticas pre-build (`stop lightdm`, `drop caches`, swap config) y monitoreo para detectar OOM inminente.
5. **systemd templated services + sudoers granular wildcard**: ¿funciona `NOPASSWD: /bin/systemctl start embebidos3-builder@*.service`? Patrón canónico para disparar instancias templated arbitrarias sin password.
6. **`huggingface_hub` SDK en Python 3.6 (Jetson Nano)**: última versión compatible, API disponible (`hf_hub_download` con revision, upload con `create_commit`/`upload_file`), límites de tamaño, cache offline.
7. **Patrones de retomar jobs persistentes tras crash del orquestador**: cómo dramatiq/rq/sidekiq/celery detectan jobs huérfanos y los retoman vs marcan abandoned. Adaptación a un caso single-job con state file.

---

## Constraints duros identificados (no negociables, derivados del contexto)

- `.pt → .onnx` **no se hace en el Nano** (Python 3.6.9, sin ultralytics, sin PyTorch moderno). Ya resuelto por respuesta a Q1.
- `.onnx → .engine` se compila **solo en el Nano** (engines TRT no son portables entre devices distintos; Maxwell `sm_53` específico).
- Antes de compilar engine: `sudo systemctl stop lightdm` + drop caches (RAM pico 3.5 GB unificada — Maxwell, ultralytics issue #14751 reporta `Killed` por OOM).
- `trtexec --workspace=1024` (1 GiB), nunca más en Nano 4 GB.
- ONNX debe ser opset 11 (TRT 8.2 max opset 13; Ultralytics defaultea opset 17, hay que forzar). El manifest confirma `opset: 11, ir_version: 6` — compatible.
- Tiempo de compilación TRT FP16 en Nano para YOLOv8n@416: **15-45 min** (impacta UX del job: no es razonable bloquear UI, de ahí la decisión de job en background).
- Si el servidor systemd se reinicia mientras hay un job de compilación corriendo, el job se pierde (a menos que el job lo lance otra unit independiente). Esto requiere decisión: ¿el job de compilación es parte del proceso `nano_server`, o un subproceso desacoplado?
