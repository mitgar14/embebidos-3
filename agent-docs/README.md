# embebidos-3 · documentación de arquitectura

Documentación interna del sistema **embebidos-3** (clasificador de residuos en Jetson Nano), enfocada en el subsistema de **pipeline dashboard + ciclo de vida del modelo**. Útil para retomar contexto entre sesiones o auditar el sistema sin leer todo el código.

Última actualización: 2026-05-16.

## Cómo usar esta documentación

Cada archivo cubre **un bloque funcional** del sistema y es independiente. El orden recomendado de lectura para alguien nuevo:

1. [01-systemd-units-sudoers.md](01-systemd-units-sudoers.md) — la base sobre la que todo se monta
2. [02-fastapi-server-trtworker.md](02-fastapi-server-trtworker.md) — el server FastAPI y el worker de inferencia TRT
3. [03-huggingface-integration.md](03-huggingface-integration.md) — cliente REST a HF Hub
4. [04-builder-trt-pipeline.md](04-builder-trt-pipeline.md) — el script orquestador del build TRT
5. [05-job-lifecycle-endpoints.md](05-job-lifecycle-endpoints.md) — endpoints HTTP que manejan jobs
6. [06-dashboard-modelo-tab.md](06-dashboard-modelo-tab.md) — pestaña "modelo" del dashboard
7. [07-ui-components-toast-modal-loading.md](07-ui-components-toast-modal-loading.md) — primitivas UI reutilizables
8. [08-visual-design-system.md](08-visual-design-system.md) — sistema visual (OKLCH, tipo, capitalización)
9. [09-recovery-watchdog.md](09-recovery-watchdog.md) — recuperación de estado y resiliencia
10. [10-validation-and-smoke-tests.md](10-validation-and-smoke-tests.md) — pruebas end-to-end y MCP

## Convenciones

- **Rutas absolutas en el Nano**: `/home/jetson/embebidos-3/...`
- **Rutas en el repo (Windows dev)**: `C:\Users\mitgar14\Documentos\embebidos-3\...`
- **Sudo password (entorno académico)**: `IAEmbebidos` — válida solo dentro de la red Tailscale del proyecto
- **Endpoint principal del server**: `http://100.100.166.120:8000` (Tailscale del Nano)

## Decisiones arquitectónicas globales

- **Proyecto académico MVP** — fluidez de la sustentación > robustez de producción
- **Python 3.6 en el Nano** (Ubuntu 18.04 ARM, JetPack 4.6) — incompatible con muchas APIs modernas (no `subprocess.run(capture_output=True)`, no f-string `=`, etc.)
- **systemd para orquestación** — el server, el builder y la recovery están en units `.service` separadas
- **trtexec FP16** — engine target es `best_fp16.engine`, workspace 512 MiB para caber en 4 GB RAM
- **HF Hub como single source of truth** — el modelo ONNX vive en `mitgar14/embebidos-3-models` y se descarga por commit SHA + verificación SHA256
- **Dashboard vanilla JS** — sin frameworks, servido como estáticos directamente

## Estado conocido al cierre de esta sesión (2026-05-16)

- ✅ Pipeline funcional end-to-end (build → validate → backup → swap → restart)
- ✅ UI con toast/modal/loading states robustos
- ✅ Builder con cleanup robustecido (server auto-restart incondicional)
- ✅ **V-1 resuelto** (2026-05-16) con estrategia A+ híbrida: backup local en `engines-archive/<ts>__<sha>/` + manifest a HF como índice remoto vía NDJSON inline. Sustentado en [`investigaciones/2026-05-16-hf-hub-upload-binarios-python36-mvp.md`](../investigaciones/2026-05-16-hf-hub-upload-binarios-python36-mvp.md).
- ⚠️ El histórico de jobs está placeholder ("próximamente")
- ⚠️ Sin circuit breaker en el polling del dashboard
- ⚠️ Sin autenticación en endpoints destructivos — confía en Tailscale como perímetro
- 📋 Vacíos restantes listados en [VACIOS.md](VACIOS.md) (V-2 a V-15 + V-1.1 follow-up backup binario remoto durable)

## Handoffs activos

Ninguno al cierre de esta sesión. El handoff V-1 (`HANDOFF-2026-05-16-fix-v1-backup-hf.md`) puede archivarse o eliminarse — quedó cerrado con la implementación documentada en VACIOS.md y la investigación referenciada arriba.
