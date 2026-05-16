# Vacíos detectados del sistema embebidos-3 (pipeline dashboard)

Estado al cierre de sesión: **2026-05-16**.

Estos son los puntos donde el sistema documentado tiene huecos, bugs latentes, o decisiones explícitamente diferidas. Priorizados por gravedad operativa.

> **Cómo leer la prioridad:**
> - 🔴 **Crítico** — bloquea la operación normal del sistema
> - 🟠 **Alto** — degrada la experiencia significativamente o tiene riesgo de inconsistencia
> - 🟡 **Medio** — vacío relevante pero sin impacto inmediato
> - 🟢 **Bajo** — mejora deseada, no urgente

---

## ✅ V-1 · El paso `backing_up_previous` del builder rompe TODOS los builds con `.previous`

**Estado: RESUELTO 2026-05-16** mediante estrategia A+ híbrida (backup local autoritativo en `engines-archive/<ts>__<sha>/` del Nano + manifest `~1 KB` a HF como índice remoto via NDJSON inline). Investigación sustentadora en `investigaciones/2026-05-16-hf-hub-upload-binarios-python36-mvp.md`. Fix incluye:
- `scripts/hf_rest.py::upload_file_inline` reescrito con shape NDJSON correcto (`Content-Type: application/x-ndjson`, payload `header` + `file`) y guard de 10 MB para forzar uso solo en manifests pequeños.
- `scripts/write_archive_manifest.py` nuevo: genera manifest del archive con SHA + size + hardware.
- `scripts/nano_build_engine.sh` paso 9 reescrito: `cp` local + `write_archive_manifest` + upload solo del manifest a HF (warning, no abort, si falla).

Síntoma original (preservado para auditoría): todo build que encontraba un `engines/.previous/best_fp16.engine.old` abortaba con `exit 4` justo después de compilar y validar exitosamente. El engine recién compilado nunca se promocionaba a activo.

**Evidencia (encontrada hoy 2026-05-16 14:53 en `logs/jobs/20260516-1943-e2f6ff.log`):**

```
[state] phase=engine_built pct=70
&&&& PASSED TensorRT.trtexec [TensorRT v8201]
[state] phase=validating pct=80
PASS (3/3 imágenes con detecciones)
[state] phase=validated pct=85
[state] phase=backing_up_previous pct=88
Traceback (most recent call last):
  File "/home/jetson/embebidos-3/scripts/hf_rest.py", line 154, in <module>
    result = upload_file_inline(...)
  File "/home/jetson/embebidos-3/scripts/hf_rest.py", line 124, in upload_file_inline
    r.raise_for_status()
requests.exceptions.HTTPError: 400 Client Error: Bad Request
   for url: https://huggingface.co/api/models/mitgar14/embebidos-3-models/commit/main
[BUILD] backup HF falló, abort cleanup
[BUILD] FAILED exit=4
```

**Causa raíz (hipotética, requiere verificación):**

`scripts/nano_build_engine.sh:158-162` invoca `hf_rest.py upload` para subir el engine viejo al directorio `engines-archive/<timestamp>__<sha>/` del repo HF. El endpoint `POST /api/models/<repo>/commit/main` necesita un payload multipart específico que `upload_file_inline` no está construyendo correctamente (header-only? base64? estructura tree?). HF devuelve 400.

**Por qué nadie lo notó antes:**
- En los primeros builds no había `.previous` (sistema nuevo, sin engine viejo) → el bloque `if [[ -f "$PREV_ENGINE" ]]` se saltaba
- Después de adoptar el engine y compilarlo una vez exitosa, queda `.previous` poblado → el siguiente build dispara el upload → falla
- El usuario empezó a notarlo en esta sesión al hacer múltiples builds consecutivos

**Severidad:** crítico. **Imposibilita actualizar el modelo** una vez que ya hay un `.previous`. La única forma de progresar es `rm -rf engines/.previous/` manualmente en el Nano antes de cada build.

**Causa raíz confirmada (post-mortem):** el cliente enviaba `{"summary":..., "files":[{"path", "encoding":"base64", "content"}]}` con `Content-Type: application/json` al endpoint `/api/models/<repo>/commit/main`. Ese shape no existe en la API actual de HF — el endpoint espera NDJSON (`Content-Type: application/x-ndjson`) con líneas `{"key":"header","value":{...}}` + `{"key":"file","value":{...}}` o `{"key":"lfsFile","value":{...}}`. Además, para binarios ≥10 MB (engine TRT pesa 13,5 MB), HF responde `uploadMode:"lfs"` en el preupload y rechaza inline en cualquier shape — el flujo correcto sería preupload + lfs/objects/batch + PUT S3 + verify + commit con `lfsFile`, ~310-460 LOC funcional en Py3.6. Por ese desbalance costo-beneficio se eligió A+ híbrida (manifest chico) en lugar de implementar LFS completo (opción B).

**Referencia:** PR [#1117](https://github.com/huggingface/huggingface_hub/pull/1117) (introducción NDJSON), foro HF [113997](https://discuss.huggingface.co/t/hf-hub-commit-api-isnt-accepting-lfs-files/113997) (caso espejo con shape NDJSON confirmado), HfHub Elixir (threshold 10 MB independientemente verificado).

---

## 🟡 V-1.1 · Backup binario remoto durable (follow-up de V-1)

**Estado: backlog, no urgente.**

El fix de V-1 deja el binario solo en la SD del Nano (con manifest remoto como índice). Si la SD se corrompe, los engines históricos se pierden. Para MVP académico es aceptable; para preservar artefactos a largo plazo, considerar un script complementario que corra **desde el laptop** (no desde el Nano), pulle los `engines-archive/<ts>__<sha>/` del Nano via SSH y los suba a **GitHub Releases** con `gh release create <tag> <archive>/*`.

Ventajas de GitHub Releases para este caso (sustentado en `investigaciones/2026-05-16-hf-hub-upload-binarios-python36-mvp.md` sección Track A.2):
- Sin límite de storage para release assets.
- Sin límite de bandwidth para downloads (confirmado en GitHub community discussion [73875](https://github.com/orgs/community/discussions/73875)).
- No requiere implementar LFS ni tocar Py3.6 del Nano.
- Patrón inmune a la futura migración HF LFS → Xet.

Estimación: ~1 h de script Python en el laptop con `subprocess` a `gh release create`, sin tocar el Nano.

---

## 🔴 V-2 · Recovery no cubre swap interrumpido a mitad

**Síntoma:** si el builder muere por SIGKILL (OOM, system shutdown) **entre los dos `mv` del paso 10 (swap atómico)**, queda sin engine activo en disco. El estado va de `ACTIVE_ENGINE` poblado a `.previous` poblado pero `ACTIVE_ENGINE` vacío antes del segundo `mv` que pondría el `.staging` como activo.

**Código vulnerable** (`scripts/nano_build_engine.sh:172-178`):
```bash
rm -f "$PREV_ENGINE" "$PREV_META"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"     # ← punto 1
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
fi
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"      # ← punto 2 (no llega)
```

Si muere entre `← punto 1` y `← punto 2`, no hay `best_fp16.engine`. El server al arrancar ve `engine_binary_present=false` y va a estado `no_model`, pero hay un `.previous` válido sin usar.

**Recovery actual no lo detecta**: solo limpia `active_job.json`, no inspecciona la coherencia entre staging/active/previous.

**Severidad:** crítico cuando ocurre, pero baja probabilidad (window <1s). Pero un OOM en el Nano durante el swap es plausible.

**Fix:**
- Recovery al startup detecta: `engine_binary_present=false` AND `previous_engine_present=true` AND `last_job.phase=swapping` → automáticamente promueve `.previous` a active y marca `degraded`.
- O cambiar el swap por una secuencia con un solo `rename(2)` atómico usando symlinks (similar a `ln -sf nuevo current && atomic_rename current best_fp16.engine`).

---

## 🟠 V-3 · El polling del dashboard sigue durante el server down

**Síntoma:** cuando el server está abajo (durante un build), el dashboard sigue haciendo `fetch('/model/state')` cada 3s. El browser maneja los errores rápido (connection refused), pero genera ruido en devtools y log del Nano.

**Falta:** circuit breaker con backoff exponencial. Después de N fallos consecutivos, espaciar reintentos a 6s, 12s, 30s, max 60s.

**Severidad:** medio. No rompe nada, pero suma ruido y consume batería del laptop del operador.

---

## 🟠 V-4 · Sin autenticación en endpoints destructivos

**Síntoma:** cualquiera con acceso a la red Tailscale del proyecto puede `curl -X DELETE /jobs/<id>` o `curl -X POST /model/rollback`. No hay header de auth, no hay CSRF, no hay rate limit.

**Aceptable para MVP académico** porque Tailscale ya da perimetraje fuerte (sin invitación al tailnet, no hay acceso). **No aceptable para producción.**

**Fix candidato:** token estático por env (`X-Embebidos-Token: ...`) o HMAC del body. Implementación es trivial; queda diferido por scope.

---

## 🟠 V-5 · Sin watchdog de staleness para builds colgados

**Síntoma:** si `trtexec` cuelga sin progresar (caso reportado en otras Jetson Nano: optimizer infinite loop en ciertos onnx), no hay quien lo mate hasta `TimeoutStartSec=2700` (45 min) de systemd. El usuario espera ~45 min antes de poder lanzar otro build.

**Falta:** un watchdog en el server que detecte `active_job.updated_at` stale >5 min y proactivamente envíe SIGTERM al unit del builder.

`active_job.json` ya incluye `updated_at` (lo escribe `builder_state.py phase`). Solo falta el polling del server.

**Severidad:** medio. Caso patológico pero real.

---

## 🟠 V-6 · `nano_correctness.py` no está en CI ni en sync automático

**Síntoma:** cualquier cambio en `scripts/nano_correctness.py` (cambia letterbox, conf threshold, etc.) requiere `scp` manual al Nano. Si se olvida, `validate_engine.py` crashea con `ModuleNotFoundError` y todos los builds fallan silenciosamente.

**Pasó en producción** (2026-05-16): el archivo se creó local sin deploy → builds fallaban con exit 3 hasta que se hizo `scp`.

**Fix:** agregar `nano_correctness.py` al `nano_install_inference.sh` y al `bootstrap.sh`. O un `Makefile` con target `make deploy` que sincronice `scripts/` completo via rsync.

---

## 🟡 V-7 · Histórico de jobs es placeholder

**Síntoma:** la sidebar de la pestaña modelo muestra "Últimos jobs (próximamente)". Los datos existen (`logs/jobs/<id>.json` con phase, exit_code, duración), pero el frontend no los lee.

**Fix:** nuevo endpoint `GET /jobs/recent?limit=10` que liste los `.json` finalizados ordenados por mtime descendente. Frontend renderiza una mini-tabla con job_id, fecha, status (success/failed/cancelled), duración.

---

## 🟡 V-8 · `_generate_job_id` no es atomic-safe

**Síntoma:** si dos clientes hacen POST `/model/build` al mismo segundo, ambos generan job_id distintos (porque incluye 6 chars hex random), pero solo uno gana el lock. El perdedor recibe 409 con `active_job_id` del ganador → su job_id queda huérfano sin ningún meta. No es bug crítico (no escribe nada en disco), solo confunde si alguien hace `GET /jobs/<huerfano>` y recibe 404.

**Fix:** generar el job_id DENTRO del lock acquire, no antes. Implica reordenar el flujo del endpoint.

---

## 🟡 V-9 · El server no valida el manifest version

**Síntoma:** `nano_build_engine.sh` lee `manifest.json` asumiendo shape conocido. Si el formato cambia (e.g. `artifacts.best_onnx.path` cambia a `artifacts.onnx.path`), el build crashea en `KeyError` sin contexto.

**Fix:** validar `manifest.version == "1"` al inicio de la fase 3. Si no coincide, abortar con mensaje claro.

---

## 🟡 V-10 · El dashboard no muestra el changelog de HF al detectar update

**Síntoma:** cuando `check-updates` devuelve "Nueva iteración disponible", muestra los SHAs viejo→nuevo. Sería útil ver el commit message del nuevo HEAD ("v3 con augmentation rotada") para decidir si vale la pena recompilar.

**Falta:** `hf_rest.py` ya tiene `commit-info` que devuelve `message` y `author`. El check-updates podría incluirlo en la response y el toast mostrarlo.

**Severidad:** baja. UX nice-to-have.

---

## 🟡 V-11 · Sin healthcheck del worker reentrante post-swap

**Síntoma:** después de un swap (`worker.request_swap(new_path)`), no hay verificación de que el worker realmente cargó el engine nuevo. Si la carga falla (engine corrupto, falta CUDA mem), el worker queda en estado inconsistente y los próximos frames fallan en runtime.

**Falta:** el `request_swap` debería ser síncrono — esperar al ACK del worker que confirmó load OK. Si falla, hacer rollback inmediato del swap (filesystem-level) y propagar el error.

---

## 🟡 V-12 · El stream SSE de logs no se reanuda al reconectar

**Síntoma:** si el browser pierde la conexión SSE durante un build (network blip), no reintenta. El usuario queda mirando logs estancados aunque el build sigue corriendo en el Nano.

**Falta:** EventSource tiene reconexión automática nativa, pero el server SSE actual no soporta `Last-Event-ID` para resumir desde una línea específica. Hay que implementar tail-from-line, o aceptar pérdida de las líneas durante el blip y solo continuar desde el tail actual.

---

## 🟢 V-13 · No hay export del meta del modelo activo

**Síntoma:** si el operador quiere copiar la metadata del engine activo (commit, hashes, fecha build) a un report académico, tiene que screenshot del dashboard o curl + jq el endpoint. No hay un botón "Copiar info" o "Descargar como JSON".

**Severidad:** baja, cosmético.

---

## 🟢 V-14 · Cache-busting manual de assets

**Síntoma:** cada deploy de `style.css` o `modelo.js` requiere bumpear manualmente `?v=YYYYMMDD-N` en `index.html`. Olvidos = browsers sirven cache stale.

**Fix:** un mini script `build.sh` que calcule el hash de los assets y reemplace el version param. O servir con header `Cache-Control: no-cache` en development.

---

## 🟢 V-15 · No hay screenshot/grabación del live durante un build

**Síntoma:** mientras el build corre, no se puede ver el modelo "viejo" en acción para comparar antes/después. El server está caído.

**Fix posible:** el builder no podría dejar el server arriba (necesita la GPU). Una alternativa es bufferear los últimos 30s de inferencia antes del build y servirlos como video estático durante.

---

## Vacíos diferidos por scope (no entran al MVP)

- Integración CI/CD (pytest + lint en cada PR)
- Métricas Prometheus / Grafana
- Logs estructurados (ahora son texto plano)
- Auth real (OAuth, JWT)
- Multi-modelo (solo soportamos un engine activo a la vez)
- A/B testing entre dos engines
- Histórico de builds en HF Hub con metadata enriched

---

## Próximos pasos sugeridos (en orden)

1. ~~**Fix V-1 (backup-to-HF)**~~ → ✅ resuelto 2026-05-16 con A+ híbrida (manifest a HF, binario local).
2. **Fix V-2 (swap interrumpido)** → mejora de robustez, baja complejidad.
3. **Implementar V-5 (watchdog staleness)** → previene deadlocks de 45 min.
4. **Implementar V-6 (deploy script)** → previene el bug del `nano_correctness` que ya pasó.
5. **Implementar V-7 (histórico)** → completa la pestaña modelo.
6. **V-1.1 (GitHub Releases backup follow-up)** → solo si surge necesidad real de archivo binario remoto.
