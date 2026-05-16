# 04 · Builder TRT pipeline

El script `scripts/nano_build_engine.sh` (196 líneas) es el **orquestador del build TRT**. Lo invoca systemd cuando arranca la unit `embebidos3-builder@<job_id>.service`. Su trabajo: descargar el ONNX desde HF, optimizarlo a un engine TRT FP16, validar correctness, y hacer swap atómico contra el engine activo.

## Archivos relacionados

| Archivo | Propósito |
|---|---|
| `scripts/nano_build_engine.sh` | El orquestador |
| `scripts/hf_rest.py` | Cliente HF (ver doc 03) |
| `scripts/builder_state.py` | Escribe/finaliza `active_job.json` y meta del job |
| `scripts/validate_engine.py` | Carga el engine, corre N inferencias test contra `test_images/`, valida bboxes |
| `scripts/parse_trtexec_progress.py` | Parsea el stdout verboso de trtexec → porcentaje aproximado |
| `scripts/write_engine_meta.py` | Genera `best_fp16.engine.meta.json` desde el manifest + métricas del build |

## Las 13 fases del builder (ordenadas)

Cada fase escribe `phase=<name> pct=<n>` en `active_job.json` vía `builder_state.py phase`. El dashboard lo poll y actualiza el progreso.

| # | Phase | pct | Acción |
|---|---|---|---|
| 1 | `acquired_lock` | 5 | `flock -n` sobre `/run/embebidos3/builder.lock` |
| 2 | `downloaded_manifest` | 8 | `hf_rest.py download manifests/manifest.json` |
| 3 | parseo en bash | 10 | Extrae `HF_REV`, `EXPECTED_SHA`, `HF_COMMIT_DATE` |
| 4 | `downloaded_onnx` | 12 | `hf_rest.py download exports/best.onnx --revision $HF_REV` |
| 5 | `verified_sha` | 15 | `sha256sum` → comparar con `EXPECTED_SHA`. Si mismatch, exit 2 |
| 6 | `stopped_server` | 18 | `sudo systemctl stop embebidos3-server.service` + sleep 3 |
| 7 | `prep_nano` | 22 | Stop lightdm, disable nvzram, recrea swap 8 GB en `/mnt/swap.img`, `vm.swappiness=100`, drop_caches |
| 8 | `trtexec_started` → `trtexec_built` | 25→75 | Lanza `trtexec --fp16 --workspace=512 --buildOnly --verbose`, parsea progreso |
| 9 | `validating` → `validated` | 80→85 | `python3 validate_engine.py <staging>.engine` con 3 imágenes test |
| 10 | `backing_up_previous` → `backed_up_previous` | 88→92 | **Si existe `.previous/`**: copia local a `engines-archive/<ts>__<sha>/`, genera manifest con `write_archive_manifest.py`, sube solo el manifest (~1 KB) a HF vía `hf_rest.py upload` (NDJSON inline). Si HF falla → warning, no abort. Ver sección "Backup viejo (fase 10)" abajo |
| 11 | `swapping` → `swapped` | 94→95 | Atomic swap: `.engine` → `.previous/.engine.old`, `staging` → `.engine`. Idem `.meta.json`. Escribe `engines/best_fp16.engine.meta.json` con `write_engine_meta.py` |
| 12 | `restoring_nano` | 97 | `sudo systemctl start lightdm`, `vm.swappiness=60` |
| 13 | `starting_server` | 99 | `sudo systemctl start embebidos3-server.service` |
| ✓ | (finalize) | 100 | `builder_state.py finalize --phase done --exit-code 0` |

## El trap cleanup

El bloque más delicado del script. Garantiza que el server queda arriba pase lo que pase.

```bash
cleanup() {
    local code=$?
    # DESACTIVAR set -e dentro del cleanup. Sin esto, una linea falla y
    # set -e aborta el trap antes de llegar al systemctl start final.
    set +e
    [[ -n "${TEGRA_PID:-}" ]] && kill "$TEGRA_PID" 2>/dev/null
    rm -f "$STAGING_ENGINE" 2>/dev/null
    sudo systemctl start lightdm.service 2>/dev/null
    sudo sysctl vm.swappiness=60 >/dev/null 2>&1
    if [[ $code -ne 0 ]]; then
        echo "[BUILD] FAILED exit=$code" >&2
        JS finalize --phase failed --exit-code "$code" 2>/dev/null
    fi
    # ARRANQUE INCONDICIONAL del server: la idempotencia la da systemd.
    sudo systemctl start embebidos3-server.service 2>/dev/null
}
trap cleanup EXIT
```

### El bug que motivó el robustecimiento (2026-05-16)

Originalmente:
```bash
if [[ $code -ne 0 ]]; then
    ...
    systemctl is-active --quiet embebidos3-server.service \
        || sudo systemctl start embebidos3-server.service
fi
```

Pero el `set -euo pipefail` del header del script estaba activo dentro del trap. Cuando algún comando intermedio fallaba (typicalmente el `JS finalize` que invoca Python), `set -e` abortaba el trap antes de llegar al `start server`. Resultado: server caído al finalizar un build fallido, dashboard mostrando "Servidor no responde" indefinidamente.

Fix: `set +e` al inicio del cleanup + `start` incondicional al final (no en la rama condicional). systemd da la idempotencia: si el server ya está activo, no-op.

## El trap SIGTERM

```bash
cancel_handler() {
    echo "[BUILD] SIGTERM recibido, cancelando..." >&2
    pkill -KILL -P $$ trtexec 2>/dev/null || true
    JS finalize --phase cancelled --exit-code 130 2>/dev/null || true
    exit 130
}
trap cancel_handler SIGTERM
```

Cuando el usuario presiona "Cancelar build" en el UI:
1. UI hace `DELETE /jobs/<id>`
2. Server hace `sudo systemctl stop embebidos3-builder@<id>.service`
3. systemd manda `SIGTERM` al script
4. `cancel_handler` mata `trtexec` (que es el child más pesado) y finaliza el meta con `cancelled`
5. `exit 130` dispara el trap `cleanup` (que arranca el server)

## Backup viejo (fase 10)

Tras el fix de V-1 (2026-05-16), esta fase usa la estrategia **A+ híbrida**:

1. Si `engines/.previous/best_fp16.engine.old` existe, calcula su SHA256 completo (`OLD_SHA_FULL`) y los primeros 8 chars hex (`OLD_SHA`).
2. Crea `engines-archive/<TIMESTAMP>__<OLD_SHA>/` (TIMESTAMP en UTC `YYYYMMDDTHHMMSSZ`).
3. `cp -p` del engine y del `.meta.json` al archive (preserva mtimes y permisos).
4. `write_archive_manifest.py` genera `manifest.json` en el archive con: SHA256 completo, size, hardware (jetson-nano-b01, jetpack 4.6.1, trt 8.2.1.8, sm_53, fp16), `binary_present_remotely: false`, e inline el `.meta.json` origen si existe.
5. `hf_rest.py upload` sube **solo el manifest** (~1 KB) a `engines-archive/<TIMESTAMP>__<OLD_SHA>/manifest.json` en el repo HF. El binario nunca sale del Nano.
6. Si el upload a HF falla → `echo "[BUILD] WARN: ..."`, **NO abort**. La fuente de verdad del backup es el archive local.

### Por qué este diseño

El intento previo subía el binario completo (~13,5 MB) al endpoint `/commit/main` con shape JSON inventado. HF rechazaba con 400 porque el shape correcto es NDJSON (`application/x-ndjson`, líneas `header`+`file`/`lfsFile`), y además por encima de 10 MB el server fuerza LFS — el protocolo correcto requiere `preupload` + `lfs/objects/batch` + PUT S3 + verify, ~310-460 LOC en Py3.6 sin lib oficial. Para un MVP académico el costo-beneficio no se sostiene. Investigación completa en `investigaciones/2026-05-16-hf-hub-upload-binarios-python36-mvp.md`.

### Limitaciones conocidas

- El SD del Nano es la única copia del binario. Pérdida del SD = pérdida de backups históricos. Aceptable para MVP académico.
- Si se quiere backup binario remoto durable en el futuro, un script complementario corriendo **desde el laptop** (no desde el Nano) puede pullar `engines-archive/` via SSH y subirlo a GitHub Releases con `gh release create`. Ver V-1.1 en `VACIOS.md`.
- El storage del Nano crece monotónicamente con cada build (~13,5 MB por engine viejo). Cuando supere un umbral razonable, agregar tarea de housekeeping para conservar solo los últimos N archives.

## El swap atómico

```bash
# 10. swap
JS phase --name swapping --pct 94
mkdir -p "$PREV_DIR"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv -f "$ACTIVE_ENGINE" "$PREV_ENGINE"
fi
if [[ -f "$ACTIVE_META" ]]; then
    mv -f "$ACTIVE_META" "$PREV_META"
fi
mv -f "$STAGING_ENGINE" "$ACTIVE_ENGINE"
```

`mv` en el mismo filesystem es atómico (rename syscall). Si el script muere entre los `mv`, queda inconsistente — pero `recover_job_state.py` (ver doc 09) detecta el caso (`active_job.json` con phase `swapping`, mtime stale) y lo limpia.

Existe **un edge case no cubierto**: si el script muere DENTRO del `mv` del active al previous (`.engine.old`) pero antes del segundo `mv`, queda sin engine activo. La recovery actual no detecta esto. Apuntado en VACIOS.md.

## Validación post-build

`validate_engine.py` corre 3 imágenes de `test_images/` por el engine recién compilado, valida que detecte ≥1 bbox por imagen, y escribe un JSON con resultados:

```json
{
  "engine_path": "/home/jetson/embebidos-3/engines/.staging/best_fp16.engine.new",
  "imgsz": 416,
  "results": [
    {"image": "...000013...jpg", "detections": 1},
    {"image": "...000014...jpg", "detections": 2},
    {"image": "...000015...jpg", "detections": 1}
  ]
}
```

El script `nano_build_engine.sh` lo invoca con stdout redirigido a `$VAL_JSON`. Si crashea, exit 3 (fase `validation_failed`).

### Dependencia oculta: `nano_correctness.py`

`validate_engine.py` importa `from nano_correctness import letterbox, postprocess, IMGSZ, CLASSES, CONF_TH, NMS_TH`. Si ese archivo no está deployado al Nano, el script crashea con `ModuleNotFoundError` y el build queda en estado `validation_failed` con un JSON vacío. Esto pasó en producción hasta el deploy de 2026-05-16. **Recordatorio**: cualquier cambio en `nano_correctness.py` requiere `scp` al Nano.

## Costo en RAM y tiempo

Build típico:
- **Duración**: 20-30 min (trtexec --verbose alargado por el logging)
- **Pico de RAM**: ~3.4 GB (de 4 GB total). El swap 8 GB en disk previene OOM-kill.
- **Disk IO**: el swap thrashea fuerte al armar el plan de optimización. Por eso `vm.swappiness=100` durante el build y `60` (default) al terminar.
- **CPU**: 4 cores al 100% mientras corre trtexec

## Cómo monitorear un build en curso

```bash
# logs en vivo (tail -f)
ssh nano "tail -f /home/jetson/embebidos-3/logs/jobs/<job_id>.log"

# estado actual
cat /run/embebidos3/active_job.json

# tegrastats (CPU/GPU/RAM/temp samples)
tail -f /home/jetson/embebidos-3/logs/jobs/<job_id>.tegrastats.log
```
