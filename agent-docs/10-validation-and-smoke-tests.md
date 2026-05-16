# 10 · Validación end-to-end & smoke tests

Cómo se valida que cada cambio no rompe el sistema. Mix de tests automatizados, scripts manuales, y herramientas MCP que exponen el Nano a Claude Code.

## Niveles de validación

| Nivel | Qué cubre | Cuándo correr |
|---|---|---|
| Unit tests (pytest) | Lógica pura del server | Antes de commit |
| `validate_engine.py` | Correctness del engine TRT | Auto, al final de cada build |
| Playwright smoke | UI del dashboard | Cambios visuales/JS |
| Manual curl | Endpoints HTTP | Cambios al backend |
| MCP exposure | Inspección remota Claude Code | Debugging asistido |

## Unit tests

```bash
uv run pytest tests/ -v
```

Mock del filesystem (`tmpfs`), del HF client (responses estáticos), y del subprocess. No corren en CI (proyecto académico, sin CI configurado), pero existen para regression testing manual.

Cobertura actual:
- `tests/test_nano_server.py` — endpoints `/model/state`, `/model/check-updates`, `/model/adopt`, error paths
- `tests/test_builder_state.py` — escritura/lectura de `active_job.json`
- `tests/test_recover_job_state.py` — los 4 caminos de recovery (vivo, zombie, finalizado, ausente)

## `validate_engine.py` — validación automática post-build

Llamada por `nano_build_engine.sh` en la fase 9 (`validating`, pct 92). Si falla, el build aborta con exit 3 y no se hace swap.

Implementación:
1. Carga el engine en TRT context separado
2. Carga 3 imágenes test de `test_images/`
3. Corre inferencia + postprocess
4. Asserta ≥1 detección por imagen
5. Escribe JSON con resultados

Si el threshold de ≥1 falla en cualquier imagen, exit 1.

Esta es la red de seguridad contra engines compilados a partir de ONNX corrupto o builds parciales que pasan trtexec pero no detectan nada.

## Playwright smoke (`playwright-cli` skill)

```bash
playwright-cli -s=smoke open http://localhost:8001/
playwright-cli -s=smoke eval "..."   # mock state, trigger acción
playwright-cli -s=smoke screenshot --filename=foo.png
playwright-cli -s=smoke close
```

Usado iterativamente para validar:
- Render del template `ready` con datos reales
- Render del template `ready` con `adopted: true` (badge visible)
- Render del template `building` (progreso + logs)
- Renderdel card-warn ("Servidor no responde")
- Toast por severidad (info, success, warn, error)
- Modal danger + primary
- Spinner inline en botón loading

No automatizado (no hay test runner Playwright en el repo), pero los comandos están registrados en el historial. Reusable copiando.

## Manual curl

Útil para validar el backend sin pasar por el UI.

```bash
# health check
curl -sS http://100.100.166.120:8000/health | jq

# estado del modelo
curl -sS http://100.100.166.120:8000/model/state | jq

# lanzar build
curl -sS -X POST http://100.100.166.120:8000/model/build \
  -H 'Content-Type: application/json' \
  -d '{"force":false}' | jq

# verificar updates contra HF Hub
curl -sS -X POST http://100.100.166.120:8000/model/check-updates | jq

# stream de logs SSE
curl -N http://100.100.166.120:8000/jobs/<job-id>/logs

# cancelar job
curl -sS -X DELETE http://100.100.166.120:8000/jobs/<job-id> | jq
```

Timing-sensitive: ver `time_total` con `-w "TIME=%{time_total}s\n"` para detectar regresiones (ej. el fix del `--no-block` bajó `/model/build` de 10s → 0.1s).

## Acceso SSH al Nano

Configurado en `~/.ssh/config` del equipo dev:

```
Host nano
    HostName 100.100.166.120
    User jetson
    IdentityFile ~/.ssh/id_ed25519_nano
```

Comandos útiles:
```bash
# logs del server
ssh nano "echo IAEmbebidos | sudo -S journalctl -u embebidos3-server.service -n 100 --no-pager"

# logs del último builder
ssh nano "tail -200 /home/jetson/embebidos-3/logs/jobs/$(ls -t /home/jetson/embebidos-3/logs/jobs/ | grep '\.log$' | head -1)"

# units cargadas
ssh nano "systemctl list-units 'embebidos3-*'"

# RAM y procesos top
ssh nano "free -m; ps aux --sort=-%mem | head -10"

# tegrastats (one-shot)
ssh nano "sudo -n tegrastats --interval 1000 --logfile /dev/stdout | head -3"
```

## MCP exposure

El proyecto expone el Nano via un MCP server custom (task #21 cerrada) para que Claude Code pueda ejecutar comandos remotos sin que el operador copy-paste. Útil durante debugging asistido.

Tools expuestas:
- `nano_ssh_exec` — comando arbitrario via SSH
- `nano_status` — estado consolidado (server, builder, recent logs)
- `nano_tail_log <jobid>` — tail del log de un job

## CI/CD: no hay

Proyecto académico. El workflow es:
1. Cambios en local (Windows)
2. `scp` o `rsync` a Nano para scripts/server
3. `sudo systemctl restart embebidos3-server.service` si tocaste el server
4. Refresh del browser (cache-bust con `?v=...`)
5. Smoke manual via UI o curl

## Reglas no-negociables al deploy

- **Cualquier cambio en `nano_correctness.py`** requiere `scp` al Nano. Es importado por `validate_engine.py`.
- **Cambios en `nano_server.py`** requieren `systemctl restart embebidos3-server.service`. uvicorn reload no está habilitado.
- **Cambios en `embebidos3-builder-launch`** requieren `sudo install -m 0755 ... /usr/local/bin/embebidos3-builder-launch`. Es un binario privilegiado (sudoers).
- **Cambios en sudoers** requieren `visudo -c` antes de aplicar (sintaxis).
- **Cambios en unit files** requieren `sudo systemctl daemon-reload` + `restart` del unit afectado.

## Test plan que se viene siguiendo (Fase H)

| Test | Estado |
|---|---|
| Smoke visual via Playwright (templates ready/building/no_model) | ✅ |
| `curl /model/state` → JSON válido | ✅ |
| `curl /model/build` → 202 con job_id, 409 si hay otro | ✅ |
| Stream SSE de logs sin colgar el browser | ✅ post-fix RAF |
| Cancelación de build con SIGTERM | ✅ |
| Rollback restaura `.previous` y marca degraded | ✅ |
| Recovery startup limpia job zombie | ✅ unit-tested |
| Reboot Nano: server arranca solo, dashboard reconnecta | 📋 pendiente |
| Build E2E completo con engine real | 🟡 en curso al cierre de sesión |
| Smoke completo post-rollback (rollback → build → rollback) | 📋 pendiente |
