# 01 · systemd units + sudoers

Esta capa define **qué corre como qué** en el Jetson Nano y **qué privilegios elevados se conceden** sin password al usuario `jetson`. Todo lo demás del sistema se monta sobre estas units.

## Archivos en el repo

| Archivo | Propósito |
|---|---|
| `systemd/embebidos3-server.service` | El proceso FastAPI/WS de inferencia |
| `systemd/embebidos3-builder@.service` | Template unit del builder TRT (`%i` = job_id) |
| `systemd/embebidos3-logs.tmpfiles.conf` | Crea `/run/embebidos3/` con permisos correctos al boot |
| `scripts/nano_install_systemd.sh` | Instalador idempotente que copia units, recarga systemd, configura sudoers |
| `scripts/embebidos3-builder-launch` | Wrapper validado para `sudo systemctl start embebidos3-builder@<job>.service` |

## En el Nano (filesystem)

- `/etc/systemd/system/embebidos3-server.service`
- `/etc/systemd/system/embebidos3-builder@.service`
- `/usr/lib/tmpfiles.d/embebidos3-logs.conf`
- `/usr/local/bin/embebidos3-builder-launch` (instalado con `install -m 0755`)
- `/etc/sudoers.d/embebidos3` — concede NOPASSWD a comandos específicos
- `/etc/embebidos3/secrets.env` — `HF_TOKEN=...` (modo 0600, root)
- `/run/embebidos3/` (tmpfs) — `builder.lock`, `active_job.json`

## Server unit

```ini
[Service]
ExecStart=/home/jetson/embebidos-3/scripts/nano_start_server.sh
User=jetson
Restart=on-failure
RestartSec=5
```

`Restart=on-failure` solo dispara si el proceso muere con exit code != 0. Si el server se detiene **limpiamente** (lo que el builder hace con `systemctl stop`), no se reinicia automáticamente. Por eso el builder tiene que arrancarlo explícitamente al final.

## Builder unit (templated)

```ini
[Service]
Type=oneshot
User=jetson
WorkingDirectory=/home/jetson/embebidos-3
EnvironmentFile=/etc/embebidos3/secrets.env
RuntimeDirectory=embebidos3
ExecStart=/home/jetson/embebidos-3/scripts/nano_build_engine.sh %i
TimeoutStartSec=2700
StandardOutput=file:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
StandardError=file:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
```

### El gotcha crítico: `Type=oneshot` y `systemctl start`

Por especificación de systemd, `systemctl start <unit-oneshot>` **bloquea hasta que el `ExecStart` termina**. Para un build TRT eso son 15-40 minutos.

Esto rompió el endpoint `/model/build` durante días: `subprocess.run(..., timeout=10)` lanzaba `TimeoutExpired`, no `CalledProcessError`, no estaba capturada, FastAPI devolvía 500 genérico que el frontend interpretaba como `launch_failed` aunque el build sí se había encolado.

**Fix definitivo** (2026-05-16): el wrapper `embebidos3-builder-launch` usa `systemctl start --no-block` para retornar inmediatamente. El endpoint también captura `TimeoutExpired` por defensa en profundidad.

## El wrapper `embebidos3-builder-launch`

```bash
#!/usr/bin/env bash
set -euo pipefail
JOBID="${1:-}"
if [[ ! "$JOBID" =~ ^[A-Za-z0-9_-]{10,40}$ ]]; then
    echo "JOBID inválido: '$JOBID'" >&2
    exit 1
fi
exec /bin/systemctl start --no-block "embebidos3-builder@${JOBID}.service"
```

Existe porque la regla sudoers concede `NOPASSWD: /usr/local/bin/embebidos3-builder-launch <arg>` con validación del arg, no `NOPASSWD: /bin/systemctl *` que sería un agujero de seguridad.

## Sudoers

Reglas otorgadas al user `jetson` sin password (instaladas por `nano_install_systemd.sh`):

```
jetson ALL=(root) NOPASSWD: /usr/local/bin/embebidos3-builder-launch *
jetson ALL=(root) NOPASSWD: /bin/systemctl start embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl disable nvzramconfig
jetson ALL=(root) NOPASSWD: /sbin/swapon /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/swapoff -a
jetson ALL=(root) NOPASSWD: /sbin/mkswap /mnt/swap.img
jetson ALL=(root) NOPASSWD: /usr/bin/fallocate -l 8G /mnt/swap.img
jetson ALL=(root) NOPASSWD: /bin/chmod 600 /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/sysctl vm.swappiness=*
jetson ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
```

### El gotcha de los paths

En Ubuntu 18.04 ARM (JetPack 4.6) `fallocate` está en `/usr/bin/fallocate`, **no** `/sbin/fallocate` como en distros más nuevas. Una versión anterior del sudoers tenía `/sbin/fallocate`, y el `prep_nano` del builder colgaba pidiendo password fantasma. Fix en commit `3a54df6`.

## Lock file y active_job

`/run/embebidos3/builder.lock` es un fd lockeado con `flock -n` por el builder. Garantiza un único builder concurrente. Se libera automáticamente al morir el proceso (no requiere cleanup).

`/run/embebidos3/active_job.json` lo escribe `builder_state.py` con el estado actual del job en curso. El server lo lee en `/model/state` y `/jobs/active`.

Ambos viven en tmpfs (`/run`) → se limpian solos al reboot. Ese es el comportamiento deseado: si reboot mata un build, no queda lock huérfano.

## Comandos de operación

```bash
# Ver units cargadas
systemctl status embebidos3-server.service
systemctl list-units 'embebidos3-builder@*'

# Restart manual del server
sudo systemctl restart embebidos3-server.service

# Reset de units en estado failed (para limpiar list-units)
sudo systemctl reset-failed 'embebidos3-builder@*'

# Cancelar un build manualmente
sudo systemctl stop 'embebidos3-builder@<job_id>.service'
```

## Decisiones que sobrevivieron

- **`Type=oneshot` en el builder** — válido para tener un estado terminal claro (active/failed/exited). Con `--no-block` el problema del bloqueo HTTP queda resuelto.
- **`RuntimeDirectoryPreserve=yes`** — el `/run/embebidos3/` se conserva entre restarts del unit (no entre reboots), útil para que el server pueda leer `active_job.json` si el builder lo escribió.
- **Logs separados**: el builder escribe `<job>.systemd.log` (capturado por StandardOutput) **y** `<job>.log` (escrito por `nano_build_engine.sh` con `tee`). El primero captura el wrapper systemd; el segundo el output crudo del script.
